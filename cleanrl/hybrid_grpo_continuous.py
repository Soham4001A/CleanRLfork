# Hybrid GRPO (True Node Sampling) for Continuous Actions – CleanRL-style
# Shadow-env probing + split buffers:
# - Sample K actions; probe one-step rewards on synchronized SHADOW envs (identical wrappers) and restore shadow base.
# - Tanh-normalize rewards across K -> per-sample policy advantages (no re-normalization in updates; K=1 uses GAE).
# - Advance real env only with the executed action (argmax or softmax over standardized rewards).
# - VALUE LOSS trains on the K-probe **expected Bellman target** per state: E_a[r + gamma * V(s')] (Huber loss).
# - POLICY trains on B*K samples; VALUE trains on B samples.
#
# Logging keys match PPO continuous (no extras).

import os
import random
import time
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import tyro
import glob
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter

# Optional: custom optimizers
try:
    from optim.sgd import AlphaGrad, DAG
except Exception:
    AlphaGrad, DAG = None, None


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    seed: int = 1
    torch_deterministic: bool = True
    cuda: bool = True
    track: bool = True
    wandb_project_name: str = "cleanRL"
    wandb_entity: str = None
    capture_video: bool = False
    save_model: bool = False
    upload_model: bool = False
    hf_entity: str = ""
    optimizer: str = "Adam"   # "Adam" | "AlphaGrad" | "DAG"
    alpha: float = 0.0        # AlphaGrad

    # Algo
    env_id: str = "HalfCheetah-v5"
    total_timesteps: int = 1_000_000
    learning_rate: float = 3e-4
    num_envs: int = 1
    num_steps: int = 2048
    anneal_lr: bool = True
    gamma: float = 0.99
    gae_lambda: float = 0.95
    num_minibatches: int = 32
    update_epochs: int = 10
    norm_adv: bool = True     # applies to realized branch only when K=1; K-adv is not re-normalized here
    clip_coef: float = 0.2
    clip_vloss: bool = True
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: float = None

    # Hybrid GRPO specifics
    num_node_samples: int = 4       # K
    tanh_temperature: float = 1.0   # tau for tanh(z)
    eps_norm: float = 1e-8
    exec_temperature: float = 0.0   # 0 => argmax, >0 => softmax over standardized rewards

    # runtime (filled in main)
    batch_size: int = 0
    minibatch_size: int = 0
    num_iterations: int = 0
    batch_size_policy: int = 0
    minibatch_size_policy: int = 0


def make_env(env_id, idx, capture_video, run_name, gamma):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, video_folder=f"videos/{run_name}")
        else:
            env = gym.make(env_id)

        # match PPO stack
        env = gym.wrappers.FlattenObservation(env)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.wrappers.ClipAction(env)
        env = gym.wrappers.NormalizeObservation(env)
        env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10))
        env = gym.wrappers.NormalizeReward(env, gamma=gamma)
        env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))
        return env
    return thunk


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        obs_dim = int(np.array(envs.single_observation_space.shape).prod())
        act_dim = int(np.prod(envs.single_action_space.shape))
        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, act_dim), std=0.01),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, act_dim))

    def get_value(self, x):
        return self.critic(x)

    def dist_params(self, x):
        mean = self.actor_mean(x)
        logstd = self.actor_logstd.expand_as(mean)
        std = torch.exp(logstd)
        return mean, std

    def get_dist(self, x):
        mean, std = self.dist_params(x)
        return Normal(mean, std)


# ---------- Low-level state copy helpers ----------
def _get_base(env):
    cur = env
    while hasattr(cur, "env"):
        cur = cur.env
    return cur

def _snapshot_base_env(base):
    snap = {}
    if hasattr(base, "np_random") and hasattr(base, "np_random") and hasattr(base.np_random, "bit_generator"):
        snap["rng"] = base.np_random.bit_generator.state
    if hasattr(base, "state"):
        s = base.state
        snap["state"] = np.array(s, copy=True) if isinstance(s, np.ndarray) else (s.copy() if hasattr(s, "copy") else s)
    for attr in ("qpos", "qvel", "qacc"):
        if hasattr(base, attr):
            try:
                snap[attr] = np.copy(getattr(base, attr))
            except Exception:
                pass
    for attr in ("steps_beyond_terminated", "t"):
        if hasattr(base, attr):
            snap[attr] = getattr(base, attr)
    return snap

def _restore_base_env(base, snap):
    if "rng" in snap and hasattr(base, "np_random") and hasattr(base.np_random, "bit_generator"):
        base.np_random.bit_generator.state = snap["rng"]
    if "state" in snap and hasattr(base, "state"):
        base.state = np.array(snap["state"], copy=True)
    for attr in ("qpos", "qvel", "qacc"):
        if attr in snap and hasattr(base, attr):
            try:
                getattr(base, attr)[:] = snap[attr]
            except Exception:
                pass
    for k, v in snap.items():
        if k in ("rng", "state", "qpos", "qvel", "qacc"):
            continue
        if hasattr(base, k):
            try:
                setattr(base, k, v)
            except Exception:
                pass

def _copy_rms(dst_rms, src_rms):
    for name in ("mean", "var"):
        if hasattr(dst_rms, name) and hasattr(src_rms, name):
            src_val = getattr(src_rms, name)
            try:
                cur = getattr(dst_rms, name)
                if isinstance(cur, np.ndarray):
                    np.copyto(cur, np.asarray(src_val))
                else:
                    setattr(dst_rms, name, float(src_val) if np.isscalar(src_val) else src_val)
            except Exception:
                try:
                    setattr(dst_rms, name, src_val)
                except Exception:
                    pass
    if hasattr(dst_rms, "count") and hasattr(src_rms, "count"):
        try:
            dst_rms.count = float(src_rms.count)
        except Exception:
            pass

def _sync_wrappers(dst_env, src_env):
    """
    Copy wrapper states (NormalizeObservation/Reward, running return) from src_env (real)
    to dst_env (shadow), and copy base state/RNG.
    """
    def walk_chain(env):
        chain = []
        cur = env
        while hasattr(cur, "env"):
            chain.append(cur)
            cur = cur.env
        return chain, cur
    src_chain, src_base = walk_chain(src_env)
    dst_chain, dst_base = walk_chain(dst_env)

    _restore_base_env(dst_base, _snapshot_base_env(src_base))

    for w_src, w_dst in zip(src_chain, dst_chain):
        if hasattr(w_src, "obs_rms") and hasattr(w_dst, "obs_rms"):
            _copy_rms(w_dst.obs_rms, w_src.obs_rms)
        if hasattr(w_src, "return_rms") and hasattr(w_dst, "return_rms"):
            _copy_rms(w_dst.return_rms, w_src.return_rms)
        if hasattr(w_src, "ret") and hasattr(w_dst, "ret"):
            try:
                w_dst.ret = float(w_src.ret)
            except Exception:
                pass

def _clip_to_box(action_np, box_space: gym.spaces.Box):
    return np.clip(action_np, box_space.low, box_space.high)

def _snapshot_wrappers(env):
    chain = []
    cur = env
    while hasattr(cur, "env"):
        chain.append(cur)
        cur = cur.env

    snap = {"_order": [type(w).__name__ for w in chain]}
    for idx, w in enumerate(chain):
        key = f"W{idx}:{type(w).__name__}"
        ws = {}

        # NormalizeObservation
        if hasattr(w, "obs_rms") and w.obs_rms is not None:
            rms = w.obs_rms
            ws["obs_rms_mean"] = np.array(getattr(rms, "mean", 0.0), copy=True)
            ws["obs_rms_var"]  = np.array(getattr(rms, "var", 1.0), copy=True)
            ws["obs_rms_count"]= float(getattr(rms, "count", 1.0))

        # NormalizeReward
        if hasattr(w, "return_rms") and w.return_rms is not None:
            rr = w.return_rms
            ws["ret_rms_mean"] = np.array(getattr(rr, "mean", 0.0), copy=True)
            ws["ret_rms_var"]  = np.array(getattr(rr, "var", 1.0), copy=True)
            ws["ret_rms_count"]= float(getattr(rr, "count", 1.0))
        if hasattr(w, "ret"):
            ws["ret"] = float(w.ret)

        # TimeLimit
        if hasattr(w, "elapsed_steps"):
            ws["elapsed_steps"] = int(w.elapsed_steps)

        if ws:
            snap[key] = ws
    return snap

def _restore_wrappers(env, snap):
    chain = []
    cur = env
    while hasattr(cur, "env"):
        chain.append(cur)
        cur = cur.env

    for idx, w in enumerate(chain):
        key = f"W{idx}:{type(w).__name__}"
        if key not in snap:
            continue
        ws = snap[key]

        # NormalizeObservation
        if "obs_rms_mean" in ws and hasattr(w, "obs_rms") and w.obs_rms is not None:
            rms = w.obs_rms
            try:
                if isinstance(rms.mean, np.ndarray):
                    np.copyto(rms.mean, ws["obs_rms_mean"])
                else:
                    rms.mean = float(np.asarray(ws["obs_rms_mean"]))
            except Exception:
                rms.mean = np.asarray(ws["obs_rms_mean"])
            try:
                if isinstance(rms.var, np.ndarray):
                    np.copyto(rms.var, ws["obs_rms_var"])
                else:
                    rms.var = float(np.asarray(ws["obs_rms_var"]))
            except Exception:
                rms.var = np.asarray(ws["obs_rms_var"])
            if hasattr(rms, "count"):
                rms.count = float(ws["obs_rms_count"])

        # NormalizeReward
        if "ret_rms_mean" in ws and hasattr(w, "return_rms") and w.return_rms is not None:
            rr = w.return_rms
            try:
                if isinstance(rr.mean, np.ndarray):
                    np.copyto(rr.mean, ws["ret_rms_mean"])
                else:
                    rr.mean = float(np.asarray(ws["ret_rms_mean"]))
            except Exception:
                rr.mean = np.asarray(ws["ret_rms_mean"])
            try:
                if isinstance(rr.var, np.ndarray):
                    np.copyto(rr.var, ws["ret_rms_var"])
                else:
                    rr.var = float(np.asarray(ws["ret_rms_var"]))
            except Exception:
                rr.var = np.asarray(ws["ret_rms_var"])
            if hasattr(rr, "count"):
                rr.count = float(ws["ret_rms_count"])

        if "ret" in ws and hasattr(w, "ret"):
            w.ret = float(ws["ret"])

        if "elapsed_steps" in ws and hasattr(w, "elapsed_steps"):
            w.elapsed_steps = int(ws["elapsed_steps"])

def _probe_one_step_with_obs_on_shadow(shadow_env, action_np):
    """
    One-step probe on shadow env with full snapshot/restore of base & wrappers.
    Returns (next_obs, reward, done).
    """
    base = _get_base(shadow_env)
    base_snap = _snapshot_base_env(base)
    wrap_snap = _snapshot_wrappers(shadow_env)

    action_clipped = _clip_to_box(action_np, shadow_env.action_space)
    try:
        obs_next, r, term, trunc, _ = shadow_env.step(action_clipped)
    except gym.error.ResetNeeded:
        shadow_env.reset()
        obs_next, r, term, trunc, _ = shadow_env.step(action_clipped)
    finally:
        _restore_base_env(base, base_snap)
        _restore_wrappers(shadow_env, wrap_snap)

    return obs_next, float(r), bool(term or trunc)
# --------------------------------------------------------------


if __name__ == "__main__":
    args = tyro.cli(Args)
    args.batch_size = int(args.num_envs * args.num_steps)  # B
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    # Policy buffer scales by K
    args.batch_size_policy = args.batch_size * args.num_node_samples
    args.minibatch_size_policy = max(1, args.minibatch_size * args.num_node_samples)

    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"

    if args.track:
        import wandb
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=False,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{k}|{v}|" for k, v in vars(args).items()])),
    )

    # Seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # Real envs
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, i, args.capture_video, run_name, args.gamma) for i in range(args.num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    # Shadow probe envs (identical wrapper stack, no video) + reset to satisfy OrderEnforcing
    probe_envs = [make_env(args.env_id, i + 10_000, False, run_name, args.gamma)() for i in range(args.num_envs)]
    for i, penv in enumerate(probe_envs):
        penv.reset(seed=args.seed + 10_000 + i)

    agent = Agent(envs).to(device)

    # ---- Split optimizers: policy vs value (critic LR a bit larger) ----
    K = max(1, args.num_node_samples)
    base_lr = args.learning_rate
    lr_pi_base = base_lr / np.sqrt(K)         # stabilize with larger K
    lr_v_base  = lr_pi_base * 1.5             # critic slightly faster

    pi_params = list(agent.actor_mean.parameters()) + [agent.actor_logstd]
    v_params  = list(agent.critic.parameters())

    if args.optimizer == "Adam":
        opt_pi = optim.Adam(pi_params, lr=lr_pi_base, eps=1e-5)
        opt_v  = optim.Adam(v_params,  lr=lr_v_base,  eps=1e-5)
    elif args.optimizer == "AlphaGrad" and AlphaGrad is not None:
        opt_pi = AlphaGrad(pi_params, lr=lr_pi_base, alpha=args.alpha, epsilon=1e-5, momentum=0.9)
        opt_v  = AlphaGrad(v_params,  lr=lr_v_base,  alpha=args.alpha, epsilon=1e-5, momentum=0.9)
    elif args.optimizer == "DAG" and DAG is not None:
        opt_pi = DAG(pi_params, lr=lr_pi_base, momentum=0.9)
        opt_v  = DAG(v_params,  lr=lr_v_base,  momentum=0.9)
    else:
        opt_pi = optim.Adam(pi_params, lr=lr_pi_base, eps=1e-5)
        opt_v  = optim.Adam(v_params,  lr=lr_v_base,  eps=1e-5)

    # Storage (PPO-compatible)
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape, device=device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape, device=device)
    logprobs = torch.zeros((args.num_steps, args.num_envs), device=device)
    rewards = torch.zeros((args.num_steps, args.num_envs), device=device)
    dones = torch.zeros((args.num_steps, args.num_envs), device=device)
    values = torch.zeros((args.num_steps, args.num_envs), device=device)

    # Critic targets from K probes: V-target per state
    v_targets = torch.zeros((args.num_steps, args.num_envs), device=device)

    # K-sample buffers
    K = args.num_node_samples
    act_dim = int(np.prod(envs.single_action_space.shape))
    all_actions = torch.zeros((args.num_steps, args.num_envs, K, act_dim), device=device)
    all_logprobs = torch.zeros((args.num_steps, args.num_envs, K), device=device)
    all_advantages = torch.zeros((args.num_steps, args.num_envs, K), device=device)

    # Rollout
    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.tensor(next_obs, device=device, dtype=torch.float32)
    next_done = torch.zeros(args.num_envs, device=device)

    real_envs_list = envs.envs  # list of single envs

    for iteration in range(1, args.num_iterations + 1):
        # Anneal LRs (both policy and value) with K-scaled bases
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lr_pi = float(frac * lr_pi_base)
            lr_v  = float(frac * lr_v_base)
            for g in opt_pi.param_groups:
                g["lr"] = lr_pi
            for g in opt_v.param_groups:
                g["lr"] = lr_v

        for step in range(args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            with torch.no_grad():
                mean, std = agent.dist_params(next_obs)                     # [E, A]
                dist = Normal(mean, std)
                k_actions = dist.sample((K,)).transpose(0, 1).contiguous()  # [E, K, A]
                k_logprobs = dist.log_prob(k_actions).sum(-1)               # [E, K]

            # sync shadow envs from real before probing
            for e in range(args.num_envs):
                _sync_wrappers(probe_envs[e], real_envs_list[e])

            # probe on shadow envs: collect rewards for policy-adv AND expected Bellman target for critic
            k_rewards = np.zeros((args.num_envs, K), dtype=np.float32)
            # accumulate expected target per env: mean_k [ r_k + gamma * (1-done_k) * V(s'_k) ]
            expected_target = torch.zeros(args.num_envs, device=device)

            for e in range(args.num_envs):
                penv = probe_envs[e]
                acc = 0.0
                for ki in range(K):
                    a_np = k_actions[e, ki].detach().cpu().numpy()
                    obs_next, r_i, done_i = _probe_one_step_with_obs_on_shadow(penv, a_np)
                    k_rewards[e, ki] = r_i

                    # critic bootstrap at s'_k
                    obs_next_t = torch.as_tensor(obs_next, device=device, dtype=torch.float32).unsqueeze(0)
                    with torch.no_grad():
                        v_sp = agent.get_value(obs_next_t).squeeze(0)  # scalar tensor
                    acc += r_i + args.gamma * (0.0 if done_i else 1.0) * v_sp.item()
                expected_target[e] = acc / K

            v_targets[step] = expected_target  # save expected Bellman target for critic

            # tanh-normalized K-advantages (numerically safe, K=1 friendly) for POLICY
            k_rewards_t = torch.tensor(k_rewards, device=device)
            mu = k_rewards_t.mean(dim=1, keepdim=True)
            stdr = k_rewards_t.std(dim=1, keepdim=True, unbiased=False).clamp_min(args.eps_norm)
            z = (k_rewards_t - mu) / stdr
            adv_k = torch.tanh(args.tanh_temperature * z)                   # [E, K]
            adv_k = torch.nan_to_num(adv_k, nan=0.0, posinf=1.0, neginf=-1.0)

            # choose executed action
            if args.exec_temperature > 0.0:
                weights = torch.softmax(z * args.exec_temperature, dim=1)
                best_idx = torch.multinomial(weights, num_samples=1).squeeze(1)
            else:
                best_idx = torch.argmax(k_rewards_t, dim=1)
            chosen_actions = k_actions[torch.arange(args.num_envs), best_idx]  # [E, A]
            chosen_logprobs = dist.log_prob(chosen_actions).sum(-1)            # [E]
            with torch.no_grad():
                value = agent.get_value(next_obs).squeeze(-1)                  # [E]

            # store realized branch & K buffers
            actions[step] = chosen_actions
            logprobs[step] = chosen_logprobs
            values[step] = value
            rewards[step] = 0.0
            all_actions[step] = k_actions
            all_logprobs[step] = k_logprobs
            all_advantages[step] = adv_k

            # step real env
            next_obs_np, reward, terminations, truncations, infos = envs.step(chosen_actions.detach().cpu().numpy())
            next_done = torch.tensor(np.logical_or(terminations, truncations), device=device, dtype=torch.float32)
            rewards[step] = torch.tensor(reward, device=device, dtype=torch.float32)
            next_obs = torch.tensor(next_obs_np, device=device, dtype=torch.float32)

            # episodic logging (gymnasium path then fallback)
            if "final_info" in infos and infos["final_info"] is not None:
                for finfo in infos["final_info"]:
                    if finfo and "episode" in finfo:
                        ep_r = finfo["episode"]["r"]
                        ep_l = finfo["episode"]["l"]
                        ep_r = float(ep_r) if hasattr(ep_r, "item") else ep_r
                        ep_l = float(ep_l) if hasattr(ep_l, "item") else ep_l
                        print(f"global_step={global_step}, episodic_return={ep_r}")
                        writer.add_scalar("charts/episodic_return", ep_r, global_step)
                        writer.add_scalar("charts/episodic_length", ep_l, global_step)
            elif "episode" in infos:
                rs = infos["episode"].get("r", [])
                ls = infos["episode"].get("l", [])
                for i in range(len(rs)):
                    ep_r = float(rs[i]) if hasattr(rs[i], "item") else rs[i]
                    ep_l = float(ls[i]) if hasattr(ls[i], "item") else ls[i]
                    print(f"global_step={global_step}, episodic_return={ep_r}")
                    writer.add_scalar("charts/episodic_return", ep_r, global_step)
                    writer.add_scalar("charts/episodic_length", ep_l, global_step)

        # GAE on realized branch (still computed for K=1 policy parity if needed)
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards, device=device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

        # Flatten realized buffers
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)          # [B, ...]
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)        # [B, A]
        b_logprobs = logprobs.reshape(-1)                                          # [B]
        b_advantages_realized = advantages.reshape(-1)                             # [B]  # used if K==1 for policy
        b_values = values.reshape(-1)                                              # [B]
        b_v_targets = v_targets.reshape(-1)                                        # [B]  # NEW: critic targets

        # Flatten K-buffers
        B = args.batch_size
        K = args.num_node_samples
        act_dim = int(np.prod(envs.single_action_space.shape))
        b_all_actions = all_actions.reshape(B, K, act_dim)                         # [B, K, A]
        b_all_logprobs = all_logprobs.reshape(B, K)                                # [B, K]
        b_all_adv = all_advantages.reshape(B, K)                                   # [B, K]

        # Policy views (B*K)
        pol_obs = b_obs.repeat_interleave(K, dim=0)                                # [B*K, ...]
        pol_actions = b_all_actions.reshape(B * K, act_dim)                        # [B*K, A]
        pol_oldlogp = b_all_logprobs.reshape(B * K)                                # [B*K]
        pol_adv = b_all_adv.reshape(B * K)                                         # [B*K]
        if K == 1:
            # PPO parity: use realized GAE for policy when K=1
            pol_adv = b_advantages_realized                                        # [B]

        # Split updates: POLICY over B*K, VALUE over B
        clipfracs = []
        policy_kl_vals, policy_oldkl_vals, policy_entropy_vals = [], [], []
        last_pg_loss = torch.tensor(0.0, device=device)
        last_v_loss = torch.tensor(0.0, device=device)
        last_entropy = torch.tensor(0.0, device=device)
        approx_kl = torch.tensor(0.0, device=device)
        old_approx_kl = torch.tensor(0.0, device=device)

        for epoch in range(args.update_epochs):
            # -------- POLICY UPDATE over B*K --------
            p_inds = np.arange(args.batch_size_policy)
            np.random.shuffle(p_inds)
            for start in range(0, args.batch_size_policy, args.minibatch_size_policy):
                end = start + args.minibatch_size_policy
                mbp = p_inds[start:end]

                mb_obs = pol_obs[mbp]
                mean_new, std_new = agent.dist_params(mb_obs)                      # [MbP, A]
                dist_new = Normal(mean_new, std_new)

                newlogp = dist_new.log_prob(pol_actions[mbp]).sum(-1)              # [MbP]
                logratio = newlogp - pol_oldlogp[mbp]
                ratio = torch.exp(logratio)

                mb_adv = pol_adv[mbp]  # no re-normalization (already tanh/ranked or GAE)
                pg_loss1 = -mb_adv * ratio
                pg_loss2 = -mb_adv * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                entropy_loss = dist_new.entropy().sum(-1).mean()
                loss_pg = pg_loss - args.ent_coef * entropy_loss

                opt_pi.zero_grad()
                loss_pg.backward()
                nn.utils.clip_grad_norm_(pi_params, args.max_grad_norm)
                opt_pi.step()

                with torch.no_grad():
                    old_approx_kl_mb = (-logratio).mean()
                    approx_kl_mb = ((ratio - 1) - logratio).mean()
                    clipfracs.append(((ratio - 1.0).abs() > args.clip_coef).float().mean().item())
                    policy_oldkl_vals.append(old_approx_kl_mb.item())
                    policy_kl_vals.append(approx_kl_mb.item())
                    policy_entropy_vals.append(entropy_loss.item())
                    last_pg_loss = pg_loss
                    last_entropy = entropy_loss

                # Keep logstd sane
                with torch.no_grad():
                    agent.actor_logstd.clamp_(min=-5.0, max=2.0)

            # Early stop on policy KL if requested
            if args.target_kl is not None and len(policy_kl_vals) > 0 and np.mean(policy_kl_vals) > args.target_kl:
                approx_kl = torch.tensor(np.mean(policy_kl_vals), device=device)
                old_approx_kl = torch.tensor(np.mean(policy_oldkl_vals), device=device)
                break

            # -------- VALUE UPDATE over B --------
            v_inds = np.arange(B)
            np.random.shuffle(v_inds)
            for start in range(0, B, args.minibatch_size):
                end = start + args.minibatch_size
                mbv = v_inds[start:end]

                mb_obs_v = b_obs[mbv]
                new_values = agent.get_value(mb_obs_v).view(-1)
                mb_targets = b_v_targets[mbv]  # expected Bellman target from K probes

                if args.clip_vloss:
                    v_loss_unclipped = F.smooth_l1_loss(new_values, mb_targets, reduction="none")
                    v_clipped = b_values[mbv] + torch.clamp(
                        new_values - b_values[mbv], -args.clip_coef, args.clip_coef
                    )
                    v_loss_clipped = F.smooth_l1_loss(v_clipped, mb_targets, reduction="none")
                    v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()
                else:
                    v_loss = 0.5 * F.smooth_l1_loss(new_values, mb_targets)

                opt_v.zero_grad()
                (args.vf_coef * v_loss).backward()
                nn.utils.clip_grad_norm_(v_params, args.max_grad_norm)
                opt_v.step()

                last_v_loss = v_loss

            # epoch averages for logging
            if len(policy_kl_vals) > 0:
                approx_kl = torch.tensor(np.mean(policy_kl_vals), device=device)
                old_approx_kl = torch.tensor(np.mean(policy_oldkl_vals), device=device)
                last_entropy = torch.tensor(np.mean(policy_entropy_vals), device=device)

        # Logging (PPO parity) — explained variance vs Bellman targets
        y_pred, y_true = b_values.detach().cpu().numpy(), b_v_targets.detach().cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1.0 - np.var(y_true - y_pred) / var_y

        # Log policy LR (matches PPO key)
        writer.add_scalar("charts/learning_rate", opt_pi.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/policy_loss", float(last_pg_loss.item()), global_step)
        writer.add_scalar("losses/value_loss", float(last_v_loss.item()), global_step)
        writer.add_scalar("losses/entropy", float(last_entropy.item()), global_step)
        writer.add_scalar("losses/old_approx_kl", float(old_approx_kl.item()), global_step)
        writer.add_scalar("losses/approx_kl", float(approx_kl.item()), global_step)
        writer.add_scalar("losses/clipfrac", float(np.mean(clipfracs) if len(clipfracs) else 0.0), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)

        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    # Save & optional eval / upload (kept from PPO script)
    if args.save_model:
        model_path = f"runs/{run_name}/{args.exp_name}.cleanrl_model"
        torch.save(agent.state_dict(), model_path)
        print(f"model saved to {model_path}")

        try:
            from cleanrl_utils.evals.ppo_eval import evaluate
            episodic_returns = evaluate(
                model_path,
                make_env,
                args.env_id,
                eval_episodes=10,
                run_name=f"{run_name}-eval",
                Model=Agent,
                device=device,
                gamma=args.gamma,
            )
            for idx, episodic_return in enumerate(episodic_returns):
                writer.add_scalar("eval/episodic_return", episodic_return, idx)
        except Exception as e:
            print("Eval skipped or failed:", e)

        if args.upload_model:
            try:
                from cleanrl_utils.huggingface import push_to_hub
                repo_name = f"{args.env_id}-{args.exp_name}-seed{args.seed}"
                repo_id = f"{args.hf_entity}/{repo_name}" if args.hf_entity else repo_name
                push_to_hub(args, episodic_returns, repo_id, "PPO", f"runs/{run_name}", f"videos/{run_name}-eval")
            except Exception as e:
                print("HF upload skipped or failed:", e)

    envs.close()
    for penv in probe_envs:
        penv.close()

    # Optional W&B video upload parity
    if args.track and args.capture_video:
        print("Attempting to upload recorded videos to W&B...")
        video_dir = f"videos/{run_name}"
        try:
            import wandb
            video_files = sorted(glob.glob(os.path.join(video_dir, "*.mp4")), key=os.path.getmtime)
            if video_files:
                for i, video_file in enumerate(video_files):
                    wandb.log({f"media/video_capture_{i}": wandb.Video(video_file, fps=4, format="mp4")}, step=global_step)
        except Exception as e:
            print("Video upload skipped or failed:", e)

    writer.close()
    if args.track:
        try:
            import wandb
            wandb.finish()
        except Exception:
            pass
