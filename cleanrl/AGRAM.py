# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_continuous_actionpy

import os
import random
import time
from dataclasses import dataclass, field
from typing import Callable, Optional # Added Optional for dynago_kappa

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
# Assuming your DAG and AlphaGrad are in an 'optim' directory relative to this script
# Make sure this path is correct or 'optim' is in PYTHONPATH
from optim.sgd import AlphaGrad, DAG # User's import
import tyro
import glob
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter
import math # For math.sqrt if needed, though torch.sqrt is usually used

@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")] + "_AdvModDAGAlpha" # Updated suffix
    """the name of this experiment"""
    seed: int = 1
    torch_deterministic: bool = True
    cuda: bool = True
    track: bool = True
    wandb_project_name: str = "cleanRL-DynAGO"
    wandb_entity: str = None
    capture_video: bool = False
    save_model: bool = False
    upload_model: bool = False
    hf_entity: str = ""
    optimizer: str = "Adam"
    alpha: float = 0.0 # For AlphaGrad optimizer

    # Algorithm specific arguments
    env_id: str = "HalfCheetah-v4"
    total_timesteps: int = 1000000
    learning_rate: float = 9e-5
    num_envs: int = 1
    num_steps: int = 2048
    anneal_lr: bool = True
    gamma: float = 0.99
    gae_lambda: float = 0.95
    num_minibatches: int = 32 #32
    update_epochs: int = 10
    norm_adv: bool = True # Applied to the *output* of dynago_transform_advantages
    clip_coef: float = 0.2
    clip_vloss: bool = True
    ent_coef: float = 0.00
    vf_coef: float = 0.4
    max_grad_norm: float = 0.5
    target_kl: float = None

    # ### DynAGO-AM (Full DAG Alpha Controller for Advantages) ###
    M_samples: int = 1
    """Number of actions to sample per state per conceptual step"""
    # Hyperparameters mirroring DAG's 'h' dict, prefixed with 'dynago_'
    dynago_tau: float = 1.25
    """Saturation threshold for Z_A (input to tanh for advantages)"""
    dynago_p_star: float = 0.10
    """Target saturation percentage for Z_A"""
    dynago_kappa: float = 2 # If None, will be derived from tau and p_star
    """Base scaling factor for advantage alpha. If None, derived like in DAG optimizer."""
    dynago_eta: float = 0.3
    """Strength of saturation feedback for advantage alpha controller"""
    dynago_rho: float = 0.1 # EMA for the main alpha_A_ema
    """EMA smoothing factor for DynAGO advantage alpha"""
    dynago_eps: float = 1e-5
    """Epsilon for numerical stability in advantage alpha controller"""
    dynago_alpha_min: float = 1e-12 # Min clamp for EMA'd alpha_A
    """Min clamp for the EMA'd advantage modulation alpha"""
    dynago_alpha_max: float = 1e12  # Max clamp for EMA'd alpha_A
    """Max clamp for the EMA'd advantage modulation alpha"""
    # EMA for observed saturation
    dynago_rho_sat: float = 0.98 # Typically a slower EMA
    """EMA smoothing factor for observed advantage saturation ratio"""
    dynago_alpha_A_init: float = 1.0 # Initial value for the alpha_A_ema state
    """Initial value for the advantage modulation alpha EMA state"""
    dynago_prev_sat_A_init: float = 0.10 # Initial guess for prev_saturation_A_ema state
    """Initial value for observed advantage saturation EMA state"""

    # to be filled in runtime
    batch_size: int = 0
    minibatch_size: int = 0
    num_iterations: int = 0
    num_total_samples_per_rollout: int = 0


# ### DynAGO-AM (Full DAG Alpha Controller) ###: Function to transform ADVANTAGES
def dynago_transform_advantages(
        raw_advantages_batch: torch.Tensor,
        dynago_params_A: dict,
        alpha_A_ema_state: torch.Tensor,
        prev_saturation_A_ema_state: torch.Tensor,
        device: torch.device,
        update_ema: bool = True,   # <‑‑ NEW
) -> torch.Tensor:
    """
    Transforms raw advantages using a DAG-optimizer-like alpha controller.
    Updates alpha_A_ema_state and prev_saturation_A_ema_state in-place.
    """
    current_raw_advantages_MB = raw_advantages_batch

    if current_raw_advantages_MB.numel() <= 1:
        return current_raw_advantages_MB

    # Retrieve hyperparameters
    kappa = dynago_params_A["kappa"] # This is the derived kappa
    tau = dynago_params_A["tau"]
    p_star = dynago_params_A["p_star"]
    eta = dynago_params_A["eta"]
    rho = dynago_params_A["rho"]
    eps = dynago_params_A["eps"]
    alpha_min = dynago_params_A["alpha_min"]
    alpha_max = dynago_params_A["alpha_max"]
    rho_sat = dynago_params_A["rho_sat"]

    # 1. Calculate Norm (N_A) and Std (sigma_A) of raw advantages in the minibatch
    N_A = torch.linalg.norm(current_raw_advantages_MB)
    if N_A < eps: # Avoid division by zero if norm is too small
        return current_raw_advantages_MB # Cannot meaningfully normalize or scale

    sigma_A = torch.std(current_raw_advantages_MB) + eps
    # Note: In DAG optimizer, sigma_cat can be pre-computed. Here, sigma_A is for the current batch.

    # 2. Retrieve current EMA states
    alpha_A_prev_ema_val = alpha_A_ema_state[0]
    prev_sat_A_ema_val = prev_saturation_A_ema_state[0]

    # 3. Calculate target alpha_A_hat (mirroring DAG's alpha_hat structure)
    # We omit dimensionality prior and global shrink (s_t) as they don't apply here.
    alpha_A_hat = (
        kappa
        * (N_A + eps) / (sigma_A + eps) # Norm/Std ratio. Note: N_A is already sum of squares, not per-element.
                                        # DAG uses norm_cat / sigma_cat. For single batch of scalars, this is N_A / sigma_A.
        * (p_star / (prev_sat_A_ema_val + eps)) ** eta
    )
    # alpha_A_hat = torch.clamp(alpha_A_hat, alpha_min, alpha_max) # Optional: clamp target alpha too

    # 4. Calculate new alpha_A using EMA
    if update_ema:
        _alpha_A_updated = (1 - rho) * alpha_A_prev_ema_val + rho * alpha_A_hat
        _alpha_A_updated = torch.clamp(_alpha_A_updated, alpha_min, alpha_max)
        alpha_A_ema_state[0] = _alpha_A_updated.detach()
        alpha_A_to_use = _alpha_A_updated # Use the new value
    else:
        alpha_A_to_use = alpha_A_prev_ema_val

    # 5. Update persistent alpha_A_ema_state
    alpha_A_ema_state[0] = alpha_A_to_use.detach()

    # 6. Normalize raw advantages for input to tanh
    normalized_advantages_A_MB = current_raw_advantages_MB / (N_A + eps)

    # 7. Calculate Z_A (input to tanh)
    Z_A_MB = alpha_A_to_use * normalized_advantages_A_MB

    # 8. Calculate current observed saturation
    current_observed_saturation_A = (Z_A_MB.abs() > tau).float().mean()

    # 9. Update persistent prev_saturation_A_ema_state for the *next* call
    if update_ema:
        prev_saturation_A_ema_state[0] = (
            (1 - rho_sat) * prev_sat_A_ema_val + rho_sat * current_observed_saturation_A
        ).detach()

    # 10. Tanh transformation
    T_A_MB = torch.tanh(Z_A_MB)

    # 11. Modulate Raw Advantages
    modulated_advantages_MB = current_raw_advantages_MB  * (args.dynago_kappa * torch.tanh(Z_A_MB) + 0.5)

    return modulated_advantages_MB
    # return current_raw_advantages_MB

# (make_env, layer_init, Agent class remain the same)
def make_env(env_id, idx, capture_video, run_name, gamma):
    def thunk():
        env = None
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array"); video_folder = f"videos/{run_name}"
            os.makedirs(video_folder, exist_ok=True); env = gym.wrappers.RecordVideo(env, video_folder=video_folder, episode_trigger=lambda x: True if idx == 0 else False)
        else: env = gym.make(env_id)
        env = gym.wrappers.FlattenObservation(env); env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.wrappers.ClipAction(env); env = gym.wrappers.NormalizeObservation(env)
        env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10), observation_space=env.observation_space)
        env = gym.wrappers.NormalizeReward(env, gamma=gamma); env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))
        return env
    return thunk

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std); torch.nn.init.constant_(layer.bias, bias_const); return layer

class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__(); obs_prod = np.array(envs.single_observation_space.shape).prod(); act_prod = np.prod(envs.single_action_space.shape)
        self.critic = nn.Sequential(layer_init(nn.Linear(obs_prod, 64)), nn.Tanh(), layer_init(nn.Linear(64, 64)), nn.Tanh(), layer_init(nn.Linear(64, 1), std=1.0))
        self.actor_mean = nn.Sequential(layer_init(nn.Linear(obs_prod, 64)), nn.Tanh(), layer_init(nn.Linear(64, 64)), nn.Tanh(), layer_init(nn.Linear(64, act_prod), std=0.01))
        self.actor_logstd = nn.Parameter(torch.zeros(1, act_prod))
    def get_value(self, x): return self.critic(x)
    def get_action_and_value(self, x, action=None):
        action_mean = self.actor_mean(x); action_logstd = self.actor_logstd.expand_as(action_mean); action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std);
        if action is None: action = probs.sample()
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x)

if __name__ == "__main__":
    args = tyro.cli(Args)
    args.num_total_samples_per_rollout = args.num_steps * args.M_samples
    args.batch_size = int(args.num_envs * args.num_total_samples_per_rollout)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // (args.num_envs * args.num_steps)

    run_name = f"{args.env_id}__{args.exp_name}-M{args.M_samples}__{args.optimizer}__{args.seed}__{int(time.time())}"
    if args.track: import wandb; wandb.init(project=args.wandb_project_name, entity=args.wandb_entity, sync_tensorboard=True, config=vars(args), name=run_name, monitor_gym=True, save_code=True)
    writer = SummaryWriter(f"runs/{run_name}"); writer.add_text("hyperparameters", "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])))

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed); torch.backends.cudnn.deterministic = args.torch_deterministic
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    envs = gym.vector.SyncVectorEnv([make_env(args.env_id, i, args.capture_video, run_name, args.gamma) for i in range(args.num_envs)])
    assert isinstance(envs.single_action_space, gym.spaces.Box)
    agent = Agent(envs).to(device)
    if args.optimizer.lower() == 'adam': optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)
    elif args.optimizer.lower() == 'alphagrad': optimizer = AlphaGrad(agent.parameters(), lr=args.learning_rate, alpha=args.alpha); print(f"AlphaGrad selected.")
    elif args.optimizer.lower() == 'dag': optimizer = DAG(agent.parameters(), lr=args.learning_rate); print("DAG selected.")
    else: print(f"Unknown optimizer '{args.optimizer}', defaulting to Adam."); optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    obs_shape = envs.single_observation_space.shape; action_shape = envs.single_action_space.shape
    obs_buffer = torch.zeros((args.num_total_samples_per_rollout, args.num_envs) + obs_shape).to(device)
    actions_buffer = torch.zeros((args.num_total_samples_per_rollout, args.num_envs) + action_shape).to(device)
    logprobs_buffer = torch.zeros((args.num_total_samples_per_rollout, args.num_envs)).to(device)
    rewards_buffer = torch.zeros((args.num_total_samples_per_rollout, args.num_envs)).to(device) # Raw rewards
    dones_buffer = torch.zeros((args.num_total_samples_per_rollout, args.num_envs)).to(device)
    values_buffer = torch.zeros((args.num_total_samples_per_rollout, args.num_envs)).to(device)
    next_obs_values_buffer = torch.zeros((args.num_total_samples_per_rollout, args.num_envs)).to(device)

    # ### DynAGO-AM (Full DAG Alpha Controller) ###: Initialize Advantage modulation state
    # Derive kappa from tau and p_star if not provided by user
    derived_kappa_A = args.dynago_kappa
    if derived_kappa_A is None:
        if args.dynago_p_star <= 0 or args.dynago_p_star >= 1:
            raise ValueError("dynago_p_star must be in (0, 1) to derive kappa.")
        inv_A = torch.distributions.Normal(0, 1).icdf(torch.tensor(1 - args.dynago_p_star / 2, device=device))
        if inv_A.abs().item() < 1e-6: # Avoid division by zero if p_star is too close to 0 or 1
            raise ValueError("Derived Z-score for p_star is too close to zero, leading to unstable kappa. Adjust p_star.")
        derived_kappa_A = args.dynago_tau / inv_A.item()
        print(f"DEBUG: Derived dynago_kappa_A = {derived_kappa_A:.4f} from tau={args.dynago_tau}, p_star={args.dynago_p_star}")


    dynago_params_A_config = {
        "kappa": derived_kappa_A, # Use the derived or user-provided kappa
        "tau": args.dynago_tau,
        "p_star": args.dynago_p_star,
        "eta": args.dynago_eta,
        "rho": args.dynago_rho,
        "eps": args.dynago_eps,
        "alpha_min": args.dynago_alpha_min,
        "alpha_max": args.dynago_alpha_max,
        "rho_sat": args.dynago_rho_sat,
    }
    # These are the EMA states, initialized once
    alpha_A_ema_state = torch.tensor([args.dynago_alpha_A_init], dtype=torch.float32, device=device)
    prev_saturation_A_ema_state = torch.tensor([args.dynago_prev_sat_A_init], dtype=torch.float32, device=device)

    global_step = 0; start_time = time.time()
    next_obs_conceptual, _ = envs.reset(seed=args.seed)
    next_obs_conceptual = torch.Tensor(next_obs_conceptual).to(device)
    next_done_conceptual = torch.zeros(args.num_envs).to(device)

    # (Data Collection Loop - s_iter: remains mostly the same structure for M-sampling,
    # ensuring raw rewards are stored in rewards_buffer. No reward modulation here.)
    for iteration in range(1, args.num_iterations + 1):
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate; optimizer.param_groups[0]["lr"] = lrnow

        for s_iter in range(0, args.num_steps):
            current_conceptual_obs_at_s_iter = next_obs_conceptual
            list_of_stacked_s_t = [None]*args.num_envs; list_of_stacked_raw_rewards = [None]*args.num_envs
            list_of_stacked_actions = [None]*args.num_envs; list_of_stacked_logprobs = [None]*args.num_envs
            list_of_stacked_values_s_t = [None]*args.num_envs; list_of_stacked_s_prime_for_commit_selection = [None]*args.num_envs
            list_of_stacked_dones_primes = [None]*args.num_envs; list_of_stacked_next_obs_prime_values = [None]*args.num_envs
            committed_actions_for_true_env_step = torch.zeros((args.num_envs,) + action_shape).to(device)

            for env_idx in range(args.num_envs):
                s_t_current_env = current_conceptual_obs_at_s_iter[env_idx:env_idx+1]
                env_instance = envs.envs[env_idx].unwrapped
                env_snapshot_qpos, env_snapshot_qvel, can_clone_state = None, None, False

                current_env_s_t_list_M = []
                current_env_raw_rewards_list_M = []
                current_env_actions_list_M = []
                current_env_logprobs_list_M = []
                current_env_values_s_t_list_M = [] # V(s_t)
                current_env_s_prime_list_M_for_commit = [] # s'_tj
                current_env_dones_primes_list_M = [] # d_tj
                current_env_next_obs_prime_values_list_M = [] # V(s'_tj)
                
                if hasattr(env_instance, "data") and hasattr(env_instance.data, "qpos") and \
                   hasattr(env_instance.data, "qvel") and hasattr(env_instance, "set_state") and \
                   callable(getattr(env_instance, "set_state")):
                    try:
                        env_snapshot_qpos = env_instance.data.qpos.copy(); env_snapshot_qvel = env_instance.data.qvel.copy()
                        can_clone_state = True
                        if s_iter == 0 and env_idx == 0: print("DEBUG: Using qpos/qvel for state cloning.")
                    except Exception as e:
                        if s_iter == 0 and env_idx == 0: print(f"WARNING: qpos/qvel get error: {e}")
                if not can_clone_state and s_iter == 0 and env_idx == 0: print("WARNING: State cloning failed. M-sampling sequential.")

                for k_sample in range(args.M_samples):
                    if can_clone_state:
                        try: env_instance.set_state(env_snapshot_qpos, env_snapshot_qvel)
                        except Exception as e:
                            if s_iter==0 and env_idx==0 and k_sample==0: print(f"WARNING: State set error: {e}")
                    with torch.no_grad(): action_k,logprob_k,_,value_s_t_k = agent.get_action_and_value(s_t_current_env)
                    action_k_np = action_k.cpu().numpy().reshape(envs.single_action_space.shape)
                    next_obs_prime_k,raw_reward_k,term_k,trunc_k,_ = env_instance.step(action_k_np)
                    done_prime_k = term_k or trunc_k
                    current_env_s_t_list_M.append(s_t_current_env.clone())
                    current_env_raw_rewards_list_M.append(torch.tensor([raw_reward_k],device=device))
                    current_env_actions_list_M.append(action_k.clone()); current_env_logprobs_list_M.append(logprob_k.clone())
                    current_env_values_s_t_list_M.append(value_s_t_k.clone())
                    next_obs_prime_k_tensor = torch.Tensor(next_obs_prime_k).to(device).unsqueeze(0)
                    current_env_s_prime_list_M_for_commit.append(next_obs_prime_k_tensor.clone())
                    current_env_dones_primes_list_M.append(torch.tensor([done_prime_k],dtype=torch.float32,device=device))
                    with torch.no_grad(): value_s_prime_k = agent.get_value(next_obs_prime_k_tensor)
                    current_env_next_obs_prime_values_list_M.append(value_s_prime_k.clone())
                
                if args.M_samples <= 0: raise ValueError("M_samples > 0")

                # For tensors that are already (1, Dim) per sample, cat along dim 0
                list_of_stacked_s_t[env_idx] = torch.cat(current_env_s_t_list_M, dim=0) # Becomes (M, Dim)
                list_of_stacked_actions[env_idx] = torch.cat(current_env_actions_list_M, dim=0) # Becomes (M, Dim)
                list_of_stacked_s_prime_for_commit_selection[env_idx] = torch.cat(current_env_s_prime_list_M_for_commit, dim=0) # Becomes (M, Dim)

                # For tensors that are (1,) per sample (scalars wrapped in a 1D tensor)
                list_of_stacked_raw_rewards[env_idx] = torch.stack(current_env_raw_rewards_list_M, dim=0).squeeze(-1) # Becomes (M,)
                list_of_stacked_logprobs[env_idx] = torch.stack(current_env_logprobs_list_M, dim=0).squeeze(-1) # Becomes (M,)
                list_of_stacked_values_s_t[env_idx] = torch.stack(current_env_values_s_t_list_M, dim=0).squeeze(-1) # Becomes (M,)
                list_of_stacked_dones_primes[env_idx] = torch.stack(current_env_dones_primes_list_M, dim=0).squeeze(-1) # Becomes (M,)
                list_of_stacked_next_obs_prime_values[env_idx] = torch.stack(current_env_next_obs_prime_values_list_M, dim=0).squeeze(-1) # Becomes (M,)

             # --- Determine committed actions and their indices ONCE PER S_ITER ---
            committed_k_indices_for_all_envs = np.zeros(args.num_envs, dtype=int)
            # temp_next_obs_list_for_commit, temp_next_done_list_for_commit = [], [] # Keep if needed for next_obs_conceptual update strategy

            for env_idx_commit in range(args.num_envs):
                values_s_prime_for_env = list_of_stacked_next_obs_prime_values[env_idx_commit]
                best_k_idx_tensor = torch.argmax(values_s_prime_for_env)
                best_k_idx_int = best_k_idx_tensor.item()
                
                committed_k_indices_for_all_envs[env_idx_commit] = best_k_idx_int
                committed_actions_for_true_env_step[env_idx_commit] = list_of_stacked_actions[env_idx_commit][best_k_idx_int]
                
                # If you are using these lists to manually set next_obs_conceptual (instead of envs.step output):
                # current_env_next_obs_for_commit = list_of_stacked_s_prime_for_commit_selection[env_idx_commit][best_k_idx_int]
                # temp_next_obs_list_for_commit.append(current_env_next_obs_for_commit.unsqueeze(0))
                # current_env_done_for_commit = list_of_stacked_dones_primes[env_idx_commit][best_k_idx_int]
                # temp_next_done_list_for_commit.append(current_env_done_for_commit.unsqueeze(0))
                

            # --- Now fill the buffers using the calculated committed_k_indices_for_all_envs ---
            for k_fill in range(args.M_samples):
                buffer_row_idx = s_iter * args.M_samples + k_fill
                for env_idx_fill in range(args.num_envs):
                    # always store obs / action / logprob
                    obs_buffer[buffer_row_idx, env_idx_fill]      = list_of_stacked_s_t[env_idx_fill][k_fill]
                    actions_buffer[buffer_row_idx, env_idx_fill]  = list_of_stacked_actions[env_idx_fill][k_fill]
                    logprobs_buffer[buffer_row_idx, env_idx_fill] = list_of_stacked_logprobs[env_idx_fill][k_fill]
                    
                    # === critic targets (Patch #1) ===
                    if k_fill == committed_k_indices_for_all_envs[env_idx_fill]:
                        # this is the real branch for env_idx_fill
                        rewards_buffer[buffer_row_idx, env_idx_fill] = list_of_stacked_raw_rewards[env_idx_fill][k_fill]
                        dones_buffer[buffer_row_idx,   env_idx_fill] = list_of_stacked_dones_primes[env_idx_fill][k_fill]
                        values_buffer[buffer_row_idx,  env_idx_fill] = list_of_stacked_values_s_t[env_idx_fill][k_fill]
                        next_obs_values_buffer[buffer_row_idx, env_idx_fill] = \
                            list_of_stacked_next_obs_prime_values[env_idx_fill][k_fill]
                    else:
                        # hypothetical branch for env_idx_fill
                        rewards_buffer[buffer_row_idx, env_idx_fill]        = 0.0
                        dones_buffer[buffer_row_idx,   env_idx_fill]        = 1.0  # treat as terminal
                        # For V(st) on uncommitted branches, it's better to store the actual V(st)
                        # The GAE for these branches will be zeroed by done=1.0 anyway.
                        # Setting V(st)=0 can make delta = reward - 0, which might be non-zero if reward isn't zeroed.
                        # Sticking to the original values_buffer for all samples, and letting done=1.0 handle GAE:
                        values_buffer[buffer_row_idx,  env_idx_fill]        = list_of_stacked_values_s_t[env_idx_fill][k_fill] 
                        next_obs_values_buffer[buffer_row_idx, env_idx_fill] = 0.0 # This V(s') is not used if done=1.0

            actual_next_obs_sv,_,actual_terminations_sv,actual_truncations_sv,infos = envs.step(committed_actions_for_true_env_step.cpu().numpy())
            actual_next_done_sv = np.logical_or(actual_terminations_sv,actual_truncations_sv)
            next_obs_conceptual = torch.Tensor(actual_next_obs_sv).to(device); next_done_conceptual = torch.Tensor(actual_next_done_sv).to(device)
            global_step += args.num_envs
            # (Logging logic for infos - same as before)
            if "final_info" in infos:
                for i, info_item in enumerate(infos["final_info"]): # infos["final_info"] is a list of dicts
                    if info_item and "episode" in info_item:
                        writer.add_scalar(f"charts/episodic_return_env{i}", info_item['episode']['r'], global_step)
                        writer.add_scalar(f"charts/episodic_length_env{i}", info_item['episode']['l'], global_step)
                        print(f"g_step={global_step}, e_idx={i}, e_ret={info_item['episode']['r']:.2f}")
            # Corrected part for the 'else' condition for SyncVectorEnv without final_info sometimes
            elif "episode" in infos and "_episode" in infos: # Check if both keys exist
                # infos["episode"] is a dict of arrays, infos["_episode"] is a boolean array
                for i in range(args.num_envs): # Iterate up to num_envs
                    if i < len(infos["_episode"]) and infos["_episode"][i]: # Check if index i is valid and the episode for env i is done
                        if i < len(infos["episode"]["r"]): # Ensure 'r' and 'l' arrays are long enough
                            writer.add_scalar(f"charts/episodic_return_env{i}", infos["episode"]["r"][i].item(), global_step)
                            writer.add_scalar(f"charts/episodic_length_env{i}", infos["episode"]["l"][i].item(), global_step)
                            print(f"g_step={global_step}, e_idx={i}, e_ret={infos['episode']['r'][i].item():.2f}")

        # GAE Calculation (Standard PPO, using raw rewards from rewards_buffer)
        with torch.no_grad():
            end_of_rollout_next_value = agent.get_value(next_obs_conceptual).reshape(1, args.num_envs)
            advantages_raw = torch.zeros_like(rewards_buffer).to(device) # Initialize advantages_raw
            lastgaelam = torch.zeros(args.num_envs).to(device)
            for t in reversed(range(args.num_total_samples_per_rollout)):
                # ... (logic inside the loop is correct now) ...
                is_last_conceptual_step_sample = (t >= (args.num_steps - 1) * args.M_samples)
                if is_last_conceptual_step_sample:
                    nextvalues_gae = end_of_rollout_next_value.squeeze(0)
                    nextnonterminal_gae = 1.0 - next_done_conceptual
                else:
                    nextvalues_gae = next_obs_values_buffer[t]
                    nextnonterminal_gae = 1.0 - dones_buffer[t]

                delta = rewards_buffer[t] + args.gamma * nextvalues_gae * nextnonterminal_gae - values_buffer[t]
                advantages_raw[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal_gae * lastgaelam

        # --- Flatten Buffers ---
        b_obs = obs_buffer.reshape((-1,) + obs_shape)
        b_logprobs = logprobs_buffer.reshape(-1)
        b_actions = actions_buffer.reshape((-1,) + action_shape)
        b_advantages_raw_flat = advantages_raw.reshape(-1) # Flat raw A_GAE
        # b_returns = returns.reshape(-1) # We calculate the new returns below
        b_values = values_buffer.reshape(-1) # V_old(s)

        # --- !!! NEW: Pre-calculate Modulated Advantages for Value Target !!! ---
        print(f"Iteration {iteration}: Pre-calculating modulated advantages for value targets...")
        start_precalc_time = time.time()
        b_advantages_mod_for_vtarget_flat = torch.zeros_like(b_advantages_raw_flat)
        # Use temporary EMA states so we don't modify the main ones needed for policy updates yet
        temp_alpha_ema = alpha_A_ema_state.clone()
        temp_sat_ema = prev_saturation_A_ema_state.clone()
        # Prepare indices for sequential processing
        precalc_inds = np.arange(args.batch_size)
        # Note: No shuffling here, process sequentially to mimic epoch 0 state evolution
        with torch.no_grad(): # No gradients needed for this target calculation
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = precalc_inds[start:end]
                mb_adv_raw = b_advantages_raw_flat[mb_inds]

                # Apply modulation using TEMPORARY EMA states
                # The function will update temp_alpha_ema and temp_sat_ema in-place
                mb_adv_mod = dynago_transform_advantages(
                        mb_adv_raw,
                        dynago_params_A_config,
                        temp_alpha_ema,
                        temp_sat_ema,
                        device,
                        update_ema=False,   # <‑‑ FREEZE
                )
                # Store the result for this minibatch
                b_advantages_mod_for_vtarget_flat[mb_inds] = mb_adv_mod
        
        # --- !!! NEW: Calculate Value Target using Modulated Advantages !!! ---
        b_returns_mod = b_advantages_mod_for_vtarget_flat + b_values # V_target_mod = A_mod + V_old

        end_precalc_time = time.time()
        print(f"Iteration {iteration}: Pre-calculation finished in {end_precalc_time - start_precalc_time:.4f} seconds.")
        # --- End of New Pre-calculation Section ---

        # --- Update Loop ---
        b_inds = np.arange(args.batch_size) # Indices for shuffling
        clipfracs = []; avg_raw_adv_epoch_list = []; avg_mod_adv_epoch_list = []

        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds) # Shuffle for stochasticity
            epoch_raw_advs, epoch_mod_advs = [], []
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size; mb_inds = b_inds[start:end]

                # Get policy outputs for the minibatch
                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]; ratio = logratio.exp()

                # Calculate KL divergence (for logging/early stopping)
                with torch.no_grad():
                    old_approx_kl = (-logratio).mean(); approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                # Get raw advantages for this minibatch (for logging)
                mb_advantages_raw_minibatch = b_advantages_raw_flat[mb_inds]
                epoch_raw_advs.append(mb_advantages_raw_minibatch.mean().item())

                # --- Apply DynAGO modulation for POLICY Loss ---
                # Uses the *actual* global EMA states, which evolve across epochs/minibatches
                mb_advantages_mod = dynago_transform_advantages(
                        mb_advantages_raw_minibatch,
                        dynago_params_A_config,
                        alpha_A_ema_state,
                        prev_saturation_A_ema_state,
                        device,
                        update_ema=False,   # <‑‑ FREEZE α THROUGHOUT THE UPDATE
                )
                epoch_mod_advs.append(mb_advantages_mod.mean().item())
                # --- End Advantage Modulation ---

                # Final advantages for policy loss (apply optional normalization)
                mb_advantages_final = mb_advantages_mod
                if args.norm_adv: # Normalize the MODULATED advantages
                    mb_advantages_final = (mb_advantages_final - mb_advantages_final.mean()) / (mb_advantages_final.std() + 1e-8)

                # --- Policy Loss Calculation ---
                pg_loss1 = -mb_advantages_final * ratio
                pg_loss2 = -mb_advantages_final * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # --- Value Loss Calculation ---
                newvalue = newvalue.view(-1)
                # !!! Use the MODULATED returns calculated before the epoch loop !!!
                mb_target_values = b_returns_mod[mb_inds]

                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - mb_target_values) ** 2
                    # Clip value difference against V_old (b_values)
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef, # Using policy clip coef for value clipping
                         args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - mb_target_values) ** 2
                    v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()
                else:
                    v_loss = 0.5 * ((newvalue - mb_target_values) ** 2).mean()

                # --- Entropy Loss Calculation ---
                entropy_loss = entropy.mean()

                # --- Total Loss ---
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                # --- Optimization Step ---
                optimizer.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm); optimizer.step()

            # --- End of Minibatch Loop ---
            if args.target_kl is not None and approx_kl > args.target_kl:
                print(f"Early stopping at epoch {epoch+1} due to reaching max KL divergence.")
                break
            if epoch_raw_advs: avg_raw_adv_epoch_list.append(np.mean(epoch_raw_advs))
            if epoch_mod_advs: avg_mod_adv_epoch_list.append(np.mean(epoch_mod_advs))
        # --- End of Epoch Loop ---

        # (Logging - as before, adding new DynAGO-AM specific logs)
        y_pred, y_true = b_values.cpu().numpy(), b_returns_mod.cpu().numpy() # Use b_returns_mod for y_true
        var_y = np.var(y_true); explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
        writer.add_scalar("charts/learning_rate",optimizer.param_groups[0]["lr"],global_step); writer.add_scalar("losses/value_loss",v_loss.item(),global_step)
        writer.add_scalar("losses/policy_loss",pg_loss.item(),global_step); writer.add_scalar("losses/entropy",entropy_loss.item(),global_step)
        writer.add_scalar("losses/old_approx_kl",old_approx_kl.item(),global_step); writer.add_scalar("losses/approx_kl",approx_kl.item(),global_step)
        writer.add_scalar("losses/clipfrac",np.mean(clipfracs),global_step); writer.add_scalar("losses/explained_variance",explained_var,global_step) # Log the new EV
        writer.add_scalar("charts/SPS",int(args.num_envs*args.num_steps/(time.time()-start_time)),global_step)
        if alpha_A_ema_state.numel()>0: writer.add_scalar("dynago_adv_mod/alpha_A_ema", alpha_A_ema_state.mean().item(), global_step)
        if prev_saturation_A_ema_state.numel()>0: writer.add_scalar("dynago_adv_mod/prev_saturation_A_ema", prev_saturation_A_ema_state.mean().item(), global_step)
        if avg_raw_adv_epoch_list: writer.add_scalar("dynago_adv_mod/avg_raw_advantage_iter", np.mean(avg_raw_adv_epoch_list), global_step)
        if avg_mod_adv_epoch_list: writer.add_scalar("dynago_adv_mod/avg_mod_advantage_iter", np.mean(avg_mod_adv_epoch_list), global_step)

    # (Save model, Evaluate, Upload, Cleanup - as before)
    if args.save_model:
        model_path=f"runs/{run_name}/{args.exp_name}.cleanrl_model"; os.makedirs(os.path.dirname(model_path),exist_ok=True)
        torch.save(agent.state_dict(),model_path); print(f"model saved to {model_path}")
        try:
            from cleanrl_utils.evals.ppo_eval import evaluate
            ret=evaluate(model_path,make_env,args.env_id,10,f"{run_name}-eval",Agent,device,args.gamma)
            for idx,item in enumerate(ret): writer.add_scalar("Training Rewards",item,idx)
            if args.upload_model:
                from cleanrl_utils.huggingface import push_to_hub
                repo_id=f"{args.hf_entity}/{args.env_id}-{args.exp_name}-seed{args.seed}" if args.hf_entity else f"{args.env_id}-{args.exp_name}-seed{args.seed}"
                push_to_hub(args=vars(args),episodic_returns=ret,repo_id=repo_id,commit_message="Upload DynAGO-AM PPO model",model_filename=model_path)
        except ImportError: print("cleanrl_utils not found, skipping eval/upload.")
        except Exception as e: print(f"Error during eval/upload: {e}")
    envs.close()
    if args.track and args.capture_video:
        print("Uploading videos..."); video_base_dir = f"videos/{run_name}"; vids = []
        try:
            vids = sorted(glob.glob(os.path.join(video_base_dir, "**", "*.mp4"), recursive=True), key=os.path.getmtime)
            if not vids: vids = sorted(glob.glob(os.path.join(video_base_dir, "*.mp4")), key=os.path.getmtime)
        except Exception as e: print(f"Error finding videos: {e}")
        if vids:
            print(f"Found {len(vids)} videos. Uploading...");
            for i, fp in enumerate(vids):
                try: wandb.log({f"media/video_train_{i}": wandb.Video(fp,fps=20,format="mp4")},step=global_step)
                except Exception as e: print(f"Error logging video {fp}: {e}")
            print("Video upload done.")
        else: print(f"No videos in {video_base_dir} or subdirs.")
    elif args.capture_video: print("Videos saved locally.")
    writer.close()
    if args.track: print("Finishing W&B run..."); wandb.finish(); print("W&B finished.")
    print("Script finished.")