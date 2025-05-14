# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_continuous_actionpy

import os
import random
import time
from dataclasses import dataclass, field # Added field
from typing import Callable, Optional, Dict # Added Dict

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
import math

@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")] + "_SimpleAGRAM"
    """the name of this experiment"""
    seed: int = 1
    torch_deterministic: bool = True
    cuda: bool = True
    track: bool = True
    wandb_project_name: str = "cleanRL-DynAGO-Simple" # Simplified project name
    wandb_entity: str = None
    capture_video: bool = False
    save_model: bool = False
    upload_model: bool = False
    hf_entity: str = ""
    optimizer: str = "Adam" # Defaulting to Adam, can be changed via CLI
    alpha: float = 0.0 # For AlphaGrad optimizer

    # Algorithm specific arguments from standard PPO
    env_id: str = "HalfCheetah-v4"
    total_timesteps: int = 1000000
    learning_rate: float = 3e-4 # Standard PPO LR
    num_envs: int = 1 # Standard PPO often uses more, but 1 is fine for debugging
    num_steps: int = 2048 # Rollout length per env
    anneal_lr: bool = True
    gamma: float = 0.99
    gae_lambda: float = 0.95
    num_minibatches: int = 32
    update_epochs: int = 10
    norm_adv: bool = True # Normalize the *output* of dynago_transform_advantages
    clip_coef: float = 0.2
    clip_vloss: bool = True
    ent_coef: float = 0.0
    vf_coef: float = 0.5 # Standard PPO vf_coef
    max_grad_norm: float = 0.5
    target_kl: float = None

    # ### AGRAM Controller Hyperparameters ###
    # M_samples is effectively 1 in this simplified script
    dynago_tau: float = 1.25
    dynago_p_star: float = 0.10
    dynago_kappa: float = 2.0
    dynago_eta: float = 0.3
    dynago_rho: float = 0.1
    dynago_eps: float = 1e-5
    dynago_alpha_min: float = 1e-12
    dynago_alpha_max: float = 1e12
    dynago_rho_sat: float = 0.98
    dynago_alpha_A_init: float = 1.0
    dynago_prev_sat_A_init: float = 0.10
    dynago_v_shift: float = 0.0 # This is unused -> please disreagd (this causes sign flippage)

    # to be filled in runtime
    batch_size: int = 0
    minibatch_size: int = 0
    num_iterations: int = 0
    # num_total_samples_per_rollout is not needed if M_samples=1 effectively

# ### AGRAM Controller Function ###
def dynago_transform_advantages(
    raw_advantages_batch: torch.Tensor,
    dynago_params_A: Dict, # Using Dict for type hint
    alpha_A_ema_state: torch.Tensor,
    prev_saturation_A_ema_state: torch.Tensor,
    # device: torch.device, # Not explicitly used if all tensors are already on device
    update_ema: bool = True,
    # Pass args directly to access args.dynago_kappa for the specific formula
    cli_args: Args = None
) -> torch.Tensor:
    current_raw_advantages_MB = raw_advantages_batch

    if current_raw_advantages_MB.numel() <= 1:
        return current_raw_advantages_MB

    kappa_controller = dynago_params_A["kappa"] # Kappa for controller's target alpha
    tau = dynago_params_A["tau"]
    p_star = dynago_params_A["p_star"]
    eta = dynago_params_A["eta"]
    rho = dynago_params_A["rho"]
    eps = dynago_params_A["eps"]
    alpha_min = dynago_params_A["alpha_min"]
    alpha_max = dynago_params_A["alpha_max"]
    rho_sat = dynago_params_A["rho_sat"]

    N_A = torch.linalg.norm(current_raw_advantages_MB)
    if N_A < eps:
        return current_raw_advantages_MB
    sigma_A = torch.std(current_raw_advantages_MB) + eps

    alpha_A_prev_ema_val = alpha_A_ema_state[0] # This is a tensor
    prev_sat_A_ema_val = prev_saturation_A_ema_state[0] # This is a tensor

    alpha_A_hat = (
        kappa_controller
        * (N_A + eps) / (sigma_A + eps)
        * (p_star / (prev_sat_A_ema_val + eps)) ** eta
    )

    alpha_A_to_use_for_Z = None
    if update_ema:
        _alpha_A_updated = (1 - rho) * alpha_A_prev_ema_val + rho * alpha_A_hat
        _alpha_A_updated = torch.clamp(_alpha_A_updated, alpha_min, alpha_max)
        alpha_A_ema_state[0] = _alpha_A_updated.detach() # Update persistent state
        alpha_A_to_use_for_Z = _alpha_A_updated
    else:
        alpha_A_to_use_for_Z = alpha_A_prev_ema_val # Use existing state, don't update

    normalized_advantages_A_MB = current_raw_advantages_MB / (N_A + eps)
    Z_A_MB = alpha_A_to_use_for_Z * normalized_advantages_A_MB

    if update_ema:
        current_observed_saturation_A = (Z_A_MB.abs() > tau).float().mean()
        prev_saturation_A_ema_state[0] = (
            (1 - rho_sat) * prev_sat_A_ema_val + rho_sat * current_observed_saturation_A
        ).detach() # Update persistent state

    # Your specific modulation formula
    # It uses args.dynago_kappa (from cli_args) for the scaling part of tanh output
    if cli_args is None:
        raise ValueError("cli_args must be provided to dynago_transform_advantages for its formula")
    
    modulation_factor = (cli_args.dynago_kappa * torch.tanh(Z_A_MB) + cli_args.dynago_v_shift)
    modulated_advantages_MB = abs(current_raw_advantages_MB) * modulation_factor

    return modulated_advantages_MB

def make_env(env_id, idx, capture_video, run_name, gamma):
    def thunk():
        env = None
        if capture_video and idx == 0:
            if env_id == "f16_intercept":
                from f16env import F16InterceptEnv
                env = F16InterceptEnv()
            else:
                env = gym.make(env_id, render_mode="rgb_array")
            video_folder = f"videos/{run_name}"
            os.makedirs(video_folder, exist_ok=True)
            env = gym.wrappers.RecordVideo(env, video_folder=video_folder)
        else:
            if env_id == "f16_intercept":
                from f16env import F16InterceptEnv
                env = F16InterceptEnv()
            else:
                env = gym.make(env_id)
        env = gym.wrappers.FlattenObservation(env)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.wrappers.ClipAction(env)
        env = gym.wrappers.NormalizeObservation(env)
        env = gym.wrappers.TransformObservation(
            env, 
            lambda obs: np.clip(obs, -10, 10), 
            observation_space=env.observation_space # Explicitly provide it
        )
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
        obs_space_shape_prod = np.array(envs.single_observation_space.shape).prod()
        act_space_shape_prod = np.prod(envs.single_action_space.shape)
        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_space_shape_prod, 64)), nn.Tanh(),
            layer_init(nn.Linear(64, 64)), nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0)
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(obs_space_shape_prod, 64)), nn.Tanh(),
            layer_init(nn.Linear(64, 64)), nn.Tanh(),
            layer_init(nn.Linear(64, act_space_shape_prod), std=0.01)
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, act_space_shape_prod))

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x)

if __name__ == "__main__":
    args = tyro.cli(Args)
    args.batch_size = int(args.num_envs * args.num_steps) # Standard PPO batch size
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    run_name = f"{args.env_id}__{args.exp_name}__{args.optimizer}__{args.seed}__{int(time.time())}"

    if args.track:
        import wandb
        wandb.init(
            project=args.wandb_project_name, entity=args.wandb_entity, sync_tensorboard=True,
            config=vars(args), name=run_name, monitor_gym=True, save_code=True # monitor_gym=True if no custom video handling
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text("hyperparameters", "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])))

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    envs = gym.vector.SyncVectorEnv([make_env(args.env_id, i, args.capture_video, run_name, args.gamma) for i in range(args.num_envs)])
    assert isinstance(envs.single_action_space, gym.spaces.Box)
    agent = Agent(envs).to(device)

    if args.optimizer.lower() == 'adam': optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)
    elif args.optimizer.lower() == 'alphagrad': optimizer = AlphaGrad(agent.parameters(), lr=args.learning_rate, alpha=args.alpha); print("AlphaGrad selected.")
    elif args.optimizer.lower() == 'dag': optimizer = DAG(agent.parameters(), lr=args.learning_rate); print("DAG selected.")
    else: print(f"Unknown optimizer '{args.optimizer}', defaulting to Adam."); optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # AGRAM Controller State Initialization
    derived_kappa_A_controller = args.dynago_kappa # Using args.dynago_kappa for controller's base scaling
    # Optional: You could still derive kappa if args.dynago_kappa was None, but now it's mandatory for your formula.
    # if derived_kappa_A_controller is None: ... (derivation logic) ...
    
    dynago_params_A_config = {
        "kappa": derived_kappa_A_controller, # This kappa is for the controller's target alpha
        "tau": args.dynago_tau, "p_star": args.dynago_p_star, "eta": args.dynago_eta,
        "rho": args.dynago_rho, "eps": args.dynago_eps, "alpha_min": args.dynago_alpha_min,
        "alpha_max": args.dynago_alpha_max, "rho_sat": args.dynago_rho_sat,
    }
    alpha_A_ema_state = torch.tensor([args.dynago_alpha_A_init], dtype=torch.float32, device=device)
    prev_saturation_A_ema_state = torch.tensor([args.dynago_prev_sat_A_init], dtype=torch.float32, device=device)

    # Standard PPO Storage setup
    obs_buffer = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions_buffer = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs_buffer = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards_buffer = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones_buffer = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values_buffer = torch.zeros((args.num_steps, args.num_envs)).to(device)

    global_step = 0; start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)

    for iteration in range(1, args.num_iterations + 1):
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate; optimizer.param_groups[0]["lr"] = lrnow

        for step in range(0, args.num_steps): # This is the rollout loop for standard PPO
            global_step += args.num_envs
            obs_buffer[step] = next_obs
            dones_buffer[step] = next_done

            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values_buffer[step] = value.flatten()
            actions_buffer[step] = action
            logprobs_buffer[step] = logprob

            next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
            next_done_np = np.logical_or(terminations, truncations)
            rewards_buffer[step] = torch.tensor(reward).to(device).view(-1)
            next_obs = torch.Tensor(next_obs).to(device)
            next_done = torch.Tensor(next_done_np).to(device)
            
            # Logging episodic returns (standard CleanRL PPO way)
            if "final_info" in infos:
                for i, info_item in enumerate(infos["final_info"]):
                    if info_item and "episode" in info_item:
                        writer.add_scalar(f"charts/episodic_return_env{i}", info_item['episode']['r'], global_step)
                        writer.add_scalar(f"charts/episodic_length_env{i}", info_item['episode']['l'], global_step)
                        if args.num_envs == 1: print(f"g_step={global_step}, e_ret={info_item['episode']['r']:.2f}")
            elif "_episode" in infos: # Fallback for some vectorized envs
                 for i in range(args.num_envs):
                    if infos["_episode"][i]:
                        writer.add_scalar(f"charts/episodic_return_env{i}", infos["episode"]["r"][i].item(), global_step)
                        writer.add_scalar(f"charts/episodic_length_env{i}", infos["episode"]["l"][i].item(), global_step)
                        if args.num_envs == 1: print(f"g_step={global_step}, e_ret={infos['episode']['r'][i].item():.2f}")


        # Standard PPO GAE Calculation
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1) # Value of S_{T}
            advantages = torch.zeros_like(rewards_buffer).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1: # If last step in rollout
                    nextnonterminal = 1.0 - next_done # Is S_{T} terminal?
                    nextvalues = next_value # V(S_{T})
                else: # Not the last step
                    nextnonterminal = 1.0 - dones_buffer[t + 1] # Is S_{t+1} terminal?
                    nextvalues = values_buffer[t + 1] # V(S_{t+1})
                delta = rewards_buffer[t] + args.gamma * nextvalues * nextnonterminal - values_buffer[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            # returns = advantages + values_buffer # V_target = A_raw + V_old(s)

        b_values = values_buffer.reshape(-1) # These are V_old(s)
        # Flatten the batch
        b_obs = obs_buffer.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs_buffer.reshape(-1)
        b_actions = actions_buffer.reshape((-1,) + envs.single_action_space.shape)
        b_advantages_raw_flat = advantages.reshape(-1) # These are A_raw
        norm_all = torch.linalg.norm(b_advantages_raw_flat) + args.dynago_eps
        Z_full   = alpha_A_ema_state * (b_advantages_raw_flat / norm_all)   # sign‑free # REMOVING ABS TO FLAT
        scale    = args.dynago_kappa * torch.tanh(Z_full) + args.dynago_v_shift   # ≥0
        b_returns_scaled = scale * b_advantages_raw_flat + b_values         # V_target # REMOVING ABS TO FLAT
        # b_returns = returns.reshape(-1) # These are standard on-policy V_targets

        # --- AGRAM Controller Update (Optional: if you want to update EMAs once per iteration) ---
        full_batch_A_mod_for_ema_update = dynago_transform_advantages(
            b_advantages_raw_flat, dynago_params_A_config, alpha_A_ema_state,
            prev_saturation_A_ema_state, update_ema=True, cli_args=args
        )

        # --- PPO Update Loop ---
        b_inds = np.arange(args.batch_size)
        clipfracs = []; avg_raw_adv_epoch_list = []; avg_mod_adv_epoch_list = []

        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            epoch_raw_advs_mb, epoch_mod_advs_mb = [], [] # Store per-minibatch means

            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size; mb_inds = b_inds[start:end]

                # Get current policy outputs for this minibatch
                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]; ratio = logratio.exp()

                with torch.no_grad():
                    old_approx_kl = (-logratio).mean(); approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_advantages_raw = b_advantages_raw_flat[mb_inds]
                epoch_raw_advs_mb.append(mb_advantages_raw.mean().item())

                # --- Apply AGRAM modulation for Policy Loss (Frozen EMA) ---
                mb_advantages_mod = dynago_transform_advantages(
                    mb_advantages_raw,
                    dynago_params_A_config,
                    alpha_A_ema_state,
                    prev_saturation_A_ema_state,
                    # device, # Not needed by function if tensors are on device
                    update_ema=False, # FREEZE EMA states during epoch updates
                    cli_args=args     # Pass full args for the formula
                )
                epoch_mod_advs_mb.append(mb_advantages_mod.mean().item())
                # --- End Advantage Modulation ---

                mb_advantages_final_policy = mb_advantages_mod
                if args.norm_adv: # Normalize the MODULATED advantages for policy
                    mb_advantages_final_policy = (mb_advantages_final_policy - mb_advantages_final_policy.mean()) / \
                                                 (mb_advantages_final_policy.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages_final_policy * ratio
                pg_loss2 = -mb_advantages_final_policy * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss (Standard PPO - uses unmodulated, on-policy returns)
                newvalue = newvalue.view(-1)
                with torch.no_grad():
                    mb_returns_scaled = mb_advantages_mod + b_values[mb_inds]   # signed & already scaled
                mb_target_values = mb_returns_scaled # PRESERVING SCALE
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - mb_target_values) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds], -args.clip_coef, args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - mb_target_values) ** 2
                    v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()
                else:
                    v_loss = 0.5 * ((newvalue - mb_target_values) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm); optimizer.step()
            
            if epoch_raw_advs_mb: avg_raw_adv_epoch_list.append(np.mean(epoch_raw_advs_mb))
            if epoch_mod_advs_mb: avg_mod_adv_epoch_list.append(np.mean(epoch_mod_advs_mb))
            if args.target_kl is not None and approx_kl > args.target_kl: break
        
        # Logging
        # y_pred, y_true = b_values.cpu().numpy(), b_returns_scaled.cpu().numpy()
        y_pred, y_true = b_values.cpu().numpy(), (full_batch_A_mod_for_ema_update + b_values).cpu().numpy() # Testing corrected explained var calculation
        var_y = np.var(y_true); explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
        
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        writer.add_scalar("charts/SPS", int(args.num_envs * args.num_steps / (time.time() - start_time)), global_step) # Correct SPS for one iteration
        start_time = time.time() # Reset start_time for next iteration's SPS

        if alpha_A_ema_state.numel() > 0: writer.add_scalar("agram_controller/alpha_A_ema", alpha_A_ema_state.item(), global_step)
        if prev_saturation_A_ema_state.numel() > 0: writer.add_scalar("agram_controller/prev_saturation_A_ema", prev_saturation_A_ema_state.item(), global_step)
        if avg_raw_adv_epoch_list: writer.add_scalar("advantages/mean_raw_adv_iteration", np.mean(avg_raw_adv_epoch_list), global_step)
        if avg_mod_adv_epoch_list: writer.add_scalar("advantages/mean_mod_adv_iteration", np.mean(avg_mod_adv_epoch_list), global_step)

    # Model saving, evaluation, and cleanup (standard CleanRL)
    if args.save_model:
        model_path = f"runs/{run_name}/{args.exp_name}.cleanrl_model"
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        torch.save(agent.state_dict(), model_path)
        print(f"model saved to {model_path}")
        try:
            from cleanrl_utils.evals.ppo_eval import evaluate
            eval_returns = evaluate(
                model_path, make_env, args.env_id, eval_episodes=10, run_name=f"{run_name}-eval",
                Model=Agent, device=device, gamma=args.gamma
            )
            writer.add_scalar("eval/mean_episodic_return", np.mean(eval_returns), global_step)
            if args.upload_model:
                from cleanrl_utils.huggingface import push_to_hub
                repo_id = f"{args.hf_entity}/{args.env_id}-{args.exp_name}-seed{args.seed}" if args.hf_entity else f"{args.env_id}-{args.exp_name}-seed{args.seed}"
                push_to_hub(args=vars(args), episodic_returns=eval_returns, repo_id=repo_id,
                            commit_message="Upload AGRAM PPO model", model_filename=model_path)
        except ImportError: print("cleanrl_utils not found, skipping eval/upload.")
        except Exception as e: print(f"Error during eval/upload: {e}")

    envs.close()
    if args.track and args.capture_video:
        # Simplified video logging, assuming RecordVideo wrapper handles naming and wandb can find them
        video_folder_to_log = f"videos/{run_name}"
        # For this to work well, RecordVideo should save videos with fixed names or wandb.Video needs specific paths
        # This is a common point of friction. CleanRL's RecordVideo default might save to a subfolder.
        # If videos are in video_folder_to_log directly:
        mp4_files = glob.glob(os.path.join(video_folder_to_log, "*.mp4"))
        for i, vid_f in enumerate(mp4_files):
             wandb.log({f"media/video_train_iter_{iteration}_ep{i}": wandb.Video(vid_f, fps=20, format="mp4")}, step=global_step)

    writer.close()
    if args.track: wandb.finish()
    print("Script finished.")