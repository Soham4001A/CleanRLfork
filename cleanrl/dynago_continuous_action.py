# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_continuous_actionpy

import os
import random
import time
from dataclasses import dataclass, field
from typing import Callable 

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

@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = True
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "cleanRL-DynAGO"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    save_model: bool = False
    """whether to save model into the `runs/{run_name}` folder"""
    upload_model: bool = False
    """whether to upload the saved model to huggingface"""
    hf_entity: str = ""
    """the user or org name of the model repository from the Hugging Face Hub"""
    optimizer: str = "Adam"
    """ Optimizer to use: Adam, AlphaGrad, DAG """
    alpha: float = 0.0 # For AlphaGrad
    """ Alpha value for AlphaGrad if selected"""

    # Algorithm specific arguments
    env_id: str = "HalfCheetah-v4"
    """the id of the environment"""
    total_timesteps: int = 1000000
    """total timesteps of the experiments"""
    learning_rate: float = 3e-4
    """the learning rate of the optimizer"""
    num_envs: int = 1
    """the number of parallel game environments (conceptual environments for DynAGO)"""
    num_steps: int = 2048 # Number of *committed* steps per rollout per env
    """the number of committed steps to run in each environment per policy rollout"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.0 # ### DynAGO ###: Defaulted to 0.0 for DynAGO's core reward processing
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 32
    """the number of mini-batches"""
    update_epochs: int = 10
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.2
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.0
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: float = None
    """the target KL divergence threshold"""

    # ### DynAGO ###: New arguments
    M_samples: int = 5
    """Number of actions to sample per state per conceptual step for DynAGO"""
    dynago_h_kappa_R: float = 1.5 
    """Base scaling for DynAGO reward alpha controller"""
    dynago_h_rho_R: float = 0.1
    """EMA smoothing factor for DynAGO reward alpha"""
    dynago_h_eps_R: float = 1e-5
    """Epsilon for DynAGO reward alpha controller"""
    dynago_alpha_R_init: float = 10.0 
    """Initial value for the DynAGO reward alpha_R EMA"""
    
    # to be filled in runtime
    batch_size: int = 0
    minibatch_size: int = 0
    num_iterations: int = 0
    num_total_samples_per_rollout: int = 0 # Total samples in buffer: num_steps * M_samples


# ### DynAGO ###: Function to transform rewards
def dynago_transform_rewards(
    raw_rewards_tensor: torch.Tensor, # Shape: (num_conceptual_envs, M_samples)
    dynago_h_params: dict,
    alpha_R_ema_tensor: torch.Tensor, # Shape: (num_conceptual_envs,) or (1,) if global
    device: torch.device,
) -> torch.Tensor:
    num_conceptual_envs, M_samples = raw_rewards_tensor.shape
    modulated_rewards_list = []

    for i in range(num_conceptual_envs):
        current_raw_rewards_M = raw_rewards_tensor[i, :] 
        N_R = torch.linalg.norm(current_raw_rewards_M)
        normalized_rewards_R_M = current_raw_rewards_M / (N_R + dynago_h_params["eps_R"])
        
        sigma_R = torch.std(current_raw_rewards_M) + dynago_h_params["eps_R"]
        if sigma_R < dynago_h_params["eps_R"] * 2 : 
             sigma_R = torch.tensor(1.0, device=device) 
        
        N_R_for_hat_alpha = N_R if N_R > dynago_h_params["eps_R"] else torch.tensor(1.0, device=device)
        hat_alpha_R = dynago_h_params["kappa_R"] * (N_R_for_hat_alpha / sigma_R)
        hat_alpha_R = torch.clamp(hat_alpha_R, 1e-3, 1e3) # Basic clamp

        current_alpha_val_idx = 0 if alpha_R_ema_tensor.shape[0] == 1 else i
        current_alpha_val = alpha_R_ema_tensor[current_alpha_val_idx]
        
        new_alpha_val = (1 - dynago_h_params["rho_R"]) * current_alpha_val + dynago_h_params["rho_R"] * hat_alpha_R
        alpha_R_ema_tensor[current_alpha_val_idx] = new_alpha_val.detach()
        
        Z_R_M = new_alpha_val * normalized_rewards_R_M
        T_R_M = torch.tanh(Z_R_M)
        r_modulated_M = current_raw_rewards_M * T_R_M
        modulated_rewards_list.append(r_modulated_M.unsqueeze(0))
        
    return torch.cat(modulated_rewards_list, dim=0)


def make_env(env_id, idx, capture_video, run_name, gamma):
    def thunk():
        env = None 
        if capture_video and idx == 0: # Only record video for the first environment (if num_envs > 1)
            print(f"Setting up environment {idx} for video capture (render_mode='rgb_array')")
            env = gym.make(env_id, render_mode="rgb_array")
            video_folder = f"videos/{run_name}"
            # Ensure video_folder exists, RecordVideo might not create it.
            os.makedirs(video_folder, exist_ok=True)
            print(f"Recording video for env {idx} to {video_folder}")
            env = gym.wrappers.RecordVideo(env, video_folder=video_folder, 
                                           # Record all episodes for the first env if capture_video is True
                                           episode_trigger=lambda x: True if idx == 0 else False) 
        else:
            env = gym.make(env_id)

        env = gym.wrappers.FlattenObservation(env)
        env = gym.wrappers.RecordEpisodeStatistics(env) 
        env = gym.wrappers.ClipAction(env)
        env = gym.wrappers.NormalizeObservation(env)
        env = gym.wrappers.TransformObservation(
            env, 
            lambda obs: np.clip(obs, -10, 10), 
            observation_space=env.observation_space
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
        obs_prod = np.array(envs.single_observation_space.shape).prod()
        act_prod = np.prod(envs.single_action_space.shape) # Works for Box, might need adjustment for other spaces
        
        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_prod, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(obs_prod, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, act_prod), std=0.01),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, act_prod))

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
    args.num_total_samples_per_rollout = args.num_steps * args.M_samples
    args.batch_size = int(args.num_envs * args.num_total_samples_per_rollout)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    # Number of PPO updates / iterations
    args.num_iterations = args.total_timesteps // (args.num_envs * args.num_steps) 

    run_name = f"{args.env_id}__{args.exp_name}-M{args.M_samples}__{args.optimizer}__{args.seed}__{int(time.time())}"
    if args.track:
        import wandb
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True, 
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, i, args.capture_video, run_name, args.gamma) for i in range(args.num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported for this script"

    agent = Agent(envs).to(device)
    if args.optimizer.lower() == 'adam':
        optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)
    elif args.optimizer.lower() == 'alphagrad':
        optimizer = AlphaGrad(agent.parameters(), lr=args.learning_rate, alpha=args.alpha) # Add other AlphaGrad params if needed
        print(f"AlphaGrad optimizer selected with alpha={args.alpha}.")
    elif args.optimizer.lower() == 'dag':
        optimizer = DAG(agent.parameters(), lr=args.learning_rate) # Add other DAG params if needed
        print("DAG optimizer selected.")
    else: # Default to Adam if unknown optimizer
        print(f"Unknown optimizer '{args.optimizer}', defaulting to Adam.")
        optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    obs_shape = envs.single_observation_space.shape
    action_shape = envs.single_action_space.shape
    
    # Buffers store all M_samples for each conceptual step, then for each env
    obs_buffer = torch.zeros((args.num_total_samples_per_rollout, args.num_envs) + obs_shape).to(device)
    actions_buffer = torch.zeros((args.num_total_samples_per_rollout, args.num_envs) + action_shape).to(device)
    logprobs_buffer = torch.zeros((args.num_total_samples_per_rollout, args.num_envs)).to(device)
    rewards_buffer = torch.zeros((args.num_total_samples_per_rollout, args.num_envs)).to(device) # Will store modulated rewards
    dones_buffer = torch.zeros((args.num_total_samples_per_rollout, args.num_envs)).to(device)
    values_buffer = torch.zeros((args.num_total_samples_per_rollout, args.num_envs)).to(device) # V(s_t)
    next_obs_values_buffer = torch.zeros((args.num_total_samples_per_rollout, args.num_envs)).to(device) # V(s'_tj)

    dynago_h_params_config = {
        "kappa_R": args.dynago_h_kappa_R,
        "rho_R": args.dynago_h_rho_R,
        "eps_R": args.dynago_h_eps_R,
    }
    # Global alpha_R EMA state. If per-env adaptation is desired later, this would be a tensor of size (args.num_envs,)
    alpha_R_ema = torch.tensor([args.dynago_alpha_R_init], dtype=torch.float32, device=device)

    global_step = 0 # Tracks committed environment steps
    start_time = time.time()
    next_obs_conceptual, _ = envs.reset(seed=args.seed) # State of conceptual envs
    next_obs_conceptual = torch.Tensor(next_obs_conceptual).to(device)
    next_done_conceptual = torch.zeros(args.num_envs).to(device)

    for iteration in range(1, args.num_iterations + 1):
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        # s_iter iterates through conceptual steps (args.num_steps is # of committed steps)
        for s_iter in range(0, args.num_steps):
            current_conceptual_obs_at_s_iter = next_obs_conceptual 
            
            # Lists to hold the final STACKED tensors of M_samples for each conceptual env
            list_of_stacked_s_t = [None] * args.num_envs
            list_of_stacked_raw_rewards = [None] * args.num_envs
            list_of_stacked_actions = [None] * args.num_envs
            list_of_stacked_logprobs = [None] * args.num_envs
            list_of_stacked_values_s_t = [None] * args.num_envs
            list_of_stacked_s_prime_for_commit_selection = [None] * args.num_envs
            list_of_stacked_dones_primes = [None] * args.num_envs
            list_of_stacked_next_obs_prime_values = [None] * args.num_envs
            
            # To store the actions that will be committed to advance the true env state
            committed_actions_for_true_env_step = torch.zeros((args.num_envs,) + action_shape).to(device)

            for env_idx in range(args.num_envs):
                s_t_current_env = current_conceptual_obs_at_s_iter[env_idx:env_idx+1] # Shape (1, obs_dim)
                env_instance = envs.envs[env_idx].unwrapped 
                env_snapshot = None
                is_mujoco_env_type = "mujoco" in str(type(env_instance)).lower()

                # Attempt to get environment state snapshot
                if is_mujoco_env_type and hasattr(env_instance, "physics") and \
                   hasattr(env_instance.physics, "get_state") and hasattr(env_instance.physics, "set_state"):
                    try:
                        env_snapshot = env_instance.physics.get_state()
                        if isinstance(env_snapshot, np.ndarray): env_snapshot = env_snapshot.copy()
                    except Exception as e:
                        if s_iter == 0 and env_idx == 0: print(f"WARNING: MuJoCo get_state failed for env {env_idx}: {e}")
                        env_snapshot = None
                elif hasattr(env_instance, "get_state") and hasattr(env_instance, "set_state"):
                    try:
                        env_snapshot = env_instance.get_state()
                        if isinstance(env_snapshot, np.ndarray): env_snapshot = env_snapshot.copy()
                    except Exception as e:
                        if s_iter == 0 and env_idx == 0: print(f"WARNING: General get_state failed for env {env_idx}: {e}")
                        env_snapshot = None
                
                if env_snapshot is None and s_iter == 0 and env_idx == 0: # Print once per run if cloning fails for the first env
                    print("WARNING: State cloning/restoration method not found or failed. M-sampling will be sequential.")

                # Temporary Python lists for the current env_idx's M samples
                current_env_s_t_list_M, current_env_raw_rewards_list_M, current_env_actions_list_M = [], [], []
                current_env_logprobs_list_M, current_env_values_s_t_list_M = [], []
                current_env_s_prime_list_M_for_commit, current_env_dones_primes_list_M = [], []
                current_env_next_obs_prime_values_list_M = [], []

                for k_sample in range(args.M_samples):
                    if env_snapshot is not None: # Attempt to restore state if snapshot exists
                        try:
                            if is_mujoco_env_type and hasattr(env_instance.physics, "set_state"):
                                env_instance.physics.set_state(env_snapshot)
                            elif hasattr(env_instance, "set_state"):
                                env_instance.set_state(env_snapshot)
                        except Exception as e:
                            if s_iter == 0 and env_idx == 0 and k_sample == 0: print(f"WARNING: State setting error for env {env_idx}, sample {k_sample}: {e}")
                    
                    with torch.no_grad():
                        action_k, logprob_k, _, value_s_t_k = agent.get_action_and_value(s_t_current_env)
                    
                    action_k_np = action_k.cpu().numpy().reshape(envs.single_action_space.shape)
                    next_obs_prime_k, raw_reward_k, term_k, trunc_k, _ = env_instance.step(action_k_np)
                    done_prime_k = term_k or trunc_k

                    current_env_s_t_list_M.append(s_t_current_env.clone()) # s_t
                    current_env_raw_rewards_list_M.append(torch.tensor([raw_reward_k], device=device))
                    current_env_actions_list_M.append(action_k.clone())
                    current_env_logprobs_list_M.append(logprob_k.clone())
                    current_env_values_s_t_list_M.append(value_s_t_k.clone()) # V(s_t)
                    
                    next_obs_prime_k_tensor = torch.Tensor(next_obs_prime_k).to(device).unsqueeze(0)
                    current_env_s_prime_list_M_for_commit.append(next_obs_prime_k_tensor.clone()) # s'_tj
                    current_env_dones_primes_list_M.append(torch.tensor([done_prime_k], dtype=torch.float32, device=device)) # d_tj

                    with torch.no_grad(): # V(s'_tj)
                        value_s_prime_k = agent.get_value(next_obs_prime_k_tensor)
                    current_env_next_obs_prime_values_list_M.append(value_s_prime_k.clone())

                if args.M_samples <= 0: raise ValueError("args.M_samples must be > 0.")
                if not current_env_s_t_list_M : # Should not happen if M_samples > 0
                    print(f"CRITICAL ERROR: No samples collected for env_idx {env_idx} in s_iter {s_iter}, M_samples={args.M_samples}")
                    # This indicates a deeper issue; for now, we'll let it error on cat if this happens.
                    # A robust solution might involve skipping or default-filling.
                
                list_of_stacked_s_t[env_idx] = torch.cat(current_env_s_t_list_M, dim=0)
                list_of_stacked_raw_rewards[env_idx] = torch.cat(current_env_raw_rewards_list_M, dim=0).squeeze(-1)
                list_of_stacked_actions[env_idx] = torch.cat(current_env_actions_list_M, dim=0)
                list_of_stacked_logprobs[env_idx] = torch.cat(current_env_logprobs_list_M, dim=0)
                list_of_stacked_values_s_t[env_idx] = torch.cat(current_env_values_s_t_list_M, dim=0).squeeze(-1)
                list_of_stacked_s_prime_for_commit_selection[env_idx] = torch.cat(current_env_s_prime_list_M_for_commit, dim=0)
                list_of_stacked_dones_primes[env_idx] = torch.cat(current_env_dones_primes_list_M, dim=0).squeeze(-1)
                list_of_stacked_next_obs_prime_values[env_idx] = torch.cat(current_env_next_obs_prime_values_list_M, dim=0).squeeze(-1)
            
            # --- Stage 2: Modulate Rewards and Select Committed Actions ---
            if any(t is None for t in list_of_stacked_raw_rewards): # Should be populated if M_samples > 0
                raise RuntimeError("Logic error: Not all conceptual environments produced stacked raw reward tensors.")
            
            all_raw_rewards_tensor_for_transform = torch.stack(list_of_stacked_raw_rewards, dim=0) # Shape: (num_envs, M_samples)
            
            all_modulated_rewards_tensor = dynago_transform_rewards( # Shape (num_envs, M_samples)
                all_raw_rewards_tensor_for_transform, dynago_h_params_config, alpha_R_ema, device
            )
            
            temp_next_obs_list_for_commit = [] # To build next_obs_from_committed_paths
            temp_next_done_list_for_commit = [] # To build next_done_from_committed_paths

            for env_idx_commit in range(args.num_envs):
                modulated_rewards_for_env = all_modulated_rewards_tensor[env_idx_commit, :] # (M_samples,)
                best_k_idx = torch.argmax(modulated_rewards_for_env) # Index of best sample for this env
                
                committed_actions_for_true_env_step[env_idx_commit] = list_of_stacked_actions[env_idx_commit][best_k_idx]
                temp_next_obs_list_for_commit.append(list_of_stacked_s_prime_for_commit_selection[env_idx_commit][best_k_idx].unsqueeze(0)) # (1, obs_dim)
                temp_next_done_list_for_commit.append(list_of_stacked_dones_primes[env_idx_commit][best_k_idx].unsqueeze(0)) # (1,)
            
            # --- Stage 3: Store ALL M_samples * num_envs into global PPO buffers ---
            for k_fill in range(args.M_samples):
                buffer_row_idx = s_iter * args.M_samples + k_fill # Row in the global buffers
                for env_idx_fill in range(args.num_envs):
                    obs_buffer[buffer_row_idx, env_idx_fill] = list_of_stacked_s_t[env_idx_fill][k_fill]
                    actions_buffer[buffer_row_idx, env_idx_fill] = list_of_stacked_actions[env_idx_fill][k_fill]
                    logprobs_buffer[buffer_row_idx, env_idx_fill] = list_of_stacked_logprobs[env_idx_fill][k_fill]
                    rewards_buffer[buffer_row_idx, env_idx_fill] = all_modulated_rewards_tensor[env_idx_fill, k_fill] 
                    dones_buffer[buffer_row_idx, env_idx_fill] = list_of_stacked_dones_primes[env_idx_fill][k_fill]
                    values_buffer[buffer_row_idx, env_idx_fill] = list_of_stacked_values_s_t[env_idx_fill][k_fill]
                    next_obs_values_buffer[buffer_row_idx, env_idx_fill] = list_of_stacked_next_obs_prime_values[env_idx_fill][k_fill]

            # --- Update conceptual env state for next s_iter using the actual committed step in SyncVectorEnv ---
            # Critical: The state of `envs.envs[i]` should ideally be `current_conceptual_obs_at_s_iter[i]`
            # before this `envs.step` call for `infos` to be perfectly aligned. This depends on effective state restoration.
            actual_next_obs_sv, _, actual_terminations_sv, actual_truncations_sv, infos = envs.step(
                committed_actions_for_true_env_step.cpu().numpy()
            )
            actual_next_done_sv = np.logical_or(actual_terminations_sv, actual_truncations_sv)

            # Update the conceptual state for the next s_iter using results from the SyncVectorEnv step
            next_obs_conceptual = torch.Tensor(actual_next_obs_sv).to(device)
            next_done_conceptual = torch.Tensor(actual_next_done_sv).to(device)
            
            global_step += args.num_envs # Tracks committed environment steps

            # Logging episode statistics from the committed path via SyncVectorEnv's infos
            if "final_info" in infos:
                for i_env_info, info_item in enumerate(infos["final_info"]):
                    if info_item is not None and "episode" in info_item:
                        # Log per-environment to avoid conflicts if args.num_envs > 1
                        writer.add_scalar(f"charts/episodic_return_env{i_env_info}", info_item['episode']['r'], global_step)
                        writer.add_scalar(f"charts/episodic_length_env{i_env_info}", info_item['episode']['l'], global_step)
                        if i_env_info == 0 : # Print for the first env to reduce console spam
                             print(f"global_step={global_step}, env_idx={i_env_info}, episodic_return={info_item['episode']['r']}")
            # Fallback for older gym/CleanRL patterns if final_info is not present but episode is
            elif "episode" in infos and isinstance(infos["episode"].get("r"), np.ndarray): 
                for i_env_info in range(len(infos["episode"]["r"])):
                    if infos["_episode"][i_env_info]: # Check if episode actually ended for this specific env
                        episodic_return = infos["episode"]["r"][i_env_info].item()
                        episodic_length = infos["episode"]["l"][i_env_info].item()
                        writer.add_scalar(f"charts/episodic_return_env{i_env_info}", episodic_return, global_step)
                        writer.add_scalar(f"charts/episodic_length_env{i_env_info}", episodic_length, global_step)
                        if i_env_info == 0 :
                             print(f"global_step={global_step}, env_idx={i_env_info}, episodic_return={episodic_return}")
        
        # GAE Calculation (Advantage and Returns)
        with torch.no_grad():
            # Bootstrap value from the state of conceptual envs after the last committed step of the rollout
            end_of_rollout_next_value = agent.get_value(next_obs_conceptual).reshape(1, args.num_envs) # V(S_T+1)

            advantages = torch.zeros_like(rewards_buffer).to(device) # Shape: (num_total_samples, num_envs)
            lastgaelam = torch.zeros(args.num_envs).to(device) # For GAE, per env

            # Iterate over all collected samples in the buffers (num_total_samples_per_rollout = num_steps * M_samples)
            for t in reversed(range(args.num_total_samples_per_rollout)):
                s_iter_of_t = t // args.M_samples # Conceptual step this sample 't' belongs to
                
                if s_iter_of_t == args.num_steps - 1: # Sample 't' is from the last conceptual step of the rollout
                    # For all M samples of this last conceptual step, the "true" next value for GAE
                    # bootstrapping comes from V(S_T+1) of the committed trajectory for that env.
                    current_target_V_s_prime = end_of_rollout_next_value.squeeze(0) # Shape (num_envs,)
                else:
                    # For samples not in the last conceptual step, their V(s') is V(s'_tj)
                    # which we stored in next_obs_values_buffer.
                    current_target_V_s_prime = next_obs_values_buffer[t] # Shape (num_envs,)
                
                nextnonterminal_for_delta = 1.0 - dones_buffer[t] # dones_buffer[t] is d_tj for sample t
                # rewards_buffer[t] is r_mod_tj; values_buffer[t] is V(s_t)
                delta = rewards_buffer[t] + args.gamma * current_target_V_s_prime * nextnonterminal_for_delta - values_buffer[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal_for_delta * lastgaelam
            
            returns = advantages + values_buffer # Target for value function update

        # Flatten the batch from (num_total_samples, num_envs, ...) to (total_batch_size, ...)
        b_obs = obs_buffer.reshape((-1,) + obs_shape)
        b_logprobs = logprobs_buffer.reshape(-1)
        b_actions = actions_buffer.reshape((-1,) + action_shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values_buffer.reshape(-1) # V(s_t) from buffer

        # Optimizing the policy and value network (Standard PPO update part)
        b_inds = np.arange(args.batch_size) # args.batch_size = num_envs * num_total_samples_per_rollout
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds], -args.clip_coef, args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None and approx_kl > args.target_kl:
                break
        
        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # Logging
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        writer.add_scalar("charts/SPS", int(args.num_envs * args.num_steps / (time.time() - start_time)), global_step) # SPS based on committed steps
        if alpha_R_ema.numel() > 0 : # Check if alpha_R_ema is not empty
            writer.add_scalar("dynago/alpha_R_ema", alpha_R_ema.mean().item(), global_step) 

    # Save model
    if args.save_model:
        model_path = f"runs/{run_name}/{args.exp_name}.cleanrl_model"
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        torch.save(agent.state_dict(), model_path)
        print(f"model saved to {model_path}")
        
        # Evaluation (assuming cleanrl_utils is available)
        try:
            from cleanrl_utils.evals.ppo_eval import evaluate 
            episodic_returns_eval = evaluate(
                model_path, make_env, args.env_id, eval_episodes=10, run_name=f"{run_name}-eval",
                Model=Agent, device=device, gamma=args.gamma,
            )
            for idx, episodic_return_eval_item in enumerate(episodic_returns_eval):
                writer.add_scalar("eval/episodic_return", episodic_return_eval_item, idx)

            if args.upload_model:
                from cleanrl_utils.huggingface import push_to_hub
                repo_name = f"{args.env_id}-{args.exp_name}-seed{args.seed}"
                repo_id = f"{args.hf_entity}/{repo_name}" if args.hf_entity else repo_name
                # Ensure videos_dir for eval is passed if eval also captures videos
                # For now, assuming eval videos are not captured or handled by `evaluate`
                push_to_hub(args=vars(args), 
                            episodic_returns=episodic_returns_eval, 
                            repo_id=repo_id, 
                            commit_message="Upload PPO model",
                            model_filename=model_path,
                            # videos_dir=f"videos/{run_name}-eval" # If eval videos exist
                            )
        except ImportError:
            print("cleanrl_utils not found, skipping model evaluation and upload.")
        except Exception as e_eval:
            print(f"Error during model evaluation or upload: {e_eval}")

    envs.close()
    # Video upload logic for training videos
    if args.track and args.capture_video:
        print("Attempting to upload recorded training videos to W&B...")
        # RecordVideo saves videos in subfolders like 'rl-video-episode-0' if triggered per episode
        # Or directly in video_folder if not using episode_trigger or a fixed trigger.
        # The glob pattern needs to be robust.
        video_base_dir = f"videos/{run_name}"
        # Search for mp4 files, **/* accounts for potential subdirectories created by RecordVideo
        video_files = sorted(glob.glob(os.path.join(video_base_dir, "**", "*.mp4"), recursive=True), key=os.path.getmtime)
        
        if not video_files: # Fallback to non-recursive if the above finds nothing
             video_files = sorted(glob.glob(os.path.join(video_base_dir, "*.mp4")), key=os.path.getmtime)

        if video_files:
            print(f"Found {len(video_files)} video file(s) in {video_base_dir} and subdirectories. Uploading...")
            for i, video_file_path in enumerate(video_files):
                # Ensure the video key is unique and descriptive
                wandb_video_key = f"media/video_capture_env0_training_{i}" 
                print(f" - Logging {video_file_path} as {wandb_video_key}")
                try:
                    wandb.log({wandb_video_key: wandb.Video(video_file_path, fps=20, format="mp4")}, step=global_step)
                except Exception as e_vid_upload:
                    print(f"Error logging video {video_file_path} to W&B: {e_vid_upload}")
            print("Video upload attempt complete.")
        else:
            print(f"Warning: No .mp4 video files found in {video_base_dir} or its subdirectories for training.")
    elif args.capture_video:
         print("Info: Video capture was enabled but W&B tracking is off. Training videos saved locally only.")

    writer.close()
    if args.track:
         print("Finishing W&B run...")
         wandb.finish()
         print("W&B run finished.")
    print("Script execution finished.")
         print("Finishing W&B run...")
         wandb.finish()
         print("W&B run finished.")
    print("Script execution finished.")