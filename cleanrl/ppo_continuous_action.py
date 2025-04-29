# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_continuous_actionpy
import os
import random
import time
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
# Assuming AlphaGrad is correctly located like this:
# from optim.sgd import AlphaGrad
# If not, adjust the import path or remove if not using AlphaGrad
try:
    from optim.sgd import AlphaGrad
except ImportError:
    print("Warning: AlphaGrad optimizer not found. Only Adam will be available.")
    AlphaGrad = None # Define as None if not found

import tyro
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter

# === REMOVED THE WANDB PATCH BLOCK ===
# No longer needed as we rely on monitor_gym=True

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
    wandb_project_name: str = "cleanRL"
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
    optimizer: str = "Adam" # Default to Adam
    """ Optimizer to use (e.g., Adam, AlphaGrad)"""
    alpha: float = 0.0
    """ Alpha value for AlphaGrad"""

    # Algorithm specific arguments
    env_id: str = "HalfCheetah-v4" # Note: You ran with v5, ensure consistency or use v4
    """the id of the environment"""
    total_timesteps: int = 1000000
    """total timesteps of the experiments"""
    learning_rate: float = 3e-4
    """the learning rate of the optimizer"""
    num_envs: int = 1
    """the number of parallel game environments"""
    num_steps: int = 2048
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.95
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

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""


def make_env(env_id, idx, capture_video, run_name, gamma):
    def thunk():
        # Set render_mode to 'rgb_array' if capture_video is True
        render_mode = "rgb_array" if capture_video else None
        print(f"[{idx}] Creating env {env_id} with render_mode='{render_mode}'") # Debug print
        env = gym.make(env_id, render_mode=render_mode)

        # === START DEBUG RENDER TEST ===
        if render_mode == "rgb_array" and idx == 0: # Only test for the env that should capture video
             print(f"[{idx}] Attempting initial render test...")
             try:
                 # For Gymnasium v0.26+, render() returns the frame directly for rgb_array
                 frame = env.render()
                 if frame is not None and isinstance(frame, np.ndarray):
                     print(f"[{idx}] Initial render successful! Frame shape: {frame.shape}, dtype: {frame.dtype}")
                 elif frame is None:
                      # In some older gym/mujoco versions, render might return None and update an internal viewer
                      # but for rgb_array it *should* return the array. This is likely an issue.
                      print(f"[{idx}] Initial render returned None. This indicates a potential rendering problem for rgb_array.")
                 else:
                      print(f"[{idx}] Initial render returned unexpected type: {type(frame)}")
             except Exception as e:
                 print(f"[{idx}] ERROR during initial render test: {e}")
                 import traceback
                 traceback.print_exc() # Print detailed traceback for rendering errors
                 # Consider exiting if render fails fundamentally
                 # raise e # Optional: Stop execution if rendering fails
        # === END DEBUG RENDER TEST ===


        # === RE-ENABLE RecordVideo (if capture_video is True) ===
        # Keep this enabled for testing direct recording
        if capture_video and idx == 0:
             video_folder = f"videos/{run_name}"
             print(f"[{idx}] Wrapping with RecordVideo, saving to: {video_folder}")
             os.makedirs(video_folder, exist_ok=True)
             env = gym.wrappers.RecordVideo(
                 env,
                 video_folder=video_folder,
                 # episode_trigger=lambda x: x % 1 == 0 # Record every episode (or default first)
                 name_prefix=f"{env_id}-ep"
             )
        # === END RecordVideo ===


        env = gym.wrappers.RecordEpisodeStatistics(env)

        # --- Make sure the rest of your wrappers are here ---
        env = gym.wrappers.FlattenObservation(env)
        env = gym.wrappers.ClipAction(env)
        env = gym.wrappers.NormalizeObservation(env)

        current_obs_space = env.observation_space
        assert isinstance(current_obs_space, gym.spaces.Box), \
            f"Expected Box space after NormalizeObservation, got {type(current_obs_space)}"
        clip_low = -10.0
        clip_high = 10.0
        low = np.full(current_obs_space.shape, clip_low, dtype=current_obs_space.dtype)
        high = np.full(current_obs_space.shape, clip_high, dtype=current_obs_space.dtype)
        new_obs_space = gym.spaces.Box(
            low=low,
            high=high,
            shape=current_obs_space.shape,
            dtype=current_obs_space.dtype
        )
        env = gym.wrappers.TransformObservation(
            env,
            lambda obs: np.clip(obs, clip_low, clip_high),
            observation_space=new_obs_space
        )

        env = gym.wrappers.NormalizeReward(env, gamma=gamma)
        env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))
        # --- End of wrappers ---

        print(f"[{idx}] Environment creation complete.") # Debug print
        return env

    return thunk
# ===================================


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, np.prod(envs.single_action_space.shape)), std=0.01),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, np.prod(envs.single_action_space.shape)))

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
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"

    if args.track:
        import wandb

        # Ensure monitor_gym=True is set (it is by default if installed)
        # W&B will now handle video recording if capture_video=True was used in make_env to set render_mode
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True, # IMPORTANT: This enables W&B video recording
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    envs = gym.vector.SyncVectorEnv(
        # Pass capture_video flag to make_env so it can set render_mode
        [make_env(args.env_id, i, args.capture_video, run_name, args.gamma) for i in range(args.num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    agent = Agent(envs).to(device)

    # === Fixed Optimizer Selection Logic ===
    if args.optimizer.lower() == 'alphagrad' and AlphaGrad is not None:
        print(f"Using AlphaGrad optimizer with alpha={args.alpha} lr={args.learning_rate}")
        optimizer = AlphaGrad(agent.parameters(), lr=args.learning_rate, alpha=args.alpha, epsilon=1e-5, momentum=0.9)
    elif args.optimizer.lower() == 'adam':
        print(f"Using Adam optimizer with lr={args.learning_rate}")
        optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)
    else:
        # Default to Adam if optimizer name is unknown or AlphaGrad is not available
        if args.optimizer.lower() not in ['adam', 'alphagrad']:
             print(f"Warning: Unknown optimizer '{args.optimizer}'. Defaulting to Adam.")
        elif args.optimizer.lower() == 'alphagrad':
             print(f"Warning: AlphaGrad optimizer selected but not found. Defaulting to Adam.")
        print(f"Using Adam optimizer with lr={args.learning_rate}")
        optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)
    # ======================================


    # ALGO Logic: Storage setup
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)

    print(f"Starting training for {args.total_timesteps} timesteps...")
    print(f"Number of iterations: {args.num_iterations}")

    for iteration in range(1, args.num_iterations + 1):
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            step_result = envs.step(action.cpu().numpy())
            next_obs, reward, terminations, truncations, infos = step_result

            # Handle VecEnv step return format (may differ slightly across gym versions)
            # Ensure 'final_info' and 'final_observation' are handled if present
            if isinstance(infos, dict) and "_final_info" in infos:
                # New Gymnasium VecEnv returns final_info and final_observation separately
                final_infos = infos.get("final_info", [{} for _ in range(args.num_envs)]) # Default to list of empty dicts
                # Filter out None entries which indicate the episode is not done
                valid_final_infos = [info for info in final_infos if info is not None and 'episode' in info]

                for info in valid_final_infos:
                    episodic_return = info["episode"]["r"].item()
                    episodic_length = info["episode"]["l"].item()
                    print(f"global_step={global_step}, episodic_return={episodic_return}")
                    writer.add_scalar("charts/episodic_return", episodic_return, global_step)
                    writer.add_scalar("charts/episodic_length", episodic_length, global_step)
                    # W&B monitor_gym also logs these automatically if RecordEpisodeStatistics is used
            elif "episode" in infos:
                 # Older or different VecEnv setup might still put episode info directly in 'infos'
                 # This part might become less relevant with latest Gymnasium but kept for compatibility
                 # Ensure accessing info items safely
                 if "r" in infos["episode"] and "l" in infos["episode"]:
                    for i in range(len(infos["episode"]["r"])):
                        # Check if the episode data corresponds to a finished episode in this step
                        if terminations[i] or truncations[i]:
                            episodic_return = infos["episode"]["r"][i].item()
                            episodic_length = infos["episode"]["l"][i].item()
                            print(f"global_step={global_step}, episodic_return={episodic_return}")
                            writer.add_scalar("charts/episodic_return", episodic_return, global_step)
                            writer.add_scalar("charts/episodic_length", episodic_length, global_step)


            next_done = np.logical_or(terminations, truncations)
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(next_done).to(device)


        # bootstrap value if not done
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
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

        # flatten the batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value network
        b_inds = np.arange(args.batch_size)
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
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
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
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
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
                # === Corrected Grad Clipping Logic ===
                if not (args.optimizer.lower() == 'alphagrad' and AlphaGrad is not None):
                    # Clip gradients only if NOT using AlphaGrad (assuming AlphaGrad handles norms internally or differently)
                     nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                # ====================================
                optimizer.step()


            if args.target_kl is not None and approx_kl > args.target_kl:
                print(f"Early stopping at epoch {epoch+1} due to reaching target KL {approx_kl.item():.4f}")
                break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        current_time = time.time()
        sps = int(args.num_steps * args.num_envs / (current_time - start_time)) # Calculate SPS for this iteration
        print(f"Iteration {iteration}/{args.num_iterations}, SPS: {sps}")
        writer.add_scalar("charts/SPS", sps, global_step)
        # Reset start_time for next iteration's SPS calculation if desired, otherwise it's cumulative average
        # start_time = current_time # Uncomment to calculate SPS per iteration

    # === EVALUATION AND SAVING (No changes needed here) ===
    if args.save_model:
        model_path = f"runs/{run_name}/{args.exp_name}.cleanrl_model"
        torch.save(agent.state_dict(), model_path)
        print(f"model saved to {model_path}")

        # Ensure cleanrl_utils is installed and importable
        try:
            from cleanrl_utils.evals.ppo_eval import evaluate

            # Modify make_env for evaluation (no video capture needed here unless desired)
            def make_env(env_id, idx, capture_video, run_name, gamma):
                def thunk():
                    # Set render_mode to 'rgb_array' if capture_video is True for wandb
                    render_mode = "rgb_array" if capture_video else None
                    print(f"[{idx}] Creating env {env_id} with render_mode='{render_mode}'") # Debug print
                    env = gym.make(env_id, render_mode=render_mode)
            
                    # === START DEBUG RENDER TEST ===
                    if render_mode == "rgb_array" and idx == 0: # Only test for the env that should capture video
                         print(f"[{idx}] Attempting initial render test...")
                         try:
                             frame = env.render()
                             if frame is not None and isinstance(frame, np.ndarray):
                                 print(f"[{idx}] Initial render successful! Frame shape: {frame.shape}, dtype: {frame.dtype}")
                             elif frame is None:
                                  print(f"[{idx}] Initial render returned None.")
                             else:
                                  print(f"[{idx}] Initial render returned unexpected type: {type(frame)}")
                         except Exception as e:
                             print(f"[{idx}] ERROR during initial render test: {e}")
                             import traceback
                             traceback.print_exc() # Print detailed traceback
                    # === END DEBUG RENDER TEST ===
            
            
                    # RecordEpisodeStatistics must be applied BEFORE the W&B wrapper
                    env = gym.wrappers.RecordEpisodeStatistics(env)
            
                    # Apply other wrappers
                    env = gym.wrappers.FlattenObservation(env)
                    env = gym.wrappers.ClipAction(env)
                    env = gym.wrappers.NormalizeObservation(env)
            
                    current_obs_space = env.observation_space
                    assert isinstance(current_obs_space, gym.spaces.Box), \
                        f"Expected Box space after NormalizeObservation, got {type(current_obs_space)}"
                    clip_low = -10.0
                    clip_high = 10.0
                    low = np.full(current_obs_space.shape, clip_low, dtype=current_obs_space.dtype)
                    high = np.full(current_obs_space.shape, clip_high, dtype=current_obs_space.dtype)
                    new_obs_space = gym.spaces.Box(
                        low=low,
                        high=high,
                        shape=current_obs_space.shape,
                        dtype=current_obs_space.dtype
                    )
                    env = gym.wrappers.TransformObservation(
                        env,
                        lambda obs: np.clip(obs, clip_low, clip_high),
                        observation_space=new_obs_space
                    )
            
                    env = gym.wrappers.NormalizeReward(env, gamma=gamma)
                    env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))
            
                    print(f"[{idx}] Environment creation complete.") # Debug print
                    return env
            
                return thunk

            episodic_returns = evaluate(
                model_path,
                make_eval_env, # Use the potentially modified eval env maker
                args.env_id,
                eval_episodes=10,
                run_name=f"{run_name}-eval",
                Model=Agent, # Pass the Agent class correctly
                device=device,
                gamma=args.gamma, # Pass gamma if needed by NormalizeReward in eval env
            )
            for idx, episodic_return in enumerate(episodic_returns):
                writer.add_scalar("eval/episodic_return", episodic_return, idx)

            if args.upload_model:
                 # Ensure cleanrl_utils is installed and importable
                 try:
                    from cleanrl_utils.huggingface import push_to_hub

                    repo_name = f"{args.env_id}-{args.exp_name}-seed{args.seed}"
                    repo_id = f"{args.hf_entity}/{repo_name}" if args.hf_entity else repo_name
                    # push_to_hub expects video folder path, but wandb handles video saving
                    # If you want to upload videos saved locally by wandb, you might need to find their path
                    # or rely on wandb's artifact logging.
                    # For now, let's assume push_to_hub primarily uploads the model and maybe tensorboard logs.
                    # Passing None for video folder might be safer if videos aren't saved where expected.
                    # Check push_to_hub documentation for specifics.
                    print("Note: W&B automatically saves videos. Check W&B run page for videos.")
                    print("Uploading model and run data to Hugging Face Hub...")
                    push_to_hub(
                        args=args, # Pass args object
                        episodic_returns=episodic_returns,
                        repo_id=repo_id,
                        algo_name="PPO", # Algorithm name string
                        basedir=f"runs/{run_name}", # Base directory of the run
                        wandb_project_name=args.wandb_project_name, # Pass wandb project name
                        wandb_entity=args.wandb_entity, # Pass wandb entity
                        # video_folder=f"videos/{run_name}" # Maybe Optional: W&B logs videos directly
                    )
                 except ImportError:
                    print("cleanrl_utils not found, skipping Hugging Face upload.")
        except ImportError:
            print("cleanrl_utils not found, skipping evaluation.")


    envs.close()
    writer.close()
    if args.track:
        wandb.finish() # Ensure wandb run finishes cleanly

print("Script finished.")
