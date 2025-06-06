# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_continuous_actionpy

# This currently manually saves the videos and uploads them to wand
# Command line to reproduce results - xvfb-run -a -s "-screen 0 1400x900x24" python ppo_continuous_action.py --env-id HalfCheetah-v5 --optimizer DAG --learning_rate 9E-5 --capture_video --track

# NOTE -> make sure you have the correct versions of the requirements installed from the initial runs -> for some reason, PPO is super finicky and can absolutely mess up over updates

import os
import random
import time
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from optim.sgd import AlphaGrad, DAG
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
    optimizer: str = ""
    """ Optimizer to use"""
    alpha: float = 0.0
    """ Alpha value for AlphaGrad"""

    # Algorithm specific arguments
    env_id: str = "HalfCheetah-v4"
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
        env = None # Initialize env variable
        # Create the environment, enabling rgb_array render mode if capturing video for the first env
        if capture_video and idx == 0:
            print(f"Setting up environment {idx} for video capture (render_mode='rgb_array')")
            env = gym.make(env_id, render_mode="rgb_array")
            # Apply the RecordVideo wrapper here, BEFORE other wrappers if possible
            # Videos will be saved in videos/{run_name}/
            # Default behavior records until env.close() is called.
            video_folder = f"videos/{run_name}"
            print(f"Recording video for env {idx} to {video_folder}")
            env = gym.wrappers.RecordVideo(env, video_folder=video_folder,
                                          # Optional: Trigger recording based on episode number
                                          # episode_trigger=lambda x: x % 50 == 0 # e.g., record every 50 episodes
                                          )
        else:
            env = gym.make(env_id)

        # Apply common wrappers AFTER potential RecordVideo
        env = gym.wrappers.FlattenObservation(env)  # deal with dm_control's Dict observation space
        env = gym.wrappers.RecordEpisodeStatistics(env) # Records episode stats like return and length
        env = gym.wrappers.ClipAction(env)
        env = gym.wrappers.NormalizeObservation(env)
        env = gym.wrappers.TransformObservation(
            env,
            lambda obs: np.clip(obs, -10, 10),
            observation_space=env.observation_space # Deprecated? Gymnasium might handle this automatically
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
        [make_env(args.env_id, i, args.capture_video, run_name, args.gamma) for i in range(args.num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    agent = Agent(envs).to(device)
    if args.optimizer == 'Adam':
        optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)
    if args.optimizer == 'AlphaGrad':
        optimizer = AlphaGrad(agent.parameters(), lr=args.learning_rate, alpha = args.alpha, epsilon=1e-5, momentum = 0.9)
    if args.optimizer == 'DAG':
        optimizer = DAG(agent.parameters(), lr=args.learning_rate, momentum = 0.9)

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
            next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
            next_done = np.logical_or(terminations, truncations)
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(next_done).to(device)

            if "episode" in infos:
                for i in range(len(infos["episode"]["r"])):
                    episodic_return = infos["episode"]["r"][i].item()
                    episodic_length = infos["episode"]["l"][i].item()
                    print(f"global_step={global_step}, episodic_return={episodic_return}")
                    writer.add_scalar("charts/episodic_return", episodic_return, global_step)
                    writer.add_scalar("charts/episodic_length", episodic_length, global_step)

        # bootstrap value if not done
        with torch.no_grad():
            # NOTE: Assumes 'rewards', 'dones', 'values' below are the DynAGO buffers
            # (rewards_buffer, dones_buffer, values_buffer) with shape (num_total_samples, num_envs)
            
            # Calculate Monte Carlo returns G_t
            returns = torch.zeros_like(rewards_buffer).to(device) # Will store G_t for each sample
            mc_return_from_here = torch.zeros(args.num_envs).to(device) # Accumulator for return from t+1 onwards
        
            # Iterate backwards through ALL collected samples in the rollout buffer
            for t in reversed(range(args.num_total_samples_per_rollout)):
                # Get the modulated reward for this step
                current_rewards = rewards_buffer[t] # r_mod_tj
                
                # Calculate G_t = r_t + gamma * G_{t+1} * (1 - d_t)
                # 'mc_return_from_here' holds G_{t+1} (return from step t+1 onwards)
                # dones_buffer[t] determines if the episode ended *after* this step t
                returns[t] = current_rewards + args.gamma * mc_return_from_here * (1.0 - dones_buffer[t])
                
                # Update the accumulator for the next iteration (t-1). This becomes G_t.
                mc_return_from_here = returns[t] 
        
            # Use the calculated Monte Carlo returns (G_t) as the advantage signal
            # No separate advantage calculation needed when baseline is effectively zero.
            advantages = returns # Shape: (num_total_samples, num_envs)
        
        # Flatten the batch - uses the correct total batch size based on DynAGO buffers
        # Ensure obs_buffer, logprobs_buffer, actions_buffer also have num_total_samples_per_rollout in first dim
        b_obs = obs_buffer.reshape((-1,) + obs_shape) # Use DynAGO's obs_buffer
        b_logprobs = logprobs_buffer.reshape(-1)       # Use DynAGO's logprobs_buffer
        b_actions = actions_buffer.reshape((-1,) + action_shape) # Use DynAGO's actions_buffer
        b_advantages = advantages.reshape(-1)         # These are now G_t
        b_returns = returns.reshape(-1)               # Also G_t, target for value loss (but vf_coef=0)
        b_values = values_buffer.reshape(-1)          # V(s_t) from buffer, used for value loss calculation (but result ignored)
        
        # Optimizing the policy network (Value network loss is disabled by vf_coef=0)
        b_inds = np.arange(args.batch_size) # Should be num_envs * num_total_samples_per_rollout
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]
        
                # We only need policy outputs: newlogprob, entropy
                # newvalue is computed but not used in the final loss calculation if vf_coef=0
                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()
        
                with torch.no_grad():
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]
        
                mb_advantages = b_advantages[mb_inds] # This is G_t for the minibatch
                if args.norm_adv:
                    # Normalizing MC returns can sometimes help stabilize REINFORCE
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)
        
                # Policy loss using G_t (or normalized G_t) as advantage
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()
        
                # Value loss calculation (will be multiplied by vf_coef=0)
                # We still compute it to avoid errors if parts of the code expect v_loss variable
                v_loss_calc = torch.tensor(0.0, device=device) # Default if not clipped
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    # Need b_values (V(s_t) prediction) for clipping calculation
                    v_clipped = b_values[mb_inds] + torch.clamp( 
                        newvalue - b_values[mb_inds], -args.clip_coef, args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss_calc = 0.5 * v_loss_max.mean()
                else:
                    v_loss_calc = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()
        
                entropy_loss = entropy.mean()
                # Total loss: policy loss + entropy bonus (value loss term is zeroed out by vf_coef)
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss_calc * args.vf_coef 
        
                optimizer.zero_grad()
                loss.backward() # Gradients only computed for policy network (and shared layers if any)
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()
        
            if args.target_kl is not None and approx_kl > args.target_kl:
                break
        
        # Logging Update
        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy() # y_pred=V(s_t), y_true=G_t
        var_y = np.var(y_true)
        # Explained variance is not meaningful as V(s_t) wasn't trained against G_t
        explained_var = np.nan 

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        # writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        # writer.add_scalar("losses/explained_variance", explained_var, global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    if args.save_model:
        model_path = f"runs/{run_name}/{args.exp_name}.cleanrl_model"
        torch.save(agent.state_dict(), model_path)
        print(f"model saved to {model_path}")
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

        if args.upload_model:
            from cleanrl_utils.huggingface import push_to_hub

            repo_name = f"{args.env_id}-{args.exp_name}-seed{args.seed}"
            repo_id = f"{args.hf_entity}/{repo_name}" if args.hf_entity else repo_name
            push_to_hub(args, episodic_returns, repo_id, "PPO", f"runs/{run_name}", f"videos/{run_name}-eval")

    envs.close()
    # --- Manual Video Upload Logic ---
    # Check if tracking AND video capture were enabled during training
    if args.track and args.capture_video:
        print("Attempting to upload recorded videos to W&B...")
        video_dir = f"videos/{run_name}" # Directory where make_env saved videos
        print(f"Searching for videos in: {video_dir}")
        try:
            # Find all mp4 files, sorted by modification time (newest last)
            video_files = sorted(glob.glob(os.path.join(video_dir, "*.mp4")), key=os.path.getmtime)

            if video_files:
                print(f"Found {len(video_files)} video file(s). Uploading...")
                # Log each video with a unique key, associating with the final global_step
                for i, video_file in enumerate(video_files):
                    wandb_key = f"media/video_capture_{i}" # Unique key: media/video_capture_0, ..._1 etc.
                    print(f" - Logging {video_file} as {wandb_key}")
                    wandb.log({wandb_key: wandb.Video(video_file, fps=4, format="mp4")}, step=global_step)
                print("Video upload attempt complete.")
            else:
                print(f"Warning: No .mp4 video files found in {video_dir}. Video capture might have failed or completed no episodes.")

        except FileNotFoundError:
            print(f"Warning: Video directory '{video_dir}' not found. Cannot upload videos.")
        except Exception as e:
             print(f"Error during video search or W&B upload: {e}")
    elif args.capture_video:
         print("Info: Video capture was enabled but W&B tracking is off. Videos saved locally only.")
    # --- End Video Upload Logic ---


    # --- Final Cleanup ---
    writer.close() # Close Tensorboard SummaryWriter

    # Finish wandb run if tracking was enabled
    if args.track:
         print("Finishing W&B run...")
         wandb.finish()
         print("W&B run finished.")

    print("Script execution finished.")
