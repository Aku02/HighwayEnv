import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from wandb.integration.sb3 import WandbCallback
import wandb
import highway_env  # noqa: F401
from gymnasium.wrappers import RecordVideo
from stable_baselines3.common.callbacks import BaseCallback
from rllte.xplore.reward import RND, RIDE, ICM, NGU
from stable_baselines3.common.base_class import BaseAlgorithm
import torch as th

import os
# os.environ["WANDB_MODE"] = "offline"
class RLeXploreWithOnPolicyRL(BaseCallback):
    """
    A custom callback for combining RLeXplore and on-policy algorithms from SB3 with W&B logging.
    """
    def __init__(self, irs, verbose=0):
        super(RLeXploreWithOnPolicyRL, self).__init__(verbose)
        self.irs = irs
        self.buffer = None

    def init_callback(self, model: BaseAlgorithm) -> None:
        super().init_callback(model)
        self.buffer = self.model.rollout_buffer

    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to `env.step()`.
        """
        observations = self.locals["obs_tensor"]
        device = observations.device
        actions = th.as_tensor(self.locals["actions"], device=device)
        rewards = th.as_tensor(self.locals["rewards"], device=device)
        dones = th.as_tensor(self.locals["dones"], device=device)
        next_observations = th.as_tensor(self.locals["new_obs"], device=device)

        # Watch the interaction for intrinsic reward signals
        self.irs.watch(observations, actions, rewards, dones, dones, next_observations)
        
        # Log intrinsic rewards in W&B
        wandb.log({"intrinsic_reward": rewards.mean().item()})
        return True

    def _on_rollout_end(self) -> None:
        # Compute the intrinsic rewards
        obs = th.as_tensor(self.buffer.observations)
        new_obs = obs.clone()
        new_obs[:-1] = obs[1:]
        new_obs[-1] = th.as_tensor(self.locals["new_obs"])
        actions = th.as_tensor(self.buffer.actions)
        rewards = th.as_tensor(self.buffer.rewards)
        dones = th.as_tensor(self.buffer.episode_starts)
        
        intrinsic_rewards = self.irs.compute(
            samples=dict(observations=obs, actions=actions, 
                         rewards=rewards, terminateds=dones, 
                         truncateds=dones, next_observations=new_obs),
            sync=True
        )

        # Add intrinsic rewards to buffer and log to W&B
        intrinsic_rewards_np = intrinsic_rewards.cpu().numpy()
        self.buffer.advantages += 0.1*intrinsic_rewards_np
        self.buffer.returns += 0.1*intrinsic_rewards_np
        wandb.log({"buffer_intrinsic_rewards_mean": intrinsic_rewards_np.mean(),
                   "buffer_intrinsic_rewards_std": intrinsic_rewards_np.std()})

def make_configure_env(**kwargs):
    env = gym.make(kwargs["id"])
    env.configure(kwargs["config"])
    env.reset()
    return env

# ==================================
#        Main script
# ==================================

if __name__ == "__main__":
    # Initialize WandB
    wandb.init(
        project="rl_attention_ppo",
        sync_tensorboard=True,
        monitor_gym=True,
        group="highway-fast-v0" + "_ppo" + "_intrinsic",
        config={
            "policy": "CnnPolicy",
            "n_steps": 64,
            "batch_size": 64,
            "n_epochs": 10,
            "learning_rate": 5e-4,
            "gamma": 0.8,
            "total_timesteps": int(2e5),
            "environment": "highway-fast-v0",
            "int":"icm"
        },
    )

    train = True
    if train:
        device = 'cuda'
        n_cpu = 6
        batch_size = 64

        # Create the environment
        env = make_vec_env("highway-fast-v0", n_envs=n_cpu, vec_env_cls=SubprocVecEnv, env_kwargs={
            "config": {
                "observation": {
                    "type": "GrayscaleObservation",
                    "observation_shape": (128, 64),
                    "stack_size": 4,
                    "weights": [0.2989, 0.5870, 0.1140],
                    "scaling": 1.75,
                },
                "vehicles_count": 15,
                "policy_frequency": 2,
                "duration": 40,
            }
        })

        # Initialize the RND (Random Network Distillation) module
        irs = ICM(env, device=device)

        # Initialize PPO model
        model = PPO(
            "CnnPolicy",
            env,
            policy_kwargs=dict(net_arch=[dict(pi=[256, 256], vf=[256, 256])]),
            n_steps=batch_size * 12 // n_cpu,
            batch_size=batch_size,
            n_epochs=10,
            learning_rate=5e-4,
            gamma=0.8,
            verbose=2,
            tensorboard_log="highway_ppo/",
        )

        # Initialize custom callback with intrinsic reward signals
        intrinsic_callback = RLeXploreWithOnPolicyRL(irs=irs)

        # Train the agent
        model.learn(
            total_timesteps=int(2e5),
            callback=[intrinsic_callback, WandbCallback(
                gradient_save_freq=100,
                model_save_path="highway_ppo/models",
                verbose=2
            )]
        )

        # Save the agent
        model.save("highway_ppo/model_intrinsic")

    # Load the trained model
    model = PPO.load("highway_ppo/model_intrinsic")
    env = gym.make("highway-fast-v0", render_mode="rgb_array", config = {
        "observation": {
            "type": "GrayscaleObservation",
            "observation_shape": (128, 64),
            "stack_size": 4,
            "weights": [0.2989, 0.5870, 0.1140],
            "scaling": 1.75,
        }
    })
    env = RecordVideo(
        env,
        video_folder="highway_ppo/videos/",
        episode_trigger=lambda e: True
    )

    # Evaluate the agent and log test rewards
    mean_rewards = []
    for episode in range(10):
        obs, info = env.reset()
        done = truncated = False
        episode_reward = 0
        while not (done or truncated):
            action, _ = model.predict(obs)
            obs, reward, done, truncated, info = env.step(action)
            episode_reward += reward
            env.render()

        # Log the episode reward to WandB
        wandb.log({"test_reward": episode_reward})
        mean_rewards.append(episode_reward)
        print(f"Episode {episode + 1} - Reward: {episode_reward}")

    # Log the mean reward of the evaluation
    env.close()
    mean_reward = sum(mean_rewards) / len(mean_rewards)
    wandb.log({"mean_test_reward": mean_reward})

    # Finish the WandB run
    wandb.finish()
