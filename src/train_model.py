from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.math_env import MathEnv

if __name__ == '__main__':
    # Create environment with random unmasking actions
    cpu = os.cpu_count() or 1
    print(f"Using {cpu} CPU(s)")
    env = SubprocVecEnv([lambda: MathEnv() for _ in range(cpu)])

    # Use PPO to optimize the denoising model
    model = PPO("MlpPolicy", env, verbose=1, device="cpu")
    model.learn(total_timesteps=10000)

    # Save the model
    model.save("ppo_masked_text")
    print("Model saved as 'ppo_masked_text'")
