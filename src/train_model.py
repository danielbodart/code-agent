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

    # Check if a saved model exists
    model_path = "ppo_masked_text.zip"
    if os.path.exists(model_path):
        model = PPO.load(model_path, env, verbose=1, device="cpu", n_steps=340, gamma=0.9053995634851253, learning_rate=0.00029203107211106104)
        print("Model loaded from", model_path)
    else:
        model = PPO("MlpPolicy", env, verbose=1, device="cpu", n_steps=340, gamma=0.9053995634851253, learning_rate=0.00029203107211106104)
    
    model.learn(total_timesteps=10000000)  

    # Save the model
    model.save("ppo_masked_text")
    print("Model saved as 'ppo_masked_text'")

    # Add logging
    print("Training completed. Evaluate the model's performance.")
