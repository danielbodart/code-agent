from stable_baselines3 import PPO
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.math_env import MathEnv

# Load the trained model
model = PPO.load("ppo_masked_text")

# Create the environment
read_env = MathEnv()

# Perform predictions on a generated example
def predict_example():
    for _ in range(10):  # Predict for a few steps
        obs, _ = read_env.reset()
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, terminated, truncated, info = read_env.step(action)
        # Print the original, masked, and guessed tokens
        print("Original: ", read_env.expression)
        print("Masked:   ", read_env.masked_expression)
        guessed = read_env.tokens_to_string(action)
        print("Guessed:  ", guessed)
        print("-------------------")

# Call the prediction function
predict_example()
