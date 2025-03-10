from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from src.math_env import MathEnv

# Create a vectorized environment
vec_env = make_vec_env(lambda: MathEnv(), n_envs=5)

# Initialize the PPO model with MlpPolicy
model = PPO('MlpPolicy', vec_env, verbose=2, device='cpu')

# Train the model
model.learn(total_timesteps=10000)

# Save the model
model.save("ppo_mathenv")

# Load the model and set the environment
model = PPO.load("ppo_mathenv")
model.set_env(vec_env)

# Evaluate the model using the vectorized environment
obs = vec_env.reset()

print("Starting evaluation...")

for _ in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, rewards, dones, info = vec_env.step(action)
    print(f"Reward: {rewards} | Done: {dones} | Info: {info} | Action: {action}")
    vec_env.render()