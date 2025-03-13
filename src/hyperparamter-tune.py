import optuna
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from src.math_env import MathEnv

def optimize_ppo(trial):
    env = DummyVecEnv([lambda: MathEnv()])

    # Define the hyperparameter search space
    n_steps = trial.suggest_int('n_steps', 128, 2048)
    gamma = trial.suggest_float('gamma', 0.9, 0.999)
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-3)

    # Create the PPO model
    model = PPO('MlpPolicy', env, n_steps=n_steps, gamma=gamma, learning_rate=learning_rate, verbose=0)

    # Train the model
    model.learn(total_timesteps=10000)

    # Evaluate the model
    mean_reward = evaluate_model(model, env)
    return mean_reward

def evaluate_model(model, env, n_eval_episodes=10):
    # Evaluate the model and return the mean reward
    total_reward = 0
    for _ in range(n_eval_episodes):
        obs = env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs)
            obs, reward, done, _ = env.step(action)
            total_reward += reward
    return total_reward / n_eval_episodes

study = optuna.create_study(direction='maximize')
study.optimize(optimize_ppo, n_trials=100)

print('Best hyperparameters:', study.best_params)