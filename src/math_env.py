import gymnasium as gym
from gymnasium import spaces
import numpy as np
from data_generator import DataGenerator

class MathEnv(gym.Env):
    def __init__(self):
        super(MathEnv, self).__init__()
        # Define action and observation space
        # They must be gymnasium.spaces objects
        # Example when using discrete actions:
        self.action_space = spaces.Discrete(10)  # 0-9 for numbers, 10 for operators
        # Example for using image as input:
        self.observation_space = spaces.Box(low=0, high=1, shape=(10,), dtype=np.float32)

    def reset(self):
        self.state = DataGenerator.generate_addition_example()
        return self.state

    def step(self, action):
        # Execute one time step within the environment
        # Example logic for handling an action
        correct_answer = 42  # Placeholder for the correct answer
        if action == correct_answer:
            reward = 1.0  # Max reward for correct answer
        elif abs(action - correct_answer) <= 1:
            reward = 0.9  # Partial reward for close answer
        else:
            reward = -1.0  # Penalty for wrong answer

        done = True  # End the episode after one step
        return self.state, reward, done, {}

    def render(self, mode='human'):
        # Render the environment to the screen
        pass

    def close(self):
        pass
