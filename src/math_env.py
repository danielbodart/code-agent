import gymnasium as gym
from gymnasium import spaces
import numpy as np
from .data_generator import generate_addition_example
from .mask_expression import mask_expression
from .reward_function import reward_function

class MathEnv(gym.Env):
    def __init__(self):
        super(MathEnv, self).__init__()
        self.action_space = spaces.Discrete(210)
        self.observation_space = spaces.Box(low=-1, high=105, shape=(5,), dtype=np.int32)

    def _get_obs(self):
        tokens = self.masked_expression.split()
        operator_encoding = {'+': 101, '-': 102, '*': 103, '/': 104, '=': 105}
        return [int(token) if token.isdigit() else operator_encoding.get(token, -1) for token in tokens]

    def reset(self, seed=None):
        super().reset(seed=seed)
        self.expression = generate_addition_example()
        self.original_token, self.masked_expression = mask_expression(self.expression, seed)
        return self._get_obs()

    def step(self, actions):
        rewards = [reward_function(self.original_token, str(action)) for action in actions]
        done = True
        return self._get_obs(), rewards, done, {} 

    def render(self, mode='human'):
        # Render the environment to the screen
        pass

    def close(self):
        pass
