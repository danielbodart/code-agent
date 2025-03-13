import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
from .data_generator import generate_addition_example
from .mask_expression import mask_expression
from .reward_function import reward_function
from .split_expression import split_expression

class MathEnv(gym.Env):
    def __init__(self, max_steps=100):
        super(MathEnv, self).__init__()
        # Define the fixed vocabulary including numbers, math symbols, and special tokens
        self.vocab = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "+", "-", "*", "/", "=", "?", "P"]
        self.vocab_size = len(self.vocab)
        # Define action and observation space
        # MultiDiscrete action space for choosing a token from the vocabulary
        self.action_space = spaces.MultiDiscrete([self.vocab_size] * 6)  # Assuming max 6 tokens
        self.observation_space = spaces.Box(low=0, high=self.vocab_size, shape=(6,), dtype=np.int32)
        self.expression = None
        self.original_token = None
        self.masked_expression = None
        self.max_steps = max_steps
        self.current_step = 0

    def _get_obs(self):
        tokens = split_expression(self.masked_expression)
        # Ensure the observation has a fixed size, padding with 'P' if necessary
        obs = [self.vocab.index(token) if token in self.vocab else self.vocab.index('?') for token in tokens]
        obs += [self.vocab.index('P')] * (6 - len(obs))  # Pad with 'P' to ensure length 6
        return obs

    def reset(self, seed=None):
        super().reset(seed=seed)
        self.expression = generate_addition_example()
        self.original_token, self.masked_expression = mask_expression(self.expression, seed)
        self.current_step = 0  # Reset current_step
        return np.array(self._get_obs()), {}

    def step(self, action):
        reward = reward_function(self.original_token, self.tokens_to_string(action))
        self.current_step += 1
        done = self.current_step >= self.max_steps or reward == 1
        return np.array(self._get_obs()), reward, done, False, {}

    def render(self, mode='human'):
        # Render the environment to the screen
        pass

    def close(self):
        pass

    def tokens_to_string(self, tokens):
        """Convert a list of token indices back into a string."""
        return ''.join([self.vocab[token] for token in tokens])
