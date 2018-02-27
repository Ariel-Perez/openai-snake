"""
Registers the Snake environment on gym.
"""
from envs.snake import SnakeEnv
from gym.envs.registration import register


register(
    id='Snake-v0',
    entry_point='envs.snake:SnakeEnv'
)
