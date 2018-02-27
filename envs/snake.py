"""
Environment for the Snake game.

The agent controls a snake on a bordered plane, which moves around eating
pellets to grow and get points. The snake has a specific length, so there
is a moving tail a fixed number of units away from the head.

The player loses when the snake runs into the screen border or itself.
"""
import sys
import random
import logging
import itertools

import numpy as np
import gym
from gym.spaces import Box, Discrete
from gym.utils import seeding, colorize
from enum import IntEnum, unique
from six import StringIO


logger = logging.getLogger(__name__)


@unique
class Direction(IntEnum):
    """Enum for possible directions."""
    UP    = 0
    LEFT  = 1
    DOWN  = 2
    RIGHT = 3


@unique
class Board(IntEnum):
    """Enum for possible board square content."""
    BLANK      = 0
    FRUIT      = 1
    SNAKE_HEAD = 2
    SNAKE_BODY = 3


class SnakeEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    COLOR_MAPPING = {
        Board.BLANK: ' ',
        Board.FRUIT: colorize('@', 'yellow'),
        Board.SNAKE_BODY: colorize('o', 'green'),
        Board.SNAKE_HEAD: colorize('o', 'green')
    }

    def __init__(self, width=80, height=60):
        """Initialize the SnakeEnv with a given size."""
        self.width = width
        self.height = height

        # Space of observed data: a grid with values as specified by Board Enum
        self.observation_space = Box(low=0.0, high=3.0, shape=(width, height))

        # Space of possible actions as specified by Directions Enum
        self.action_space = Discrete(4)

        # Space of possible rewards
        self.reward_range = (0.0, np.inf)

        # Set the rng
        self.seed()
        self.reset()

    def _get_obs(self):
        """Get the current observation."""
        board = np.full((self.width, self.height), fill_value=Board.BLANK)
        board[self.fruit] = Board.FRUIT
        board[self.snake[0]] = Board.SNAKE_HEAD
        for position in self.snake[1:]:
            board[position] = Board.SNAKE_BODY

        return board

    def _step(self, action):
        """Act on a given action."""
        direction = action[0]
        if direction == Direction.UP:
            new_position = self.snake[0][0], self.snake[0][1] - 1
        elif direction == Direction.DOWN:
            new_position = self.snake[0][0], self.snake[0][1] + 1
        elif direction == Direction.LEFT:
            new_position = self.snake[0][0] - 1, self.snake[0][1]
        elif direction == Direction.RIGHT:
            new_position = self.snake[0][0] + 1, self.snake[0][1]
        else:
            raise ValueError("Invalid action!")

        collision = (
            new_position[0] < 0
            or new_position[0] >= self.width
            or new_position[1] < 0
            or new_position[1] >= self.height
            or new_position in self.snake)

        done = collision

        eat = new_position == self.fruit
        reward = int(eat)

        if not collision:
            self.snake = [new_position] + (self.snake if eat else self.snake[:-1])

        self.fruit = self.place_fruit() if eat else self.fruit

        self.episode_total_reward += reward
        self.lastaction = direction

        obs = self._get_obs()
        return (obs, reward, done, {})

    def render(self, mode='human'):
        """Render the current observation with [0, 0] as the top-left corner."""
        outfile = StringIO() if mode == 'ansi' else sys.stdout

        out = self._get_obs().tolist()
        out = [[self.COLOR_MAPPING[c] for c in line] for line in out]

        outfile.write("\n".join(["".join(row) for row in out])+"\n")
        if self.lastaction is not None:
            outfile.write("  ({})\n".format(Board[self.lastaction]))
        else: outfile.write("\n")

        # No need to return anything for human
        if mode != 'human':
            return outfile

    def place_fruit(self):
        """Get a random location to place a fruit."""
        return random.sample(self.void, 1)[0]

    def _reset(self):
        """
        Reset the environment run.

        Restocks the wallet, and starts the time at a new point.
        Rewards and valuations are reset.
        """
        self.snake = [
            (self.np_random.randint(self.width),
             self.np_random.randint(self.height))
        ]

        self.void = {
            (x, y) for x, y in
            itertools.product(range(self.width), range(self.height))
            if (x, y) not in self.snake
        }

        self.fruit = self.place_fruit()

        self.lastaction = None

        obs = self._get_obs()

        self.episode_total_reward = 0.0
        return obs

    def _seed(self, seed=None):
        """Set the seed for the random number generators."""
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
