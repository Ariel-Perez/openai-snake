"""
Implementation of Deep Q Learning.
"""
import random
import logging
import collections
import numpy as np
from keras.models import Model, load_model, clone_model
from keras.layers import Input, Dense, Flatten, Activation
from keras.layers import Conv2D
from keras.optimizers import RMSprop


logger = logging.getLogger(__name__)
logger.setLevel('INFO')


class DQN:

    DEFAULT_HYPERPARAMETERS = {
        'minibatch_size': 32,
        'replay_memory_size': 1e6,
        # 'agent_history_length': 4,
        'network_update_frequency': 1e4,
        'discount_factor': 0.99,
        # 'frame_skip': 4,
        'learning_rate': 0.00025,
        'gradient_momentum': 0.95,
        'initial_exploration': 1.0,
        'final_exploration': 0.1,
        'final_exploration_frame': 1e6,
        'replay_start': 5e4
    }

    def __init__(self, env, **kwargs):
        """Initialize the model."""
        self.env = env
        self.action_space = env.action_space
        self.n_actions = env.action_space.n
        self.observation_space = env.observation_space.shape
        self.input_shape = (
            self.observation_space[0],
            self.observation_space[1],
            1
        )

        self.params = self.DEFAULT_HYPERPARAMETERS.copy()
        self.params.update(kwargs)

        self.discount_factor = self.params['discount_factor']
        optimizer = RMSprop(
            lr=self.params['learning_rate'],
            rho=self.params['gradient_momentum']
        )

        self.exploration_schedule = np.linspace(
            self.params['initial_exploration'],
            self.params['final_exploration'],
            self.params['final_exploration_frame']
        )

        self.minibatch_size = self.params['minibatch_size']
        self.replay_start = self.params['replay_start']
        self.replay_memory_size = int(self.params['replay_memory_size'])
        self.replay_memory = collections.deque(
            [], maxlen=self.replay_memory_size
        )

        self.model = self.model_architecture()
        self.model.compile(optimizer=optimizer, loss='mean_squared_error')
        self.inference_model = clone_model(self.model)
        self.network_update_frequency = self.params['network_update_frequency']
        self.counter = 0

    @property
    def exploration(self):
        """Get the current exploration parameter."""
        idx = self.counter\
            if self.counter < len(self.exploration_schedule) else -1

        return self.exploration_schedule[idx]

    def transform_reward(self, reward):
        """Transform rewards to -1, 0, 1."""
        return np.sign(reward)

    def _expand(self, obs, dims):
        """Expand an observation adding the specified dimensions."""
        dims = iter(dims)
        for dim in dims:
            obs = np.expand_dims(obs, dim)

        return obs

    def predict(self, observations):
        """Predict Q values for a batch of observations."""
        return self.inference_model.predict(observations)

    def predict_single(self, obs):
        """Predict Q values for a given observation."""
        obs = self._expand(obs, (0,))
        return self.predict(obs)[0]

    def act(self, obs):
        """Make a move for a given observation."""
        q_values = self.predict_single(obs)
        righteousness = np.random.rand()
        if righteousness > self.exploration:
            return np.argmax(q_values)
        else:
            return self.action_space.sample()

    def replay(self):
        """Replay a minibatch from the memory."""
        batch = random.sample(
            self.replay_memory, self.minibatch_size
        )

        observations, actions, rewards, results, keep = zip(*batch)

        observations = np.array(observations)
        rewards = np.array(rewards)
        results = np.array(results)
        keep = np.array(keep)

        q_values = self.predict(observations)

        next_predicted = self.predict(results)
        best_next_predicted = np.amax(next_predicted, axis=1)

        actual_values = rewards + self.discount_factor * best_next_predicted * keep
        q_values[np.arange(self.minibatch_size), actions] = actual_values

        self.model.fit(observations, q_values, epochs=1, verbose=False)

    def train(self, steps):
        """Train on the environment for a given number of steps."""
        step = 0
        while step < steps:
            observation = self.env.reset()
            done = False
            while not done:
                action = self.act(observation)
                result, reward, done, info = self.env.step(action)

                # Clip rewards
                reward = self.transform_reward(reward)

                # Remember
                self.replay_memory.append(
                    (observation, action, reward, result, not done)
                )

                observation = result
                step += 1

                if step >= self.replay_start and step % self.minibatch_size == 0:
                    self.replay()

                if step % self.network_update_frequency == 0:
                    self.inference_model = clone_model(self.model)

                if step % 1e3 == 0:
                    logger.info("%i / %i" % (step, steps))

    def model_architecture(self):
        """Build the model architecture."""
        inputs = Input(self.input_shape)
        conv1  = Conv2D(32, 3, strides=1, padding='valid')(inputs)
        act1   = Activation('relu')(conv1)
        conv2  = Conv2D(64, 3, strides=1, padding='valid')(act1)
        act2   = Activation('relu')(conv2)
        conv3  = Conv2D(128, 3, strides=1, padding='valid')(act2)
        act3   = Activation('relu')(conv3)
        flat   = Flatten()(act3)
        dense1 = Dense(512, activation='relu', name='dense1')(flat)
        out    = Dense(self.n_actions, activation='relu', name='output')(dense1)
        return Model(inputs=inputs, outputs=out)

    def save(self, path):
        """Save the model to the given path."""
        self.model.save(path)

    @classmethod
    def load(cls, env, path):
        """Load the model from the given path."""
        ql = cls(env)
        ql.model = ql.inference_model = load_model(path)
        ql.counter = len(ql.exploration_schedule)
        return ql
