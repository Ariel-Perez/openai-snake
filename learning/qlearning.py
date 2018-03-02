"""
Basic implementation of Q Learning.
"""
import numpy as np
from keras.models import Model, load_model
from keras.layers import Input, Dense, Flatten, Activation
from keras.layers import Conv2D


class QLearning:

    def __init__(self, env, gamma=0.9):
        """Initialize the Q learning."""
        self.n_actions = env.action_space.n
        self.board_shape = env.observation_space.shape
        self.input_shape = (self.board_shape[0], self.board_shape[1], 1)
        self.model = self.model_architecture()
        self.model.compile(optimizer='adadelta', loss='mean_squared_error')
        self.gamma = 0.9

    def _expand(self, obs, dims):
        """Expand an observation adding the specified dimensions."""
        dims = iter(dims)
        for dim in dims:
            obs = np.expand_dims(obs, dim)

        return obs

    def predict(self, obs):
        """Predict Q values for a given observation."""
        obs = self._expand(obs, (0, -1))
        return self.model.predict(obs)

    def act(self, obs):
        """Make a move for a given observation."""
        q_values = self.predict(obs)
        return np.argmax(q_values)

    def train(self, obs, action, result, reward):
        """Perform one training step for a given SAS."""
        q_obs = self.predict(obs)
        q_next = self.predict(result)

        q_obs[0][action] = reward + self.gamma * np.amax(q_next)

        x = np.expand_dims(obs, -1)
        y = q_obs
        self.model.fit(x, y, verbose=False)

    def model_architecture(self):
        """Build the model architecture."""
        inputs = Input(self.input_shape)
        conv1  = Conv2D(64, 5, padding='valid')(inputs)
        act1   = Activation('relu')(conv1)
        conv2  = Conv2D(64, 3, padding='valid')(act1)
        act2   = Activation('relu')(conv2)
        flat   = Flatten()(act2)
        dense1 = Dense(128, activation='relu')(flat)
        out    = Dense(self.n_actions, activation='relu')(dense1)
        return Model(inputs=inputs, outputs=out)

    def save(self, path):
        """Save the model to the given path."""
        self.model.save(path)

    @classmethod
    def load(cls, env, path):
        """Load the model from the given path."""
        ql = cls(env)
        ql.model = load_model(path)
        return ql
