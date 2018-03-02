"""
Script for training an agent for the game.
"""
if __name__ == '__main__':
    import gym
    import envs
    import time
    import logging
    from learning import DQN

    logger = logging.getLogger(__name__)
    logger.setLevel('INFO')

    env = gym.make("Snake-v0")
    dqn = DQN(env, replay_start=1000)

    dqn.train(100000)

    path = "dqn.hdf5"
    dqn.save(path)
    logger.info("Saved to %s" % path)
