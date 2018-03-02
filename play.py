"""
Script for playing the game visually.
"""
if __name__ == '__main__':
    import gym
    import envs
    import time
    from learning import DQN

    env = gym.make("Snake-v0")
    ql = DQN.load(env, "dqn.hdf5")

    observation = env.reset()
    env.render()
    done = False
    while not done:
        time.sleep(0.1)
        action = ql.act(observation)
        observation, reward, done, info = env.step(action)
        env.render()
