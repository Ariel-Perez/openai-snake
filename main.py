"""
Environment for the Snake game.

The agent controls a snake on a bordered plane, which moves around eating
pellets to grow and get points. The snake has a specific length, so there
is a moving tail a fixed number of units away from the head.

The player loses when the snake runs into the screen border or itself.
"""
if __name__ == '__main__':
    import gym
    import envs
    import time

    env = gym.make("Snake-v0")
    observation = env.reset()
    env.render()
    done = False
    while not done:
        time.sleep(0.1)
        action = env.action_space.sample() # your agent here (this takes random actions)
        observation, reward, done, info = env.step(action)
        env.render()
