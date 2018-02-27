# OpenAI Snake

Warm-up exercise from OpenAI's [Requests for Research](https://blog.openai.com/requests-for-research-2/).

### Task
Implement a clone of the classic Snake game as a Gym environment, and solve it with a reinforcement learning algorithm of your choice. Tweet us videos of the agent playing. Were you able to train a policy that wins the game?

### Solution
Created the Snake Environment, but don't have an agent yet.

```python
    import gym
    import envs
    import time

    env = gym.make("Snake-v0")
    observation = env.reset()
    env.render()
    done = False
    while not done:
        time.sleep(0.1)
        action = env.action_space.sample() # (take random actions)
        observation, reward, done, info = env.step(action)
        env.render()
```
