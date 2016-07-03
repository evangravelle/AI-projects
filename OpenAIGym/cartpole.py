import gym
# env = gym.make('CartPole-v0')
env = gym.make('MountainCar-v0')
env.monitor.start('./tmp/cartpole-experiment-1', force=True)
env.reset()
for _ in range(1000):
    env.render()
    observation, reward, done, info = env.step(env.action_space.sample())  # take a random action
    if done:
        break

env.monitor.close()
