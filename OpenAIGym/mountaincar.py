import gym
env = gym.make('MountainCar-v0')
# There are 3 discrete actions, presumably accelerate left, right, or do nothing
# There are 2 observations, one in [-1.2, 0.6] and the other in [-0.07, 0.07]
# Possibly x location and speed
# env.monitor.start('./tmp/cartpole-experiment-1', force=True)

for ep in range(1):
    observation = env.reset()
    # print observation
    for t in range(200):
        env.render()
        # action = env.action_space.sample()  # take a random action
        if t < 50:
            action = 0
        else:
            action = 2
        observation, reward, done, info = env.step(action)
        # if t % 10 == 0:
        print "t = ", t
        print "obs = ", observation
        print "reward = ", reward
        if reward != -1.0:
            print reward

        if done:
            break

# env.monitor.close()
