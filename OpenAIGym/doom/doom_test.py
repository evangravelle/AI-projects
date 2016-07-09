import gym
env = gym.make('DoomBasic-v0')

env.monitor.start('./tmp/doom-experiment', force=True)
env.reset()
action = [0] * 43
action[0] = 1
action[10] = 1
for _ in range(100):
    env.render()
    # sample_action = env.action_space.sample()
    observation, reward, done, info = env.step(action)
    print observation.shape
    if done:
        break

env.monitor.close()
