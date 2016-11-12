# Written by Evan Gravelle
# 11/7/2016

import gym
import gym_pull
env = gym.make('ppaquette/DoomBasic-v0')

# num_actions = env.action_space.n
# env.monitor.start('tmp/doom-experiment', force=True)
env.reset()
action = [0] * 43
action[0] = 1  # shoot
action[10] = 1  # right
action[11] = 1  # left
for _ in range(100):
    env.render()
    # sample_action = env.action_space.sample()
    observation, reward, done, info = env.step(action)
    print observation.shape
    if done:
        break

# env.monitor.close()
