import gym
import gym_pull     # Only required once, envs will be loaded with import gym_pull afterwards
env = gym.make('ppaquette/DoomBasic-v0')

# num_actions = env.action_space.n
# env.monitor.start('tmp/doom-experiment', force=True)
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

# env.monitor.close()
