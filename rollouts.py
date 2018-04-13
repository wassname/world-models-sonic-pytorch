import gym
import torch


def random_rollouts(env, num_rollouts, render=False):
    """
        This function collects random rollouts from a given environment.
    """
    obs = env.reset()
    num_obs = 0
    rollouts = {'observations': [obs], 'actions': ['0']}
    while num_obs != num_rollouts:
        if render:
            env.render()
        num_obs += 1
        action = env.action_space.sample()
        obs, reward, done, _ = env.step(action)
        rollouts['observations'].append(obs)
        rollouts['actions'].append(action)

        if done:
            obs = env.reset()
            rollouts['observations'].append(obs)
            rollouts['actions'].append('0')
    return rollouts


def save_rollouts(rollouts):
    torch.save(rollouts, 'rollouts.data')


def load_rollouts(fname):
    return torch.load(fname)


