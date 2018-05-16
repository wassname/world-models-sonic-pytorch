# xvfb-run -s "-screen 0 1400x900x24" python generate_data.py sonic

import numpy as np
from custom_envs.env import make_env

import argparse


def generate_data_action(t, current_action):
    """Default action used to 01 generate data."""
    if t % 4 == 0:
        return [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]  # right
    elif t % 5 == 0:
        return [0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0]  # down right
    elif t % 6 == 0:
        return [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]  # jump and right
    elif t % 7 == 0:
        return [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]  # down jump
    return current_action


def main(args):

    env_name = args.env_name
    total_episodes = args.total_episodes
    start_batch = args.start_batch
    time_steps = args.time_steps
    render = args.render
    batch_size = args.batch_size
    run_all_envs = args.run_all_envs

    envs_to_generate = [env_name]

    for current_env_name in envs_to_generate:
        print("Generating data for env {}".format(current_env_name))

        s = 0
        batch = start_batch

        batch_size = min(batch_size, total_episodes)

        while s < total_episodes:
            obs_data = []
            action_data = []

            for i_episode in range(batch_size):
                env = make_env(current_env_name)
                print('-----')
                observation = env.reset()

                if render:
                    env.render()
                done = False
                t = 0
                obs_sequence = []
                action_sequence = []

                while t < time_steps:  # and not done:
                    t = t + 1

                    action = env.action_space.sample()
                    action = generate_data_action(t, action)

                    obs_sequence.append(observation)
                    action_sequence.append(action)

                    observation, reward, done, info = env.step(action)

                    if render:
                        env.render()

                obs_data.append(obs_sequence)
                action_data.append(action_sequence)

                print("Batch {} Episode {} finished after {} timesteps".format(batch, i_episode, t + 1))
                print("Current dataset contains {} observations".format(sum(map(len, obs_data))))

                s = s + 1
                env.close()

            print("Saving dataset for batch {}".format(batch))
            np.savez('/data/vae/obs_data_' + current_env_name + '_' + str(batch), obs_data)
            np.savez('/data/vae/action_data_' + current_env_name + '_' + str(batch), action_data)

            batch = batch + 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=('Create new training data'))
    parser.add_argument('env_name', type=str, help='name of environment')
    parser.add_argument('--total_episodes', type=int, default=2000, help='total number of episodes to generate')
    parser.add_argument('--start_batch', type=int, default=0, help='start_batch number')
    parser.add_argument('--time_steps', type=int, default=300, help='how many timesteps at start of episode?')
    parser.add_argument('--render', action='store_true', help='render the env as data is generated')
    parser.add_argument('--batch_size', type=int, default=10, help='how many episodes in a batch (one file)')
    parser.add_argument('--run_all_envs', action='store_true', help='if true, will ignore env_name and loop over all envs in train_envs variables in config.py')

    args = parser.parse_args()
    main(args)
