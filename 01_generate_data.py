# xvfb-run -s "-screen 0 1400x900x24" python generate_data.py sonic
# sudo /home/wassname/.pyenv/shims/python 01_generate_data.py sonic256 --batch_size 3 --total_episodes 500 --render --state GreenHillZone.Act1 --time_steps 1200

import numpy as np
import os
import argparse
import glob
import dask.array as da

from world_models_sonic.custom_envs.env import make_env
from world_models_sonic import config
from world_models_sonic.record_human_plays import keyboard_to_actions
import keyboard

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

    if args.human:
        keyboard.start_recording()
        config.base_vae_data_dir = os.path.join(config.base_vae_data_dir, 'human')

    env_name = args.env_name
    total_episodes = args.total_episodes
    time_steps = args.time_steps
    render = args.render
    batch_size = args.batch_size

    envs_to_generate = [env_name]

    for current_env_name in envs_to_generate:

        # Instead of overwriting batches let's add more
        filenames = sorted(glob.glob(os.path.join(config.base_vae_data_dir, 'obs_data_' + env_name + '_*.npz')))
        batches = [int(os.path.splitext(file)[0].split('_')[-1]) for file in filenames]
        start_batch = max(batches) + 1 if len(batches) else 0

        print("Generating data for env {}".format(current_env_name))

        s = 0
        batch = start_batch

        batch_size = min(batch_size, total_episodes)

        while s < total_episodes:
            if args.human:
                print('-----Waiting for enter-----')
                keyboard.wait('ENTER')
            obs_data = []
            action_data = []
            reward_data = []
            done_data = []

            for i_episode in range(batch_size):
                env = make_env(current_env_name)
                observation = env.reset()
                print('-----')

                if render or args.human:
                    env.render()
                done = False
                t = 0
                obs_sequence = []
                action_sequence = []
                reward_sequence = []
                done_sequence = []

                while t < time_steps:  # and not done:
                    t = t + 1

                    if args.human:
                        action = keyboard_to_actions()
                    else:
                        action = env.action_space.sample()
                        action = generate_data_action(t, action)

                    obs_sequence.append(observation)
                    action_sequence.append(action)

                    observation, reward, done, info = env.step(action)
                    reward_sequence.append(reward)
                    done_sequence.append(done)

                    if render or args.human:
                        env.render()

                obs_data.append(obs_sequence)
                action_data.append(action_sequence)
                reward_data.append(reward_sequence)
                done_data.append(done_sequence)

                print("Batch {} Episode {} finished after {} timesteps".format(batch, i_episode, t + 1))
                print("Current dataset contains {} observations".format(sum(map(len, obs_data))))

                s = s + 1
                env.close()

            print(config.base_vae_data_dir)
            print("Saving dataset for batch {}".format(batch))

            np.save(os.path.join(config.base_vae_data_dir, 'obs_data_' + current_env_name + '_' + str(batch)), obs_data)
            np.save(os.path.join(config.base_vae_data_dir, 'action_data_' + current_env_name + '_' + str(batch)), action_data)
            np.save(os.path.join(config.base_vae_data_dir, 'reward_data_' + current_env_name + '_' + str(batch)), reward_data)
            np.save(os.path.join(config.base_vae_data_dir, 'done_data_' + current_env_name + '_' + str(batch)), done_data)

            batch = batch + 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=('Create new training data'))
    parser.add_argument('env_name', type=str, default='sonc256', help='name of environment')
    parser.add_argument('--state', type=str, default=None, help='choose a sonic level to play e.g. "GreenHillZone.Act1"')
    parser.add_argument('--total_episodes', type=int, default=2000, help='total number of episodes to generate')
    parser.add_argument('--time_steps', type=int, default=300, help='how many timesteps at start of episode?')
    parser.add_argument('--render', action='store_true', help='render the env as data is generated')
    parser.add_argument('--human', action='store_true', help='render the env and let a human control as data is generated')
    parser.add_argument('--batch_size', type=int, default=10, help='how many episodes in a batch (one file)')
    parser.add_argument('--run_all_envs', action='store_true', help='if true, will ignore env_name and loop over all envs in train_envs variables in config.py')

    args = parser.parse_args()
    main(args)
