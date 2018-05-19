# xvfb-run -s "-screen 0 1400x900x24" python generate_data.py sonic
# sudo /home/wassname/.pyenv/shims/python 01_generate_data.py sonic256 --batch_size 1 --total_episodes 500 --state GreenHillZone --game SonicTheHedgehog-Genesis --time_steps 3600 --human
# python 01_generate_data.py sonic256 --batch_size 4 --total_episodes 500 --game SonicTheHedgehog-Genesis --time_steps 600


import numpy as np
import os
import time
import argparse
import glob
import zarr

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

        # # Instead of overwriting batches let's add more
        # filenames = sorted(glob.glob(os.path.join(config.base_vae_data_dir, 'obs_data_' + env_name + '_*.np*')))
        # batches = [int(os.path.splitext(file)[0].split('_')[-1]) for file in filenames]
        # start_batch = max(batches) + 1 if len(batches) else 0

        print("Generating data for env {}".format(current_env_name))

        s = 0
        z_obs = None
        batch = 0

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
                env = make_env(current_env_name, state=args.state, game=args.game, slow=args.human)

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

                # since we stopped it early let's add a done signal
                done_sequence[-1] = True

                obs_data.append(obs_sequence)
                action_data.append(action_sequence)
                reward_data.append(reward_sequence)
                done_data.append(done_sequence)

                print("Batch {} Episode {} finished after {} timesteps".format(batch, i_episode, t + 1))
                print("Current dataset contains {} observations".format(sum(map(len, obs_data))))

                s = s + 1
                env.close()

            print('=========')
            print("Saving dataset for batch {} in {}".format(batch, config.base_vae_data_dir))

            obs_data = (np.concatenate(obs_data, axis=0) * 255.0).astype(np.uint8)
            action_data = np.concatenate(action_data, axis=0).astype(np.uint8)
            done_data = np.concatenate(done_data, axis=0).astype(np.uint8)
            reward_data = np.concatenate(reward_data, axis=0).astype(np.float32)

            # I'm using zarr since I can append and in a slicable format, and it's fast and small
            # Chunk sizes between 10MB-1GB are common https://dask.pydata.org/en/latest/array-creation.html#chunks
            chunk_size = 500

            obs_mb = obs_data.size * obs_data.itemsize * 1e-6
            chunk_mb = np.prod((chunk_size,) + obs_data.shape[1:]) * obs_data.itemsize * 1e-6
            print("Saving %s, %d MB, chunk_size %d MB" % (obs_data.shape, obs_mb, chunk_mb))

            # Open zarray if not already open
            if not z_obs:
                z_obs = zarr.open(os.path.join(config.base_vae_data_dir, 'obs_data.zarr'), mode='a',
                                  shape=obs_data.shape, chunks=(chunk_size,) + obs_data.shape[1:], dtype=obs_data.dtype)
                z_act = zarr.open(os.path.join(config.base_vae_data_dir, 'action_data.zarr'), mode='a',
                                  shape=action_data.shape, chunks=(chunk_size,) + action_data.shape[1:], dtype=action_data.dtype)
                z_don = zarr.open(os.path.join(config.base_vae_data_dir, 'done_data.zarr'), mode='a',
                                  shape=done_data.shape, chunks=(chunk_size,) + done_data.shape[1:], dtype=done_data.dtype)
                z_rew = zarr.open(os.path.join(config.base_vae_data_dir, 'reward_data.zarr'), mode='a',
                                  shape=reward_data.shape, chunks=(chunk_size,) + reward_data.shape[1:], dtype=reward_data.dtype)
                print('opened file', z_obs)

            # append new data
            z_obs.append(obs_data)
            z_act.append(action_data)
            z_don.append(done_data)
            z_rew.append(reward_data)
            print('appended file', z_obs)

            # np.save(os.path.join(config.base_vae_data_dir, 'obs_data_' + current_env_name + '_' + str(batch)), obs_data)
            # np.save(os.path.join(config.base_vae_data_dir, 'action_data_' + current_env_name + '_' + str(batch)), action_data)
            # np.save(os.path.join(config.base_vae_data_dir, 'reward_data_' + current_env_name + '_' + str(batch)), reward_data)
            # np.save(os.path.join(config.base_vae_data_dir, 'done_data_' + current_env_name + '_' + str(batch)), done_data)

            # hdf5_file = os.path.join(config.base_vae_data_dir, 'obs_data_' + current_env_name + '_' + str(batch)+'.hdf5')
            # chunk_size = 1000
            # da.to_hdf5(hdf5_file, dict(
            #     obs_data = da.from_array(obs_data, chunks=(chunk_size,)+obs_data.shape[1:]),
            #     action_data = da.from_array(action_data, chunks=(chunk_size,)+action_data.shape[1:]),
            #     reward_data = da.from_array(reward_data, chunks=(chunk_size,)+reward_data.shape[1:]),
            #     done_data = da.from_array(done_data, chunks=(chunk_size,)+done_data.shape[1:]),
            # ))

            batch = batch + 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=('Create new training data'))
    parser.add_argument('env_name', type=str, default='sonc256', help='name of environment')
    parser.add_argument('--state', type=str, default=None,
                        help='choose a sonic level to play e.g. "GreenHillZone.Act1", you can provide a partial string e.g. "GreenHillZone" to get a random match')
    parser.add_argument('--game', type=str, default=None,
                        help='choose a sonic game to play e.g. "SonicTheHedgehog-Genesis". Full list in data/sonice_env/sonic-train.csv')
    parser.add_argument('--total_episodes', type=int, default=2000, help='total number of episodes to generate')
    parser.add_argument('--time_steps', type=int, default=300, help='how many timesteps at start of episode?')
    parser.add_argument('--render', action='store_true', help='render the env as data is generated')
    parser.add_argument('--human', action='store_true', help='render the env and let a human control as data is generated')
    parser.add_argument('--batch_size', type=int, default=10, help='how many episodes in a batch (one file)')
    parser.add_argument('--run_all_envs', action='store_true', help='if true, will ignore env_name and loop over all envs in train_envs variables in config.py')

    args = parser.parse_args()
    main(args)
