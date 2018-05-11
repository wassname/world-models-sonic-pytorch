"""
Run on linux with "sudo `which python` live.py"

Records human playthroughs as bk2 files.
"""
# util
import os
import sys
import datetime

# numeric
import numpy as np
import pandas as pd

# sys.path.append('..')
import keyboard
import logging

import gym
import retro
import retro.data
from retro_contest.local import make

# PARAMS

TRAJ_DIR = './outputs/trajectories'
ROM_DIRS = ['./roms', retro.data_path()]

# INIT
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__file__)
logger.setLevel(logging.INFO)

train_states = pd.read_csv('./data/sonic-train.csv')
validation_states = pd.read_csv('./data/sonic-validation.csv')

# keyboard._os_keyboard.build_tables()


def roms_import_paths(paths):
    """Import sega roms from a list of paths"""
    potential_roms = []
    for path in paths:
        for base, _, files in os.walk(path):
            potential_roms.extend([os.path.join(base, file) for file in files])
    print('Importing %i potential games...' % len(potential_roms))
    retro.data.merge(*potential_roms, quiet=False)

# Import roms (place in './roms)
roms_import_paths(['./roms'])

def keyboard_to_actions():
    """Transform pressed keys to actions."""

    # set the keys you want to use for actions here
    # see http://www.gadgetreview.com/wp-content/uploads/2015/02/Sega-Genesis-Mk2-6button.jpg
    # to list all key names `import keyboard; keyboard._os_keyboard.build_tables(); keyboard._os_keyboard.from_name.keys()`
    button_names = [
        "5", #"B",
        "4", # A",
        "SHIFT", # "MODE",
        "ENTER", # "START",
        "w", # "UP",
        "s", #"DOWN",
        "a", # "LEFT",
        "d", # "RIGHT",
        "6", # "C",
        "7", #"Y",
        "8", # "X",
        "9", # "Z"
    ]
    button_names = [name.lower() for name in button_names]
    action = np.zeros((12,))
    logger.debug('keyboard._pressed_events %s', keyboard._pressed_events)
    for code, e in keyboard._pressed_events.items():
        if e.event_type == 'down':
            if e.name.lower() in button_names:
                i = button_names.index(e.name)
                action[i] = 1
    return action


class TrajectoryRecorder(gym.Wrapper):
    """
    Save play trajectories.

    Files are saved to csv's similar to the [atarigrand project ](https://github.com/yobibyte/atarigrandchallenge).
    """
    def __init__(self, env, directory):
        super().__init__(env)
        self.trajectory = []

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        self.trajectory.append([
            self.env._elapsed_steps,
            reward,
            info['score'],
            done,
            action
        ])
        return observation, reward, done, info

    def reset(self):
        # Save to file
        if len(self.trajectory) > 0:
            df = pd.DataFrame(self.trajectory, columns=["frame", "reward", "score", "terminal", "action"])
            timestamp = datetime.datetime.utcnow().strftime('%Y%m%d_%H-%M-%S')
            trajectory_file = os.path.join(TRAJ_DIR, 'csv', self.env.unwrapped.gamename, self.env.unwrapped.statename, '{}.csv'.format(timestamp))

            trajectory_dir = os.path.dirname(trajectory_file)
            if not os.path.isdir(trajectory_dir):
                os.makedirs(trajectory_dir)

            df.to_csv(trajectory_file)
            logger.info("saved trajectory %s", trajectory_file)

        # reset
        self.trajectory = []
        return self.env.reset()

def main():
    # Start random train game
    start_state = train_states.sample().iloc[0]
    logger.info("starting %s", start_state.to_dict())

    timestamp = datetime.datetime.utcnow().strftime('%Y%m%d_%H-%M-%S')
    bk2dir = os.path.join(TRAJ_DIR, 'bk2', timestamp + "_human")
    if not os.path.isdir(bk2dir):
        os.makedirs(bk2dir)
    logger.info("saving recordings to %s", bk2dir)

    env = make(game=start_state.game, state=start_state.state, bk2dir=bk2dir)
    env = TrajectoryRecorder(env, TRAJ_DIR)
    obs = env.reset()
    keyboard.start_recording()

    while True:
        action = keyboard_to_actions()
        logger.debug(action)
        obs, rew, done, info = env.step(action)

        env.render()
        if done:
            # Reset
            print(dict(
                score=info['score'],
                reward=rew,
                elapsed_steps=env.env._elapsed_steps
            ))
            print('\n')
            obs = env.reset()


if __name__ == "__main__":
    main()
