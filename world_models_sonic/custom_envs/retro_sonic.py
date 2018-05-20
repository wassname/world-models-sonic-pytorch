# util
import os
import sys
import datetime

# numeric
import numpy as np
import pandas as pd

import logging

import gym
import retro
import retro.data
from retro_contest.local import make
import retro_contest
from . import wrappers

# INIT
ROM_DIRS = ['./roms', retro.data_path()]
train_states = pd.read_csv('./data/sonic_env/sonic-train.csv')
validation_states = pd.read_csv('./data/sonic_env/sonic-validation.csv')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__file__)
logger.setLevel(logging.INFO)

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


def make(game, state, discrete_actions=False, bk2dir=None):
    """Make the competition environment."""
    print('game:', game, 'state:', state)
    use_restricted_actions = retro.ACTIONS_FILTERED
    if discrete_actions:
        use_restricted_actions = retro.ACTIONS_DISCRETE
    try:
        env = retro.make(game, state, scenario='contest', use_restricted_actions=use_restricted_actions)
    except Exception:
        env = retro.make(game, state, use_restricted_actions=use_restricted_actions)
    if bk2dir:
        env.auto_record(bk2dir)
    env = retro_contest.StochasticFrameSkip(env, n=4, stickprob=0.25)
    env = gym.wrappers.TimeLimit(env, max_episode_steps=4500)
    return env


def make_sonic(game=None, state=None, image_size=128, discrete_actions=False, bk2dir=None, slow=False):
    """My function to make the environment."""
    if game is not None:
        # restrict to this game
        game_states = train_states[train_states.game == game]
    else:
        game_states = train_states
    if state is not None:
        # Restrict to state that contain the provided string
        game_states = game_states[game_states.state.str.contains(state)]

    # choose a random matching game/state
    start_state = game_states.sample().iloc[0]

    env = make(game=start_state.game, state=start_state.state)
    env = wrappers.RewardScaler(env)
    env = wrappers.SonicDiscretizer(env)
    env = wrappers.AllowBacktracking(env)
    env = wrappers.ScaledFloatFrame(env)
    env = wrappers.WarpFrame(env, image_size, image_size, to_gray=False)
    if slow:
        env = wrappers.SlowFrames(env)
    return env


ACTION_MEANING = dict(enumerate(['B', 'A', 'MODE', 'START', 'UP', 'DOWN', 'LEFT', 'RIGHT', 'C', 'Y', 'X', 'Z']))
