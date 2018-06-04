import os
import numpy as np
import gym
from gym import spaces
import gzip
import cv2
import time
import skimage.color
import torch
import retro
from collections import deque
import logging
import sys

cv2.ocl.setUseOpenCL(False)

from .data import sonic_buttons, discrete_actions, train_states, validation_states


class RenderWrapper(gym.Wrapper):
    def __init__(self, env, mode='world_model'):
        """Render each step."""
        super().__init__(env)
        self.mode = mode

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        self.render(mode=self.mode)
        return observation, reward, done, info

    def reset(self):
        observation = self.env.reset()
        self.render(mode=self.mode)
        return observation


class ScaledFloatFrame(gym.ObservationWrapper):
    def __init__(self, env):
        gym.ObservationWrapper.__init__(self, env)
        old_obs_space = env.observation_space
        self.observation_space = spaces.Box(low=0, high=1,
                                            shape=old_obs_space.shape, dtype=np.float32)

    def observation(self, observation):
        # careful! This undoes the memory optimization, use
        # with smaller replay buffers only.
        return np.array(observation).astype(np.float32) / 255.0


class SlowFrames(gym.Wrapper):
    def __init__(self, env, fps=60 * 4, frames_per_timestep=4):
        super().__init__(env)
        self.t0 = time.time()
        self.ms_per_step = frames_per_timestep / fps

    def render(self, mode='human'):
        # Note: one timestep is 4 frames
        time_passed = time.time() - self.t0
        if time_passed < self.ms_per_step:
            time.sleep(self.ms_per_step - time_passed)
        self.t0 = time.time()
        return self.env.render(mode)


class WarpFrame(gym.ObservationWrapper):
    def __init__(self, env, width=128, height=128, to_gray=True):
        """Warp frames to 84x84 as done in the Nature paper and later work."""
        # note native sonic size is (224, 320, 3)
        gym.ObservationWrapper.__init__(self, env)
        self.width = width
        self.height = height
        self.to_gray = to_gray
        self.observation_space = spaces.Box(low=0, high=255,
                                            shape=(self.height, self.width, 1 if to_gray else 3), dtype=np.uint8)

    def observation(self, frame):
        if self.to_gray:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)[:, :, None]
        frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)
        return frame


class RandomGameReset(gym.Wrapper):
    def __init__(self, env, state=None):
        """Use the world model to give next latent state as observation."""
        super().__init__(env)
        self.state = state

    def step(self, action):
        return self.env.step(action)

    def reset(self):
        # Reset to a random level (but don't change the game)
        try:
            game = self.env.unwrapped.gamename
        except AttributeError:
            print('no game name')
            pass
        else:
            game_path = retro.get_game_path(game)

            # pick a random state that's in the same game
            game_states = train_states[train_states.game == game]
            if self.state:
                game_states = game_states[game_states.state.str.contains(self.state)]

            # Load
            state = game_states.sample().iloc[0].state + '.state'
            print('reseting to', state)
            with gzip.open(os.path.join(game_path, state), 'rb') as fh:
                self.env.unwrapped.initial_state = fh.read()

        return self.env.reset()


# wrappers below are from from https://github.com/openai/retro-baselines/blob/master/agents/sonic_util.py
class SonicDiscretizer(gym.ActionWrapper):
    """
    Wrap a gym-retro environment and make it use discrete
    actions for the Sonic game.
    """

    def __init__(self, env):
        super(SonicDiscretizer, self).__init__(env)
        self._actions = []
        for action in discrete_actions:
            arr = np.array([False] * 12)
            for button in action:
                arr[sonic_buttons.index(button)] = True
            self._actions.append(arr)
        self.action_space = gym.spaces.Discrete(len(self._actions))

    def action(self, a):  # pylint: disable=W0221
        a = self._actions[a].copy()
        return a


class RewardScaler(gym.RewardWrapper):
    """
    Bring rewards to a reasonable scale for PPO.
    This is incredibly important and effects performance
    drastically.
    """

    def __init__(self, env, scale=0.005):
        gym.RewardWrapper.__init__(self, env)
        self.scale = scale

    def reward(self, reward):
        return reward * self.scale


class AllowBacktracking(gym.Wrapper):
    """
    Use deltas in max(X) as the reward, rather than deltas
    in X. This way, agents are not discouraged too heavily
    from exploring backwards if there is no way to advance
    head-on in the level.
    """

    def __init__(self, env):
        super(AllowBacktracking, self).__init__(env)
        self._cur_x = 0
        self._max_x = 0

    def reset(self, **kwargs):  # pylint: disable=E0202
        self._cur_x = 0
        self._max_x = 0
        return self.env.reset(**kwargs)

    def step(self, action):  # pylint: disable=E0202
        obs, rew, done, info = self.env.step(action)
        self._cur_x += rew
        rew = max(0, self._cur_x - self._max_x)
        self._max_x = max(self._max_x, self._cur_x)
        return obs, rew, done, info

    
class StochasticFrameSkip2(gym.Wrapper):
    """Just allows us to double wrap durign testing."""

    def __init__(self, env, n, stickprob):
        gym.Wrapper.__init__(self, env)
        self.n = n
        self.stickprob = stickprob
        self.curac = None
        self.rng = np.random.RandomState()

    def reset(self, **kwargs):
        self.curac = None
        return self.env.reset(**kwargs)

    def step(self, ac):
        done = False
        totrew = 0
        for i in range(self.n):
            # First step after reset, use action
            if self.curac is None:
                self.curac = ac
            # First substep, delay with probability=stickprob
            elif i == 0:
                if self.rng.rand() > self.stickprob:
                    self.curac = ac
            # Second substep, new action definitely kicks in
            elif i == 1:
                self.curac = ac
            ob, rew, done, info = self.env.step(self.curac)
            totrew += rew
            if done:
                break
        return ob, totrew, done, info
    

# from https://github.com/openai/baselines/blob/master/baselines/common/atari_wrappers.py
class FrameStack(gym.Wrapper):
    def __init__(self, env, k):
        """Stack k last frames.
        Returns array.
        """
        gym.Wrapper.__init__(self, env)
        self.k = k
        self.frames = deque([], maxlen=k)
        shp = env.observation_space.shape
        self.observation_space = spaces.Box(low=0, high=255, shape=(shp[0], shp[1], shp[2] * k), dtype=np.uint8)

    def reset(self):
        ob = self.env.reset()
        for _ in range(self.k):
            self.frames.append(ob)
        return self._get_ob()

    def step(self, action):
        ob, reward, done, info = self.env.step(action)
        self.frames.append(ob)
        return self._get_ob(), reward, done, info

    def _get_ob(self):
        assert len(self.frames) == self.k
        return np.concatenate(list(self.frames), -1)