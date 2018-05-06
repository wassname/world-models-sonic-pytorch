import numpy as np
import gym
from gym import spaces
import cv2
cv2.ocl.setUseOpenCL(False)


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


class WarpFrame(gym.ObservationWrapper):
    def __init__(self, env, width=128, height=128, to_gray=True):
        """Warp frames to 84x84 as done in the Nature paper and later work."""
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


class RewardScaler(gym.RewardWrapper):
    """
    Bring rewards to a reasonable scale for PPO.
    This is incredibly important and effects performance
    drastically.
    """

    def __init__(self, env, scale=0.01):
        gym.RewardWrapper.__init__(self, env)
        self.scale = scale

    def reward(self, reward):
        return reward * self.scale
