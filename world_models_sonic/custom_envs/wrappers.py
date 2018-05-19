import numpy as np
import gym
from gym import spaces
import cv2
import time
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

# class TrajectoryRecorder(gym.Wrapper):
#     """
#     Save play trajectories.
#
#     Files are saved to csv's similar to the [atarigrand project ](https://github.com/yobibyte/atarigrandchallenge).
#     """
#     def __init__(self, env, directory):
#         super().__init__(env)
#         self.trajectory = []
#
#     def step(self, action):
#         observation, reward, done, info = self.env.step(action)
#         self.trajectory.append([
#             self.env._elapsed_steps,
#             reward,
#             info['score'],
#             done,
#             action
#         ])
#         return observation, reward, done, info
#
#     def reset(self):
#         # Save to file
#         if len(self.trajectory) > 0:
#             df = pd.DataFrame(self.trajectory, columns=["frame", "reward", "score", "terminal", "action"])
#             timestamp = datetime.datetime.utcnow().strftime('%Y%m%d_%H-%M-%S')
#             trajectory_file = os.path.join(TRAJ_DIR, 'csv', self.env.unwrapped.gamename, self.env.unwrapped.statename, '{}.csv'.format(timestamp))
#
#             trajectory_dir = os.path.dirname(trajectory_file)
#             if not os.path.isdir(trajectory_dir):
#                 os.makedirs(trajectory_dir)
#
#             df.to_csv(trajectory_file)
#             logger.info("saved trajectory %s", trajectory_file)
#
#         # reset
#         self.trajectory = []
#         return self.env.reset()
