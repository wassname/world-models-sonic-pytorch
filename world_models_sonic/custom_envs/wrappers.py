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


class WorldModelWrapper(gym.Wrapper):
    def __init__(self, env, world_model, cuda=True, state=None):
        """Use the world model to give next latent state as observation."""
        super().__init__(env)
        self.world_model = world_model
        self.max_hidden_states = 6
        self.state = state
        self.img_z = None
        self.img_z_next_pred = None
        self.hidden_state = None
        self.cuda = cuda
        # old_obs_space = env.observation_space
        self.observation_space = spaces.Box(low=-1000, high=1000,
                                            shape=(world_model.mdnrnn.z_dim + world_model.mdnrnn.hidden_size,), dtype=np.float32)

    def process_obs(self, observation, action=None):
        if action is None:
            action = self.env.action_space.sample()
        action = torch.from_numpy(np.array(action)).unsqueeze(0).unsqueeze(0).float()
        observation = torch.from_numpy(observation).unsqueeze(0).transpose(1, 3)
        if self.cuda:
            action = action.cuda()
            observation = observation.cuda()

        z_next, z, hidden_state = self.world_model.forward(observation, action, hidden_state=self.hidden_state)
        z = z.squeeze(0).cpu().data.numpy()
        z_next = z_next.squeeze(0).cpu().data.numpy()
        latest_hidden = hidden_state[-1].squeeze(0).squeeze(0).cpu().data.numpy()

        self.z = z
        self.z_next = z_next
        hidden_state = [h.data for h in hidden_state]  # Otherwise it doesn't garbge collect
        if self.max_hidden_states == 1:
            self.hidden_state = hidden_state[-1][None, :]
        else:
            self.hidden_state = hidden_state[-self.max_hidden_states:]

        return np.concatenate([z, latest_hidden])

    def step(self, action):
        # action = action.round(0).astype(int)
        observation, reward, done, info = self.env.step(action)
        observation = self.process_obs(observation, action)
        return observation, reward, done, info

    def reset(self):
        # Reset to a random level (but don't change the game)
        game = self.env.unwrapped.gamename
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

        # Reset
        self.hidden_state = None
        observation = self.env.reset()
        observation = self.process_obs(observation)
        return observation

    def render(self, mode='human', close=False):
        env = self.env.unwrapped
        if close:
            for viewer in [self.viewer_z, self.viewer_z_next, self.viewer_img_z, self.viewer_img_z_next]:
                viewer.close()
            if env.viewer:
                env.viewer.close()
            return
        if mode == 'world_model':
            # Render while showing latent vectors and decoded latent vectors
            if env.viewer is None:
                from gym.envs.classic_control.rendering import SimpleImageViewer
                import pyglet
                margin_vert = 60  # for window border
                # TODO add margin_horiz
                mult = 3

                env.viewer = SimpleImageViewer()
                env.viewer.window = pyglet.window.Window(width=160 * mult, height=128 * mult, vsync=False, resizable=True, caption='Game output')
                env.viewer.window.set_location(0 * mult, 0 * mult)

                self.viewer_img_z = SimpleImageViewer()
                self.viewer_img_z.window = pyglet.window.Window(width=128 * mult, height=128 * mult, vsync=False, resizable=True, caption='Decoded image')
                self.viewer_img_z.window.set_location(160 * mult, 0 * mult)

                self.viewer_img_z_next = SimpleImageViewer()
                self.viewer_img_z_next.window = pyglet.window.Window(width=128 * mult, height=128 * mult,
                                                                     vsync=False, resizable=True, caption='Decoded predicted image')
                self.viewer_img_z_next.window.set_location((160 + 128) * mult, 0 * mult)

                self.viewer_z = SimpleImageViewer()
                self.viewer_z.window = pyglet.window.Window(width=128 * mult, height=128 * mult, vsync=False, resizable=True, caption='latent vector')
                self.viewer_z.window.set_location(160 * mult, 128 * mult + margin_vert)

                self.viewer_z_next = SimpleImageViewer()
                self.viewer_z_next.window = pyglet.window.Window(width=128 * mult, height=128 * mult, vsync=False,
                                                                 resizable=True, caption='latent predicted vector')
                self.viewer_z_next.window.set_location((160 + 128) * mult, 128 * mult + margin_vert)

            # Decode latent vector for display

            # to pytorch
            zv = torch.from_numpy(self.z)[None, :]
            zv_next = torch.from_numpy(self.z_next)[None, :]
            if self.cuda:
                zv = zv.cuda()
                zv_next = zv_next.cuda()

            # Decode
            img_z = self.world_model.vae.decode(zv)
            img_z_next = self.world_model.vae.decode(zv_next)

            # to numpy images
            img_z = img_z.squeeze(0).transpose(0, 2)
            img_z = img_z.data.cpu().numpy()
            img_z = (img_z * 255.0).astype(np.uint8)
            img_z_next = img_z_next.squeeze(0).transpose(0, 2).clamp(0, 1)
            img_z_next = img_z_next.data.cpu().numpy()
            img_z_next = (img_z_next * 255.0).astype(np.uint8)

            z_uint8 = ((self.z + 0.5) * 255).astype(np.uint8).reshape((16, 16))
            z_uint8 = skimage.color.gray2rgb(z_uint8)  # Make it rgb to avoid problems with pyglet
            z_uint8 = cv2.resize(z_uint8, dsize=(256, 256), interpolation=cv2.INTER_NEAREST)  # Resize manually to avoid interp of pixels

            z_next_uint8 = ((self.z_next + 0.5) * 255).astype(np.uint8).reshape((16, 16))
            z_next_uint8 = skimage.color.gray2rgb(z_next_uint8)
            z_next_uint8 = cv2.resize(z_next_uint8, dsize=(256, 256), interpolation=cv2.INTER_NEAREST)

            # Display
            env.viewer.imshow(env.img)
            self.viewer_img_z.imshow(img_z)
            self.viewer_img_z_next.imshow(img_z_next)
            self.viewer_z.imshow(z_uint8)
            self.viewer_z_next.imshow(z_next_uint8)
            return env.viewer.isopen

        return self.env.render(mode=mode, close=close)


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
        return self._actions[a].copy()


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
