from deep_rl.component.task import BaseTask

from ...custom_envs.wrappers import RenderWrapper


class SonicWorldModelDeepRL(BaseTask):
    """Sonic environment wrapper for deep_rl."""

    def __init__(self, env_fn, name='sonic', max_steps=10000, log_dir=None, cuda=True, verbose=False):
        BaseTask.__init__(self)
        self.name = name
        self.env = env_fn()
        self.env._max_episode_steps = max_steps
        self.action_dim = self.env.action_space.n
        self.state_dim = self.env.observation_space.shape[0]
        self.env = self.set_monitor(self.env, log_dir)
        if verbose:
            self.env = RenderWrapper(self.env, mode='world_model')
