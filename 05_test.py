import gym_remote.exceptions as gre
import gym_remote.client as grc
import torch
import os
import sys
import logging
from torch.nn import functional as F
from torch import optim

from deep_rl.utils.logger import get_logger
from deep_rl.network.network_bodies import FCBody
from deep_rl.utils.normalizer import RunningStatsNormalizer, RescaleNormalizer
from deep_rl.component.task import ParallelizedTask

from world_models_sonic.custom_envs import wrappers
from world_models_sonic.helpers.deep_rl import PPOAgent, run_iterations, SonicWorldModelDeepRL, CategoricalWorldActorCriticNet, Config
from world_models_sonic.models.rnn import MDNRNN
from world_models_sonic.models.vae import VAE5
from world_models_sonic.models.inverse_model import InverseModel
from world_models_sonic.models.world_model import WorldModel

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger(__name__)


def make_env(stack=True, scale_rew=True, image_size=256, to_gray=False):
    """Create an environment with some standard wrappers."""
    env = grc.RemoteEnv('tmp/sock')
    if scale_rew:
        env = wrappers.RewardScaler(env)
    env = wrappers.SonicDiscretizer(env)
    env = wrappers.AllowBacktracking(env)
    env = wrappers.ScaledFloatFrame(env)
    env = wrappers.WarpFrame(env, image_size, image_size, to_gray=to_gray)
    env = wrappers.StochasticFrameSkip2(env, n=4, stickprob=0)
    if stack:
        env = wrappers.FrameStack(env, 4)
    # env = BatchedFrameStack(BatchedGymEnv([[env]]), num_images=4, concat=False)
    return env


def load_world_models(deep_rl_logger, image_size=128, cuda=True, action_dim=10, z_dim=512, channels=3*4):
    # Load VAE
    vae = VAE5(image_size=image_size, z_dim=128, conv_dim=64, code_dim=8, k_dim=z_dim, channels=channels)

    # Load MDRNN
    action_dim, hidden_size, n_mixture, temp = action_dim, z_dim * 2, 5, 0.0
    mdnrnn = MDNRNN(z_dim, action_dim, hidden_size, n_mixture, temp)

    # Finv
    finv = InverseModel(z_dim, action_dim, hidden_size=z_dim * 2)

    world_model = WorldModel(vae, mdnrnn, finv, logger=deep_rl_logger, lambda_vae_kld=1 / 1024., lambda_finv=1 / 100, lambda_vae=1 / 16, lambda_loss=1000)
    world_model = world_model.train()
    if cuda:
        world_model = world_model.cuda()
    return world_model


def main():
    verbose = False
    NAME = 'wassname_ppo'
    channels = 3 * 4
    z_dim = 512
    image_size = 128
    action_dim = 10

    log_dir = './results'
    for dir in ['log', 'results', 'outputs', 'models', 'data', log_dir]:
        if not os.path.isdir(dir):
            os.makedirs(dir)
    cuda = torch.cuda.is_available()
    print('cuda', cuda)
    print('log_dir', log_dir)

    ppo_save_file = './data/checkpoints/PPOAgent-vanilla-model-sonic_cpu.pkl'
    deep_rl_logger = get_logger(
        NAME[-10:],
        file_name='deep_rl_ppo.log',
        level=logging.INFO,
        log_dir=log_dir, )
    world_model = load_world_models(deep_rl_logger, image_size=image_size, cuda=cuda, action_dim=action_dim, z_dim=z_dim, channels=channels)
    optimizer = optim.Adam(world_model.parameters(), lr=0)
    world_model.optimizer = optimizer

    z_state_dim = world_model.mdnrnn.z_dim + world_model.mdnrnn.hidden_size + world_model.mdnrnn.action_dim
    print(log_dir)

    def task_fn(log_dir):
        return SonicWorldModelDeepRL(
            env_fn=lambda: make_env('sonic256', to_gray=False, image_size=image_size),
            log_dir=log_dir,
            verbose=verbose
        )

    config = Config()

    config.num_workers = 1
    config.task_fn = lambda: ParallelizedTask(
        task_fn, config.num_workers, single_process=config.num_workers == 1)
    config.optimizer_fn = lambda params: torch.optim.RMSprop(params, 3e-4)
    config.network_fn = lambda state_dim, action_dim: CategoricalWorldActorCriticNet(
        state_dim, action_dim, FCBody(z_state_dim, hidden_units=(64, 64), gate=F.relu), gpu=0 if cuda else -1, world_model_fn=lambda: world_model,
        render=(config.num_workers == 1 and verbose),
        z_shape=(32, 16)
    )
    config.discount = 0.99
    config.logger = deep_rl_logger
    config.use_gae = True
    config.gae_tau = 0.95
    config.entropy_weight = 0.0001
    config.gradient_clip = 0.4
    config.rollout_length = 64
    config.optimization_epochs = 10
    config.num_mini_batches = 8
    config.ppo_ratio_clip = 0.2
    config.iteration_log_interval = 1

    config.train_world_model = False
    config.world_model_batch_size = 1

    # I tuned these so the intrinsic reward was 1) within an order of magnitude of the extrinsic. 2) smaller, 3) negative when stuck
    # TODO use reward normalisers etc to reduce the need for these hyperparameters
    config.curiosity = False
    config.curiosity_only = False
    config.curiosity_weight = 0.1
    config.curiosity_boredom = 0  # how many standard deviations above the mean does it's new experience need to be, so it's not bored
    config.intrinsic_reward_normalizer = RescaleNormalizer()

    agent = PPOAgent(config)
    if os.path.isfile(ppo_save_file):
        print('loading', ppo_save_file)
        agent.load(ppo_save_file)

    try:
        run_iterations(agent, log_dir=log_dir)
    except:
        if config.num_workers == 1:
            agent.task.tasks[0].env.close()
        else:
            [t.close() for t in agent.task.tasks]
        print("saving", ppo_save_file)
        agent.save(ppo_save_file)
        raise


if __name__ == '__main__':
    try:
        main()
    except gre.GymRemoteError as exc:
        print('exception', exc)
