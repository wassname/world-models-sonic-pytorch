import gym_remote.exceptions as gre
import gym_remote.client as grc
import torch
import os
import sys
import logging

from deep_rl.utils import Config
from deep_rl.utils.logger import get_logger, get_default_log_dir

from deep_rl.network.network_heads import CategoricalActorCriticNet
from deep_rl.network.network_bodies import FCBody

from deep_rl.component.task import ParallelizedTask

from world_models_sonic.custom_envs import wrappers
from world_models_sonic.helpers.deep_rl import PPOAgent, SonicWorldModelDeepRL, run_iterations
from world_models_sonic.models.rnn import MDNRNN2
from world_models_sonic.models.vae import VAE6, VAE5
from world_models_sonic.models.inverse_model import InverseModel
from world_models_sonic.models.world_model import WorldModel

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger(__name__)


def make_env(stack=False, scale_rew=True, image_size=256):
    """Create an environment with some standard wrappers."""
    env = grc.RemoteEnv('tmp/sock')
    if scale_rew:
        env = wrappers.RewardScaler(env)
    env = wrappers.SonicDiscretizer(env)
    env = wrappers.AllowBacktracking(env)
    env = wrappers.ScaledFloatFrame(env)
    env = wrappers.WarpFrame(env, image_size, image_size, to_gray=False)
    # if stack:
    #     env = FrameStack(env, 4)
    # env = BatchedFrameStack(BatchedGymEnv([[env]]), num_images=4, concat=False)
    return env


def load_world_models(save_file_vae, save_file_rnn, save_file_finv, image_size=256, cuda=True, action_dim=10, z_dim=512):

    # Load VAE
    vae = VAE5(image_size=image_size, z_dim=128, conv_dim=64, code_dim=8, k_dim=z_dim)
    # vae = VAE6(image_size=image_size, z_dim=32, conv_dim=48, code_dim=8, k_dim=z_dim)
    if cuda:
        vae.cuda()

    state_dict = torch.load(save_file_vae)
    vae.load_state_dict(state_dict)
    print('loaded save_file {save_file}'.format(save_file=save_file_vae))

    # Load MDRNN
    action_dim, hidden_size, n_mixture, temp = action_dim, 512, 2, 0.0
    mdnrnn = MDNRNN2(z_dim, action_dim, hidden_size, n_mixture, temp)
    if cuda:
        mdnrnn = mdnrnn.cuda()

    state_dict = torch.load(save_file_rnn)
    mdnrnn.load_state_dict(state_dict)
    print('loaded {save_file}'.format(save_file=save_file_rnn))

    finv = InverseModel(z_dim, action_dim, hidden_size=256)
    if cuda:
        finv = finv.cuda()
    state_dict = torch.load(save_file_finv)
    finv.load_state_dict(state_dict)
    print('loaded {save_file}'.format(save_file=save_file_finv))

    world_model = WorldModel(vae, mdnrnn, finv)
    return world_model


def main():
    log_dir = './results' # get_default_log_dir('results')
    for dir in ['log', 'results', 'outputs', 'models', 'data', log_dir]:
        if not os.path.isdir(dir):
            os.makedirs(dir)
    cuda = torch.cuda.is_available()
    print('cuda', cuda)
    print('log_dir', log_dir)

    ppo_save_file = './data/checkpoints/PPO_greenfields_256z_v2_cpu.pkl'
    ppo_save_file_reward_norm = ppo_save_file.replace('_cpu.pkl',
                                                      '') + '_reward_norm.pkl'
    ppo_save_file_state_norm = ppo_save_file.replace('_cpu.pkl',
                                                     '') + '_state_norm.pkl'
    save_file_rnn = './data/checkpoints/mdnrnn_state_dict_cpu.pkl'
    save_file_vae = './data/checkpoints/vae_state_dict_cpu.pkl'
    save_file_finv = './data/checkpoints/finv_state_dict_cpu.pkl'

    world_model = load_world_models(save_file_vae, save_file_rnn, save_file_finv, image_size=256, cuda=cuda, action_dim=10, z_dim=512)

    def task_fn(log_dir):
        return SonicWorldModelDeepRL(
            env_fn=lambda: make_env(),
            max_steps=10000,
            log_dir=log_dir,
            world_model_func=lambda: world_model,
            verbose=False,
            cuda=cuda
        )

    config = Config()

    config.num_workers = 1
    config.task_fn = lambda: ParallelizedTask(task_fn, config.num_workers, single_process=config.num_workers == 1)
    config.optimizer_fn = lambda params: torch.optim.RMSprop(params, 1e-3)
    config.network_fn = lambda state_dim, action_dim: CategoricalActorCriticNet(
        state_dim, action_dim, FCBody(state_dim), gpu=0 if cuda else -1)
    config.discount = 0.99
    config.logger = get_logger('deep_rl_ppo', file_name='deep_rl_ppo.log', level=logging.INFO, log_dir=log_dir)
    config.use_gae = True
    config.gae_tau = 0.95
    config.entropy_weight = 0.0001
    config.gradient_clip = 0.4
    config.rollout_length = 128
    config.optimization_epochs = 10
    config.num_mini_batches = 4
    config.ppo_ratio_clip = 0.2
    config.iteration_log_interval = 1
    agent = PPOAgent(config)
    if os.path.isfile(ppo_save_file):
        print('loading', ppo_save_file)
        agent.load(ppo_save_file)
        agent.config.state_normalizer.load_state_dict(torch.load(ppo_save_file_state_norm))
        agent.config.reward_normalizer.load_state_dict(torch.load(ppo_save_file_reward_norm))
        print('running')
    run_iterations(agent, log_dir)


if __name__ == '__main__':
    try:
        main()
    except gre.GymRemoteError as exc:
        print('exception', exc)
