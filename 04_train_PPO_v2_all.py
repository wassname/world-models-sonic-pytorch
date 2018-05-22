
# coding: utf-8

import deep_rl

import torch
from torch.nn import functional as F
from torch.autograd import Variable
from torch import nn, optim
import torch.utils.data

# load as dask array
import dask.array as da
import dask
import h5py

import os
import glob
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm_notebook as tqdm

from world_models_sonic.models.vae import VAE5, loss_function_vae
from world_models_sonic.helpers.summarize import TorchSummarizeDf
from world_models_sonic.helpers.dataset import load_cache_data
from world_models_sonic.custom_envs.wrappers import RenderWrapper, WorldModelWrapper
from world_models_sonic.models.rnn import MDNRNN2
from world_models_sonic.models.inverse_model import InverseModel
from world_models_sonic.models.world_model import WorldModel
from world_models_sonic import config
from world_models_sonic.custom_envs.env import make_env

import logging
import sys

# logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.basicConfig(
    level=logging.INFO, 
    format='[{%(filename)s:%(lineno)d} %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(filename='log/tmp5a.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

# # Init
cuda=torch.cuda.is_available()
env_name='sonic256'
z_dim = 512 # latent dimensions

# RNN
action_dim = 10
image_size=256

verbose=False # Set this true to render (and makes it go slower)

NAME='RNN_v3b_256im_512z_1512_v6_VAE5_all'
ppo_save_file = './outputs/models/PPO_greenfields_256z_v2.pkl'

# # Load Data

# # Load VAE

vae = VAE5(image_size=image_size, z_dim=128, conv_dim=64, code_dim=8, k_dim=512)
if cuda:
    vae.cuda()

# # Resume
save_file = './outputs/models/{NAME}-vae_state_dict2.pkl'.format(NAME=NAME)
if os.path.isfile(save_file):
    state_dict = torch.load(save_file)
    vae.load_state_dict(state_dict)
    print(f'loaded save_file {save_file}')


# Load MDRNN
action_dim, hidden_size, n_mixture, temp = action_dim, 512, 2, 0.0
mdnrnn = MDNRNN2(z_dim, action_dim, hidden_size, n_mixture, temp)

if cuda:
    mdnrnn = mdnrnn.cuda()

save_file = './outputs/models/{NAME}-mdnrnn_state_dict2.pkl'.format(NAME=NAME)
if os.path.isfile(save_file):
    state_dict = torch.load(save_file)
    mdnrnn.load_state_dict(state_dict)
    print('loaded {save_file}'.format(save_file=save_file))


# # FInverse Model

finv = InverseModel(z_dim, action_dim, hidden_size=256)
if cuda:
    finv.cuda()
finv.eval()

# Resume?
save_file = './outputs/models/{NAME}-finv_state_dict.pkl'.format(NAME=NAME)
if os.path.isfile(save_file):
    state_dict = torch.load(save_file)
    finv.load_state_dict(state_dict)
    print('loaded {save_file}'.format(save_file=save_file))

# # Init

world_model = WorldModel(vae, mdnrnn, finv)
world_model=world_model.eval()
world_model


# # Env wrappers


from deep_rl.utils import Config
from deep_rl.utils.logger import get_logger, get_default_log_dir

from deep_rl.network.network_heads import CategoricalActorCriticNet, QuantileNet, OptionCriticNet, DeterministicActorCriticNet, GaussianActorCriticNet
from deep_rl.network.network_bodies import FCBody

from deep_rl.agent.PPO_agent import PPOAgent
from deep_rl.component.task import ParallelizedTask, BaseTask
from deep_rl.utils.misc import run_episodes, run_iterations


class SonicWorldModelDeepRL(BaseTask):
    """Sonic environment wrapper for deep_rl."""
    def __init__(self, name='sonic256', max_steps=4500, log_dir=None, world_model_func=None, state=None, game=None):
        BaseTask.__init__(self)
        self.name = name
        self.world_model = world_model_func()
        self.env = WorldModelWrapper(make_env(self.name, state=state, game=game), self.world_model, cuda=cuda, state=state)
        self.env._max_episode_steps = max_steps
        self.action_dim = self.env.action_space.n
        self.state_dim = self.env.observation_space.shape[0]
        self.env = self.set_monitor(self.env, log_dir)
        if verbose:
            self.env = RenderWrapper(self.env, mode='world_model') # world_model_next


# # Train
log_dir = get_default_log_dir(os.path.basename(ppo_save_file))
print(log_dir)
task_fn = lambda log_dir: SonicWorldModelDeepRL(
    'sonic256',
    max_steps=200,
    log_dir=log_dir,
    world_model_func=lambda :world_model,
#    state='GreenHillZone.Act1',
#    game='SonicTheHedgehog-Genesis'
)

config = Config()

config.num_workers = 1
config.task_fn = lambda: ParallelizedTask(task_fn, config.num_workers, single_process=config.num_workers==1)
config.optimizer_fn = lambda params: torch.optim.RMSprop(params, 1e-3)
config.network_fn = lambda state_dim, action_dim: CategoricalActorCriticNet(
        state_dim, action_dim, FCBody(state_dim), gpu=0 if cuda else -1)
config.discount = 0.99
config.logger = get_logger(NAME, level=logging.INFO)
config.use_gae = True
config.gae_tau = 0.95
config.entropy_weight = 0.0001
config.gradient_clip = 0.5
config.rollout_length = 128
config.optimization_epochs = 10
config.num_mini_batches = 4
config.ppo_ratio_clip = 0.2
config.iteration_log_interval = 1
agent=PPOAgent(config)
# env = agent.task.tasks[0].env
if os.path.isfile(ppo_save_file):
    print('loading', ppo_save_file)
    agent.load(ppo_save_file)


try:
    run_iterations(agent)
except:
    agent.task.tasks[0].env.close()
    agent.save(ppo_save_file)
    raise
