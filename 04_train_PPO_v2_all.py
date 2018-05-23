# coding: utf-8
import deep_rl

import torch
from torch.nn import functional as F
from torch.autograd import Variable
from torch import nn, optim
import torch.utils.data

import os
import glob
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm_notebook as tqdm

from deep_rl.utils import Config
from deep_rl.utils.logger import get_logger, get_default_log_dir
from deep_rl.network.network_heads import CategoricalActorCriticNet
from deep_rl.network.network_bodies import FCBody
from deep_rl.component.task import ParallelizedTask

from world_models_sonic.models.vae import VAE5, loss_function_vae
from world_models_sonic.helpers.summarize import TorchSummarizeDf
from world_models_sonic.helpers.dataset import load_cache_data
from world_models_sonic.models.rnn import MDNRNN2
from world_models_sonic.models.inverse_model import InverseModel
from world_models_sonic.models.world_model import WorldModel
from world_models_sonic import config
from world_models_sonic.helpers.deep_rl import PPOAgent, run_iterations, SonicWorldModelDeepRL

import logging
import sys


# # Init
cuda=torch.cuda.is_available()
env_name='sonic256'
z_dim = 512 # latent dimensions

# RNN
action_dim = 10
image_size=256

# TODO get this working with forwarding on a headless server
verbose=False # Set this true to render (and makes it go slower)

NAME='RNN_v3b_256im_512z_1512_v6_VAE5_all'
ppo_save_file = f'./outputs/{NAME}/PPO_greenfields_256z_v2.pkl'
ppo_save_file_reward_norm = ppo_save_file.replace('.pkl', '') + '_reward_norm.pkl'
ppo_save_file_state_norm = ppo_save_file.replace('.pkl', '') + '_state_norm.pkl'
save_file_rnn = './outputs/{NAME}/mdnrnn_state_dict.pkl'.format(NAME=NAME)
save_file_vae = './outputs/{NAME}/vae_state_dict.pkl'.format(NAME=NAME)
save_file_finv = './outputs/{NAME}/finv_state_dict.pkl'.format(NAME=NAME)
if not os.path.isdir(f'./outputs/{NAME}'):
    os.makedirs(f'./outputs/{NAME}')

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger(NAME)


# # Load VAE
vae = VAE5(image_size=image_size, z_dim=128, conv_dim=64, code_dim=8, k_dim=512)
if cuda:
    vae.cuda()
vae.eval()

# # Resume
if os.path.isfile(save_file_vae):
    state_dict = torch.load(save_file_vae)
    vae.load_state_dict(state_dict)
    print(f'loaded save_file {save_file_vae}')


# Load MDRNN
action_dim, hidden_size, n_mixture, temp = action_dim, 512, 2, 0.0
mdnrnn = MDNRNN2(z_dim, action_dim, hidden_size, n_mixture, temp)
mdnrnn.eval()
if cuda:
    mdnrnn = mdnrnn.cuda()

if os.path.isfile(save_file_rnn):
    state_dict = torch.load(save_file_rnn)
    mdnrnn.load_state_dict(state_dict)
    print('loaded {save_file}'.format(save_file=save_file_rnn))


# # FInverse Model
finv = InverseModel(z_dim, action_dim, hidden_size=256)
if cuda:
    finv.cuda()
finv.eval()

# Resume
if os.path.isfile(save_file_finv):
    state_dict = torch.load(save_file_finv)
    finv.load_state_dict(state_dict)
    print('loaded {save_file}'.format(save_file=save_file_finv))


# World model wrapper module
world_model = WorldModel(vae, mdnrnn, finv)
world_model = world_model.eval()
world_model


# # Train
import datetime
log_dir=f'./outputs/{NAME}'
print('log_dir', log_dir)

# FIXME multiple workers doesn't work because I've included pytorch in the env wrapper

task_fn = lambda log_dir: SonicWorldModelDeepRL(
    'sonic256',
    max_steps=1000, # FIXME doesn't do anything
    log_dir=log_dir,
    world_model_func=lambda :world_model,
#    state='GreenHillZone.Act1',
#    game='SonicTheHedgehog-Genesis',
    cuda=cuda,
    verbose=verbose,
)

config = Config()

config.num_workers = 1
config.task_fn = lambda: ParallelizedTask(task_fn, config.num_workers, single_process=config.num_workers==1)
config.optimizer_fn = lambda params: torch.optim.RMSprop(params, 1e-3)
config.network_fn = lambda state_dim, action_dim: CategoricalActorCriticNet(
        state_dim, action_dim, FCBody(state_dim), gpu=0 if cuda else -1)
config.discount = 0.99
config.logger = get_logger(NAME, file_name='agent.log', level=logging.INFO, log_dir=f'./outputs/{NAME}')
config.use_gae = True
config.gae_tau = 0.95
config.entropy_weight = 0.0001
config.gradient_clip = 0.4
config.rollout_length = 128
config.optimization_epochs = 10
config.num_mini_batches = 4
config.ppo_ratio_clip = 0.2
config.iteration_log_interval = 10

agent=PPOAgent(config)

if os.path.isfile(ppo_save_file):
    print('loading', ppo_save_file)
    agent.load(ppo_save_file)
    agent.config.state_normalizer.load_state_dict(torch.load(ppo_save_file_state_norm))
    agent.config.reward_normalizer.load_state_dict(torch.load(ppo_save_file_reward_norm))

try:
    run_iterations(agent, f'./outputs/{NAME}')
except:
    print("saving", ppo_save_file)
    agent.save(ppo_save_file)
    torch.save(agent.config.state_normalizer.state_dict(), ppo_save_file_state_norm)
    torch.save(agent.config.reward_normalizer.state_dict(), ppo_save_file_reward_norm)
    if config.num_workers==1:
        agent.task.tasks[0].env.close()
    raise
