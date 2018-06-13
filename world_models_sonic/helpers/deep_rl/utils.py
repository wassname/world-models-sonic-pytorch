#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################


import time
import numpy as np
import pickle
from collections import defaultdict


def run_iterations(agent, log_dir):
    # From https://github.com/ShangtongZhang/DeepRL/blob/master/deep_rl/utils/misc.py#L51
    # Modified to save if a differen't location
    config = agent.config
    agent_name = agent.__class__.__name__
    iteration = 0
    history = defaultdict(list)
    t0 = time.time()
    while True:
        steps0 = agent.total_steps
        agent.iteration()
        history['steps'].append(agent.total_steps - steps0)
        history['rewards'] += agent.last_episode_rewards.tolist()
        history['times'].append(history['steps'][-1] / (time.time() - t0))
        history['loss'].append(agent.network.world_model.last_loss)
        history['loss_vae'].append(agent.network.world_model.last_loss_vae)
        history['loss_KLD'].append(agent.network.world_model.last_loss_KLD)
        history['loss_recon'].append(agent.network.world_model.last_loss_recon)
        history['loss_mdn'].append(agent.network.world_model.last_loss_mdn)
        history['loss_inv'].append(agent.network.world_model.last_loss_inv)
        t0 = time.time()
        if iteration % config.iteration_log_interval == 0:
            msg1 = '\n  steps: {}, steps/s: {:2.2f}'.format(
                agent.total_steps,
                np.mean(history['times'][-500:]),
            )
            msg2 = '\n  world model losses: loss={loss:2.2f} loss_rnn={loss_mdn:2.2f}, loss_inv= {loss_inv2:2.2f}={lambda_finv:2.2f} * {loss_inv:2.2f}, loss_vae={loss_vae:2.2f}={lambda_vae:2.2f} * ({loss_recon:2.2f} + {lambda_vae_kld:2.2f} * {loss_KLD:2.2f})'.format(
                loss=np.mean(history['loss'][-500:]),
                loss_mdn=np.mean(history['loss_mdn'][-500:]),
                loss_recon=np.mean(history['loss_recon'][-500:]),
                loss_KLD=np.mean(history['loss_KLD'][-500:]),
                loss_vae=agent.network.world_model.lambda_vae *
                (np.mean(history['loss_recon'][-500:]) + agent.network.world_model.lambda_vae_kld * np.mean(history['loss_KLD'][-500:])),
                loss_inv=np.mean(history['loss_inv'][-500:]),
                loss_inv2=np.mean(history['loss_inv'][-500:]) * agent.network.world_model.lambda_finv,
                lambda_vae_kld=agent.network.world_model.lambda_vae_kld,
                lambda_finv=agent.network.world_model.lambda_finv,
                lambda_vae=agent.network.world_model.lambda_vae,
            )
            msg3 = '\n  epoch reward:       %2.4f/%2.4f/%2.4f [n=%d] (min/mean/max)' % (
                np.min(agent.last_episode_rewards),
                np.mean(agent.last_episode_rewards),
                np.max(agent.last_episode_rewards),
                len(agent.last_episode_rewards)
            )
            msg4 = '\n  running reward:     %2.4f/%2.4f/%2.4f [n=%s]' % (
                np.min(history['rewards'][-500:]),
                np.mean(history['rewards'][-500:]),
                np.max(history['rewards'][-500:]),
                len(history['rewards'][-500:])
            )
            config.logger.info(msg1 + msg2 + msg3 + msg4)
        if iteration % (config.iteration_log_interval * 100) == 0:
            with open('%s/stats-%s-%s-online-stats-%s.pkl' % (log_dir, agent_name, config.tag, agent.task.name), 'wb') as f:
                pickle.dump({'rewards': history['rewards'],
                             'steps': history['steps']}, f)
            agent.save('%s/%s-%s-model-%s.pkl' % (log_dir, agent_name, config.tag, agent.task.name))
            # torch.save(agent.network.world_model.state_dict(), '%s/%s-%s-world_model-%s.pkl' % (log_dir, agent_name, config.tag, agent.task.name))
        iteration += 1
        if config.max_steps and agent.total_steps >= config.max_steps:
            agent.close()
            break

    return history
