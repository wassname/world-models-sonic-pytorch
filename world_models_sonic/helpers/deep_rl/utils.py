import time
import numpy as np
import pickle
import torch


def run_iterations(agent, log_dir):
    # From https://github.com/ShangtongZhang/DeepRL/blob/master/deep_rl/utils/misc.py#L51
    # Modified to save if a differen't location
    # TODO add tqdm
    config = agent.config
    agent_name = agent.__class__.__name__
    iteration = 0
    steps = []
    rewards = []
    times = []
    t0 = time.time()
    while True:
        agent.iteration()
        steps.append(agent.total_steps)
        rewards += agent.last_episode_rewards.tolist()
        times.append((time.time() - t0) / len(agent.last_episode_rewards))
        t0 = time.time()
        if iteration % config.iteration_log_interval == 0:
            config.logger.info('loss_vae  %2.4f loss_KLD  %2.4f loss_recon  %2.4f loss_mdn %2.4f loss_inv  %2.4f' % (
                agent.network.world_model.last_loss_vae,
                agent.network.world_model.last_loss_KLD,
                agent.network.world_model.last_loss_recon,
                agent.network.world_model.last_loss_mdn,
                agent.network.world_model.last_loss_inv
            ))

            config.logger.info('total steps %d, min/mean/max reward %2.4f/%2.4f/%2.4f of %d' % (
                agent.total_steps,
                np.min(agent.last_episode_rewards),
                np.mean(agent.last_episode_rewards),
                np.max(agent.last_episode_rewards),
                len(agent.last_episode_rewards)
            ))
            config.logger.info('running min/mean/max reward %2.4f/%2.4f/%2.4f of %d %2.4f s/rollout' % (
                np.min(rewards[-500:]),
                np.mean(rewards[-500:]),
                np.max(rewards[-500:]),
                len(rewards[-500:]),
                np.mean(times[-500:]),
            ))
        if iteration % (config.iteration_log_interval * 100) == 0:
            with open('%s/stats-%s-%s-online-stats-%s.pkl' % (log_dir, agent_name, config.tag, agent.task.name), 'wb') as f:
                pickle.dump({'rewards': rewards,
                             'steps': steps}, f)
            agent.save('%s/%s-%s-model-%s.pkl' % (log_dir, agent_name, config.tag, agent.task.name))
            torch.save(agent.network.world_model.state_dict(), '%s/%s-%s-world_model-%s.pkl' % (log_dir, agent_name, config.tag, agent.task.name))
        iteration += 1
        if config.max_steps and agent.total_steps >= config.max_steps:
            agent.close()
            break

    return steps, rewards
