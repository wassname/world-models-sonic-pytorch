import time
import numpy as np
import pickle


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
    # with tqdm(mininterval=1, unit='it', total=config.max_steps, leave=True) as prog:
    while True:
        agent.iteration()
        steps.append(agent.total_steps)
        rewards.append(np.mean(agent.last_episode_rewards))
        times.append((time.time() - t0) / len(agent.last_episode_rewards))
        t0 = time.time()
        if iteration % config.iteration_log_interval == 0:
            config.logger.info('total steps %d, min/mean/max reward %2.4f/%2.4f/%2.4f of %d' % (
                agent.total_steps,
                np.min(agent.last_episode_rewards),
                np.mean(agent.last_episode_rewards),
                np.max(agent.last_episode_rewards),
                len(agent.last_episode_rewards)
            ))
            config.logger.info('running min/mean/max reward %2.4f/%2.4f/%2.4f of %d %2.4f s/rollout' % (
                np.min(rewards[-100:]),
                np.mean(rewards[-100:]),
                np.max(rewards[-100:]),
                len(rewards[-100:]),
                np.mean(times[-100:]),
            ))
        if iteration % (config.iteration_log_interval * 100) == 0:
            with open('%s/stats-%s-%s-online-stats-%s.pkl' % (log_dir, agent_name, config.tag, agent.task.name), 'wb') as f:
                pickle.dump({'rewards': rewards,
                             'steps': steps}, f)
            agent.save('%s/%s-%s-model-%s.pkl' % (log_dir, agent_name, config.tag, agent.task.name))
        # prog.desc = 'total steps %d, mean/max/min reward %f/%f/%f of %d' % (
        #     agent.total_steps, np.mean(rewards[-config.iteration_log_interval:]),
        #     np.max(rewards[-config.iteration_log_interval:]),
        #     np.min(rewards[-config.iteration_log_interval:]),
        #     len(rewards[-config.iteration_log_interval:])
        # )
        # prog.update(1)
        iteration += 1
        if config.max_steps and agent.total_steps >= config.max_steps:
            agent.close()
            break

    return steps, rewards
