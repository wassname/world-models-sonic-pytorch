"""
Helpers for https://github.com/ShangtongZhang/DeepRL
"""
import torch
import time
from torch import nn, optim
import numpy as np
from tqdm import tqdm
import pickle

from deep_rl.agent import BaseAgent, Batcher
from deep_rl.component.task import BaseTask

from ..custom_envs.wrappers import RenderWrapper, WorldModelWrapper
from ..custom_envs.env import make_env


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


# modified from to log to tensorboard https://github.com/ShangtongZhang/DeepRL/blob/master/deep_rl/agent/PPO_agent.py
class PPOAgent(BaseAgent):
    def __init__(self, config):
        BaseAgent.__init__(self, config)
        self.config = config
        self.task = config.task_fn()
        self.network = config.network_fn(self.task.state_dim, self.task.action_dim)
        self.opt = config.optimizer_fn(self.network.parameters())
        self.total_steps = 0
        self.episode_rewards = np.zeros(config.num_workers)
        self.last_episode_rewards = np.zeros(config.num_workers)
        self.states = self.task.reset()
        self.states = config.state_normalizer(self.states)

    def iteration(self):
        config = self.config
        rollout = []
        states = self.states
        for _ in range(config.rollout_length):
            actions, log_probs, _, values = self.network.predict(states)
            next_states, rewards, terminals, _ = self.task.step(actions.cpu().detach().numpy())
            self.episode_rewards += rewards
            rewards = config.reward_normalizer(rewards)
            for i, terminal in enumerate(terminals):
                if terminals[i]:
                    self.last_episode_rewards[i] = self.episode_rewards[i]
                    self.episode_rewards[i] = 0
            next_states = config.state_normalizer(next_states)
            rollout.append([states, values.detach(), actions.detach(), log_probs.detach(), rewards, 1 - terminals])
            states = next_states

        self.states = states
        pending_value = self.network.predict(states)[-1]
        rollout.append([states, pending_value, None, None, None, None])

        processed_rollout = [None] * (len(rollout) - 1)
        advantages = self.network.tensor(np.zeros((config.num_workers, 1)))
        returns = pending_value.detach()
        for i in reversed(range(len(rollout) - 1)):
            states, value, actions, log_probs, rewards, terminals = rollout[i]
            terminals = self.network.tensor(terminals).unsqueeze(1)
            rewards = self.network.tensor(rewards).unsqueeze(1)
            actions = self.network.tensor(actions)
            states = self.network.tensor(states)
            next_value = rollout[i + 1][1]
            returns = rewards + config.discount * terminals * returns
            if not config.use_gae:
                advantages = returns - value.detach()
            else:
                td_error = rewards + config.discount * terminals * next_value.detach() - value.detach()
                advantages = advantages * config.gae_tau * config.discount * terminals + td_error
            processed_rollout[i] = [states, actions, log_probs, returns, advantages]

            # config.logger.scalar_summary('td_error', td_error.mean(), self.total_steps+i)
            # config.logger.scalar_summary('returns', returns.mean(), self.total_steps+i)
            # config.logger.scalar_summary('actions', actions, self.total_steps+i)

        states, actions, log_probs_old, returns, advantages = map(lambda x: torch.cat(x, dim=0), zip(*processed_rollout))
        advantages = (advantages - advantages.mean()) / advantages.std()

        batcher = Batcher(states.size(0) // config.num_mini_batches, [np.arange(states.size(0))])
        for _ in range(config.optimization_epochs):
            batcher.shuffle()
            while not batcher.end():
                batch_indices = batcher.next_batch()[0]
                batch_indices = self.network.tensor(batch_indices).long()
                sampled_states = states[batch_indices]
                sampled_actions = actions[batch_indices]
                sampled_log_probs_old = log_probs_old[batch_indices]
                sampled_returns = returns[batch_indices]
                sampled_advantages = advantages[batch_indices]

                _, log_probs, entropy_loss, values = self.network.predict(sampled_states, sampled_actions)
                ratio = (log_probs - sampled_log_probs_old).exp()
                obj = ratio * sampled_advantages
                obj_clipped = ratio.clamp(1.0 - self.config.ppo_ratio_clip,
                                          1.0 + self.config.ppo_ratio_clip) * sampled_advantages
                policy_loss = -torch.min(obj, obj_clipped).mean()
                entropy_loss = - config.entropy_weight * entropy_loss.mean()

                value_loss = 0.5 * (sampled_returns - values).pow(2).mean()

                self.opt.zero_grad()
                loss = (policy_loss + value_loss + entropy_loss)
                loss.backward()
                grad_norm = nn.utils.clip_grad_norm_(self.network.parameters(), config.gradient_clip)
                self.opt.step()

                config.logger.scalar_summary('loss_value', value_loss)
                config.logger.scalar_summary('loss_policy', policy_loss)
                config.logger.scalar_summary('loss_entropy', entropy_loss)
                config.logger.scalar_summary('grad_norm', grad_norm)
                config.logger.scalar_summary('ratio', ratio.mean())
        config.logger.writer.file_writer.flush()

        steps = config.rollout_length * config.num_workers
        self.total_steps += steps

# From https://github.com/ShangtongZhang/DeepRL/blob/master/deep_rl/utils/misc.py#L51
# Modified to save if a differen't location


def run_iterations(agent, log_dir):
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
                np.min(rewards[-config.iteration_log_interval:]),
                np.mean(rewards[-config.iteration_log_interval:]),
                np.max(rewards[-config.iteration_log_interval:]),
                len(rewards[-config.iteration_log_interval:]),
                np.mean(times[-config.iteration_log_interval:]),
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
