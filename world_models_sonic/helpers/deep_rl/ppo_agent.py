import torch
from torch import nn
import numpy as np

from deep_rl.agent import BaseAgent, Batcher


class PPOAgent(BaseAgent):
    # modified from to log to tensorboard https://github.com/ShangtongZhang/DeepRL/blob/master/deep_rl/agent/PPO_agent.py
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
        hidden_states = None

        for _ in range(config.rollout_length):
            with torch.no_grad():
                actions, log_probs, _, values, hidden_states = self.network.predict(states, hidden_state=hidden_states)
            next_states, rewards, terminals, _ = self.task.step(actions.cpu().detach().numpy())
            self.episode_rewards += rewards
            rewards = config.reward_normalizer(rewards)
            for i, terminal in enumerate(terminals):
                if terminals[i]:
                    self.last_episode_rewards[i] = self.episode_rewards[i]
                    self.episode_rewards[i] = 0
            next_states = config.state_normalizer(next_states)
            rollout.append([states, values.detach(), actions.detach(), log_probs.detach(), rewards, 1 - terminals, next_states, hidden_states.detach()])
            states = next_states

        self.states = states
        pending_value = self.network.predict(states)[-2]
        rollout.append([states, pending_value, None, None, None, None, None, None])

        processed_rollout = [None] * (len(rollout) - 1)
        advantages = self.network.tensor(np.zeros((config.num_workers, 1)))
        returns = pending_value.detach()
        for i in reversed(range(len(rollout) - 1)):
            states, value, actions, log_probs, rewards, terminals, next_states, hidden_states = rollout[i]
            terminals = self.network.tensor(terminals).unsqueeze(1)
            rewards = self.network.tensor(rewards).unsqueeze(1)
            actions = self.network.tensor(actions)
            states = self.network.tensor(states)
            next_states = self.network.tensor(next_states)
            hidden_states = self.network.tensor(hidden_states)

            # For hidden states we will pad them to a consistent length then stack

            # hidden_states = self.network.tensor(torch.stack(hidden_states, 1)) # (batch, num_hidden_states, z_dims)
            # hidden_states = self.network.tensor(torch.stack(hidden_states, dim=0))
            next_value = rollout[i + 1][1]
            returns = rewards + config.discount * terminals * returns
            if not config.use_gae:
                advantages = returns - value.detach()
            else:
                td_error = rewards + config.discount * terminals * next_value.detach() - value.detach()
                advantages = advantages * config.gae_tau * config.discount * terminals + td_error
            processed_rollout[i] = [states, actions, log_probs, returns, advantages, next_states, hidden_states]

            # config.logger.scalar_summary('td_error', td_error.mean(), self.total_steps+i)
            # config.logger.scalar_summary('returns', returns.mean(), self.total_steps+i)
            # config.logger.scalar_summary('actions', actions, self.total_steps+i)

        # Concat the rollout vars
        states, actions, log_probs_old, returns, advantages, next_states, hidden_states = map(lambda x: torch.cat(x, dim=0), zip(*processed_rollout))
        # states, actions, log_probs_old, returns, advantages, next_states, hidden_states = zip(*processed_rollout)
        # states, actions, log_probs_old, returns, advantages, next_states = map(lambda x: torch.cat(x, dim=0), [states, actions, log_probs_old, returns, advantages, next_states])
        # hidden_states2 = list(map(lambda x: torch.cat(x, dim=0), zip(*hidden_states)))
        # print(len(hidden_states2), hidden_states2[0].shape)

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
                sampled_next_states = next_states[batch_indices]
                sampled_hidden_states = hidden_states[batch_indices]

                _, log_probs, entropy_loss, values, hidden_state = self.network.predict(sampled_states, sampled_actions, sampled_next_states, sampled_hidden_states)
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
