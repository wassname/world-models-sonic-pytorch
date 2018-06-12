#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

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
        self.hidden_states = None

    def process_rollout(self, rollout, pending_value):
        config = self.config
        processed_rollout = [None] * (len(rollout) - 1)
        advantages = self.network.tensor(np.zeros((config.num_workers, 1)))
        returns = pending_value.detach()
        for i in reversed(range(len(rollout) - 1)):
            states, value, actions, log_probs, rewards, terminals, next_states, hidden_states = rollout[i]

            next_value = rollout[i + 1][1]
            returns = rewards + config.discount * terminals * returns
            if not config.use_gae:
                advantages = returns - value.detach()
            else:
                td_error = rewards + config.discount * terminals * next_value.detach() - value.detach()
                advantages = advantages * config.gae_tau * config.discount * terminals + td_error

            inds = zip([i] * len(rewards), range(len(rewards)))
            inds = self.network.tensor(list(inds)).long()
            processed_rollout[i] = [states, actions, log_probs, returns, advantages, next_states, hidden_states, inds]

        # Concat the rollout vars
        states, actions, log_probs_old, returns, advantages, next_states, hidden_states, inds = map(lambda x: torch.cat(x, dim=0), zip(*processed_rollout))

        advantages = (advantages - advantages.mean()) / advantages.std()
        return states, actions, log_probs_old, returns, advantages, next_states, hidden_states, inds

    def iteration(self):
        config = self.config
        rollout = []
        states = self.states
        hidden_states = self.hidden_states

        for _ in range(config.rollout_length):
            with torch.no_grad():
                actions, log_probs, _, values, hidden_states = self.network.predict(states, hidden_states=hidden_states)
            next_states, rewards, terminals, _ = self.task.step(actions.cpu().detach().numpy())
            self.episode_rewards += rewards
            rewards = config.reward_normalizer(rewards[:, None])[:, 0]
            for i, terminal in enumerate(terminals):
                if terminals[i]:
                    self.last_episode_rewards[i] = self.episode_rewards[i]
                    self.episode_rewards[i] = 0
            next_states = config.state_normalizer(next_states)
            rollout.append([
                self.network.tensor(states),
                values.detach(),
                actions.detach(),
                log_probs.detach(),
                self.network.tensor(rewards).unsqueeze(1),
                self.network.tensor(1 - terminals).unsqueeze(1),
                self.network.tensor(next_states),
                hidden_states.detach()
            ])
            states = next_states

        _, _, _, pending_value, hidden_states = self.network.predict(states, hidden_states=hidden_states)
        rollout.append([states, pending_value, None, None, None, None, hidden_states])
        self.states = states
        self.hidden_states = hidden_states.detach()

        if config.train_world_model:

            # Train world model on rollouts, that way the mdn-rnn get's a sequence
            # Stack rollouts into (seq_len, batch_size, ..)
            states, value, actions, log_probs, rewards, terminals, next_states, hidden_states = map(lambda x: torch.stack(x, dim=1), zip(*rollout[:-1]))
            instrinsic = self.network.tensor(np.zeros(states.size(0)))
            extrinsic = torch.cat([roll[4] for roll in rollout[:-1]]).mean()
            batcher = Batcher(config.world_model_batch_size, [np.arange(states.size(0))])
            batcher.shuffle()
            while not batcher.end():
                batch_indices = batcher.next_batch()[0]
                batch_indices = self.network.tensor(batch_indices).long()
                sampled_states = states[batch_indices]
                sampled_actions = actions[batch_indices]
                sampled_next_states = next_states[batch_indices]
                sampled_hidden_states = hidden_states[batch_indices]

                _, _, _, initial_loss = self.network.train_world_model(sampled_states, sampled_actions, sampled_next_states, sampled_hidden_states, train=True)
                if config.curiosity:
                    # Train world model here and update values with curiosity reward, before we calculate advantages
                    # For an intro to the idea see :https://arxiv.org/abs/1705.05363 . But my approach is to make the reward
                    # the reduction of loss from a state, similar to mentioned here http://people.idsia.ch/~juergen/creativity.html
                    _, _, _, loss = self.network.train_world_model(sampled_states, sampled_actions, sampled_next_states, sampled_hidden_states, train=False)
                    loss = loss.view((config.world_model_batch_size, -1))
                    initial_loss = initial_loss.view((config.world_model_batch_size, -1))
                    intrinsic_rewards = initial_loss - loss

                    # Update reward
                    for i, k in enumerate(batch_indices):
                        intrinsic_reward = config.intrinsic_reward_normalizer(intrinsic_rewards[i].cpu().numpy()[:, None])[:, 0]
                        intrinsic_reward = (intrinsic_reward - config.curiosity_boredom) * config.curiosity_weight
                        intrinsic_reward = self.network.tensor(intrinsic_reward)
                        for j in range(len(intrinsic_reward)):
                            if config.curiosity_only:
                                rollout[j][4][k] = intrinsic_reward[j].detach()
                            else:
                                rollout[j][4][k] += intrinsic_reward[j].detach()

            # Log
            if config.curiosity:
                extrinsic_after = torch.cat([roll[4] for roll in rollout[:-1]])
                if config.curiosity_only:
                    instrinsic = extrinsic_after
                else:
                    instrinsic = extrinsic_after - extrinsic

                config.logger.scalar_summary('reward_extrinsic', extrinsic.mean())
                config.logger.scalar_summary('reward_intrinsic', instrinsic.mean())
                # print('rollout extrinsic, intrinsic reward [min/mean/max]: {:2.4f}/{:2.4f}/{:2.4f}, {:2.4f}/{:2.4f}/{:2.4f}'.format(
                #     extrinsic.min().cpu().item(),
                #     extrinsic.mean().cpu().item(),
                #     extrinsic.max().cpu().item(),
                #     instrinsic.min().cpu().item(),
                #     instrinsic.mean().cpu().item(),
                #     instrinsic.max().cpu().item()
                # ))

        # Calculate advantages again now that we have changed the rewards
        states, actions, log_probs_old, returns, advantages, next_states, hidden_states, rewards = self.process_rollout(rollout, pending_value)

        # Now train PPO
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

                _, log_probs, entropy_loss, values, _ = self.network.predict(sampled_states, sampled_actions, sampled_next_states, sampled_hidden_states)
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
                config.logger.scalar_summary('sampled_returns', sampled_returns.mean())
        config.logger.writer.file_writer.flush()

        steps = config.rollout_length * config.num_workers
        self.total_steps += steps
