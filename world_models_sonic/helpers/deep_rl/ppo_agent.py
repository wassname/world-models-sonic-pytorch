#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

import torch
from torch import nn
import numpy as np
import collections

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

        # Get default values
        self.states = self.task.reset()
        self.states = config.state_normalizer(self.states)
        self.hidden_states = self.network.default_hidden_state(self.network.tensor(self.states))
        self.z_states = self.network.process_obs(self.states, hidden_states=self.hidden_states)

        self.last_info_world_model = None

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

            processed_rollout[i] = [states, actions, log_probs, returns, advantages, next_states, hidden_states, value]

        # Concat the rollout vars
        states, actions, log_probs_old, returns, advantages, next_states, hidden_states, values = map(lambda x: torch.cat(x, dim=0), zip(*processed_rollout))

        advantages = (advantages - advantages.mean()) / advantages.std()
        return states, actions, log_probs_old, returns, advantages, next_states, hidden_states, values

    def iteration(self):
        config = self.config
        rollout = []
        world_model_rollout = []
        states = self.states
        z_states = self.z_states
        hidden_states = self.hidden_states
        steps = config.rollout_length * config.num_workers

        for _ in range(config.rollout_length):
            with torch.no_grad():
                actions, log_probs, _, values, next_hidden_states = self.network.predict(z_states, hidden_states=hidden_states)
            next_states, rewards, terminals, _ = self.task.step(actions.cpu().detach().numpy())
            self.episode_rewards += rewards
            rewards = config.reward_normalizer(rewards[:, None])[:, 0]
            for i, terminal in enumerate(terminals):
                if terminals[i]:
                    self.last_episode_rewards[i] = self.episode_rewards[i]
                    self.episode_rewards[i] = 0
            next_states = config.state_normalizer(next_states)
            next_z_states = self.network.process_obs(next_states, hidden_states=next_hidden_states)

            # Store values
            rollout.append([
                z_states,
                values.detach(),
                actions.detach(),
                log_probs.detach(),
                self.network.tensor(rewards).unsqueeze(1),
                self.network.tensor(1 - terminals).unsqueeze(1),
                next_z_states,
                hidden_states.detach()
            ])
            world_model_rollout.append([
                self.network.tensor(states),
                actions.detach(),
                self.network.tensor(next_states),
                hidden_states.detach()
            ])

            # Store for next step
            z_states = next_z_states
            states = next_states
            hidden_states = next_hidden_states

        # Save next values, since we will need them to compute next_obs and advantages
        actions, _, _, pending_value, _ = self.network.predict(z_states, hidden_states=hidden_states)
        rollout.append([z_states, pending_value, None, None, None, None, hidden_states])
        world_model_rollout.append([states, actions, None, hidden_states])

        # Save state/hidden_state for next iteration
        self.states = states
        self.z_states = z_states
        self.hidden_states = hidden_states.detach()

        if config.train_world_model:
            # Let's train the world model on rollouts, and also encode the observation
            # We stack rollouts into (seq_len, batch_size, ..) that way the mdn-rnn get's a sequence when processing a rollout
            states, actions, next_states, hidden_states = map(lambda x: torch.stack(x, dim=1), zip(*world_model_rollout[:-1]))

            if config.curiosity:
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

                # Get initial loss in deterministic mode
                if config.curiosity:
                    initial_loss = self.network.train_world_model(sampled_states, sampled_actions, sampled_next_states, sampled_hidden_states, train=False)['loss']

                # Train
                info_world_model = self.network.train_world_model(sampled_states, sampled_actions, sampled_next_states, sampled_hidden_states, train=True)
                self.last_info_world_model = info_world_model
                # Log
                for key, val in info_world_model.items():
                    config.logger.scalar_summary('wm/' + key, val.mean())

                if config.curiosity:
                    # Train world model here and update values with curiosity reward, before we calculate advantages
                    # For an intro to the idea see :https://arxiv.org/abs/1705.05363 . But my approach is to make the reward
                    # the reduction of loss from a state, similar to mentioned here http://people.idsia.ch/~juergen/creativity.html
                    loss = self.network.train_world_model(sampled_states, sampled_actions, sampled_next_states, sampled_hidden_states, train=False)['loss']
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

            if config.curiosity:
                # Log curiosity
                extrinsic_after = torch.cat([roll[4] for roll in rollout[:-1]])
                if config.curiosity_only:
                    instrinsic = extrinsic_after
                else:
                    instrinsic = extrinsic_after - extrinsic

                config.logger.scalar_summary('curiosity/reward_extrinsic', extrinsic.mean())
                config.logger.scalar_summary('curiosity/reward_intrinsic', instrinsic.mean())
                if (self.total_steps // steps) % config.iteration_log_interval == 0:
                    config.logger.info('rollout extrinsic, intrinsic reward [min/mean/max]: {:2.4f}/{:2.4f}/{:2.4f}, {:2.4f}/{:2.4f}/{:2.4f}'.format(
                        extrinsic.min().cpu().item(),
                        extrinsic.mean().cpu().item(),
                        extrinsic.max().cpu().item(),
                        instrinsic.min().cpu().item(),
                        instrinsic.mean().cpu().item(),
                        instrinsic.max().cpu().item()
                    ))

            del states, actions, next_states, hidden_states

        # Calculate advantages again now that we have changed the rewards
        z_states, actions, log_probs_old, returns, advantages, next_z_states, hidden_states, values_old = self.process_rollout(rollout, pending_value)

        # Now train PPO
        batcher = Batcher(z_states.size(0) // config.num_mini_batches, [np.arange(z_states.size(0))])
        for _ in range(config.optimization_epochs):
            batcher.shuffle()
            while not batcher.end():
                # Sample rollout
                batch_indices = batcher.next_batch()[0]
                batch_indices = self.network.tensor(batch_indices).long()
                sampled_z_states = z_states[batch_indices]
                sampled_actions = actions[batch_indices]
                sampled_log_probs_old = log_probs_old[batch_indices]
                sampled_returns = returns[batch_indices]
                sampled_old_values = values_old[batch_indices]
                sampled_advantages = advantages[batch_indices]
                sampled_next_z_states = next_z_states[batch_indices]
                sampled_hidden_states = hidden_states[batch_indices]

                # Train controller
                _, log_probs, entropy, values, _ = self.network.predict(sampled_z_states, sampled_actions, sampled_next_z_states, sampled_hidden_states)
                ratio = (log_probs - sampled_log_probs_old).exp()
                obj = ratio * sampled_advantages
                obj_clipped = ratio.clamp(1.0 - self.config.ppo_ratio_clip,
                                          1.0 + self.config.ppo_ratio_clip) * sampled_advantages
                policy_loss = -torch.min(obj, obj_clipped).mean()
                entropy_loss = - config.entropy_weight * entropy.mean()

                # # Do the values with importance sampling, like in https://github.com/openai/baselines/blob/24fe3d65/baselines/ppo2/ppo2.py#L149
                values_clipped = sampled_old_values + (values - sampled_old_values).clamp(- config.ppo_ratio_clip, config.ppo_ratio_clip)
                vf_losses1 = (values - sampled_returns).pow(2)
                vf_losses2 = (values_clipped - sampled_returns).pow(2)
                value_loss = 0.5 * torch.max(vf_losses1, vf_losses2).mean()
                #
                # value_loss = 0.5 * (sampled_returns - values).pow(2).mean()
                loss = (policy_loss + value_loss * config.value_weight + entropy_loss).mean()

                # Backward
                self.opt.zero_grad()
                loss.backward()
                grad_norm = nn.utils.clip_grad_norm_(self.network.parameters(), config.gradient_clip)
                self.opt.step()

                # Log values from training minibatch
                config.logger.scalar_summary('ppo/loss_value', value_loss)
                config.logger.scalar_summary('ppo/loss_policy', policy_loss)
                config.logger.scalar_summary('ppo/loss_entropy', entropy_loss)  # Lets us tune/check config.entropy_weight
                config.logger.scalar_summary('ppo/grad_norm', grad_norm)  # Lets us check config.gradient_clip
                # config.logger.scalar_summary('ratio_max', ratio.max())
                # config.logger.scalar_summary('ratio_min', ratio.min())
                config.logger.scalar_summary('ppo/ratio/abs_mean', (ratio - 1).abs().mean())
                config.logger.scalar_summary('ppo/ratio/mean', (ratio - 1).mean())
                config.logger.scalar_summary('ppo/ratio/abs_max', (ratio - 1).abs().max())
                # config.logger.scalar_summary('ratio', ratio.mean())
                # Lets us check how important/relevant the training experience is. Lets us tune rollout size, optimizer_epochs, etc

        del z_states, actions, log_probs_old, returns, advantages, next_z_states, hidden_states

        # Log values from rollout
        _, values, actions, log_probs, rewards, terminals, _, _ = map(lambda x: torch.stack(x, dim=1), zip(*rollout[:-1]))
        if (self.total_steps // steps) % config.iteration_log_interval == 0:
            config.logger.info('action counts: {}'.format(
                dict(collections.Counter(actions.view(-1).cpu().numpy().tolist()))
            ))
        self.total_steps += steps
        config.logger.histo_summary('rollout/actions', actions, self.total_steps)  # How often it takes each action
        config.logger.scalar_summary('rollout/terminals', terminals.mean(), self.total_steps)  # Lets us know how often it's dying
        config.logger.scalar_summary('rollout/log_probs', log_probs.mean(), self.total_steps)  # How certain it is
        config.logger.scalar_summary('rollout/rewards', rewards.mean(), self.total_steps)  # Raw reward

        config.logger.writer.file_writer.flush()
