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

    def process_rollout(self, rollout, pending_value):
        config = self.config
        processed_rollout = [None] * (len(rollout) - 1)
        advantages = self.network.tensor(np.zeros((config.num_workers, 1)))
        returns = pending_value.detach()
        for i in reversed(range(len(rollout) - 1)):
            states, value, actions, log_probs, rewards, terminals, next_states, hidden_states = rollout[i]
            # terminals = self.network.tensor(terminals).unsqueeze(1)
            # rewards = self.network.tensor(rewards).unsqueeze(1)
            # actions = self.network.tensor(actions)
            # states = self.network.tensor(states)
            # next_states = self.network.tensor(next_states)
            # hidden_states = self.network.tensor(hidden_states)

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
        hidden_states = None

        for _ in range(config.rollout_length):
            with torch.no_grad():
                actions, log_probs, _, values, hidden_states, loss_reduction = self.network.predict(states, hidden_state=hidden_states)
            next_states, rewards, terminals, _ = self.task.step(actions.cpu().detach().numpy())
            self.episode_rewards += rewards
            rewards = config.reward_normalizer(rewards)
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

        self.states = states
        _, _, _, pending_value, _, _ = self.network.predict(states)
        rollout.append([states, pending_value, None, None, None, None, None])

        # Train world model here and update values with curiosity reward, before we calculate advantages
        # For an intro to the idea see :https://arxiv.org/abs/1705.05363 . But my approach is to make the reward
        # the reduction of loss from a state, similar to mentioned here http://people.idsia.ch/~juergen/creativity.html
        states, actions, log_probs_old, returns, advantages, next_states, hidden_states, inds = self.process_rollout(rollout, pending_value)
        batcher = Batcher(states.size(0) // config.num_mini_batches, [np.arange(states.size(0))])
        extrinsic = torch.cat([roll[4] for roll in rollout[:-1]]).mean()
        # initial_loss = torch.zeros(states.size(0))
        batcher.shuffle()
        while not batcher.end():
            batch_indices = batcher.next_batch()[0]
            batch_indices = self.network.tensor(batch_indices).long()

            sampled_states = states[batch_indices]
            sampled_actions = actions[batch_indices]
            sampled_next_states = next_states[batch_indices]
            sampled_hidden_states = hidden_states[batch_indices]

            _, log_probs, entropy_loss, values, hidden_state, loss_reduction = self.network.predict(sampled_states, sampled_actions, sampled_next_states, sampled_hidden_states)

            # Update reward in the rollout, using reducing in loss: curiosity
            for k, (i, j) in enumerate(inds[batch_indices]):
                if config.curiosity_only:
                    rollout[i][4][j] = (loss_reduction[k] - config.curiosity_baseline) * config.curiosity_weight
                else:
                    rollout[i][4][j] += (loss_reduction[k] - config.curiosity_baseline) * config.curiosity_weight

        # Log
        extrinsic_after = torch.cat([roll[4] for roll in rollout[:-1]]).mean()
        if config.curiosity_only:
            instrinsic = extrinsic_after
        else:
            instrinsic = extrinsic_after - extrinsic
        config.logger.scalar_summary('reward_extrinsic', extrinsic.mean())
        config.logger.scalar_summary('reward_intrinsic', instrinsic.mean())
        print('rollout extrinsic vs intrinsic reward {:2.4f} {:2.4f}'.format(extrinsic.mean().cpu().item(), instrinsic.mean().cpu().item() * config.curiosity_weight))

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

                _, log_probs, entropy_loss, values, hidden_state, _ = self.network.predict(sampled_states, sampled_actions, sampled_next_states, sampled_hidden_states, model_train=False)
                # sampled_returns += loss_reduction  # curiosity reward, lets update just for training the value function?
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
