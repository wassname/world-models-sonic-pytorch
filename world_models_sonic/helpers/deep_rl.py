"""
Helpers for https://github.com/ShangtongZhang/DeepRL
"""
import torch
import time
from torch import nn
from torch.nn import functional as F
import numpy as np
import pickle


import skimage.color
import cv2

from deep_rl.network.network_utils import BaseNet
from deep_rl.agent import BaseAgent, Batcher
from deep_rl.network.network_heads import ActorCriticNet
from deep_rl.component.task import BaseTask

from ..custom_envs.wrappers import RenderWrapper


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


class CategoricalWorldActorCriticNet(nn.Module, BaseNet):
    """
    A custom network head with world models and rendering.

    Putting it here instead of as an environement wrapper allows multithreading
    because it removes all pytorch operations from the rollout phase.
    """

    def __init__(self,
                 state_dim,
                 action_dim,
                 phi_body=None,
                 actor_body=None,
                 critic_body=None,
                 world_model_fn=None,
                 gpu=-1,
                 render=False
                 ):
        super().__init__()
        self.world_model = world_model_fn()
        self.z_state_dim = self.world_model.mdnrnn.z_dim + self.world_model.mdnrnn.hidden_size
        self.network = ActorCriticNet(self.z_state_dim, action_dim, phi_body, actor_body, critic_body)

        self.render = render
        self.viewer = None
        self.max_hidden_states = 6
        self.hidden_state = None
        self.set_gpu(gpu)

    def predict(self, obs, action=None):
        self.img = obs
        with torch.no_grad():
            # Input is (batch, samples...)
            obs = self.tensor(obs).transpose(1, 3)

            # Use world model to transform obs
            if self.hidden_state and (obs.size(0) != self.hidden_state[0].size(0) or obs.size(1) != self.hidden_state[0].size(1)):
                self.hidden_state = None
            z_next, z, hidden_state = self.world_model.forward(obs, hidden_state=self.hidden_state)

            hidden_state = [h.data for h in hidden_state]  # Otherwise it doesn't garbge collect
            self.hidden_state = hidden_state[-self.max_hidden_states:]
            self.z = z.data
            self.z_next = z_next.data

            latest_hidden = hidden_state[-1].squeeze(0)  # squeeze so we can concat
        obs = torch.cat([z, latest_hidden], -1)
        obs.requires_grad = True

        # Predict next action and value
        phi = self.network.phi_body(obs)
        phi_a = self.network.actor_body(phi)
        phi_v = self.network.critic_body(phi)
        prob = F.softmax(self.network.fc_action(phi_a), dim=-1)
        v = self.network.fc_critic(phi_v)
        dist = torch.distributions.Categorical(probs=prob)
        if action is None:
            action = dist.sample()
            if self.render:
                self.render()
        log_prob = dist.log_prob(action).unsqueeze(-1)

        return action, log_prob, dist.entropy().unsqueeze(-1), v

    def render(self, mode='world_model', close=False):
        if close:
            for viewer in [self.viewer_z, self.viewer_z_next, self.viewer_img_z, self.viewer_img_z_next]:
                viewer.close()
            if self.viewer:
                self.viewer.close()
            return
        if mode == 'world_model':
            # Render while showing latent vectors and decoded latent vectors
            if self.viewer is None:
                from gym.envs.classic_control.rendering import SimpleImageViewer
                import pyglet
                margin_vert = 60  # for window border
                # TODO add margin_horiz
                mult = 3

                self.viewer = SimpleImageViewer()
                self.viewer.window = pyglet.window.Window(width=160 * mult, height=128 * mult, vsync=False, resizable=True, caption='Game output')
                self.viewer.window.set_location(0 * mult, 0 * mult)

                self.viewer_img_z = SimpleImageViewer()
                self.viewer_img_z.window = pyglet.window.Window(width=128 * mult, height=128 * mult, vsync=False, resizable=True, caption='Decoded image')
                self.viewer_img_z.window.set_location(160 * mult, 0 * mult)

                self.viewer_img_z_next = SimpleImageViewer()
                self.viewer_img_z_next.window = pyglet.window.Window(width=128 * mult, height=128 * mult,
                                                                     vsync=False, resizable=True, caption='Decoded predicted image')
                self.viewer_img_z_next.window.set_location((160 + 128) * mult, 0 * mult)

                self.viewer_z = SimpleImageViewer()
                self.viewer_z.window = pyglet.window.Window(width=128 * mult, height=128 * mult, vsync=False, resizable=True, caption='latent vector')
                self.viewer_z.window.set_location(160 * mult, 128 * mult + margin_vert)

                self.viewer_z_next = SimpleImageViewer()
                self.viewer_z_next.window = pyglet.window.Window(width=128 * mult, height=128 * mult, vsync=False,
                                                                 resizable=True, caption='latent predicted vector')
                self.viewer_z_next.window.set_location((160 + 128) * mult, 128 * mult + margin_vert)

            # Decode latent vector for display
            with torch.no_grad():

                # to pytorch
                zv = self.z
                zv_next = self.z_next
                if self.cuda:
                    zv = zv.cuda()
                    zv_next = zv_next.cuda()

                # Decode
                img_z = self.world_model.vae.decode(zv)
                img_z_next = self.world_model.vae.decode(zv_next)

                # to numpy images
                img_z = img_z.squeeze(0).transpose(0, 2)
                img_z = img_z.data.cpu().numpy()
                img_z = (img_z * 255.0).astype(np.uint8)
                img_z_next = img_z_next.squeeze(0).transpose(0, 2).clamp(0, 1)
                img_z_next = img_z_next.data.cpu().numpy()
                img_z_next = (img_z_next * 255.0).astype(np.uint8)

                z_uint8 = ((self.z[0].data.cpu().numpy() + 0.5) * 255).astype(np.uint8).reshape((16, 16))
                z_uint8 = skimage.color.gray2rgb(z_uint8)  # Make it rgb to avoid problems with pyglet
                z_uint8 = cv2.resize(z_uint8, dsize=(256, 256), interpolation=cv2.INTER_NEAREST)  # Resize manually to avoid interp of pixels

                z_next_uint8 = ((self.z_next[0].data.cpu().numpy() + 0.5) * 255).astype(np.uint8).reshape((16, 16))
                z_next_uint8 = skimage.color.gray2rgb(z_next_uint8)
                z_next_uint8 = cv2.resize(z_next_uint8, dsize=(256, 256), interpolation=cv2.INTER_NEAREST)

                # Display
                img = self.img[0] * 255
                self.viewer.imshow(img.astype(np.uint8))
                self.viewer_img_z.imshow(img_z)
                self.viewer_img_z_next.imshow(img_z_next)
                self.viewer_z.imshow(z_uint8)
                self.viewer_z_next.imshow(z_next_uint8)
                return self.viewer.isopen
