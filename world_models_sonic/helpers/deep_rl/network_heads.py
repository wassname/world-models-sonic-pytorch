#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################


import skimage.color
import cv2
import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from deep_rl.network.network_utils import BaseNet
from deep_rl.network.network_heads import ActorCriticNet


class CategoricalWorldActorCriticNet(nn.Module, BaseNet):
    """
    A custom network head with world models and rendering.

    Putting it here instead of as an environement wrapper allows multithreading
    because it removes all pytorch operations from the rollout phase.

    Modified from CategoricalActorCriticNet https://github.com/ShangtongZhang/DeepRL/blob/master/deep_rl/network/network_heads.py
    """

    def __init__(self,
                 state_dim,
                 action_dim,
                 phi_body=None,
                 actor_body=None,
                 critic_body=None,
                 world_model_fn=None,
                 gpu=-1,
                 render=False,
                 z_shape=(16, 16),
                 logger=None
                 ):
        super().__init__()
        self.world_model = world_model_fn()
        self.z_state_dim = self.world_model.mdnrnn.z_dim + self.world_model.mdnrnn.hidden_size + self.world_model.mdnrnn.action_dim
        self.network = ActorCriticNet(self.z_state_dim, action_dim, phi_body, actor_body, critic_body)

        self._render = render
        self.z_shape = z_shape
        self.viewer = None
        self.max_hidden_states = 2
        self.set_gpu(gpu)

    def pad_hidden_states(self, hidden_states):
        """Pad hidden states to (batch, max_hidden_states, z_dim)."""
        hidden_states = torch.cat(hidden_states, dim=0).transpose(0, 1).contiguous()
        batch_size = hidden_states.size(0)
        pad = self.max_hidden_states - hidden_states.size(1)
        if pad > 0:
            print('pad', pad)
            padding = torch.zeros(batch_size, pad, self.world_model.mdnrnn.hidden_size)
            if hidden_states.is_cuda:
                padding = padding.cuda()
                hidden_states = torch.cat([padding, hidden_states], dim=1)
        return hidden_states

    def train_world_model(self, obs, action=None, next_obs=None, hidden_states=None, train=True):
        obs = self.tensor(obs).transpose(2, 4).contiguous()
        next_obs = self.tensor(next_obs).transpose(2, 4).contiguous()

        # Only pass in the first hidden state. Expects [(1, batch, z_dim)]*2
        hidden_states = [h[None, :] for h in hidden_states[:, 0].transpose(1, 0).contiguous().detach()] if hidden_states is not None else None

        if train:
            z_next, z, hidden_states, info = self.world_model.forward_train(
                obs.detach(),
                action.detach(),
                next_obs.detach(),
                hidden_states
            )
        else:
            with torch.no_grad():
                z_next, z, hidden_states, info = self.world_model.forward_train(
                    obs.detach(),
                    action.detach(),
                    next_obs.detach(),
                    hidden_states,
                    test=True
                )
        return z_next.detach(), z.detach(), self.pad_hidden_states(hidden_states[-self.max_hidden_states:]).detach(), info['loss'].detach()

    def predict(self, obs, action=None, next_obs=None, hidden_states=None):
        # In rollout (non training mode) when no next_obs is provided
        is_rollout = next_obs is None
        self.img = obs
        batch_size = obs.shape[0]

        obs = self.tensor(obs).transpose(1, 3).contiguous()

        # Process the observation to get the latent space
        # Input is (batch, height, width, channels)
        with torch.no_grad():
            _, mu_vae, logvar_vae = self.world_model.vae.forward(obs.detach())
            z = self.world_model.vae.sample(mu_vae, logvar_vae)

        # Now use the last hidden state and the current latent space z to input to the controller
        if hidden_states is None:
            hidden_states = self.pad_hidden_states(self.world_model.mdnrnn.default_hidden_state(z[:, None])[-self.max_hidden_states:])
        hidden_states = [h[None, :].detach() for h in hidden_states.transpose(1, 0).contiguous().detach()] if hidden_states is not None else None
        latest_hidden = hidden_states[-1].squeeze(0)  # squeeze so we can concat
        # FIXME get rid of this (legacy_action_append) code, and don't append action to state
        # (since we don't know the action in advance when rolling out)
        # and then retrain model
        legacy_action_append = True
        if legacy_action_append:
            action = torch.zeros(batch_size, 1)  # Null vector untill we remove this code
            action_1hot = torch.eye(self.world_model.mdnrnn.action_dim)[action.long().squeeze()]
            action_1hot = action_1hot.view(batch_size, -1)
            cuda = next(iter(self.parameters())).is_cuda
            if cuda:
                action_1hot = action_1hot.cuda()

            obs_z = torch.cat([z, latest_hidden, action_1hot], -1).detach()  # Gradient block between world model and controller
        else:
            obs_z = torch.cat([z, latest_hidden], -1).detach()  # Gradient block between world model and controller

        # Predict next action and value
        phi = self.network.phi_body(obs_z)
        phi_a = self.network.actor_body(phi)
        phi_v = self.network.critic_body(phi)
        prob = F.softmax(self.network.fc_action(phi_a), dim=-1)
        v = self.network.fc_critic(phi_v)
        dist = torch.distributions.Categorical(probs=prob)
        action = dist.sample()
        log_prob = dist.log_prob(action).unsqueeze(-1)

        # MDNRNN forward to get next hidden state (now that we have the action)
        with torch.no_grad():
            pi, mu, sigma, hidden_states = self.world_model.mdnrnn.forward(z[:, None], action[:, None].detach(), hidden_state=hidden_states)
            z_next = self.world_model.mdnrnn.sample(pi, mu, sigma)
            z_next = z_next.squeeze(1)
            self.z = z.data
            self.z_next = z_next.data

        if is_rollout and self._render:
            self.render()

        hidden_states = self.pad_hidden_states(hidden_states[-self.max_hidden_states:]).detach()

        return action, log_prob, dist.entropy().unsqueeze(-1), v, hidden_states

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
                self.viewer_z.window = pyglet.window.Window(
                    width=self.z_shape[0] * 4 * mult, height=self.z_shape[1] * 4 * mult, vsync=False, resizable=True, caption='latent vector')
                self.viewer_z.window.set_location(160 * mult, 128 * mult + margin_vert)

                self.viewer_z_next = SimpleImageViewer()
                self.viewer_z_next.window = pyglet.window.Window(width=self.z_shape[0] * 4 * mult, height=self.z_shape[1] * 4 * mult, vsync=False,
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
                img_z = img_z.view((img_z.size(0), img_z.size(1), 3, 4)).mean(-1)
                img_z = img_z.data.cpu().numpy()
                img_z = (img_z * 255.0).astype(np.uint8)
                img_z_next = img_z_next.squeeze(0).transpose(0, 2).clamp(0, 1)
                img_z_next = img_z_next.view((img_z_next.size(0), img_z_next.size(1), 3, 4)).mean(-1)
                img_z_next = img_z_next.data.cpu().numpy()
                img_z_next = (img_z_next * 255.0).astype(np.uint8)

                z_uint8 = ((self.z[0].data.cpu().numpy() + 0.5) * 255).astype(np.uint8)
                z_uint8 = z_uint8.reshape(self.z_shape)
                z_uint8 = skimage.color.gray2rgb(z_uint8)  # Make it rgb to avoid problems with pyglet
                z_uint8 = cv2.resize(z_uint8, dsize=(self.z_shape[0] * 4, self.z_shape[1] * 4),
                                     interpolation=cv2.INTER_NEAREST)  # Resize manually to avoid interp of pixels

                z_next_uint8 = ((self.z_next[0].data.cpu().numpy() + 0.5) * 255).astype(np.uint8)
                z_next_uint8 = z_next_uint8.reshape(self.z_shape)
                z_next_uint8 = skimage.color.gray2rgb(z_next_uint8)
                z_next_uint8 = cv2.resize(z_next_uint8, dsize=(self.z_shape[0] * 4, self.z_shape[1] * 4), interpolation=cv2.INTER_NEAREST)

                # Display
                img = self.img[0, :, :, -3:] * 255
                self.viewer.imshow(img.astype(np.uint8))
                self.viewer_img_z.imshow(img_z)
                self.viewer_img_z_next.imshow(img_z_next)
                self.viewer_z.imshow(z_uint8)
                self.viewer_z_next.imshow(z_next_uint8)
                return self.viewer.isopen
