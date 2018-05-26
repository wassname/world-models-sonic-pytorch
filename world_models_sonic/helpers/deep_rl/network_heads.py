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
        self.z_state_dim = self.world_model.mdnrnn.z_dim + self.world_model.mdnrnn.hidden_size
        self.network = ActorCriticNet(self.z_state_dim, action_dim, phi_body, actor_body, critic_body)

        self._render = render
        self.z_shape = z_shape
        self.viewer = None
        self.max_hidden_states = 6
        self.hidden_state = None
        self.set_gpu(gpu)

    def pad_hidden_states(self, hidden_states):
        """Pad hidden states to (batch, max_hidden_states, z_dim)."""
        hidden_states = torch.cat(hidden_states, dim=0).transpose(0, 1).contiguous()
        batch_size = hidden_states.size(0)
        pad = self.max_hidden_states - hidden_states.size(1)
        padding = torch.zeros(batch_size, pad, self.world_model.mdnrnn.z_dim)
        if hidden_states.is_cuda:
            padding = padding.cuda()
        hidden_states = torch.cat([padding, hidden_states], dim=1)
        return hidden_states

    def process_obs(self, obs, action=None, next_obs=None, hidden_state=None):
        """Convert observation to latent space using world model."""
        is_rollout = next_obs is None
        cuda = next(iter(self.parameters())).is_cuda
        batch_size = obs.shape[0]
        self.img = obs
        if hidden_state is None:
            hidden_state = torch.zeros(batch_size, self.max_hidden_states, self.world_model.mdnrnn.z_dim)
            if cuda:
                hidden_state = hidden_state.cuda()

        # # We will keep it in the train phase the whole time so it samples randomly
        # # (in eval phase it just takes the mean)
        # self.world_model.train()

        # Input is (batch, height, width, channels)
        obs = self.tensor(obs).transpose(1, 3).contiguous()
        if is_rollout:
            # when we don't get an action (e.g. rollout) just use null action to make prediction
            action = torch.zeros(batch_size, 1)
            hidden_state = [h.detach() for h in hidden_state.transpose(1, 0).detach()]
            z_next, z, hidden_state = self.world_model.forward(
                obs.detach(),
                action.detach(),
                hidden_state=hidden_state
            )
            # self.hidden_state = hidden_state[-self.max_hidden_states:]
            self.z = z.data
            self.z_next = z_next.data
        else:
            # Zero the hidden state if we are in an optimization iter since they arn't ordered
            # self.hidden_state = None

            next_obs = self.tensor(next_obs).transpose(1, 3).contiguous()
            # if hidden_state is not None:
            # raise Exception('check hidden state should be list of (6,512)s')
            # hidden_state = [h.detach() for h in hidden_state.transpose(1, 0).detach()]
            z_next, z, hidden_state, info_world_model = self.world_model.forward_train(
                obs.detach(),
                action.detach(),
                next_obs.detach(),
                hidden_state.transpose(1, 0).contiguous().detach()
            )

        latest_hidden = hidden_state[-1].squeeze(0)  # squeeze so we can concat
        obs = torch.cat([z, latest_hidden], -1).detach()  # Gradient block between world model and controller
        return obs, self.pad_hidden_states(hidden_state[-self.max_hidden_states:])

    def predict(self, obs, action=None, next_obs=None, hidden_state=None):
        # In rollout (non training mode) when no next_obs is provided
        is_rollout = next_obs is None

        obs_z, hidden_state = self.process_obs(obs, action, next_obs, hidden_state)

        # Predict next action and value
        phi = self.network.phi_body(obs_z)
        phi_a = self.network.actor_body(phi)
        phi_v = self.network.critic_body(phi)
        prob = F.softmax(self.network.fc_action(phi_a), dim=-1)
        v = self.network.fc_critic(phi_v)
        dist = torch.distributions.Categorical(probs=prob)
        if action is None:
            action = dist.sample()
        if is_rollout and self._render:
                self.render()
        log_prob = dist.log_prob(action).unsqueeze(-1)

        return action, log_prob, dist.entropy().unsqueeze(-1), v, hidden_state

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
                img_z = img_z.data.cpu().numpy()
                img_z = (img_z * 255.0).astype(np.uint8)
                img_z_next = img_z_next.squeeze(0).transpose(0, 2).clamp(0, 1)
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
                img = self.img[0] * 255
                self.viewer.imshow(img.astype(np.uint8))
                self.viewer_img_z.imshow(img_z)
                self.viewer_img_z_next.imshow(img_z_next)
                self.viewer_z.imshow(z_uint8)
                self.viewer_z_next.imshow(z_next_uint8)
                return self.viewer.isopen
