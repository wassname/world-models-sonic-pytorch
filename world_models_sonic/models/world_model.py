import torch
import torch.nn
from torch.nn import functional as F


class WorldModel(torch.nn.modules.Module):
    def __init__(self, vae, mdnrnn, finv, optimizer=None, scheduler=None, logger=None, lambda_vae_kld=1 / 4., C=0, lambda_finv=1, lambda_vae=1 / 20.):
        """Predicts next latent state."""
        super().__init__()
        self.vae = vae
        self.mdnrnn = mdnrnn
        self.finv = finv

        self.optimizer = optimizer
        self.scheduler = scheduler
        self.lambda_vae_kld = lambda_vae_kld
        self.C = C
        self.lambda_finv = lambda_finv
        self.lambda_vae = lambda_vae
        self.logger = logger

    def forward(self, x, action=None, hidden_state=None, X_next=None):
        _, mu_vae, logvar_vae = self.vae.forward(x)
        z = self.vae.sample(mu_vae, logvar_vae)
        pi, mu, sigma, hidden_state = self.mdnrnn.forward(z[:, None], action, hidden_state=hidden_state)
        z_next_pred = self.mdnrnn.sample(pi, mu, sigma)
        return z_next_pred.squeeze(1), z, hidden_state

    def forward_train(self, X, actions=None, X_next=None, hidden_state=None):
        seq_len = 1
        batch_size = X.size(0)
        cuda = next(iter(self.parameters())).is_cuda

        # VAE forward
        Y, mu_vae, logvar = self.vae.forward(X)
        Y_next, mu_vae_next, logvar_next = self.vae.forward(X_next)

        loss_recon, loss_KLD = self.vae.loss(Y, X, mu_vae, logvar)
        loss_vae = loss_recon + self.lambda_vae_kld * torch.abs(loss_KLD - self.C)
        loss_vae = loss_vae.mean()  # mean along the batches

        # MDNRNN Forward
        z_obs = self.vae.sample(mu_vae, logvar)
        z_obs = z_obs.view(batch_size, seq_len, -1)
        z_obs_next = self.vae.sample(mu_vae_next, logvar_next).detach()  # The RNN can't change the future
        z_obs_next = z_obs_next.view(batch_size, seq_len, -1)
        actions = actions.view(batch_size, seq_len).float()
        if cuda:
            z_obs = z_obs.cuda()
            z_obs_next = z_obs_next.cuda()
            actions = actions.cuda()
        pi, mu, sigma, hidden_state = self.mdnrnn.forward(z_obs, actions, hidden_state=hidden_state)

        # We are evaluating how the output distribution for the next step
        # matches the real next step. So we have to discard the last step in the
        # sequence which has no next step.
        # z_true_next = z_obs[:, 1:]
        loss_mdn = self.mdnrnn.rnn_loss(z_obs_next, pi, mu, sigma).mean()

        # Finv forward
        z_next_pred = self.mdnrnn.sample(pi, mu, sigma)
        action_pred = self.finv(z_obs, z_next_pred).float()
        actions_hot = torch.eye(self.mdnrnn.action_dim)[actions.long()].cuda()
        loss_inv = F.binary_cross_entropy_with_logits(action_pred, actions_hot)
        loss_inv = loss_inv.mean()

        loss = self.lambda_vae * loss_vae + loss_mdn + self.lambda_finv * loss_inv

        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        # self.scheduler.step()

        # Record
        if self.logger:
            self.logger.scalar_summary('loss_vae', loss_vae.mean())
            self.logger.scalar_summary('loss_recon', loss_recon.mean())
            self.logger.scalar_summary('loss_KLD', loss_KLD.mean())
            self.logger.scalar_summary('loss_mdn', loss_mdn.mean())
            self.logger.scalar_summary('loss_inv', loss_inv.mean())
            self.logger.scalar_summary('loss_world_model', loss.mean())

        return z_next_pred.squeeze(1), z_obs.squeeze(1), hidden_state, {}
