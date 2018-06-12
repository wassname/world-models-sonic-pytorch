import torch
import torch.nn
from torch.nn import functional as F


class WorldModel(torch.nn.modules.Module):
    def __init__(self, vae, mdnrnn, finv, optimizer=None, scheduler=None, logger=None, lambda_vae_kld=1 / 4., lambda_finv=0.01, lambda_vae=1, lambda_loss=1000):
        """Predicts next latent state."""
        super().__init__()
        self.vae = vae
        self.mdnrnn = mdnrnn
        self.finv = finv

        self.optimizer = optimizer
        self.scheduler = scheduler
        self.lambda_vae_kld = lambda_vae_kld
        self.lambda_finv = lambda_finv
        self.lambda_vae = lambda_vae
        self.lambda_loss = lambda_loss
        self.logger = logger

        self.last_loss_vae = 0
        self.last_loss_KLD = 0
        self.last_loss_recon = 0
        self.last_loss_mdn = 0
        self.last_loss_inv = 0

    def forward_train(self, X, actions=None, X_next=None, hidden_state=None, test=False):
        batch_size = X.size(0)
        seq_len = X.size(1)
        cuda = next(iter(self.parameters())).is_cuda

        # VAE forward
        X_flat = X.view((batch_size * seq_len, *X.size()[2:]))
        X_next_flat = X_next.view((batch_size * seq_len, *X.size()[2:]))
        Y, mu_vae, logvar = self.vae.forward(X_flat)
        Y_next, mu_vae_next, logvar_next = self.vae.forward(X_next_flat)

        loss_recon, loss_KLD = self.vae.loss(Y, X_flat, mu_vae, logvar)

        # MDNRNN Forward
        z_obs = self.vae.sample(mu_vae, logvar)
        z_obs = z_obs.view(batch_size, seq_len, -1)
        z_obs_next = self.vae.sample(mu_vae_next, logvar_next)  # .detach()  # The RNN can't change the future?
        z_obs_next = z_obs_next.view(batch_size, seq_len, -1)
        actions = actions.view(batch_size, seq_len).float()
        if cuda:
            z_obs = z_obs.cuda()
            z_obs_next = z_obs_next.cuda()
            actions = actions.cuda()
        logpi, mu, logsigma, hidden_state = self.mdnrnn.forward(z_obs, actions, hidden_state=hidden_state)

        # We are evaluating how the output distribution for the next step
        # matches the real next step. So we have to discard the last step in the
        # sequence which has no next step.
        loss_mdn = self.mdnrnn.rnn_loss(z_obs_next, logpi, mu, logsigma).view((-1))

        # Finv forward
        z_next_pred = self.mdnrnn.sample(logpi, mu, logsigma)
        action_pred = self.finv(z_obs, z_next_pred).float()
        loss_inv = F.nll_loss(
            action_pred.view(batch_size * seq_len, self.mdnrnn.action_dim),
            actions.view(-1,).long(),
            reduce=False
        )

        # To reduce the need for hyperparameters which balance the losses we will
        # normalise for number of pixels, action_dim, z_dim etc
        pixels = X.size(1) * X.size(2) * X.size(3)
        loss_recon = self.lambda_loss * loss_recon / pixels
        loss_KLD = self.lambda_loss * loss_KLD / pixels
        loss_vae = loss_recon + self.lambda_vae_kld * torch.abs(loss_KLD)
        loss_mdn = self.lambda_loss * loss_mdn / self.mdnrnn.z_dim
        loss_inv = self.lambda_loss * loss_inv / self.mdnrnn.action_dim
        loss = self.lambda_vae * loss_vae + loss_mdn + self.lambda_finv * loss_inv

        # TODO ideally should pass the losses back to the agent and do logging and backprop there
        if not test:
            loss.mean().backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            # self.scheduler.step()

            self.last_loss_vae = loss_vae.mean().data.cpu().item()
            self.last_loss_KLD = loss_KLD.mean().data.cpu().item()
            self.last_loss_recon = loss_recon.mean().data.cpu().item()
            self.last_loss_mdn = loss_mdn.mean().data.cpu().item()
            self.last_loss_inv = loss_inv.mean().data.cpu().item()

            # Record
            if self.logger:
                self.logger.scalar_summary('loss_vae', loss_vae.mean())
                self.logger.scalar_summary('loss_recon', loss_recon.mean())
                self.logger.scalar_summary('loss_KLD', loss_KLD.mean())
                self.logger.scalar_summary('loss_mdn', loss_mdn.mean())
                self.logger.scalar_summary('loss_inv', loss_inv.mean())
                self.logger.scalar_summary('loss_world_model', loss.mean())

        return z_next_pred.squeeze(1), z_obs.squeeze(1), hidden_state, {'loss': loss}
