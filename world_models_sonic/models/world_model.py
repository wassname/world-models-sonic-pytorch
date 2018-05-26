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

    def forward(self, x, action=None, hidden_state=None):
        _, mu_vae, logvar_vae = self.vae.forward(x)
        z = self.vae.sample(mu_vae, logvar_vae)
        pi, mu, sigma, hidden_state = self.mdnrnn.forward(z[:, None], action, hidden_state=hidden_state)
        z_next_pred = self.mdnrnn.sample(pi, mu, sigma)
        return z_next_pred.squeeze(1), z, hidden_state

    def forward_train(self, X, actions=None, X_next=None, hidden_state=None):
        seq_len = 1
        batch_size = X.size(0)
        cuda = next(iter(self.parameters())).is_cuda

        # bit of a HACK I should record action and next state in ppo then pass in
        if actions is None:
            actions = torch.zeros(batch_size, seq_len)
        if cuda:
            actions = actions.cuda()

        # Seperate controller and world model by gradient block
        X = X.detach()
        actions = actions.detach()
        if hidden_state:
            hidden_state = [h.detach() for h in hidden_state]

        # VAE forward
        Y, mu_vae, logvar = self.vae.forward(X)

        loss_recon, loss_KLD = self.vae.loss(Y, X, mu_vae, logvar)
        loss_vae = loss_recon + self.lambda_vae_kld * torch.abs(loss_KLD - self.C)
        loss_vae = loss_vae.mean()  # mean along the batches

        # MDNRNN Forward
        z_obs = self.vae.sample(mu_vae, logvar)
        z_obs = z_obs.view(batch_size, seq_len, -1)
        actions = actions.view(batch_size, seq_len).float()
        if cuda:
            z_obs = z_obs.cuda()
        pi, mu, sigma, hidden_state = self.mdnrnn.forward(z_obs, actions, hidden_state=hidden_state)

        # TODO train mdnn we could remember the last one, and do loss based on that
        # but only if it's the same size. This is because we get passed each single obs during rollout
        # then we get batches of random samples which are unordered?

        # # We are evaluating how the output distribution for the next step
        # # matches the real next step. So we have to discard the last step in the
        # # sequence which has no next step.
        # z_true_next = z_obs[:, 1:]
        # loss_mdn = self.mdnrnn.rnn_loss(z_true_next, pi[:, :-1], mu[:, :-1], sigma[:, :-1]).mean()
        #
        # # Finv forward
        z_next_pred = self.mdnrnn.sample(pi, mu, sigma)
        # action_pred = self.finv(z_obs[:, 1:], z_next_pred[:, :-1]).float()
        # actions_hot = torch.eye(self.mdnrnn.action_dim)[actions.long()].cuda()
        # loss_inv = F.binary_cross_entropy_with_logits(action_pred, actions_hot[:, 1:])
        # loss_inv = loss_inv.mean()

        loss = self.lambda_vae * loss_vae  # loss_mdn + self.lambda_finv * loss_inv +

        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        # self.scheduler.step()

        # Record
        if self.logger:
            self.logger.scalar_summary('loss_vae', loss_vae.mean())
            self.logger.scalar_summary('loss_recon', loss_recon.mean())
            self.logger.scalar_summary('loss_KLD', loss_KLD.mean())
            # self.logger.scalar_summary('loss_mdn', loss_mdn.mean())
            # self.logger.scalar_summary('loss_inv', loss_inv.mean())
            self.logger.scalar_summary('loss_world_model', loss.mean())

        hidden_state = [h.detach() for h in hidden_state]
        return z_next_pred.squeeze(1).detach(), z_obs.squeeze(1).detach(), hidden_state, {}
