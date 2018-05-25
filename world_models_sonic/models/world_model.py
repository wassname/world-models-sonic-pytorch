import torch
import torch.nn


class WorldModel(torch.nn.modules.Module):
    def __init__(self, vae, mdnrnn, finv):
        """Predicts next latent state."""
        super().__init__()
        self.vae = vae
        self.mdnrnn = mdnrnn
        self.finv = finv

    def forward(self, x, action=None, hidden_state=None):
        _, mu_vae, logvar_vae = self.vae.forward(x)
        z = self.vae.sample(mu_vae, logvar_vae)
        pi, mu, sigma, hidden_state = self.mdnrnn.forward(z[:, None], action, hidden_state=hidden_state)
        z_next_pred = self.mdnrnn.sample(pi, mu, sigma)
        return z_next_pred.squeeze(1), z, hidden_state
