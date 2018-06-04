import torch
from torch.nn import functional as F
from torch.autograd import Variable
import torch.nn as nn
import math
import numpy as np

from ..config import eps

class ConvBlock4(torch.nn.Module):
    def __init__(self, inpt_kernel, output_kernel, kernel_size=4, stride=1, padding=0):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=inpt_kernel, out_channels=output_kernel, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(output_kernel)
        self.act = nn.LeakyReLU(inplace=True)
#         self.drp = nn.Dropout2d(0.3)

        gain = nn.init.calculate_gain('leaky_relu')
        nn.init.xavier_uniform_(self.conv.weight, gain=gain)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
#         x = self.drp(x)
        x = self.act(x)
        return x


class DeconvBlock4(torch.nn.Module):
    def __init__(self, inpt_kernel, output_kernel, kernel_size=4, stride=1, padding=0):
        super().__init__()
        self.deconv = nn.ConvTranspose2d(in_channels=inpt_kernel, out_channels=output_kernel, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(output_kernel)
        self.act = nn.LeakyReLU(inplace=True)
#         self.drp = nn.Dropout2d(0.3)

        gain = nn.init.calculate_gain('leaky_relu')
        nn.init.xavier_uniform_(self.deconv.weight, gain=gain)

    def forward(self, x):
        x = self.deconv(x)
        x = self.bn(x)
#         x = self.drp(x)
        x = self.act(x)
        return x


class VAE5(nn.Module):
    """
    VAE. Vector Quantised Variational Auto-Encoder.

    Refs:
    - https://github.com/nakosung/VQ-VAE/blob/master/model.py
    - https://github.com/JunhongXu/world-models-pytorch/blob/master/vae.py
    """

    def __init__(self, image_size=64, z_dim=32, conv_dim=64, code_dim=16, k_dim=256, channels=3):
        """
        Args:
        - image_size (int) height and weight of image
        - conv_dim (int) the amound of output channels in the first conv layer (all others are multiples)
        - z_dim (int) the channels in the encoded output
        - code_dim (int) the height and width in the encoded output
        - k_dim (int) dimensions of the latent vector
        """
        super().__init__()

        self.k_dim = k_dim
        self.z_dim = z_dim
        self.code_dim = code_dim

        hidden_size = z_dim * code_dim * code_dim
        latent_vector_dim = k_dim
        self.logvar = nn.Linear(hidden_size, latent_vector_dim)
        self.mu = nn.Linear(hidden_size, latent_vector_dim)
        self.z = nn.Linear(latent_vector_dim, hidden_size)

        nn.init.xavier_uniform_(self.logvar.weight)
        nn.init.xavier_uniform_(self.mu.weight)
        nn.init.xavier_uniform_(self.z.weight)

        # Encoder (increasing #filter linearly)
        layers = []
        layers.append(ConvBlock4(channels, conv_dim, kernel_size=3, padding=1))

        repeat_num = int(math.log2(image_size / code_dim))
        curr_dim = conv_dim
        for i in range(repeat_num):
            layers.append(ConvBlock4(curr_dim, conv_dim * (i + 2), kernel_size=4, stride=2, padding=1))
            curr_dim = conv_dim * (i + 2)

        # Now we have (code_dim,code_dim,curr_dim)
        layers.append(nn.Conv2d(curr_dim, z_dim, kernel_size=1))

        # (code_dim,code_dim,z_dim)
        self.encoder = nn.Sequential(*layers)

        # Decoder (320 - 256 - 192 - 128 - 64)
        layers = []

        layers.append(DeconvBlock4(z_dim, curr_dim, kernel_size=1))

        for i in reversed(range(repeat_num)):
            layers.append(DeconvBlock4(curr_dim, conv_dim * (i + 1), kernel_size=4, stride=2, padding=1))
            curr_dim = conv_dim * (i + 1)

        layers.append(nn.Conv2d(curr_dim, channels, kernel_size=3, padding=1))
        self.decoder = nn.Sequential(*layers)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """Returns reconstructed image, mean, and log variance."""
        mu, logvar = self.encode(x)
        z = self.sample(mu, logvar)
        x = self.decode(z)
        return x, mu, logvar

    def encode(self, x):
        """Returns mean and log variance, which describe the distributions of Z"""
        x = self.encoder(x)
        x = x.view(x.size()[0], -1)
        return self.mu(x), self.logvar(x).clamp(np.log(eps),-np.log(eps))

    def decode(self, z):
        """Reconstruct image X using z sampled from Z."""
        z = self.z(z)
        n, d = z.size()
        z = z.view(n, -1, self.code_dim, self.code_dim)
        reconstruction = self.decoder(z)
        reconstruction = self.sigmoid(reconstruction)
        return reconstruction

    def sample(self, mu, logvar):
        """Sample z from Z."""
        if self.training:
            std = logvar.exp()
            std = std * Variable(std.data.new(std.size()).normal_())
            return mu + std
        else:
            return mu

    def loss(self, *args, **kwargs):
        return loss_function_vae(*args, **kwargs)


def loss_function_vae(recon_x, x, mu, logvar):
    # Reconstruction + KL divergence losses summed over all elements and batch
    # https://github.com/pytorch/examples/blob/master/vae/main.py
    n, c, h, w = recon_x.size()

    recon_x = recon_x.view(n, -1)
    x = x.view(n, -1)

    # L2 distance
    loss_recon = torch.sum(torch.pow(recon_x - x, 2), 1)

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    loss_KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), 1)
    return loss_recon, loss_KLD


if __name__ == '__main__':
    import numpy as np
    from matplotlib import pyplot as plt

    img = np.random.randn(64, 64, 3)
    gpu_img = Variable(torch.from_numpy(img[np.newaxis].transpose(0, 3, 1, 2))).float().cuda()

    vae = VAE()
    vae.cuda()
    x, mu, logvar = vae.forward(gpu_img)
    print(x.size())
    print(loss_function(x, gpu_img, mu, logvar))
    x = x.data.cpu().numpy()[0].transpose(1, 2, 0)

    plt.imshow(img)
    plt.title('original')
    plt.show()

    plt.imshow(x)
    plt.title('reconstructed')
    plt.show()
