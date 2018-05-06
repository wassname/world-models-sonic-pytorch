import torch
from torch.nn import functional as F
from torch.autograd import Variable
import torch.nn as nn


class ConvBlock(torch.nn.Module):
    def __init__(self, inpt_kernel, output_kernel, kernel_size=4, stride=2):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=inpt_kernel, out_channels=output_kernel, kernel_size=kernel_size, stride=stride)
        self.bn = nn.BatchNorm2d(output_kernel)
        self.act = nn.ReLU(inplace=True)

        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_uniform(self.conv.weight, gain=gain)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x


class DeconvBlock(torch.nn.Module):
    def __init__(self, inpt_kernel, output_kernel, kernel_size=4, stride=2):
        super().__init__()
        self.deconv = nn.ConvTranspose2d(in_channels=inpt_kernel, out_channels=output_kernel, kernel_size=kernel_size, stride=stride)
        self.bn = nn.BatchNorm2d(output_kernel)
        self.act = nn.ReLU(inplace=True)

        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_uniform(self.deconv.weight, gain=gain)

    def forward(self, x):
        x = self.deconv(x)
        x = self.bn(x)
        x = self.act(x)
        return x


# Reconstruction + KL divergence losses summed over all elements and batch
# https://github.com/pytorch/examples/blob/master/vae/main.py
def loss_function(recon_x, x, mu, logvar):
    n, c, h, w = recon_x.size()
    recon_x = recon_x.view(n, -1)
    x = x.view(n, -1)
    # L2 distance
    l2_dist = torch.mean(torch.sqrt(torch.sum(torch.pow(recon_x - x, 2), 1)))
    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return l2_dist + KLD


n = 2  # multiplier to multiply input dims

class VAE(nn.Module):
    def __init__(self, latent_vector_dim=32 * n):
        super(VAE, self).__init__()
        # encoder part

        self.conv1 = ConvBlock(3, 32 * n, 4 * n, 2 * n)
        self.conv2 = ConvBlock(32 * n, 64 * n, 4, 2)
        self.conv3 = ConvBlock(64 * n, 128 * n, 4, 2)
        self.conv4 = ConvBlock(128 * n, 256 * n, 4, 2)

        self.mu = nn.Linear(1024 * n, latent_vector_dim)
        self.logvar = nn.Linear(1024 * n, latent_vector_dim)

        self.z = nn.Linear(latent_vector_dim, 1024 * n)

        # decoder part
        self.deconv1 = DeconvBlock(1024 * n, 128 * n, 5, 2)
        self.deconv2 = DeconvBlock(128 * n, 64 * n, 5, 2)
        self.deconv3 = DeconvBlock(64 * n, 32 * n, 6, 2)
        self.deconv4 = DeconvBlock(32 * n, 3, 6 * n, 2 * n)

        self.sigmoid = nn.Sigmoid()

    def encode(self, x):
        """
            Returns mean and log variance
        """
        x = self.conv4(self.conv3(self.conv2(self.conv1(x))))
        x = x.view(x.size()[0], -1)
        return self.mu(x), self.logvar(x)

    def sample(self, mu, logvar):
        if self.training:
            std = logvar.exp()
            std = std * Variable(std.data.new(std.size()).normal_())
            return mu + std
        else:
            return mu

    def decode(self, z):
        z = self.z(z)
        n, d = z.size()
        z = z.view(n, d, 1, 1)
        reconstruction = self.deconv4(self.deconv3(self.deconv2(self.deconv1(z))))
        reconstruction = self.sigmoid(reconstruction)
        return reconstruction

    def forward(self, x):
        """
            Returns reconstructed image, mean, and log variance
        """
        mu, logvar = self.encode(x)
        z = self.sample(mu, logvar)
        x = self.decode(z)
        return x, mu, logvar


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

    # cv2.imshow('original', img)
    # cv2.imshow('reconstructed', x)
    # cv2.waitKey()
