import torch
from torch.nn import functional as F
from torch.autograd import Variable
import torch.nn as nn


def make_conv_relu(inpt_kernel, output_kernel, kernel_size=4):
    return nn.Sequential(
        nn.Conv2d(in_channels=inpt_kernel, out_channels=output_kernel, kernel_size=kernel_size, stride=2),
        nn.ReLU(inplace=True)
    )


def make_deconv_relu(inpt_kernel, output_kernel, kernel_size, use_activation=True):
    return nn.Sequential(
        nn.ConvTranspose2d(in_channels=inpt_kernel, out_channels=output_kernel, kernel_size=kernel_size, stride=2),
        nn.ReLU(inplace=True)
    ) if use_activation \
        else nn.ConvTranspose2d(in_channels=inpt_kernel, out_channels=output_kernel, kernel_size=kernel_size, stride=2)


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


class VAE(nn.Module):
    def __init__(self, latent_vector_dim=32):
        super(VAE, self).__init__()
        # encoder part
        self.conv1 = make_conv_relu(3, 32)
        self.conv2 = make_conv_relu(32, 64)
        self.conv3 = make_conv_relu(64, 128)
        self.conv4 = make_conv_relu(128, 256)

        self.mu = nn.Linear(1024, latent_vector_dim)
        self.logvar = nn.Linear(1024, latent_vector_dim)

        self.z = nn.Linear(latent_vector_dim, 1024)

        # decoder part
        self.deconv1 = make_deconv_relu(1024, 128, 5)
        self.deconv2 = make_deconv_relu(128, 64, 5)
        self.deconv3 = make_deconv_relu(64, 32, 6)
        self.deconv4 = make_deconv_relu(32, 3, 6)

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


# if __name__ == '__main__':
#     import numpy as np
#     import cv2
#
#     img = np.random.randn(64, 64, 3)
#     gpu_img = Variable(torch.from_numpy(img[np.newaxis].transpose(0, 3, 1, 2))).float().cuda()
#
#     vae = VAE()
#     vae.cuda()
#     x, mu, logvar = vae.forward(gpu_img)
#     print(x.size())
#     print(loss_function(x, gpu_img, mu, logvar))
#     x = x.data.cpu().numpy()[0].transpose(1, 2, 0)
#
#     cv2.imshow('original', img)
#     cv2.imshow('reconstructed', x)
#     cv2.waitKey()