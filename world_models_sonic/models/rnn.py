import torch
import torch.nn as nn
from torch.nn import functional as F
from torch import normal, multinomial
import torch.distributions
from torch.autograd import Variable
from torch import normal, multinomial
import math
import numpy as np

from ..config import eps
logeps = math.log(eps)


class MDNRNN2(nn.Module):
    def __init__(self, z_dim, action_dim, hidden_size, n_mixture, temperature):
        """
            :param z_dim: the dimension of VAE latent variable
            :param hidden_size: hidden size of RNN
            :param n_mixture: number of Gaussian Mixture Models to be used
            :param temperature: controls the randomness of the model

            MDNRNN stands for Mixture Density Network - RNN.
            The output of this model is [mean, sigma^2, K],
            where mean and sigma^2 have z_dim * n_mixture elements and
            K has n_mixture elements.
        """
        super(MDNRNN2, self).__init__()
        # define rnn
        self.inpt_size = z_dim + action_dim
        self.hidden_size = hidden_size
        self.action_dim = action_dim
        self.n_mixture = n_mixture
        self.z_dim = z_dim
        self.rnn = nn.LSTM(input_size=self.inpt_size, hidden_size=hidden_size, batch_first=True)

        # define MDN as fully connected layer
        self.ln1 = nn.Linear(hidden_size, hidden_size)
        self.ln2 = nn.Linear(hidden_size, hidden_size * 5)
        self.mdn = nn.Linear(hidden_size * 5, n_mixture * z_dim * 3)
        self.tau = temperature

    def forward(self, inpt, action_discrete, hidden_state=None):
        """
        :param inpt: a tensor of size (batch_size, seq_len, D)
        :param hidden_state: two tensors of size (1, batch_size, hidden_size)
        :param action: a tensor of (batch_size, seq_len, action_dim)
        :return: pi, mean, sigma, hidden_state
        """
        batch_size, seq_len, _ = inpt.size()
        # one hot code the action
        action = torch.eye(self.action_dim)[action_discrete.long()]
        cuda = list(self.parameters())[0].is_cuda
        if cuda:
            action = action.cuda()

        if hidden_state is None:
            # use new so that we do not need to know the tensor type explicitly.
            hidden_state = (Variable(inpt.data.new(1, batch_size, self.hidden_size)),
                            Variable(inpt.data.new(1, batch_size, self.hidden_size)))

        # concatenate input and action, maybe we can use an extra fc layer to project action to a space same
        # as inpt?
        concat = torch.cat((inpt, action), dim=-1)
        output, hidden_state = self.rnn(concat, hidden_state)
        pi, mean, sigma = self.get_mixture_coef(output)
        return pi, mean, sigma, hidden_state

    def get_mixture_coef(self, output):
        batch_size, seq_len, _ = output.size()
        # N, seq_len, n_mixture * z_dim * 2 + n_mixture
        output = output.contiguous()
        output = output.view(-1, self.hidden_size)

        output = F.leaky_relu(self.ln1(output))
        output = F.leaky_relu(self.ln2(output))
        mixture = self.mdn(output)
        mixture = mixture.view(batch_size, seq_len, -1)

        # Split output into mean, logsigma, pi

        # N * seq_len
        mu = mixture[..., :self.n_mixture * self.z_dim]
        logsigma = mixture[..., self.n_mixture * self.z_dim: self.n_mixture * self.z_dim * 2]
        pi = mixture[..., self.n_mixture * self.z_dim * 2:self.n_mixture * self.z_dim * 3]

        # Reshape
        mu = mu.view((-1, seq_len, self.n_mixture, self.z_dim))
        logsigma = logsigma.view((-1, seq_len, self.n_mixture, self.z_dim)).clamp(np.log(eps), -np.log(eps))
        pi = pi.view((-1, seq_len, self.n_mixture, self.z_dim))

        # Transform
        sigma = torch.exp(logsigma)
        pi = F.softmax(pi, 2)  # Weights over n_mixtures should sum to one

        # add temperature
        if self.tau > 0:
            pi /= self.tau
            sigma *= self.tau ** 0.5

        return pi, mu, sigma

    def normal_prob(self, y_true, mu, sigma, pi):
        """Probability of a value given the distribution."""
        rollout_length = y_true.size(1)

        # Repeat, for number of repeated mixtures
        y_true = y_true.unsqueeze(2).repeat((1, 1, self.n_mixture, 1))

        # Use pytorches normal dist class to calc the probs
        z_normals = torch.distributions.Normal(mu, sigma)
        z_prob = z_normals.log_prob(y_true).exp()  # .clamp(logeps, -logeps).exp()
        return (z_prob * pi).sum(2)  # weight, then sum over the mixtures

    def multinomial_on_axis(self, pi, axis=2):
        """
        Take the multinomial along one dimenionself.

        Returns an array with the same shape as the input but the chosen axis
        has one element set to one, and the other to zero. So you can multiply
        another array then sum, to choose an axis.

        e.g. k * mu = [0, 0, 1, 0] * [1.4, 1.5, 0.2, 3] = [0, 0, 0.2, 0]
        """
        # Reshape pi, so we can get the multinomial along the mixture dimension
        batch, seq, mixtures, z_dim = pi.size()
        axis_size = pi.size(axis)
        pi = pi.transpose(axis, 3).contiguous()
        pi_flat = pi.view(-1, axis_size)
        # np.testing.assert_almost_equal(pi_flat.sum(-1).cpu().data.numpy(), 1, decimal=4, err_msg='should reshape right axis')
        # sample
        k = torch.distributions.Multinomial(1, pi_flat).sample()
        # reshape back
        k = k.view(*pi.size()).transpose(axis, 3).contiguous()
        assert (k.sum(axis) == 1).all(), 'should sum to one'
        assert (k.max(axis)[0] == 1).all(), 'max should be one'
        return k

    def sample(self, pi, mu, sigma):
        """Sample z from Z."""
        k = self.multinomial_on_axis(pi, axis=2)
        mu = (mu * k).sum(2)
        sigma = (sigma * k).sum(2)
        z_normals = torch.distributions.Normal(mu, sigma)
        if self.training:
            z_sample = z_normals.rsample()
        else:
            z_sample = z_normals.sample()
        return z_sample

    def rnn_r_loss(self, y_true, pi, mu, sigma):
        # See https://github.com/hardmaru/pytorch_notebooks/blob/master/mixture_density_networks.ipynb
        # and https://github.com/AppliedDataSciencePartners/WorldModels/blob/master/rnn/arch.py#L39
        # and https://github.com/JunhongXu/world-models-pytorch

        # probability shape [batch, seq, num_mixtures, z_dim]
        batch_size, seq_len, _ = y_true.size()
        prob = self.normal_prob(y_true, mu, sigma, pi)
        loss = -torch.log(prob + eps)

        # mean over seq and z dim and num_mixtures
        batch_size = y_true.size(0)
        loss = loss.view((batch_size, seq_len, -1)).mean(2)

        # We want the loss to +ve and approach zero. But, since we clip to eps(ilon)
        # it's approaching `log(eps)`` E.g. `log(1e-7)=-16.12`. So let's shift it to approach zero from above.
        loss -= np.log(eps)
        return loss

    def rnn_loss(self, y_true, pi, mu, sigma):
        r_loss = self.rnn_r_loss(y_true, pi, mu, sigma)
        return r_loss
