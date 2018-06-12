import torch
import torch.nn as nn
import torch.distributions
import math
import numpy as np
from torch.nn import functional as F

from ..config import eps
logeps = math.log(eps)


logSqrtTwoPI = np.log(np.sqrt(2.0 * np.pi))
debug = False


def assert_finite(x):
    """Quick pytorch test that there are no nan's or infs."""
    assert ((x + 1) != x).all(), 'contains infs: {}'.format(x)
    assert (x == x).all(), 'contains nans: {}'.format(x)


def lognormal(y, mean, logstd):
    return -0.5 * ((y - mean) / torch.exp(logstd)) ** 2 - logstd - logSqrtTwoPI


def logsumexp(inputs, dim=None, keepdim=False):
    """Numerically stable logsumexp.

    Args:
        inputs: A Variable with any shape.
        dim: An integer.
        keepdim: A boolean.

    Returns:
        Equivalent of log(sum(exp(inputs), dim=dim, keepdim=keepdim)).

    From https://github.com/pytorch/pytorch/issues/2591#issuecomment-364474328
    TODO replace with native torch.logsumexp after v0.4.1 of torch

    """
    # For a 1-D array x (any array along a single dimension),
    # log sum exp(x) = s + log sum exp(x - s)
    # with s = max(x) being a common choice.
    if dim is None:
        inputs = inputs.view(-1)
        dim = 0
    s, _ = torch.max(inputs, dim=dim, keepdim=True)
    outputs = s + (inputs - s).exp().sum(dim=dim, keepdim=True).log()
    if not keepdim:
        outputs = outputs.squeeze(dim)
    return outputs


class MDNRNN(nn.Module):
    def __init__(self, z_dim, action_dim, hidden_size, n_mixture, temperature):
        """
        MDNRNN stands for Mixture Density Network - RNN.

        :param z_dim: the dimension of VAE latent variable
        :param hidden_size: hidden size of RNN
        :param n_mixture: number of Gaussian Mixture Models to be used
        :param temperature: controls the randomness of the model

        The output of this model is [mean, sigma^2, K],
        where mean and sigma^2 have z_dim * n_mixture elements and
        K has n_mixture elements.
        """
        super(MDNRNN, self).__init__()
        self.inpt_size = z_dim + action_dim
        self.hidden_size = hidden_size
        self.action_dim = action_dim
        self.n_mixture = n_mixture
        self.z_dim = z_dim
        self.rnn = nn.LSTM(input_size=self.inpt_size, hidden_size=hidden_size, batch_first=True)

        # define MDN as fully connected layer
        self.mdn = nn.Linear(hidden_size, n_mixture * z_dim * 3)
        self.tau = temperature

    def default_hidden_state(self, z):
        # you init the state to some constant value
        # https://www.cs.toronto.edu/~hinton/csc2535/notes/lec10new.pdf
        batch_size, seq_len, _ = z.size()
        cuda = z.is_cuda
        hidden_state = (
            torch.zeros((1, batch_size, self.hidden_size), dtype=z.dtype),
            torch.zeros((1, batch_size, self.hidden_size), dtype=z.dtype),
        )
        if cuda:
            hidden_state = [hs.cuda() for hs in hidden_state]
        return hidden_state

    def forward(self, inpt, action_discrete=None, hidden_state=None):
        """
        Forward.

        :param inpt: a tensor of size (batch_size, seq_len, D)
        :param hidden_state: two tensors of size (1, batch_size, hidden_size)
        :param action: a tensor of (batch_size, seq_len, action_dim)
        :return: pi, mean, sigma, hidden_state
        """
        batch_size, seq_len, _ = inpt.size()
        if action_discrete is None:
            action_discrete = torch.zeros(batch_size, seq_len)
        # one hot code the action
        action = torch.eye(self.action_dim)[action_discrete.long()]
        cuda = next(iter(self.parameters())).is_cuda
        if cuda:
            action = action.cuda()

        if hidden_state is None:
            # you init the state to some constant value
            # https://www.cs.toronto.edu/~hinton/csc2535/notes/lec10new.pdf
            hidden_state = self.default_hidden_state(inpt)

        for hs in hidden_state:
            assert_finite(hs)

        # concatenate input and action
        concat = torch.cat((inpt, action), dim=-1)
        output, hidden_state = self.rnn(concat, hidden_state)
        logpi, mean, sigma = self.get_mixture_coef(output)
        return logpi, mean, sigma, hidden_state

    def get_mixture_coef(self, output):
        batch_size, seq_len, _ = output.size()
        # N, seq_len, n_mixture * z_dim * 2 + n_mixture
        output = output.contiguous()
        output = output.view(-1, self.hidden_size)

        mixture = self.mdn(output)
        mixture = mixture.view(batch_size, seq_len, -1)

        # Split output into mean, logsigma, pi

        # N * seq_len
        mu = mixture[..., :self.n_mixture * self.z_dim]
        logsigma = mixture[..., self.n_mixture * self.z_dim: self.n_mixture * self.z_dim * 2]
        logpi = mixture[..., self.n_mixture * self.z_dim * 2:self.n_mixture * self.z_dim * 3]

        # Reshape
        mu = mu.view((-1, seq_len, self.n_mixture, self.z_dim))
        logsigma = logsigma.view((-1, seq_len, self.n_mixture, self.z_dim)).clamp(np.log(eps), -np.log(eps))
        logpi = logpi.view((-1, seq_len, self.n_mixture, self.z_dim)).clamp(logeps, -logeps)

        # A stable log domain softmax
        logpi = F.log_softmax(logpi, 2)  # Weights over n_mixtures should sum to one

        # add temperature
        if self.tau > 0:
            logpi -= torch.log(self.tau)
            logsigma += torch.log(self.tau ** 0.5)

        if debug:
            assert_finite(logpi)
            assert_finite(logsigma)
            assert_finite(mu)

        return logpi, mu, logsigma

    def multinomial_on_axis(self, logpi, axis=2):
        """
        Take the multinomial along one dimenionself.

        Returns an array with the same shape as the input but the chosen axis
        has one element set to one, and the other to zero. So you can multiply
        another array then sum, to choose an axis.

        e.g. k * mu = [0, 0, 1, 0] * [1.4, 1.5, 0.2, 3] = [0, 0, 0.2, 0]
        """
        # Reshape pi, so we can get the multinomial along the mixture dimension
        batch, seq, mixtures, z_dim = logpi.size()
        if debug:
            assert_finite(logpi.exp())
            assert ((logpi.sum(axis) - 1) < 0.01).all(), 'pi should be softmaxed along axis'
        axis_size = logpi.size(axis)
        logpi = logpi.transpose(axis, 3).contiguous()
        logpi_flat = logpi.view(-1, axis_size).clamp(1e-7)
        if debug:
            assert ((logpi_flat.sum(-1) - 1) < 0.01).all(), 'should reshape the correct axis'
        # sample
        k = torch.distributions.Multinomial(1, logpi_flat).sample()
        # reshape back
        k = k.view(*logpi.size()).transpose(axis, 3).contiguous()
        if debug:
            assert (k.sum(axis) == 1).all(), 'should sum to one'
            assert (k.max(axis)[0] == 1).all(), 'max should be one'
        return k

    def sample(self, logpi, mu, logsigma):
        """Sample z from Z."""
        # Select the sampled distribution
        k = self.multinomial_on_axis(logpi.exp(), axis=2)
        mu = (mu * k).sum(2)
        sigma = (logsigma.exp() * k).sum(2)
        # Sample from the distribution
        z_normals = torch.distributions.Normal(mu, sigma)
        if self.training:
            z_sample = z_normals.rsample()
        else:
            z_sample = z_normals.sample()
        if debug:
            assert_finite(z_sample)
        return z_sample

    def rnn_r_loss(self, y_true, logpi, mu, logsigma):
        # see https://github.com/hardmaru/WorldModelsExperiments/blob/c0cb2dee69f4b05d9494bc0263eca25a7f90d555/carracing/rnn/rnn.py#L139
        #     https://github.com/hardmaru/pytorch_notebooks/blob/master/mixture_density_networks.ipynb
        #     https://github.com/AppliedDataSciencePartners/WorldModels/blob/master/rnn/arch.py#L39
        #     https://github.com/JunhongXu/world-models-pytorch

        batch_size, seq_len, _ = y_true.size()
        y_true = y_true.unsqueeze(2).repeat((1, 1, self.n_mixture, 1))
        # probability shape [batch, seq, num_mixtures, z_dim]
        logprob = lognormal(y_true, mu, logsigma).clamp(logeps, -logeps)
        v = logpi + logprob
        loss = -logsumexp(v, dim=2, keepdim=True)

        # mean over seq and z dim and num_mixtures
        loss = loss.view((batch_size, seq_len, -1)).mean(2)

        # We want the loss to +ve and approach zero. But, since we add eps(ilon)
        # it's approaching `log(eps)`` E.g. `log(1e-7)=-16.12`. So let's shift it to approach zero from above.
        loss -= np.log(eps)
        return loss

    def rnn_loss(self, y_true, logpi, mu, logsigma):
        r_loss = self.rnn_r_loss(y_true, logpi, mu, logsigma)
        return r_loss
