import torch
import torch.nn as nn
from torch.nn import functional as F
from torch import normal, multinomial
from torch.autograd import Variable


class MDNRNN(nn.Module):
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
        super(MDNRNN, self).__init__()
        # define rnn
        self.inpt_size = z_dim + action_dim
        self.hidden_size = hidden_size
        self.n_mixture = n_mixture
        self.z_dim = z_dim
        self.rnn = nn.LSTM(input_size=self.inpt_size, hidden_size=hidden_size, batch_first=True)

        # define MDN as fully connected layer
        self.mdn = nn.Linear(hidden_size, n_mixture * z_dim * 2 + n_mixture)
        self.tau = temperature

    def forward(self, inpt, action, hidden_state=None):
        """
        :param inpt: a tensor of size (batch_size, seq_len, D)
        :param hidden_state: two tensors of size (1, batch_size, hidden_size)
        :param action: a tensor of (batch_size, seq_len, action_dim)
        :return: pi, mean, sigma, hidden_state
        """
        batch_size, seq_len, _ = inpt.size()
        if hidden_state is None:
            # use new so that we do not need to know the tensor type explicitly.
            hidden_state = (Variable(inpt.data.new(1, batch_size, self.hidden_size)),
                            Variable(inpt.data.new(1, batch_size, self.hidden_size)))

        # concatenate input and action, maybe we can use an extra fc layer to project action to a space same
        # as inpt?
        concat = torch.cat((inpt, action), dim=-1)
        output, hidden_state = self.rnn(concat, hidden_state)
        output = output.contiguous()
        output = output.view(-1, self.hidden_size)
        # N, seq_len, n_mixture * z_dim * 2 + n_mixture
        mixture = self.mdn(output)
        mixture = mixture.view(batch_size, seq_len, -1)
        # N * seq_len, n_mixture * z_dim
        mean = mixture[..., :self.n_mixture * self.z_dim]
        sigma = mixture[..., self.n_mixture * self.z_dim: self.n_mixture * self.z_dim*2]
        sigma = torch.exp(sigma)
        # N * seq_len, n_mixture
        pi = mixture[..., -self.n_mixture:]
        pi = F.softmax(pi, -1)

        # add temperature
        if self.tau > 0:
            pi /= self.tau
            sigma *= self.tau ** 0.5
        return pi, mean, sigma, hidden_state

    def sample(self, inpt, action, hidden_state=None):
        """
        Sample from a mixture of Gaussians. This function is only in testing, so batch_size=seq_len=1 for now.
        parameters same as forward
        :return:
        """
        # forward and get pi, mean. sigma, hidden_state
        pi, mean, sigma, hidden_state = self.forward(inpt, action, hidden_state)
        batch_size, seq_len, _ = inpt.size()
        pi, mean, sigma = pi.contiguous().view(-1), mean.contiguous().view(-1), sigma.contiguous().view(-1)
        # randomly draw a mixture model
        k = multinomial(pi, 1)

        selected_mean = mean[int(self.z_dim * k): int(self.z_dim * (k+1))]
        selected_sigma = sigma[int(self.z_dim * k): int(self.z_dim * (k+1))]

        # sample from normal dist
        z = normal(selected_mean, selected_sigma)
        return z, hidden_state


# if __name__ == '__main__':
#     from torch.autograd import Variable
#     z_dim, action_dim, hidden_size, n_mixture, temp = 32, 2, 256, 5, 0.0
#     batch_size = 1
#     seq_len = 1
#     mdnrnn = MDNRNN(z_dim, action_dim, hidden_size, n_mixture, temp)
#     mdnrnn.cuda()
#     prev_z = Variable(torch.randn(batch_size, seq_len, z_dim)).cuda()
#     action = Variable(torch.randn(batch_size, seq_len, action_dim)).cuda()
#
#     new_z, new_hidden_state = mdnrnn.sample(prev_z, action)
#     print(new_z)
#     pi, mean, sigma, hidden_state = mdnrnn.forward(prev_z, action)
#     print(sigma)