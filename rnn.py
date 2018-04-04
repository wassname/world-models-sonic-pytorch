import torch.nn as nn
from torch import normal, multinomial



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
        self.rnn = nn.LSTM(input_size=self.inpt_size, hidden_size=hidden_size, batch_first=True)

        # define MDN as fully connected layer
        self.mdn = nn.Linear(hidden_size, n_mixture * z_dim * 2 + n_mixture)
        self.tau = temperature

    def forward(self, inpt, hidden_state, action):
        """
        :param inpt: a tensor of size (batch_size, seq_len, D)
        :param hidden_state: two tensors of size (1, batch_size, hidden_size)
        :param action: a tensor of (batch_size, seq_len, D*)
        :return: pi, mean, sigma, hidden_state
        """
        # TODO: Add forward function
        return

    def sample(self, inpt, hidden_state, action):
        """
        parameters same as forward
        :return:
        """
        # forward and get pi, mean. sigma, hidden_state
        pi, mean, sigma, hidden_state = self.forward(inpt, hidden_state, action)

        # randomly draw a mixture model
        k = multinomial(pi, 1)
        selected_mean = mean[..., k]
        selected_sigma = sigma[..., k]

        # sample from normal dist
        z = normal(selected_mean, selected_sigma)
        return z, hidden_state