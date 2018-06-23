import torch
import torch.nn as nn
from torch.nn import functional as F


class InverseModel(torch.nn.modules.Module):
    def __init__(self, z_dim, action_dim, hidden_size):
        """
        Inverse model from https://arxiv.org/abs/1804.10689.

        Takes in z and z' and outputs predicted action
        """
        super().__init__()
        self.ln1 = nn.Linear(z_dim * 2, hidden_size)
        self.ln2 = nn.Linear(hidden_size, hidden_size)
        self.ln3 = nn.Linear(hidden_size, action_dim)

    def forward(self, z_now, z_next):
        x = torch.cat((z_now, z_next), dim=-1)
        x = F.relu(self.ln1(x))
        x = F.relu(self.ln2(x))
        x = self.ln3(x)
        return x


class InverseModel2(torch.nn.modules.Module):
    def __init__(self, z_dim, action_dim, hidden_size):
        """
        Inverse model from https://arxiv.org/abs/1804.10689.

        Takes in z and z' and outputs predicted action
        """
        super().__init__()
        self.ln1 = nn.Linear(z_dim * 2, hidden_size)
        self.ln3 = nn.Linear(hidden_size, action_dim)

    def forward(self, z_now, z_next):
        x = torch.cat((z_now, z_next), dim=-1)
        x = F.relu(self.ln1(x))
        x = self.ln3(x)
        return x
