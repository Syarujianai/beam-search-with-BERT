import torch.nn as nn
import torch.nn.functional as f


class ValueNet(nn.Module):

    def __init__(self, in_dim, hidden=256):
        super(ValueNet, self).__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, 1)

    def forward(self, state):
        value = self.fc1(state)
        value = f.relu(value)
        return value