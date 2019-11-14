import torch.nn as nn
import torch.nn.functional as f


class ValueNet(nn.modules):

    def __init(self, in_dim, hidden=256):
        super(ValueNet, self).init()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, 1)

    def forward(self, state):
        value = self.fc1(state)
        value = f.relu(value)
        return value