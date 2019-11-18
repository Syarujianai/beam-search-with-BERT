import torch.nn as nn
import torch.nn.functional as f


class PolicyNet(nn.Module):

    def __init__(self, in_dim, action_number, hidden=256):
        super(PolicyNet, self).__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, action_number)

    def forward(self, state):
        logits = self.fc1(state)
        logits = f.relu(logits)
        return logits
