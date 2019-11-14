import torch.nn as nn
import torch.nn.functional as f


class PolicyNet(nn.modules):

    def __init(self, in_dim, action_number, hidden=256):
        super(PolicyNet, self).init()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, action_number)

    def forward(self, state):
        logits = self.fc1(state)
        logits = f.relu(logits)
        return logits
