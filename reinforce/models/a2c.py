import os
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from reinforce.models.policy_net import PolicyNet
from reinforce.models.value_net import ValueNet
from torch.distributions import Categorical


class A2CAgent(object):

    def __init__(self, state_dim, action_number, lr, device):
        self.device = torch.device("cuda:%d" % device if torch.cuda.is_available() else "cpu")
        self.policy_net = PolicyNet(state_dim, action_number)
        self.value_net = ValueNet(state_dim, 1)

        self.policy_net.to(self.device)
        self.value_net.to(self.device)

        self.optimizer = optim.Adam([self.value_net.parameters(), self.policy_net.parameters()], lr=lr)
        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=lr)
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)

    def get_action(self, state, pos, pos_index):
        """
        get scalar action based on state and pos constraints
        :param state:
        :param pos:
        :param pos_index:
        :return: scalar value
        """
        state = torch.FloatTensor(state).to(self.device)
        logits = self.policy_net.forward(state)
        start, end = pos_index[pos]
        logits = logits[start:end + 1]
        prob = F.softmax(logits, dim=0)
        prob = Categorical(prob)
        action = prob.sample().cpu().detach().item()
        return action

    def optimize(self, state, action, reward):
        reward = torch.FloatTensor([reward]).to(self.device)
        state = torch.FloatTensor(state).to(self.device)
        action = Variable(torch.FloatTensor([action]).to(self.device))
        value = self.value_net.forward(state)
        advantage = reward - value

        value_loss = F.mse_loss(value, reward)

        logits = self.policy_net.forward(state)
        probs = F.softmax(logits, dim=0)
        probs = Categorical(probs)
        policy_loss = -probs.log_prob(action) * advantage

        loss = policy_loss + value_loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        # self.value_optimizer.zero_grad()
        # value_loss.backward()
        # self.value_optimizer.step()
        #
        # self.policy_optimizer.zero_grad()
        # policy_loss.backward()
        # self.policy_optimizer.step()

    def save(self, path):
        print("save checkpoint to ", path)
        torch.save(self.policy_net, os.path.join(path, "policy.pt"))
        torch.save(self.value_net, os.path.join(path, "value.pt"))

    def load(self, path):
        self.policy_net = torch.load(os.path.join(path, "policy.pt"))
        self.value_net = torch.load(os.path.join(path, "value.pt"))

