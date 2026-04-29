import torch
import torch.nn as nn
import torch.nn.functional as F


def init_weights(module: nn.Module):
    """Initialize network weights similar to common TD3/DDPG practice."""
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        nn.init.zeros_(module.bias)


class Actor(nn.Module):
    """TD3 Actor network.

    Input:
        state: [batch_size, state_dim]

    Output:
        action: [batch_size, action_dim], bounded to [-1, 1]
    """

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()

        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)

        self.apply(init_weights)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        action = torch.tanh(self.fc3(x))
        return action


class Critic(nn.Module):
    """TD3 Critic network with twin Q functions."""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()

        # Q1 network
        self.q1_state = nn.Linear(state_dim, hidden_dim // 2)
        self.q1_action = nn.Linear(action_dim, hidden_dim // 2)
        self.q1_hidden = nn.Linear(hidden_dim, hidden_dim)
        self.q1_out = nn.Linear(hidden_dim, 1)

        # Q2 network
        self.q2_state = nn.Linear(state_dim, hidden_dim // 2)
        self.q2_action = nn.Linear(action_dim, hidden_dim // 2)
        self.q2_hidden = nn.Linear(hidden_dim, hidden_dim)
        self.q2_out = nn.Linear(hidden_dim, 1)

        self.apply(init_weights)

    def forward(self, state: torch.Tensor, action: torch.Tensor):
        q1 = self.q1_forward(state, action)
        q2 = self.q2_forward(state, action)
        return q1, q2

    def q1_forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        xs = F.relu(self.q1_state(state))
        xa = F.relu(self.q1_action(action))
        x = torch.cat([xs, xa], dim=1)
        x = F.relu(self.q1_hidden(x))
        q1 = self.q1_out(x)
        return q1

    def q2_forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        xs = F.relu(self.q2_state(state))
        xa = F.relu(self.q2_action(action))
        x = torch.cat([xs, xa], dim=1)
        x = F.relu(self.q2_hidden(x))
        q2 = self.q2_out(x)
        return q2