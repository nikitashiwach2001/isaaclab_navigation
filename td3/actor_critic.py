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


class ActorGRU(nn.Module):
    """TD3 Actor with residual GRU for temporal memory.

    The GRU adds a learned correction to fc2's output via a zero-initialised
    gate layer.  At the start of training gru_gate outputs exactly zero, so
    fc3 receives the same input as the plain MLP Actor and loaded v6 weights
    work without any warm-up degradation.  The GRU contribution grows
    naturally as gru_gate's weights are updated by gradient descent.

    Same fc1/fc2/fc3 layer shapes as Actor so those weights transfer cleanly
    from a pre-trained MLP checkpoint with strict=False loading.

    Input:
        state:  [batch_size, state_dim]
        hidden: [batch_size, hidden_dim] or None (zeros)

    Output:
        action: [batch_size, action_dim], bounded to [-1, 1]
        hidden: [batch_size, hidden_dim] updated GRU state
    """

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.gru  = nn.GRUCell(hidden_dim, hidden_dim)
        # Projects GRU hidden state to a residual correction added to fc2 output.
        # Zero-initialised so the policy starts identical to the MLP baseline.
        self.gru_gate = nn.Linear(hidden_dim, hidden_dim)
        self.fc3  = nn.Linear(hidden_dim, action_dim)

        self.apply(init_weights)
        nn.init.zeros_(self.gru_gate.weight)
        nn.init.zeros_(self.gru_gate.bias)

    def forward(
        self,
        state: torch.Tensor,
        hidden: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if hidden is None:
            hidden = torch.zeros(state.shape[0], self.hidden_dim, device=state.device)
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        new_hidden: torch.Tensor = self.gru(x, hidden)
        # gru_gate is zero at init → contribution starts at 0 → fc3 sees pure fc2 output
        gru_contribution = self.gru_gate(new_hidden)
        action = torch.tanh(self.fc3(x + gru_contribution))
        return action, new_hidden


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