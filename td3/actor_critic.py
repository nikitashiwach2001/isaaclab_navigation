import os

import torch
import torch.nn as nn
import torch.nn.functional as F


# Frames-per-stack must match the env's observation spec. Default 6 matches
# the Stage 4/5/6 observation; set LIDAR_STACK_FRAMES in the shell to override.
_N_LIDAR_FRAMES = int(os.environ.get("LIDAR_STACK_FRAMES", "6"))


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


class ConvLidarEncoder(nn.Module):
    """1D conv encoder over the lidar ray axis.

    Input  : [batch, n_frames * n_rays]  flat lidar_stacked output
    Reshape: [batch, n_frames, n_rays]   (frames as conv channels)
    Output : [batch, output_dim]         dense lidar features

    Uses circular padding because the 360-degree lidar wraps around — the ray
    at index 89 is spatially adjacent to the ray at index 0. Zero padding would
    create artificial edges at the wraparound point that the network has to
    learn around.

    Two stride-2 stages downsample 90 -> 45 -> 23 rays while keeping local
    spatial structure. Final projection flattens and reduces to a fixed-width
    feature vector that the rest of the MLP consumes alongside the other
    14 observation dims.
    """

    def __init__(self, n_frames: int = _N_LIDAR_FRAMES, n_rays: int = 90, output_dim: int = 128):
        super().__init__()
        self.n_frames = n_frames
        self.n_rays   = n_rays
        self.output_dim = output_dim

        self.conv1 = nn.Conv1d(n_frames, 32, kernel_size=5, padding=2, padding_mode="circular")
        self.conv2 = nn.Conv1d(32, 32, kernel_size=5, stride=2, padding=2, padding_mode="circular")
        self.conv3 = nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2, padding_mode="circular")

        # 90 -> ceil(90/2)=45 -> ceil(45/2)=23  (with stride 2 + padding 2 + kernel 5)
        self._flat_dim = 64 * 23
        self.proj = nn.Linear(self._flat_dim, output_dim)

    def forward(self, lidar_flat: torch.Tensor) -> torch.Tensor:
        x = lidar_flat.view(-1, self.n_frames, self.n_rays)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.flatten(1)
        x = F.relu(self.proj(x))
        return x


_N_LIDAR_DIMS = _N_LIDAR_FRAMES * 90  # matches observations.lidar_stacked (varies with LIDAR_STACK_FRAMES)


class ConvActor(nn.Module):
    """TD3 Actor with 1D conv lidar encoder.

    Splits the input state into (lidar_540, other_14). Lidar goes through
    ConvLidarEncoder -> dense features. Other features pass through unchanged.
    Concatenated representation feeds the same fc1/fc2/fc3 MLP as the plain
    Actor.

    Same action_dim, same hidden_dim, same tanh output as Actor — drop-in
    replacement at the agent level.
    """

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256,
                 n_lidar: int = _N_LIDAR_DIMS, lidar_feat_dim: int = 128):
        super().__init__()
        self.n_lidar = n_lidar
        other_dim = state_dim - n_lidar
        if other_dim < 0:
            raise ValueError(f"state_dim {state_dim} < n_lidar {n_lidar}")

        self.lidar_encoder = ConvLidarEncoder(output_dim=lidar_feat_dim)

        self.fc1 = nn.Linear(lidar_feat_dim + other_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)

        self.apply(init_weights)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        lidar = state[:, :self.n_lidar]
        other = state[:, self.n_lidar:]
        lidar_feat = self.lidar_encoder(lidar)
        x = torch.cat([lidar_feat, other], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return torch.tanh(self.fc3(x))


class ConvCritic(nn.Module):
    """TD3 Critic with twin Q-functions, each with its own ConvLidarEncoder.

    Following the TD3 paper's spirit, Q1 and Q2 are fully independent — no
    shared encoder. Each has its own conv weights so the twin-min bias
    reduction works as intended.
    """

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256,
                 n_lidar: int = _N_LIDAR_DIMS, lidar_feat_dim: int = 128):
        super().__init__()
        self.n_lidar = n_lidar
        other_dim = state_dim - n_lidar
        if other_dim < 0:
            raise ValueError(f"state_dim {state_dim} < n_lidar {n_lidar}")

        effective_state_dim = lidar_feat_dim + other_dim

        # Q1
        self.q1_encoder = ConvLidarEncoder(output_dim=lidar_feat_dim)
        self.q1_state  = nn.Linear(effective_state_dim, hidden_dim // 2)
        self.q1_action = nn.Linear(action_dim, hidden_dim // 2)
        self.q1_hidden = nn.Linear(hidden_dim, hidden_dim)
        self.q1_out    = nn.Linear(hidden_dim, 1)

        # Q2
        self.q2_encoder = ConvLidarEncoder(output_dim=lidar_feat_dim)
        self.q2_state  = nn.Linear(effective_state_dim, hidden_dim // 2)
        self.q2_action = nn.Linear(action_dim, hidden_dim // 2)
        self.q2_hidden = nn.Linear(hidden_dim, hidden_dim)
        self.q2_out    = nn.Linear(hidden_dim, 1)

        self.apply(init_weights)

    def _split(self, state: torch.Tensor):
        return state[:, :self.n_lidar], state[:, self.n_lidar:]

    def forward(self, state: torch.Tensor, action: torch.Tensor):
        return self.q1_forward(state, action), self.q2_forward(state, action)

    def q1_forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        lidar, other = self._split(state)
        lidar_feat = self.q1_encoder(lidar)
        s = torch.cat([lidar_feat, other], dim=1)
        xs = F.relu(self.q1_state(s))
        xa = F.relu(self.q1_action(action))
        x = torch.cat([xs, xa], dim=1)
        x = F.relu(self.q1_hidden(x))
        return self.q1_out(x)

    def q2_forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        lidar, other = self._split(state)
        lidar_feat = self.q2_encoder(lidar)
        s = torch.cat([lidar_feat, other], dim=1)
        xs = F.relu(self.q2_state(s))
        xa = F.relu(self.q2_action(action))
        x = torch.cat([xs, xa], dim=1)
        x = F.relu(self.q2_hidden(x))
        return self.q2_out(x)


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