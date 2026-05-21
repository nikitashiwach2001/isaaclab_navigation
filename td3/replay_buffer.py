import torch


class ReplayBuffer:
    """Replay buffer for vectorized Isaac Lab TD3 training."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        max_size: int,
        device: str | torch.device,
        priv_dim: int = 0,
    ):
        self.max_size = int(max_size)
        self.device = torch.device(device)
        # Privileged obs for asymmetric actor-critic. priv_dim=0 → buffers are
        # width-0, sample() returns empty priv slices, downstream cat is a no-op.
        self.priv_dim = int(priv_dim)

        self.ptr = 0
        self.size = 0

        self.states = torch.zeros((self.max_size, state_dim), dtype=torch.float32, device=self.device)
        self.actions = torch.zeros((self.max_size, action_dim), dtype=torch.float32, device=self.device)
        self.rewards = torch.zeros((self.max_size, 1), dtype=torch.float32, device=self.device)
        self.next_states = torch.zeros((self.max_size, state_dim), dtype=torch.float32, device=self.device)
        self.dones = torch.zeros((self.max_size, 1), dtype=torch.float32, device=self.device)
        self.priv_states = torch.zeros((self.max_size, self.priv_dim), dtype=torch.float32, device=self.device)
        self.priv_next_states = torch.zeros((self.max_size, self.priv_dim), dtype=torch.float32, device=self.device)

    def add(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        dones: torch.Tensor,
        priv_states: torch.Tensor | None = None,
        priv_next_states: torch.Tensor | None = None,
    ):
        """Add a batch of transitions from vectorized envs.

        Expected shapes:
            states:      [num_envs, state_dim]
            actions:     [num_envs, action_dim]
            rewards:     [num_envs]
            next_states: [num_envs, state_dim]
            dones:       [num_envs]
            priv_states / priv_next_states: [num_envs, priv_dim] (required if priv_dim > 0)
        """

        states = states.detach()
        actions = actions.detach()
        rewards = rewards.detach().view(-1, 1)
        next_states = next_states.detach()
        dones = dones.detach().float().view(-1, 1)

        batch_size = states.shape[0]

        indices = (torch.arange(batch_size, device=self.device) + self.ptr) % self.max_size

        self.states[indices] = states
        self.actions[indices] = actions
        self.rewards[indices] = rewards
        self.next_states[indices] = next_states
        self.dones[indices] = dones
        if self.priv_dim > 0:
            self.priv_states[indices] = priv_states.detach()
            self.priv_next_states[indices] = priv_next_states.detach()

        self.ptr = (self.ptr + batch_size) % self.max_size
        self.size = min(self.size + batch_size, self.max_size)

    def sample(self, batch_size: int):
        """Sample a random batch."""

        if self.size < batch_size:
            raise ValueError(f"Not enough samples in buffer: {self.size} < {batch_size}")

        indices = torch.randint(0, self.size, (batch_size,), device=self.device)

        return (
            self.states[indices],
            self.actions[indices],
            self.rewards[indices],
            self.next_states[indices],
            self.dones[indices],
            self.priv_states[indices],
            self.priv_next_states[indices],
        )

    def __len__(self):
        return self.size