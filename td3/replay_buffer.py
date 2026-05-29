import torch


class ReplayBuffer:
    """Replay buffer for vectorized Isaac Lab TD3 training.

    By default the buffer is stored on CPU and batches are transferred to the
    compute device on `sample()`. This lets the buffer be sized in the millions
    without competing with the policy/env for GPU memory. Pass
    `storage_device="cuda"` to put the buffer back on GPU at the cost of GPU
    memory.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        max_size: int,
        device: str | torch.device,
        priv_dim: int = 0,
        storage_device: str | torch.device | None = None,
    ):
        self.max_size = int(max_size)
        self.device = torch.device(device)
        # Storage device: where the big tensors live. Defaults to CPU to keep
        # GPU memory free for the policy and environment.
        self.storage_device = torch.device(storage_device if storage_device is not None else "cpu")
        # Privileged obs for asymmetric actor-critic. priv_dim=0 → buffers are
        # width-0, sample() returns empty priv slices, downstream cat is a no-op.
        self.priv_dim = int(priv_dim)

        self.ptr = 0
        self.size = 0

        def _alloc(shape):
            return torch.zeros(shape, dtype=torch.float32, device=self.storage_device)

        self.states           = _alloc((self.max_size, state_dim))
        self.actions          = _alloc((self.max_size, action_dim))
        self.rewards          = _alloc((self.max_size, 1))
        self.next_states      = _alloc((self.max_size, state_dim))
        self.dones            = _alloc((self.max_size, 1))
        self.priv_states      = _alloc((self.max_size, self.priv_dim))
        self.priv_next_states = _alloc((self.max_size, self.priv_dim))

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

        # Move incoming GPU tensors to storage_device (CPU by default) once,
        # then write into the persistent buffers.
        states      = states.detach().to(self.storage_device, non_blocking=True)
        actions     = actions.detach().to(self.storage_device, non_blocking=True)
        rewards     = rewards.detach().view(-1, 1).to(self.storage_device, non_blocking=True)
        next_states = next_states.detach().to(self.storage_device, non_blocking=True)
        dones       = dones.detach().float().view(-1, 1).to(self.storage_device, non_blocking=True)

        batch_size = states.shape[0]

        indices = (torch.arange(batch_size, device=self.storage_device) + self.ptr) % self.max_size

        self.states[indices]      = states
        self.actions[indices]     = actions
        self.rewards[indices]     = rewards
        self.next_states[indices] = next_states
        self.dones[indices]       = dones
        if self.priv_dim > 0:
            self.priv_states[indices]      = priv_states.detach().to(self.storage_device, non_blocking=True)
            self.priv_next_states[indices] = priv_next_states.detach().to(self.storage_device, non_blocking=True)

        self.ptr = (self.ptr + batch_size) % self.max_size
        self.size = min(self.size + batch_size, self.max_size)

    def sample(self, batch_size: int):
        """Sample a random batch. Returned tensors live on `self.device`
        (the compute device); the buffer itself stays on `self.storage_device`."""

        if self.size < batch_size:
            raise ValueError(f"Not enough samples in buffer: {self.size} < {batch_size}")

        indices = torch.randint(0, self.size, (batch_size,), device=self.storage_device)

        return (
            self.states[indices].to(self.device, non_blocking=True),
            self.actions[indices].to(self.device, non_blocking=True),
            self.rewards[indices].to(self.device, non_blocking=True),
            self.next_states[indices].to(self.device, non_blocking=True),
            self.dones[indices].to(self.device, non_blocking=True),
            self.priv_states[indices].to(self.device, non_blocking=True),
            self.priv_next_states[indices].to(self.device, non_blocking=True),
        )

    def __len__(self):
        return self.size
