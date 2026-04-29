# td3/noise.py

import torch


class OUNoise:
    """Ornstein-Uhlenbeck noise for smoother TD3 exploration."""

    def __init__(
        self,
        num_envs: int,
        action_dim: int,
        device: str | torch.device,
        mu: float = 0.0,
        theta: float = 0.15,
        sigma: float = 0.15,
    ):
        self.num_envs = num_envs
        self.action_dim = action_dim
        self.device = torch.device(device)

        self.mu = mu
        self.theta = theta
        self.sigma = sigma

        self.state = torch.full(
            (self.num_envs, self.action_dim),
            self.mu,
            device=self.device,
            dtype=torch.float32,
        )

    def reset(self, env_ids: torch.Tensor | None = None):
        """Reset noise state. If env_ids is provided, reset only those envs."""

        if env_ids is None:
            self.state = torch.full_like(self.state, self.mu)
        else:
            env_ids = env_ids.to(device=self.device, dtype=torch.long)
            state = self.state.detach().clone()
            state[env_ids] = self.mu
            self.state = state

    def sample(self) -> torch.Tensor:
        """Generate OU noise sample with shape [num_envs, action_dim]."""

        dx = self.theta * (self.mu - self.state)
        dx = dx + self.sigma * torch.randn_like(self.state)

        self.state = (self.state + dx).detach().clone()

        return self.state.clone()