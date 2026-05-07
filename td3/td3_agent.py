import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

from td3.actor_critic import Actor, Critic


class TD3Agent:
    """Clean TD3 implementation for Isaac Lab vectorized env."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        device: str | torch.device,
        hidden_dim: int = 256,
        actor_lr: float = 1e-4,
        critic_lr: float = 1e-3,
        gamma: float = 0.99,
        tau: float = 0.005,
        policy_noise: float = 0.2,
        noise_clip: float = 0.5,
        policy_delay: int = 2,
    ):
        self.device = torch.device(device)

        self.state_dim = state_dim
        self.action_dim = action_dim

        self.gamma = gamma
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_delay = policy_delay

        self.total_it = 0

        self.actor = Actor(state_dim, action_dim, hidden_dim).to(self.device)
        self.actor_target = copy.deepcopy(self.actor)

        self.critic = Critic(state_dim, action_dim, hidden_dim).to(self.device)
        self.critic_target = copy.deepcopy(self.critic)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.last_actor_loss = torch.tensor(0.0, device=self.device)
        self.last_critic_loss = torch.tensor(0.0, device=self.device)

    @torch.no_grad()
    def select_action(self, state: torch.Tensor) -> torch.Tensor:
        """Select action from actor.

        Args:
            state: [num_envs, state_dim]

        Returns:
            action: [num_envs, action_dim] in [-1, 1]
        """

        state = state.to(self.device)
        action = self.actor(state)
        action = torch.clamp(action, -1.0, 1.0)
        return action

    def train(self, replay_buffer, batch_size: int):
        """One TD3 training update."""

        self.total_it += 1

        state, action, reward, next_state, done = replay_buffer.sample(batch_size)

        with torch.no_grad():
            noise = torch.randn_like(action) * self.policy_noise
            noise = torch.clamp(noise, -self.noise_clip, self.noise_clip)

            next_action = self.actor_target(next_state) + noise
            next_action = torch.clamp(next_action, -1.0, 1.0)

            target_q1, target_q2 = self.critic_target(next_state, next_action)
            target_q = torch.min(target_q1, target_q2)

            target_q = reward + (1.0 - done) * self.gamma * target_q

        current_q1, current_q2 = self.critic(state, action)

        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
        self.critic_optimizer.step()

        self.last_critic_loss = critic_loss.detach()

        if self.total_it % self.policy_delay == 0:
            actor_loss = -self.critic.q1_forward(state, self.actor(state)).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=2.0)
            self.actor_optimizer.step()

            self._soft_update(self.actor_target, self.actor)
            self._soft_update(self.critic_target, self.critic)

            self.last_actor_loss = actor_loss.detach()

        return {
            "critic_loss": float(self.last_critic_loss.detach().cpu()),
            "actor_loss": float(self.last_actor_loss.detach().cpu()),
        }

    def _soft_update(self, target_net: torch.nn.Module, source_net: torch.nn.Module):
        for target_param, source_param in zip(target_net.parameters(), source_net.parameters()):
            target_param.data.copy_(
                self.tau * source_param.data + (1.0 - self.tau) * target_param.data
            )

    def save(self, path: str):
        torch.save(
            {
                "actor": self.actor.state_dict(),
                "actor_target": self.actor_target.state_dict(),
                "critic": self.critic.state_dict(),
                "critic_target": self.critic_target.state_dict(),
                "actor_optimizer": self.actor_optimizer.state_dict(),
                "critic_optimizer": self.critic_optimizer.state_dict(),
                "total_it": self.total_it,
            },
            path,
        )

    def load(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)

        self.actor.load_state_dict(checkpoint["actor"])
        self.actor_target.load_state_dict(checkpoint["actor_target"])
        self.critic.load_state_dict(checkpoint["critic"])
        self.critic_target.load_state_dict(checkpoint["critic_target"])
        self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
        self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])
        self.total_it = checkpoint["total_it"]

    def load_actor_only(self, path: str):
        """Load actor weights only; critic stays freshly initialised.

        Use this when the reward scale changes between runs so that the old
        critic's Q-value estimates don't cause immediate divergence.
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint["actor"])
        self.actor_target.load_state_dict(checkpoint["actor_target"])
        self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
        self.total_it = checkpoint["total_it"]
        print(f"[TD3] Loaded actor only from {path}. Critic is fresh.")