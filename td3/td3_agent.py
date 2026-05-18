import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

from td3.actor_critic import Actor, ActorGRU, Critic


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
        use_gru: bool = False,
    ):
        self.device = torch.device(device)

        self.state_dim = state_dim
        self.action_dim = action_dim

        self.gamma = gamma
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_delay = policy_delay
        self.use_gru = use_gru

        self.total_it = 0

        if use_gru:
            self.actor = ActorGRU(state_dim, action_dim, hidden_dim).to(self.device)
        else:
            self.actor = Actor(state_dim, action_dim, hidden_dim).to(self.device)
        self.actor_target = copy.deepcopy(self.actor)

        self.critic = Critic(state_dim, action_dim, hidden_dim).to(self.device)
        self.critic_target = copy.deepcopy(self.critic)

        if use_gru:
            # Freeze the MLP backbone (fc1/fc2/fc3) — only GRU and gru_gate are trained.
            # This prevents the speed-collapse that corrupts fc3 during fine-tuning
            # and lets the GRU learn temporal corrections on top of the frozen v6 policy.
            for name, param in self.actor.named_parameters():
                if "gru" not in name:
                    param.requires_grad = False
            gru_params = [p for p in self.actor.parameters() if p.requires_grad]
            self.actor_optimizer = torch.optim.Adam(gru_params, lr=actor_lr)
        else:
            self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.last_actor_loss = torch.tensor(0.0, device=self.device)
        self.last_critic_loss = torch.tensor(0.0, device=self.device)

        # Per-env GRU hidden state — initialized lazily on first select_action call
        self._actor_hidden: torch.Tensor | None = None

    def reset_hidden(self, env_ids):
        """Zero out the GRU hidden state for the given environment indices."""
        if self._actor_hidden is not None and len(env_ids) > 0:
            # Clone to convert inference tensor → regular tensor before inplace update
            self._actor_hidden = self._actor_hidden.clone()
            self._actor_hidden[env_ids] = 0.0

    @torch.no_grad()
    def select_action(self, state: torch.Tensor) -> torch.Tensor:
        """Select action from actor.

        Args:
            state: [num_envs, state_dim]

        Returns:
            action: [num_envs, action_dim] in [-1, 1]
        """
        state = state.to(self.device)

        if self.use_gru:
            assert isinstance(self.actor, ActorGRU)
            if self._actor_hidden is None or self._actor_hidden.shape[0] != state.shape[0]:
                self._actor_hidden = torch.zeros(
                    state.shape[0], self.actor.hidden_dim, device=self.device
                )
            action, self._actor_hidden = self.actor(state, self._actor_hidden)
        else:
            assert isinstance(self.actor, Actor)
            action = self.actor(state)

        action = torch.clamp(action, -1.0, 1.0)
        return action

    def train(self, replay_buffer, batch_size: int, freeze_actor: bool = False):
        """One TD3 training update.

        If freeze_actor=True, only the critic is updated this step. Useful for
        the warm-up phase after --reset_critic so the critic learns the new
        reward's Q-landscape under the loaded policy without the actor
        drifting toward random-Q gradients.
        """

        self.total_it += 1

        state, action, reward, next_state, done = replay_buffer.sample(batch_size)

        with torch.no_grad():
            noise = torch.randn_like(action) * self.policy_noise
            noise = torch.clamp(noise, -self.noise_clip, self.noise_clip)

            if self.use_gru:
                assert isinstance(self.actor_target, ActorGRU)
                next_action, _ = self.actor_target(next_state, hidden=None)
            else:
                assert isinstance(self.actor_target, Actor)
                next_action = self.actor_target(next_state)
            next_action = torch.clamp(next_action + noise, -1.0, 1.0)

            target_q1, target_q2 = self.critic_target(next_state, next_action)
            target_q = torch.min(target_q1, target_q2)

            target_q = reward + (1.0 - done) * self.gamma * target_q

        current_q1, current_q2 = self.critic(state, action)

        # Huber loss (delta=20): quadratic for |error|<20, linear above.
        # Prevents spike when a batch contains many terminal collision events (-200 reward)
        # which would create TD errors of 200+ and explosive MSE gradients.
        critic_loss = F.huber_loss(current_q1, target_q, delta=20.0) + \
                      F.huber_loss(current_q2, target_q, delta=20.0)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=0.3)
        self.critic_optimizer.step()

        self.last_critic_loss = critic_loss.detach()

        if self.total_it % self.policy_delay == 0 and not freeze_actor:
            if self.use_gru:
                assert isinstance(self.actor, ActorGRU)
                actor_out, _ = self.actor(state, hidden=None)
            else:
                assert isinstance(self.actor, Actor)
                actor_out = self.actor(state)

            actor_loss = -self.critic.q1_forward(state, actor_out).mean()

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
                "use_gru": self.use_gru,
            },
            path,
        )

    def load(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)

        # If checkpoint architecture differs (e.g. MLP → GRU), load FC weights only
        ckpt_use_gru = checkpoint.get("use_gru", False)
        strict = (ckpt_use_gru == self.use_gru)

        self.actor.load_state_dict(checkpoint["actor"], strict=strict)
        self.actor_target.load_state_dict(checkpoint["actor_target"], strict=strict)
        self.critic.load_state_dict(checkpoint["critic"])
        self.critic_target.load_state_dict(checkpoint["critic_target"])

        self.total_it = checkpoint.get("total_it", 0)

        if strict:
            self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
            self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])
        else:
            print("[TD3] Partial load: GRU weights init randomly, FC weights from checkpoint.")

    def load_actor_only(self, path: str):
        """Load actor weights only; critic stays freshly initialised."""
        checkpoint = torch.load(path, map_location=self.device)

        ckpt_use_gru = checkpoint.get("use_gru", False)
        strict = (ckpt_use_gru == self.use_gru)

        self.actor.load_state_dict(checkpoint["actor"], strict=strict)
        self.actor_target.load_state_dict(checkpoint["actor_target"], strict=strict)

        self.total_it = checkpoint.get("total_it", 0)

        if strict:
            self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])

        if not strict:
            print("[TD3] Partial load: GRU weights init randomly, FC weights from checkpoint.")
        print(f"[TD3] Loaded actor only from {path}. Critic is fresh.")
