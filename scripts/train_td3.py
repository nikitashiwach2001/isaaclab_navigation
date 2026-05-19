# scripts/train_td3.py

import argparse
import os
import sys
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from isaaclab.app import AppLauncher


# -------------------------
# CLI
# -------------------------
parser = argparse.ArgumentParser(description="Train TD3 on Isaac Lab TurtleBot navigation task.")

parser.add_argument("--task", type=str, required=True, help="Isaac Lab task name.")
parser.add_argument("--num_envs", type=int, default=4, help="Number of parallel envs.")
parser.add_argument("--disable_fabric", action="store_true", default=False)

parser.add_argument("--total_steps", type=int, default=200_000)
parser.add_argument("--start_steps", type=int, default=5_000)
parser.add_argument("--batch_size", type=int, default=256)
parser.add_argument("--buffer_size", type=int, default=500_000)

parser.add_argument("--hidden_dim", type=int, default=256)
parser.add_argument("--actor_lr", type=float, default=1e-4)
parser.add_argument("--critic_lr", type=float, default=1e-3)
parser.add_argument("--gamma", type=float, default=0.99)
parser.add_argument("--tau", type=float, default=0.005)

parser.add_argument("--policy_noise", type=float, default=0.2)
parser.add_argument("--noise_clip", type=float, default=0.5)
parser.add_argument("--policy_delay", type=int, default=2)
parser.add_argument("--expl_noise", type=float, default=0.1)
parser.add_argument("--updates_per_step", type=int, default=1,
                    help="Gradient updates per outer loop. Scale with num_envs to keep update-to-data ratio sane (rule: ~num_envs/16).")

parser.add_argument("--save_interval", type=int, default=25_000)

parser.add_argument("--run_name", type=str, default="td3_turtlebot_nav")
parser.add_argument("--load_checkpoint", type=str, default=None, help="Path to TD3 checkpoint to load for finetuning.")
parser.add_argument("--reset_critic", action="store_true", default=False, help="Load actor weights only; re-init critic fresh. Use when reward scale changes between runs.")
parser.add_argument("--use_gru", action="store_true", default=False, help="Use GRU actor for temporal memory.")
parser.add_argument("--use_conv", action="store_true", default=False, help="Use 1D conv lidar encoder (ConvActor/ConvCritic). Not weight-compatible with MLP checkpoints.")

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()


# -------------------------
# Launch Isaac Sim first
# -------------------------
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


# -------------------------
# Imports after SimulationApp
# -------------------------
import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

import isaaclab_scene.tasks  # noqa: F401

from td3.replay_buffer import ReplayBuffer
from td3.td3_agent import TD3Agent
from td3.noise import OUNoise


class TrainingLogger:
    """Mirrors every call to both stdout and a persistent log file."""

    def __init__(self, log_path: str):
        self._f = open(log_path, "w", buffering=1)

    def __call__(self, msg: str = ""):
        print(msg)
        self._f.write(msg + "\n")

    def close(self):
        self._f.close()


# -------------------------
# Terminal reward constants
# -------------------------
SUCCESS_REWARD    =  300.0
COLLISION_PENALTY =  200.0
TUMBLE_PENALTY    =  200.0
# A time limit is NOT a failure (Pardo et al. 2018). Penalising it as one,
# while ALSO cutting the bootstrap on truncation (see replay_buffer.add below),
# trained the critic that ~every state is worth ≈ -100 with no future.
# Speed is already incentivised by the per-step time cost + gamma discounting.
TIMEOUT_PENALTY   =  0.0


def get_env_buffer(env, name: str, num_envs: int, device) -> torch.Tensor:
    """Safely get a bool buffer from env.unwrapped."""
    return getattr(
        env.unwrapped,
        name,
        torch.zeros(num_envs, dtype=torch.bool, device=device),
    )


def pct(count: int, total: int) -> float:
    return 100.0 * count / max(total, 1)


def print_checkpoint_summary(
    *,
    global_step,
    ckpt_path,
    window_start_episode,
    episode_count,
    window_counts,
    window_reward_sum,
    window_step_sum,
    window_env_steps,
    window_action_linear_sum,
    window_action_angular_abs_sum,
    window_reward_step_sum,
    window_done_sum,
    last_losses,
    replay_buffer,
    window_start_time,
    logger,
):
    window_total = sum(window_counts.values())
    elapsed = max(time.time() - window_start_time, 1e-6)
    fps = window_env_steps / elapsed

    logger(f"\n========== CHECKPOINT SUMMARY | step {global_step} ==========")
    logger(f"Checkpoint: {ckpt_path}")
    logger(f"Replay buffer: {len(replay_buffer)}")
    logger(f"Window env steps: {window_env_steps}")
    logger(f"Training FPS: {fps:.1f}")

    if window_env_steps > 0:
        logger(f"Mean reward / env-step: {window_reward_step_sum / window_env_steps:.4f}")
        logger(f"Mean done / env-step: {window_done_sum / window_env_steps:.4f}")
        logger(f"Mean action linear: {window_action_linear_sum / window_env_steps:.4f}")
        logger(f"Mean |action angular|: {window_action_angular_abs_sum / window_env_steps:.4f}")

    logger(f"Critic loss: {last_losses.get('critic_loss', 0.0):.6f}")
    logger(f"Actor loss:  {last_losses.get('actor_loss', 0.0):.6f}")

    if window_total > 0:
        avg_reward = window_reward_sum / window_total
        avg_steps = window_step_sum / window_total

        logger("-" * 58)
        logger(f"Episodes in window: {window_total}  ({window_start_episode + 1} - {episode_count})")
        logger(f"Avg episode reward: {avg_reward:.2f}")
        logger(f"Avg episode steps:  {avg_steps:.1f}")

        total_coll = (
            window_counts['coll_dynamic'] + window_counts['coll_static']
            + window_counts['coll_boundary'] + window_counts['coll_unknown']
        )
        has_coll_subtypes = (
            window_counts['coll_dynamic'] > 0
            or window_counts['coll_static'] > 0
            or window_counts['coll_boundary'] > 0
        )

        logger(f"SUCCESS:       {window_counts['success']:<8} ({pct(window_counts['success'], window_total):.2f}%)")
        if has_coll_subtypes:
            logger(f"COLL_DYNAMIC:  {window_counts['coll_dynamic']:<8} ({pct(window_counts['coll_dynamic'], window_total):.2f}%)")
            logger(f"COLL_STATIC:   {window_counts['coll_static']:<8} ({pct(window_counts['coll_static'], window_total):.2f}%)")
            logger(f"COLL_BOUNDARY: {window_counts['coll_boundary']:<8} ({pct(window_counts['coll_boundary'], window_total):.2f}%)")
            if window_counts['coll_unknown'] > 0:
                logger(f"COLL_UNKNOWN:  {window_counts['coll_unknown']:<8} ({pct(window_counts['coll_unknown'], window_total):.2f}%)")
        else:
            logger(f"COLLISION:     {total_coll:<8} ({pct(total_coll, window_total):.2f}%)")
        logger(f"TIMEOUT:       {window_counts['timeout']:<8} ({pct(window_counts['timeout'], window_total):.2f}%)")
        logger(f"TUMBLE:        {window_counts['tumble']:<8} ({pct(window_counts['tumble'], window_total):.2f}%)")
        logger(f"TERMINATED:    {window_counts['terminated']:<8} ({pct(window_counts['terminated'], window_total):.2f}%)")
    else:
        logger("-" * 58)
        logger("No completed episodes in this checkpoint window.")

    logger("=" * 58 + "\n")


def main():
    device = args_cli.device

    env_cfg = parse_env_cfg(
        args_cli.task,
        device=device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )

    env = gym.make(args_cli.task, cfg=env_cfg)

    env.action_space = spaces.Box(
        low=-1.0,
        high=1.0,
        shape=env.action_space.shape,
        dtype=np.float32,
    )

    obs, _ = env.reset()
    state = obs["policy"]

    num_envs = env.unwrapped.num_envs
    state_dim = state.shape[1]
    action_dim = env.action_space.shape[1]

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join("logs", args_cli.run_name, timestamp)
    os.makedirs(save_dir, exist_ok=True)

    logger = TrainingLogger(os.path.join(save_dir, "training.log"))

    logger(f"[INFO] Saving checkpoints to: {save_dir}")
    logger(f"[INFO] Run name:     {args_cli.run_name}")
    logger(f"[INFO] Checkpoint:   {args_cli.load_checkpoint}")
    logger(f"[INFO] reset_critic: {args_cli.reset_critic}")
    logger(f"[INFO] Obs space:    {env.observation_space}")
    logger(f"[INFO] Action space: {env.action_space}")
    logger(f"[INFO] num_envs:     {num_envs}  state_dim: {state_dim}  action_dim: {action_dim}")
    logger(f"[INFO] actor_lr:     {args_cli.actor_lr}  critic_lr: {args_cli.critic_lr}")
    logger(f"[INFO] policy_delay: {args_cli.policy_delay}  tau: {args_cli.tau}  expl_noise: {args_cli.expl_noise}")
    logger(f"[INFO] batch_size:   {args_cli.batch_size}  buffer_size: {args_cli.buffer_size}  total_steps: {args_cli.total_steps}")
    logger(f"[INFO] use_conv:    {args_cli.use_conv}  use_gru: {args_cli.use_gru}")

    agent = TD3Agent(
        state_dim=state_dim,
        action_dim=action_dim,
        device=device,
        hidden_dim=args_cli.hidden_dim,
        actor_lr=args_cli.actor_lr,
        critic_lr=args_cli.critic_lr,
        gamma=args_cli.gamma,
        tau=args_cli.tau,
        policy_noise=args_cli.policy_noise,
        noise_clip=args_cli.noise_clip,
        policy_delay=args_cli.policy_delay,
        use_gru=args_cli.use_gru,
        use_conv=args_cli.use_conv,
    )

    if args_cli.load_checkpoint is not None:
        if args_cli.reset_critic:
            agent.load_actor_only(args_cli.load_checkpoint)
            logger(f"[INFO] Loaded ACTOR ONLY from: {args_cli.load_checkpoint} (critic is fresh)")
        else:
            agent.load(args_cli.load_checkpoint)
            logger(f"[INFO] Loaded full checkpoint: {args_cli.load_checkpoint}")

        # load_state_dict restores the saved lr, overriding what was passed to TD3Agent.
        # Force the CLI-specified LRs to take effect after loading.
        for pg in agent.actor_optimizer.param_groups:
            pg['lr'] = args_cli.actor_lr
        for pg in agent.critic_optimizer.param_groups:
            pg['lr'] = args_cli.critic_lr
        logger(f"[INFO] LR override after load — actor_lr: {args_cli.actor_lr}  critic_lr: {args_cli.critic_lr}")

    replay_buffer = ReplayBuffer(
        state_dim=state_dim,
        action_dim=action_dim,
        max_size=args_cli.buffer_size,
        device=device,
    )

    ou_noise = OUNoise(
        num_envs=num_envs,
        action_dim=action_dim,
        device=device,
        sigma=args_cli.expl_noise,
    )

    global_step = 0
    episode_count = 0

    episode_reward_sum = torch.zeros(num_envs, device=device)
    episode_step_count = torch.zeros(num_envs, dtype=torch.long, device=device)

    total_counts = {
        "success": 0,
        "coll_dynamic": 0,
        "coll_static": 0,
        "coll_boundary": 0,
        "coll_unknown": 0,
        "timeout": 0,
        "tumble": 0,
        "terminated": 0,
    }

    window_counts = {k: 0 for k in total_counts}

    window_reward_sum = 0.0
    window_step_sum = 0
    window_start_episode = 0

    window_env_steps = 0
    window_action_linear_sum = 0.0
    window_action_angular_abs_sum = 0.0
    window_reward_step_sum = 0.0
    window_done_sum = 0.0
    window_dyn_vicinity = 0   # env-steps with a moving obstacle within 1.0 m of the robot
    window_dyn_close = 0      # env-steps with a moving obstacle within 0.6 m of the robot
    window_start_time = time.time()

    # Cache moving-obstacle scene keys once (entities named obstacle_*)
    _dyn_obstacle_keys = [k for k in env.unwrapped.scene.keys() if k.startswith("obstacle_")]

    last_losses = {
        "critic_loss": 0.0,
        "actor_loss": 0.0,
    }

    # Heartbeat: compact health line every ~10 s of wall clock so you can
    # tail -f the log and see in real time whether training is healthy.
    HEARTBEAT_INTERVAL_S = 10.0
    hb_start_time = time.time()
    hb_env_steps = 0
    hb_reward_sum = 0.0
    hb_action_linear_sum = 0.0
    hb_action_angular_abs_sum = 0.0
    hb_episode_count = 0
    hb_counts = {k: 0 for k in total_counts}

    while simulation_app.is_running() and global_step < args_cli.total_steps:
        with torch.inference_mode():
            if global_step < args_cli.start_steps:
                action = torch.empty((num_envs, action_dim), device=device).uniform_(-1.0, 1.0)
            else:
                action = agent.select_action(state)

        if global_step >= args_cli.start_steps:
            noise = ou_noise.sample()
            action = torch.clamp(action + noise, -1.0, 1.0)

        with torch.inference_mode():
            next_obs, reward, terminated, truncated, info = env.step(action)
            next_state = next_obs["policy"]
            done = terminated | truncated

        goal_reached_buf = get_env_buffer(env, "goal_reached_buf", num_envs, device)
        collision_buf = get_env_buffer(env, "collision_buf", num_envs, device)
        tumble_buf = get_env_buffer(env, "tumble_buf", num_envs, device)

        collision_dynamic_buf = get_env_buffer(env, "collision_dynamic_buf", num_envs, device)
        collision_static_buf = get_env_buffer(env, "collision_static_buf", num_envs, device)
        collision_boundary_buf = get_env_buffer(env, "collision_boundary_buf", num_envs, device)

        reward = reward.clone()

        reward = torch.where(done & goal_reached_buf, reward + SUCCESS_REWARD, reward)
        reward = torch.where(done & collision_buf, reward - COLLISION_PENALTY, reward)
        reward = torch.where(done & tumble_buf, reward - TUMBLE_PENALTY, reward)
        if TIMEOUT_PENALTY != 0.0:
            reward = torch.where(truncated & ~goal_reached_buf & ~collision_buf & ~tumble_buf, reward - TIMEOUT_PENALTY, reward)

        # Bootstrap mask must use `terminated` ONLY, never `done`.
        # `terminated` = goal/collision/tumble (true terminals: no future, so
        # zero the bootstrap). `truncated` = time_out, an ARTIFICIAL cutoff —
        # the robot's future value is real, so it MUST bootstrap. Passing the
        # combined `done` here zeroed the bootstrap on every timeout; with
        # ~98-100% timeouts that severed value propagation and collapsed the
        # critic. `done` is still used below for episode bookkeeping only.
        replay_buffer.add(
            states=state,
            actions=action,
            rewards=reward,
            next_states=next_state,
            dones=terminated,
        )

        episode_reward_sum += reward
        episode_step_count += 1

        reward_sum_step = reward.sum().item()
        done_sum_step = done.sum().item()
        action_linear_sum_step = action[:, 0].sum().item()
        action_angular_abs_sum_step = torch.abs(action[:, 1]).sum().item() if action_dim > 1 else 0.0

        # Window step stats (used by checkpoint summary)
        window_env_steps += num_envs
        window_reward_step_sum += reward_sum_step
        window_done_sum += done_sum_step
        window_action_linear_sum += action_linear_sum_step
        window_action_angular_abs_sum += action_angular_abs_sum_step

        # Heartbeat accumulators (reset every HB print, ~10 s cadence)
        hb_env_steps += num_envs
        hb_reward_sum += reward_sum_step
        hb_action_linear_sum += action_linear_sum_step
        hb_action_angular_abs_sum += action_angular_abs_sum_step

        # Dynamic-obstacle exposure: distance robot ↔ nearest moving obstacle (entity positions, not lidar)
        if _dyn_obstacle_keys:
            robot_xy_w = env.unwrapped.scene["robot"].data.root_pos_w[:, :2]
            min_dyn_dist = torch.full((num_envs,), 1e6, device=device)
            for okey in _dyn_obstacle_keys:
                obs_xy_w = env.unwrapped.scene[okey].data.root_pos_w[:, :2]
                min_dyn_dist = torch.minimum(min_dyn_dist, torch.norm(robot_xy_w - obs_xy_w, dim=-1))
            window_dyn_vicinity += int((min_dyn_dist < 1.0).sum().item())
            window_dyn_close += int((min_dyn_dist < 0.6).sum().item())

        if done.any():
            done_env_ids = torch.where(done)[0]
            ou_noise.reset(done_env_ids)
            agent.reset_hidden(done_env_ids)

            goal_reached_buf = get_env_buffer(env, "goal_reached_buf", num_envs, device)
            collision_buf = get_env_buffer(env, "collision_buf", num_envs, device)
            tumble_buf = get_env_buffer(env, "tumble_buf", num_envs, device)

            collision_dynamic_buf = get_env_buffer(env, "collision_dynamic_buf", num_envs, device)
            collision_static_buf = get_env_buffer(env, "collision_static_buf", num_envs, device)
            collision_boundary_buf = get_env_buffer(env, "collision_boundary_buf", num_envs, device)

            for env_id in done_env_ids.tolist():
                episode_count += 1

                epi_reward = episode_reward_sum[env_id].item()
                epi_steps = episode_step_count[env_id].item()

                if truncated[env_id]:
                    outcome = "timeout"

                elif goal_reached_buf[env_id]:
                    outcome = "success"

                elif collision_buf[env_id]:
                    if collision_dynamic_buf[env_id]:
                        outcome = "coll_dynamic"
                    elif collision_static_buf[env_id]:
                        outcome = "coll_static"
                    elif collision_boundary_buf[env_id]:
                        outcome = "coll_boundary"
                    else:
                        outcome = "coll_unknown"

                elif tumble_buf[env_id]:
                    outcome = "tumble"

                else:
                    outcome = "terminated"

                total_counts[outcome] += 1
                window_counts[outcome] += 1
                hb_counts[outcome] += 1
                hb_episode_count += 1

                window_reward_sum += epi_reward
                window_step_sum += epi_steps

                episode_reward_sum[env_id] = 0.0
                episode_step_count[env_id] = 0

        state = next_state
        global_step += num_envs

        if len(replay_buffer) >= args_cli.batch_size and global_step >= args_cli.start_steps:
            for _ in range(args_cli.updates_per_step):
                losses = agent.train(replay_buffer, args_cli.batch_size)
                if losses is not None:
                    last_losses = losses

        if global_step % args_cli.save_interval < num_envs:
            ckpt_path = os.path.join(save_dir, f"td3_step_{global_step}.pt")
            agent.save(ckpt_path)

            print_checkpoint_summary(
                global_step=global_step,
                ckpt_path=ckpt_path,
                window_start_episode=window_start_episode,
                episode_count=episode_count,
                window_counts=window_counts,
                window_reward_sum=window_reward_sum,
                window_step_sum=window_step_sum,
                window_env_steps=window_env_steps,
                window_action_linear_sum=window_action_linear_sum,
                window_action_angular_abs_sum=window_action_angular_abs_sum,
                window_reward_step_sum=window_reward_step_sum,
                window_done_sum=window_done_sum,
                last_losses=last_losses,
                replay_buffer=replay_buffer,
                window_start_time=window_start_time,
                logger=logger,
            )

            if window_env_steps > 0:
                logger(f"Dynamic-obstacle exposure | vicinity (<1.0m): "
                       f"{100.0 * window_dyn_vicinity / window_env_steps:.2f}%   "
                       f"close (<0.6m): {100.0 * window_dyn_close / window_env_steps:.2f}%")

            window_counts = {k: 0 for k in total_counts}
            window_reward_sum = 0.0
            window_step_sum = 0
            window_start_episode = episode_count

            window_env_steps = 0
            window_action_linear_sum = 0.0
            window_action_angular_abs_sum = 0.0
            window_reward_step_sum = 0.0
            window_done_sum = 0.0
            window_dyn_vicinity = 0
            window_dyn_close = 0
            window_start_time = time.time()

        # Heartbeat: print compact health line every ~10 s of wall clock.
        # Tail -f the training.log to watch in real time. Key signals:
        #   lin > 0 → policy is moving forward (not crawling)
        #   c_loss stable / not spiking 10× → critic is converging
        #   r/s trending positive → reward signal is healthy
        #   SR climbing checkpoint-to-checkpoint → policy is improving
        now = time.time()
        hb_elapsed = now - hb_start_time
        if hb_elapsed >= HEARTBEAT_INTERVAL_S:
            fps = hb_env_steps / max(hb_elapsed, 1e-6)
            if hb_env_steps > 0:
                r_step = hb_reward_sum / hb_env_steps
                lin_mean = hb_action_linear_sum / hb_env_steps
                ang_mean = hb_action_angular_abs_sum / hb_env_steps
            else:
                r_step = lin_mean = ang_mean = 0.0
            ep = max(hb_episode_count, 1)
            sr = 100.0 * hb_counts['success'] / ep
            cs = 100.0 * hb_counts['coll_static'] / ep
            cd = 100.0 * hb_counts['coll_dynamic'] / ep
            to = 100.0 * hb_counts['timeout'] / ep

            logger(
                f"[HB] step={global_step:>9d} fps={fps:>5.0f} "
                f"r/s={r_step:+.3f} lin={lin_mean:+.3f} |ang|={ang_mean:.3f} "
                f"c={last_losses['critic_loss']:8.3f} a={last_losses['actor_loss']:+8.2f} "
                f"| epi={hb_episode_count:>4d} SR={sr:5.1f}% "
                f"S={cs:4.1f}% D={cd:4.1f}% T={to:4.1f}%"
            )

            hb_start_time = now
            hb_env_steps = 0
            hb_reward_sum = 0.0
            hb_action_linear_sum = 0.0
            hb_action_angular_abs_sum = 0.0
            hb_episode_count = 0
            hb_counts = {k: 0 for k in total_counts}

    total_finished = max(episode_count, 1)

    total_coll_final = (
        total_counts['coll_dynamic'] + total_counts['coll_static']
        + total_counts['coll_boundary'] + total_counts['coll_unknown']
    )
    has_coll_subtypes_final = (
        total_counts['coll_dynamic'] > 0
        or total_counts['coll_static'] > 0
        or total_counts['coll_boundary'] > 0
    )

    logger("\n========== TRAINING SUMMARY ==========")
    logger(f"Total episodes: {episode_count}")
    logger(f"SUCCESS:       {total_counts['success']:<8} ({pct(total_counts['success'], total_finished):.2f}%)")
    if has_coll_subtypes_final:
        logger(f"COLL_DYNAMIC:  {total_counts['coll_dynamic']:<8} ({pct(total_counts['coll_dynamic'], total_finished):.2f}%)")
        logger(f"COLL_STATIC:   {total_counts['coll_static']:<8} ({pct(total_counts['coll_static'], total_finished):.2f}%)")
        logger(f"COLL_BOUNDARY: {total_counts['coll_boundary']:<8} ({pct(total_counts['coll_boundary'], total_finished):.2f}%)")
        if total_counts['coll_unknown'] > 0:
            logger(f"COLL_UNKNOWN:  {total_counts['coll_unknown']:<8} ({pct(total_counts['coll_unknown'], total_finished):.2f}%)")
    else:
        logger(f"COLLISION:     {total_coll_final:<8} ({pct(total_coll_final, total_finished):.2f}%)")
    logger(f"TIMEOUT:       {total_counts['timeout']:<8} ({pct(total_counts['timeout'], total_finished):.2f}%)")
    logger(f"TUMBLE:        {total_counts['tumble']:<8} ({pct(total_counts['tumble'], total_finished):.2f}%)")
    logger(f"TERMINATED:    {total_counts['terminated']:<8} ({pct(total_counts['terminated'], total_finished):.2f}%)")
    logger("======================================\n")

    final_path = os.path.join(save_dir, "td3_final.pt")
    agent.save(final_path)
    logger(f"[INFO] Saved final checkpoint: {final_path}")

    logger.close()
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()