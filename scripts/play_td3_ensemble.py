# scripts/play_td3_ensemble.py
#
# Inference-time ensemble evaluator.
# Loads multiple TD3 checkpoints and averages their action outputs every step.
# Use to combine two or three policies that each peak around the same SR
# but have different per-state failure modes — averaging smooths the failures.

import argparse
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Ensemble eval of multiple TD3 policies. Actions are averaged across checkpoints each step.")

parser.add_argument("--task", type=str, required=True)
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument(
    "--checkpoints", type=str, nargs="+", required=True,
    help="One or more checkpoint paths. Actions are averaged uniformly across all provided checkpoints.",
)
parser.add_argument("--disable_fabric", action="store_true", default=False)
parser.add_argument("--hidden_dim", type=int, default=256)
parser.add_argument("--eval_episodes", type=int, default=100)
parser.add_argument("--use_gru", action="store_true", default=False)
parser.add_argument("--use_conv", action="store_true", default=False)
parser.add_argument("--seed", type=int, default=0, help="Deterministic eval seed.")
parser.add_argument(
    "--mode", type=str, default="mean", choices=["mean", "qselect"],
    help="mean: average actions across checkpoints. "
         "qselect: each critic scores every proposed action, pick the best per env.",
)

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()


app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

import isaaclab_scene.tasks  # noqa: F401

from td3.td3_agent import TD3Agent


def get_env_buffer(env, name: str, num_envs: int, device) -> torch.Tensor:
    return getattr(
        env.unwrapped,
        name,
        torch.zeros(num_envs, dtype=torch.bool, device=device),
    )


def print_eval_summary(label: str, episode_count: int, counts: dict):
    total = max(episode_count, 1)
    pct = lambda n: f"{100.0 * n / total:.2f}%"

    print("\n========== ENSEMBLE EVALUATION SUMMARY ==========")
    print(f"Ensemble: {label}")
    print(f"Total episodes: {episode_count}")
    print(f"SUCCESS:        {counts['success']:<8} ({pct(counts['success'])})")
    print(f"COLL_DYNAMIC:   {counts['coll_dynamic']:<8} ({pct(counts['coll_dynamic'])})")
    print(f"COLL_STATIC:    {counts['coll_static']:<8} ({pct(counts['coll_static'])})")
    print(f"COLL_BOUNDARY:  {counts['coll_boundary']:<8} ({pct(counts['coll_boundary'])})")
    print(f"TIMEOUT:        {counts['timeout']:<8} ({pct(counts['timeout'])})")
    print(f"TUMBLE:         {counts['tumble']:<8} ({pct(counts['tumble'])})")
    print(f"TERMINATED:     {counts['terminated']:<8} ({pct(counts['terminated'])})")
    print("=================================================\n")


def ensemble_select_action(agents, state):
    """Mean of per-agent actions. Each agent.select_action returns shape (num_envs, action_dim)."""
    actions = [agent.select_action(state) for agent in agents]
    stacked = torch.stack(actions, dim=0)  # (K, num_envs, action_dim)
    mean_action = stacked.mean(dim=0)
    return torch.clamp(mean_action, -1.0, 1.0)


def qselect_action(agents, state):
    """Q-selection ensemble.

    Each agent proposes an action. Then every agent's critic scores every
    proposed action. Per env, the action with the best cross-critic score wins.

    Scale handling: critics from different runs have different Q magnitudes.
    Each critic's scores are mean-centered across the candidate actions (removes
    per-critic baseline) and divided by that critic's overall Q spread (balances
    critics with different scales). Then summed across critics."""
    n = state.shape[0]
    k = len(agents)

    proposals = torch.stack([agent.select_action(state) for agent in agents], dim=0)  # (K, N, act)

    # scores[critic_j, action_i] = Q_j(state, proposal_i)
    scores = torch.empty((k, k, n), device=state.device)
    for j, agent in enumerate(agents):
        for i in range(k):
            q = agent.critic.q1_forward(state, proposals[i])
            scores[j, i] = q.reshape(n)

    centered = scores - scores.mean(dim=1, keepdim=True)               # remove per-critic baseline
    scale = scores.std(dim=(1, 2), keepdim=True).clamp(min=1e-6)        # per-critic spread
    total = (centered / scale).sum(dim=0)                              # (K_actions, N)

    best_idx = total.argmax(dim=0)                                     # (N,)
    chosen = proposals[best_idx, torch.arange(n, device=state.device)] # (N, act)
    return torch.clamp(chosen, -1.0, 1.0)


def main():
    device = args_cli.device

    torch.manual_seed(args_cli.seed)
    np.random.seed(args_cli.seed)

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

    GOAL_DIST_IDX  = 270 + 8
    GOAL_ANGLE_IDX = 270 + 8 + 1
    MAX_GOAL_DIST  = 7.07106781187

    # -------------------------
    # Load every checkpoint into its own TD3Agent
    # -------------------------
    agents = []
    for ckpt_path in args_cli.checkpoints:
        agent = TD3Agent(
            state_dim=state_dim,
            action_dim=action_dim,
            device=device,
            hidden_dim=args_cli.hidden_dim,
            use_gru=args_cli.use_gru,
            use_conv=args_cli.use_conv,
        )
        agent.load(ckpt_path)
        agent.actor.eval()
        agent.critic.eval()
        agents.append(agent)
        print(f"[INFO] Loaded checkpoint: {ckpt_path}")

    print(f"[INFO] Ensemble size: {len(agents)}  mode: {args_cli.mode}")
    print(f"[INFO] state_dim: {state_dim}  action_dim: {action_dim}  hidden_dim: {args_cli.hidden_dim}")
    print(f"[INFO] num_envs: {num_envs}  eval_episodes: {args_cli.eval_episodes}  seed: {args_cli.seed}")

    ensemble_label = " + ".join(os.path.basename(os.path.dirname(p)) + "/" + os.path.basename(p) for p in args_cli.checkpoints)

    episode_reward_sum = torch.zeros(num_envs, device=device)
    episode_step_count = torch.zeros(num_envs, dtype=torch.long, device=device)

    episode_count = 0
    total_steps   = 0

    counts = dict(
        success=0,
        coll_dynamic=0,
        coll_static=0,
        coll_boundary=0,
        timeout=0,
        tumble=0,
        terminated=0,
    )

    from collections import defaultdict
    per_goal_counts = defaultdict(lambda: defaultdict(int))

    while simulation_app.is_running():
        with torch.inference_mode():
            if args_cli.mode == "qselect":
                action = qselect_action(agents, state)
            else:
                action = ensemble_select_action(agents, state)

            obs, reward, terminated, truncated, _ = env.step(action)
            state = obs["policy"]

            done = terminated | truncated

            episode_reward_sum += reward
            episode_step_count += 1
            total_steps += num_envs

            if done.any():
                done_env_ids = torch.where(done)[0]
                for agent in agents:
                    agent.reset_hidden(done_env_ids)

                goal_reached_buf       = get_env_buffer(env, "goal_reached_buf",       num_envs, device)
                collision_buf          = get_env_buffer(env, "collision_buf",           num_envs, device)
                collision_dynamic_buf  = get_env_buffer(env, "collision_dynamic_buf",   num_envs, device)
                collision_static_buf   = get_env_buffer(env, "collision_static_buf",    num_envs, device)
                collision_boundary_buf = get_env_buffer(env, "collision_boundary_buf",  num_envs, device)
                tumble_buf             = get_env_buffer(env, "tumble_buf",              num_envs, device)

                env_origins = env.unwrapped.scene.env_origins[:, :2]
                goal_pos_w  = env.unwrapped.goal_pos_w
                goal_local  = goal_pos_w - env_origins

                for env_id in done_env_ids.tolist():
                    episode_count += 1

                    epi_reward = episode_reward_sum[env_id].item()
                    epi_steps  = episode_step_count[env_id].item()

                    goal_dist_real  = state[env_id, GOAL_DIST_IDX].item()  * MAX_GOAL_DIST
                    goal_angle_norm = state[env_id, GOAL_ANGLE_IDX].item()

                    gx = round(goal_local[env_id, 0].item(), 1)
                    gy = round(goal_local[env_id, 1].item(), 1)
                    goal_key = (gx, gy)

                    if goal_reached_buf[env_id]:
                        outcome = "SUCCESS"
                        counts["success"] += 1
                    elif collision_dynamic_buf[env_id]:
                        outcome = "COLL_DYNAMIC"
                        counts["coll_dynamic"] += 1
                    elif collision_boundary_buf[env_id]:
                        outcome = "COLL_BOUNDARY"
                        counts["coll_boundary"] += 1
                    elif collision_static_buf[env_id]:
                        outcome = "COLL_STATIC"
                        counts["coll_static"] += 1
                    elif collision_buf[env_id]:
                        outcome = "COLL_STATIC"
                        counts["coll_static"] += 1
                    elif tumble_buf[env_id]:
                        outcome = "TUMBLE"
                        counts["tumble"] += 1
                    elif truncated[env_id]:
                        outcome = "TIMEOUT"
                        counts["timeout"] += 1
                    else:
                        outcome = "TERMINATED"
                        counts["terminated"] += 1

                    per_goal_counts[goal_key]["total"] += 1
                    per_goal_counts[goal_key][outcome] += 1

                    print(
                        f"Epi: {episode_count:<5} "
                        f"env: {env_id:<3} "
                        f"R: {epi_reward:<8.0f} "
                        f"outcome: {outcome:<14} "
                        f"steps: {epi_steps:<6} "
                        f"goal_dist: {goal_dist_real:<6.3f} "
                        f"goal_angle: {goal_angle_norm:<6.3f}"
                    )

                    episode_reward_sum[env_id] = 0.0
                    episode_step_count[env_id] = 0

                    if episode_count >= args_cli.eval_episodes:
                        print_eval_summary(ensemble_label, episode_count, counts)

                        print("\n========== PER-GOAL BREAKDOWN ==========")
                        print(f"{'goal (x,y)':<14} {'N':>4} {'SR':>7} {'STATIC':>7} {'DYN':>5} {'TIMEOUT':>8}")
                        for g, cs in sorted(per_goal_counts.items(), key=lambda kv: kv[1].get("SUCCESS", 0) / max(kv[1]["total"], 1)):
                            n = cs["total"]
                            sr = 100.0 * cs.get("SUCCESS", 0) / max(n, 1)
                            cs_static = 100.0 * cs.get("COLL_STATIC", 0) / max(n, 1)
                            cs_dyn = 100.0 * cs.get("COLL_DYNAMIC", 0) / max(n, 1)
                            cs_to = 100.0 * cs.get("TIMEOUT", 0) / max(n, 1)
                            print(f"({g[0]:>4.1f},{g[1]:>4.1f})  {n:>4} {sr:>6.1f}% {cs_static:>6.1f}% {cs_dyn:>4.1f}% {cs_to:>7.1f}%")
                        print("========================================\n")

                        env.close()
                        return

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
