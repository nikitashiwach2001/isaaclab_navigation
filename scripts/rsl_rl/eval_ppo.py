"""Evaluate a trained RSL-RL PPO policy — same summary format as play_td3.py."""

import argparse
import os
import sys

from isaaclab.app import AppLauncher
import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Evaluate RSL-RL PPO policy over N episodes.")
parser.add_argument("--task",          type=str, required=True)
parser.add_argument("--num_envs",      type=int, default=32)
parser.add_argument("--eval_episodes", type=int, default=100)
parser.add_argument("--agent",         type=str, default="rsl_rl_cfg_entry_point")

cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ── post-launch imports ───────────────────────────────────────────────────────
import gymnasium as gym
import torch
from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import ManagerBasedRLEnvCfg, DirectRLEnvCfg, DirectMARLEnvCfg

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import isaaclab_scene.tasks  # noqa: F401


# ── helpers ───────────────────────────────────────────────────────────────────

def _get_buf(env, name, num_envs, device):
    return getattr(env.unwrapped, name,
                   torch.zeros(num_envs, dtype=torch.bool, device=device))


def _print_summary(checkpoint, episode_count, counts):
    total = max(episode_count, 1)
    pct = lambda n: f"{100.0 * n / total:.2f}%"
    print("\n========== EVALUATION SUMMARY ==========")
    print(f"Checkpoint: {checkpoint}")
    print(f"Total episodes: {episode_count}")
    print(f"SUCCESS:        {counts['success']:<8} ({pct(counts['success'])})")
    print(f"COLL_DYNAMIC:   {counts['coll_dynamic']:<8} ({pct(counts['coll_dynamic'])})")
    print(f"COLL_STATIC:    {counts['coll_static']:<8} ({pct(counts['coll_static'])})")
    print(f"COLL_BOUNDARY:  {counts['coll_boundary']:<8} ({pct(counts['coll_boundary'])})")
    print(f"TIMEOUT:        {counts['timeout']:<8} ({pct(counts['timeout'])})")
    print(f"TUMBLE:         {counts['tumble']:<8} ({pct(counts['tumble'])})")
    print("========================================\n")


# ── main ──────────────────────────────────────────────────────────────────────

@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg,
         agent_cfg: RslRlBaseRunnerCfg):

    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = agent_cfg.seed

    device = args_cli.device if args_cli.device else "cuda"

    # ── resolve checkpoint ────────────────────────────────────────────────────
    log_root = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    _ckpt = agent_cfg.load_checkpoint
    if _ckpt and os.path.isfile(_ckpt):
        resume_path = _ckpt                                    # direct absolute path
    else:
        resume_path = get_checkpoint_path(log_root, agent_cfg.load_run, _ckpt)

    print(f"[INFO] task:       {args_cli.task}")
    print(f"[INFO] checkpoint: {resume_path}")
    print(f"[INFO] num_envs:   {args_cli.num_envs}")
    print(f"[INFO] episodes:   {args_cli.eval_episodes}")

    # ── build env ─────────────────────────────────────────────────────────────
    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    num_envs = env.unwrapped.num_envs

    # ── load policy ───────────────────────────────────────────────────────────
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=device)
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=device)

    try:
        policy_nn = runner.alg.policy
    except AttributeError:
        policy_nn = runner.alg.actor_critic

    # ── eval buffers ──────────────────────────────────────────────────────────
    ep_reward = torch.zeros(num_envs, device=device)
    ep_steps  = torch.zeros(num_envs, dtype=torch.long, device=device)
    episode_count = 0
    counts = dict(success=0, coll_dynamic=0, coll_static=0,
                  coll_boundary=0, timeout=0, tumble=0)

    obs, _ = env.reset()

    # ── evaluation loop ───────────────────────────────────────────────────────
    while simulation_app.is_running():
        with torch.inference_mode():
            actions              = policy(obs)
            obs, rewards, dones, extras = env.step(actions)
            time_outs            = extras.get("time_outs", torch.zeros_like(dones))

            policy_nn.reset(dones)

            ep_reward += rewards
            ep_steps  += 1

            if dones.any():
                done_ids   = torch.where(dones)[0]
                goal_buf   = _get_buf(env, "goal_reached_buf",       num_envs, device)
                coll_buf   = _get_buf(env, "collision_buf",           num_envs, device)
                coll_dyn   = _get_buf(env, "collision_dynamic_buf",   num_envs, device)
                coll_sta   = _get_buf(env, "collision_static_buf",    num_envs, device)
                coll_bnd   = _get_buf(env, "collision_boundary_buf",  num_envs, device)
                tumble_buf = _get_buf(env, "tumble_buf",              num_envs, device)

                for eid in done_ids.tolist():
                    episode_count += 1

                    if goal_buf[eid]:
                        outcome = "SUCCESS";       counts["success"]      += 1
                    elif coll_dyn[eid]:
                        outcome = "COLL_DYNAMIC";  counts["coll_dynamic"] += 1
                    elif coll_bnd[eid]:
                        outcome = "COLL_BOUNDARY"; counts["coll_boundary"]+= 1
                    elif coll_sta[eid] or coll_buf[eid]:
                        outcome = "COLL_STATIC";   counts["coll_static"]  += 1
                    elif tumble_buf[eid]:
                        outcome = "TUMBLE";        counts["tumble"]       += 1
                    elif time_outs[eid]:
                        outcome = "TIMEOUT";       counts["timeout"]      += 1
                    else:
                        outcome = "TIMEOUT";       counts["timeout"]      += 1

                    print(
                        f"Epi: {episode_count:<5} "
                        f"env: {eid:<3} "
                        f"R: {ep_reward[eid].item():<8.1f} "
                        f"outcome: {outcome:<14} "
                        f"steps: {ep_steps[eid].item():<6}"
                    )

                    ep_reward[eid] = 0.0
                    ep_steps[eid]  = 0

                    if episode_count >= args_cli.eval_episodes:
                        _print_summary(resume_path, episode_count, counts)
                        env.close()
                        return

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
