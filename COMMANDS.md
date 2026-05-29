# Commands — training, evaluation, diagnostics

Quick reference for the commands used in this project. Fill in `<checkpoint>`
with the path you want. Training code is never edited — difficulty and modes
are all controlled by the env vars listed at the bottom.

Current Stage 5 deliverable checkpoint:
`logs/may_21/stage5_mixedspawn_finetune_ext/20260521_061213/td3_step_11500032.pt`

---

## 1. Training / finetuning

### Finetune (e.g. more dynamic-obstacle avoidance)
Resume from a checkpoint and keep training. `OBSTACLE_SPEED_SCALE=1.0` is full
obstacle speed (hardest); raise it (2.0, 3.0) for an easier curriculum step.
Drop `--reset_critic` when continuing the same reward/obs; add it only if the
reward or the privileged-obs dimension changed.

```bash
OBSTACLE_SPEED_SCALE=1.0 python scripts/train_td3.py \
  --task IsaaclabScene-Stage5-MixedSpawn-v0 \
  --num_envs 128 \
  --total_steps 10000000 \
  --start_steps 2000 \
  --batch_size 512 \
  --buffer_size 2000000 \
  --hidden_dim 512 \
  --actor_lr 0.00005 \
  --critic_lr 0.001 \
  --gamma 0.99 --tau 0.005 \
  --policy_noise 0.2 --noise_clip 0.5 --policy_delay 4 --expl_noise 0.15 \
  --save_interval 500000 \
  --use_conv --privileged_critic \
  --load_checkpoint <checkpoint> \
  --run_name may_XX/stage5_dynobj_finetune \
  --headless
```

Training keeps `STAGE5_OBSTACLE_PHASE_RESET` ON (default) — per-episode phase
randomization is wanted for training.

### Stage 7 curriculum — train from scratch, one obstacle at a time
Trained from scratch (no warm-start) so the policy isn't dragged by another
stage's habits. Curriculum steps:

**Step 1 — single obstacle (diagnostic that we know works, 91% eval):**
```bash
STAGE7_ACTIVE_OBSTACLES="2" STAGE7_SPEED_RANDOMIZE=0 OBSTACLE_SPEED_SCALE=4.0 \
python scripts/train_td3.py \
  --task IsaaclabScene-Stage7-v0 \
  --num_envs 128 --total_steps 3500000 \
  --start_steps 25000 \
  --hidden_dim 512 --use_conv --privileged_critic \
  --actor_lr 0.00005 --critic_lr 0.001 \
  --batch_size 512 --buffer_size 2000000 \
  --policy_noise 0.2 --noise_clip 0.5 --policy_delay 4 --expl_noise 0.15 \
  --save_interval 250000 \
  --run_name may_XX/stage7_curric_1obs \
  --headless
```

**Step 2 — add obstacle_3 (corridor exit). Warm-start from Step 1's best ckpt;
do NOT reset critic (same env type, just one more crossing):**
```bash
LIDAR_STACK_FRAMES=12 STAGE7_ACTIVE_OBSTACLES="2,3" STAGE7_SPEED_RANDOMIZE=0 OBSTACLE_SPEED_SCALE=4.0 \
python scripts/train_td3.py \
  --task IsaaclabScene-Stage7-v0 \
  --num_envs 128 --total_steps 2000000 \
  --start_steps 2000 \
  --hidden_dim 512 --use_conv --privileged_critic \
  --actor_lr 0.00003 --critic_lr 0.001 \
  --batch_size 512 --buffer_size 2000000 \
  --policy_noise 0.2 --noise_clip 0.5 --policy_delay 4 --expl_noise 0.15 \
  --save_interval 250000 \
  --load_checkpoint <step1_best_checkpoint> \
  --run_name may_XX/stage7_curric_2obs \
  --headless
```

**Step 2.5 — stabilize 2-obstacle policy + small speed bump (4.0 → 3.0).** Same
two obstacles, but harder crossings. Tighter actor LR and lower exploration
noise to break the oscillation seen in Step 2. Don't add obstacle_1 until this
sits above ~80% stable.
```bash
STAGE7_ACTIVE_OBSTACLES="2,3" STAGE7_SPEED_RANDOMIZE=0 OBSTACLE_SPEED_SCALE=3.0 \
python scripts/train_td3.py \
  --task IsaaclabScene-Stage7-v0 \
  --num_envs 128 --total_steps 1500000 \
  --start_steps 2000 \
  --hidden_dim 512 --use_conv --privileged_critic \
  --actor_lr 0.00002 --critic_lr 0.001 \
  --batch_size 512 --buffer_size 2000000 \
  --policy_noise 0.2 --noise_clip 0.5 --policy_delay 4 --expl_noise 0.10 \
  --save_interval 250000 \
  --load_checkpoint <step2_best_checkpoint> \
  --run_name may_XX/stage7_curric_2obs_speed3 \
  --headless
```

**Step 3 — add obstacle_1 (closest to spawn, hardest). Same pattern, warm-start
from Step 2:**
```bash
LIDAR_STACK_FRAMES=12 STAGE7_ACTIVE_OBSTACLES="1,2,3" STAGE7_SPEED_RANDOMIZE=0 OBSTACLE_SPEED_SCALE=4.0 \
python scripts/train_td3.py \
  --task IsaaclabScene-Stage7-v0 \
  --num_envs 128 --total_steps 2000000 \
  --start_steps 2000 \
  --hidden_dim 512 --use_conv --privileged_critic \
  --actor_lr 0.00003 --critic_lr 0.001 \
  --batch_size 512 --buffer_size 2000000 \
  --policy_noise 0.2 --noise_clip 0.5 --policy_delay 4 --expl_noise 0.15 \
  --save_interval 250000 \
  --load_checkpoint <step2_best_checkpoint> \
  --run_name may_XX/stage7_curric_3obs \
  --headless
```

**Step 4 (final) — turn on per-episode speed randomization for robustness:**
```bash
LIDAR_STACK_FRAMES=12 STAGE7_ACTIVE_OBSTACLES="1,2,3" STAGE7_SPEED_RANDOMIZE=1 OBSTACLE_SPEED_SCALE=2.0 \
python scripts/train_td3.py \
  ... \
  --load_checkpoint <step3_best_checkpoint> \
  --run_name may_XX/stage7_curric_final \
  --headless
```

### Eval any curriculum checkpoint
Match the env to the training conditions of that checkpoint (e.g. for the
2-obstacle eval, set `STAGE7_ACTIVE_OBSTACLES="2,3"`).

```bash
LIDAR_STACK_FRAMES=12 STAGE7_ACTIVE_OBSTACLES="2,3" STAGE7_OBSTACLE_PHASE_RESET=0 STAGE7_SPEED_RANDOMIZE=0 \
OBSTACLE_SPEED_SCALE=4.0 \
python scripts/play_td3.py \
  --task IsaaclabScene-Stage7-v0 \
  --checkpoint <checkpoint> \
  --hidden_dim 512 --use_conv \
  --num_envs 256 --eval_episodes 1500 --headless
```

### Stage 7 corridor finetune (side-pass + wait behavior)
Warm-start from the current Stage 5 checkpoint and train on the Stage 7
corridor. `STAGE7_SPEED_RANDOMIZE=1` makes each training episode pick a random
obstacle-speed multiplier in [0.5, 2.5]×, so the policy sees a wide range of
crossing paces. Lower actor LR (we're refining, not retraining); same critic
LR. Same obs/reward as Stage 5/6 — do NOT pass `--reset_critic`.

Collision termination stays ON for training: a wall/obstacle hit ends the
episode, as in every prior stage.

```bash
STAGE7_SPEED_RANDOMIZE=1 OBSTACLE_SPEED_SCALE=2.0 \
python scripts/train_td3.py \
  --task IsaaclabScene-Stage7-v0 \
  --num_envs 128 \
  --total_steps 3000000 \
  --start_steps 2000 \
  --batch_size 512 \
  --buffer_size 2000000 \
  --hidden_dim 512 \
  --actor_lr 0.00003 \
  --critic_lr 0.001 \
  --gamma 0.99 --tau 0.005 \
  --policy_noise 0.2 --noise_clip 0.5 --policy_delay 4 --expl_noise 0.15 \
  --save_interval 500000 \
  --use_conv --privileged_critic \
  --load_checkpoint <stage5_checkpoint> \
  --run_name may_XX/stage7_corridor_finetune \
  --headless
```

After every save, regression-check against Stage 5 (open arena) to make sure
the corridor finetune isn't eating its old skills:

```bash
STAGE5_OBSTACLE_PHASE_RESET=0 OBSTACLE_SPEED_SCALE=1.0 \
python scripts/play_td3.py \
  --task IsaaclabScene-Stage5-v0 \
  --checkpoint <new_stage7_checkpoint> \
  --hidden_dim 512 --use_conv \
  --num_envs 256 --eval_episodes 500 --headless
```

And of course the Stage 7 success metric itself:

```bash
STAGE7_OBSTACLE_PHASE_RESET=0 STAGE7_SPEED_RANDOMIZE=0 OBSTACLE_SPEED_SCALE=1.0 \
python scripts/play_td3.py \
  --task IsaaclabScene-Stage7-v0 \
  --checkpoint <new_stage7_checkpoint> \
  --hidden_dim 512 --use_conv \
  --num_envs 256 --eval_episodes 500 --headless
```

---

## 2. Evaluation

### Standard eval (success-rate numbers)
```bash
STAGE5_OBSTACLE_PHASE_RESET=0 OBSTACLE_SPEED_SCALE=1.0 \
python scripts/play_td3.py \
  --task IsaaclabScene-Stage5-v0 \
  --checkpoint <checkpoint> \
  --hidden_dim 512 --use_conv \
  --num_envs 256 --eval_episodes 1500 --headless
```

- Always prepend `STAGE5_OBSTACLE_PHASE_RESET=0` for eval (obstacles drift, no
  per-episode teleport).
- Evaluate at the `OBSTACLE_SPEED_SCALE` the checkpoint was trained at.
- Eval Stage 5 on `IsaaclabScene-Stage5-v0` (fixed spawn) — keep the eval task
  the same across all compared checkpoints.

---

## 3. Diagnostics

### Waypoint following — hardcoded random paths (Step 1)
```bash
STAGE5_OBSTACLE_PHASE_RESET=0 \
python scripts/play_td3_waypoints.py \
  --task IsaaclabScene-Stage5-v0 \
  --checkpoint <checkpoint> \
  --hidden_dim 512 --use_conv \
  --num_envs 16 --eval_episodes 50
```

### A* planner + waypoint-following policy (Steps 2 + 3)
The planner routes a dense waypoint path around the walls; the policy follows
it. If an obstacle stops on the path, the planner replans around it once
(Step 3 — automatic; add `--no_replan` to disable and get pure Step 2).
```bash
STAGE5_OBSTACLE_PHASE_RESET=0 OBSTACLE_SPEED_SCALE=1.0 \
python scripts/play_td3_planner.py \
  --task IsaaclabScene-Stage5-v0 \
  --checkpoint <checkpoint> \
  --hidden_dim 512 --use_conv \
  --num_envs 16 --eval_episodes 50
```

### Planner on Stage 6 — five obstacles (four moving + one static blocker)
The planner demo is task-agnostic; point it at Stage 6 to stress-test it with a
denser field. obstacle_4 is a permanent static blocker the planner routes around.
```bash
STAGE6_OBSTACLE_PHASE_RESET=0 OBSTACLE_SPEED_SCALE=1.0 \
python scripts/play_td3_planner.py \
  --task IsaaclabScene-Stage6-v0 \
  --checkpoint <checkpoint> \
  --hidden_dim 512 --use_conv \
  --num_envs 16 --eval_episodes 50
```

### Planner on Stage 7 — corridor with crossing obstacles
2 m-wide corridor along X; robot spawns at the back facing the exit, three
cylinders cross the corridor in front of it, goals sit past the exit. The
planner reroutes through the corridor when a sensed obstacle blocks the path.
```bash
STAGE7_OBSTACLE_PHASE_RESET=0 OBSTACLE_SPEED_SCALE=1.0 \
python scripts/play_td3_planner.py \
  --task IsaaclabScene-Stage7-v0 \
  --checkpoint <checkpoint> \
  --hidden_dim 512 --use_conv \
  --num_envs 16 --eval_episodes 50
```

### Planner replan test — obstacle_3 stops on the path
Adds `STAGE5_OBS3_BLOCK=1` so obstacle_3 freezes on the robot's path; watch
whether the replanned path leads the robot around it.
```bash
STAGE5_OBS3_BLOCK=1 STAGE5_OBSTACLE_PHASE_RESET=0 OBSTACLE_SPEED_SCALE=1.0 \
python scripts/play_td3_planner.py \
  --task IsaaclabScene-Stage5-v0 \
  --checkpoint <checkpoint> \
  --hidden_dim 512 --use_conv \
  --num_envs 16 --eval_episodes 50
```

### Blocking-obstacle test (obstacle_3 stops in front of the robot)
```bash
STAGE5_OBS3_BLOCK=1 STAGE5_OBSTACLE_PHASE_RESET=0 OBSTACLE_SPEED_SCALE=1.0 \
python scripts/play_td3.py \
  --task IsaaclabScene-Stage5-v0 \
  --checkpoint <checkpoint> \
  --hidden_dim 512 --use_conv \
  --num_envs 256 --eval_episodes 1500 --headless
```

### Generalization test (obstacle_3 on a never-trained right-edge patrol)
```bash
STAGE5_OBS3_ALT_TRAJ=1 STAGE5_OBSTACLE_PHASE_RESET=0 OBSTACLE_SPEED_SCALE=1.0 \
python scripts/play_td3.py \
  --task IsaaclabScene-Stage5-v0 \
  --checkpoint <checkpoint> \
  --hidden_dim 512 --use_conv \
  --num_envs 256 --eval_episodes 1500 --headless
```

For watching (visuals): drop `--headless`, use a small `--num_envs` (2–16), and
make sure no training run is using the GPU. Add `--show_path` to the planner
demo to draw the planned route as a green line — off by default.

---

## 4. Environment variables

| Variable | Default | Effect |
|----------|---------|--------|
| `OBSTACLE_SPEED_SCALE` | `2.0` | Obstacle speed: `1.0` = full speed (hardest), `2.0` = half, `3.0` = third. |
| `STAGE5_OBSTACLE_PHASE_RESET` | `1` | `1` = per-episode obstacle phase reset (training). `0` = obstacles drift (eval). |
| `STAGE5_OBS3_BLOCK` | `0` | `1` = obstacle_3 seeks ahead of the robot then freezes — a path blocker (eval only). |
| `STAGE5_OBS3_ALT_TRAJ` | `0` | `1` = obstacle_3 follows a right-edge vertical patrol — generalization test (eval only). |
| `STAGE6_OBSTACLE_PHASE_RESET` | `1` | Same as STAGE5 variant but for Stage 6. Set to `0` for eval. |
| `STAGE7_OBSTACLE_PHASE_RESET` | `1` | Same as STAGE5 variant but for Stage 7. Set to `0` for eval. |
| `STAGE7_SPEED_RANDOMIZE` | `0` | `1` = per-episode random obstacle speed multiplier in [0.5, 2.5]×. ON for Stage 7 finetuning, OFF for eval. |
| `STAGE7_ACTIVE_OBSTACLES` | `1,2,3` | Comma list of active Stage 7 obstacles; the rest are hidden at z=-10. Use `"2"` for the single-obstacle diagnostic, `"2,3"` for the two-obstacle curriculum step, `"1,2,3"` for full Stage 7. |
| `LIDAR_STACK_FRAMES` | `6` | Number of lidar frames stacked along the conv input axis. Set to `12` for Stage 7 training/eval — gives the policy enough temporal evidence to infer obstacle motion direction at slow speeds. Stage 4/5/6 stay at the default 6 (existing checkpoints unchanged). |

These are read at process start — set them on the command line, never in code.



ANGULAR_GAIN=1.5 LINEAR_GAIN=1.1 LIDAR_STACK_FRAMES=6 OBSTACLE_SPEED_SCALE=3.0 STAGE6_OBSTACLE_PHASE_RESET=0 python scripts/play_td3_planner.py   --task IsaaclabScene-Stage6a-v0   --num_envs 1  --checkpoint logs/may_13/new_obervation_space/stage1_6frames_randstart/20260513_080515/td3_final.pt --hidden_dim 512 --eval_episodes 200   --inflate 0.33 --block_radius 0.20 --cluster_dist 0.30   --replan_every 2 --replan_lookahead_m 1.5   --switch_radius 0.18 --waypoint_spacing 0.15   --episode_length_s 75.0 --no_progress_timeout_s 15.0 --show_path 

`BEST POLICY`
## cp logs/ict_bot/stage5_ictbot_orbit_finetune/20260528_081745/td3_final.pt \
  # logs/ict_bot/ictbot_stage5_best_orbit_94.8.pt


ROBOT=ictbot LIDAR_STACK_FRAMES=6 OBSTACLE_SPEED_SCALE=1.5 STAGE6_OBSTACLE_PHASE_RESET=0 \
python scripts/play_td3.py \
  --task IsaaclabScene-Stage6a-v0 --num_envs 16 \
  --checkpoint logs/ict_bot/stage5_ictbot_orbit_finetune/20260528_081745/td3_final.pt \
  --use_conv --hidden_dim 512 --eval_episodes 200 --headless


## test policy with planner

ROBOT=ictbot LIDAR_STACK_FRAMES=6 OBSTACLE_SPEED_SCALE=1.5 STAGE6_OBSTACLE_PHASE_RESET=0 python scripts/play_td3_planner.py   --task IsaaclabScene-Stage6a-v0 --num_envs 32 --checkpoint logs/ict_bot/stage5_ictbot_orbit_finetune/20260528_081745/td3_final.pt   --use_conv --hidden_dim 512 --eval_episodes 100   --inflate 0.30 --block_radius 0.40 --cluster_dist 0.30   --replan_every 2 --replan_lookahead_m 2.5 --proximity_replan_m 1.2   --switch_radius 0.25 --waypoint_spacing 0.10   --episode_length_s 90.0 --no_progress_timeout_s 25.0 --headless
