# Isaac Lab Scene — Navigation Policy Training & Evaluation

TD3-based differential-drive navigation policy trained in Isaac Lab. Supports
two robots (TurtleBot3 Burger and ict_bot) via a single env-var switch, six
curriculum stages with progressively harder obstacle fields, and an A*
planner + waypoint-following deployment pipeline.

- **Algorithm:** TD3 with asymmetric (privileged) critic
- **Actor input:** 540-dim stacked lidar + goal angle/distance + body velocity + last action (554 dims)
- **Robots:** TurtleBot3 Burger (default) and ict_bot — switched via `ROBOT=…`
- **Best deployed policy:** `logs/ict_bot/stage5_ictbot_orbit_finetune/20260528_081745/td3_final.pt` (94.8% SR on Stage 5, ~85% SR on Stage 6a with planner)
- **Active branch:** `planner_policy_test` — latest training + planner-eval changes (ict_bot adaptation, orbit-penalty reward, Stage 6a scene, planner backward recovery)

---

## 1. Setup

### 1.1 Conda environment

The project uses Isaac Lab's recommended conda flow.

```bash
# Create the env (Python 3.10/3.11 — match your Isaac Lab build)
conda create -n env_isaaclab python=3.11 -y
conda activate env_isaaclab

# Install Isaac Lab — follow the official guide once per machine:
#   https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html
# (this installs Isaac Sim's Python bindings into the conda env)

# Clone this project anywhere OUTSIDE the IsaacLab/ directory
cd ~/Documents
git clone <this-repo-url> isaaclab_scene
cd isaaclab_scene

# Switch to the active development branch
git checkout planner_policy_test

# Install the project as an editable extension
python -m pip install -e source/isaaclab_scene
```

### 1.2 Verify the install

```bash
# List the tasks this project registers
python scripts/list_envs.py | grep IsaaclabScene
```

You should see `IsaaclabScene-Stage1-v0` through `IsaaclabScene-Stage6a-v0`.

### 1.3 Pick a robot

Every command below works for both robots — just prepend `ROBOT=turtlebot`
(default) or `ROBOT=ictbot`. The registry in [`robot_registry.py`](source/isaaclab_scene/isaaclab_scene/tasks/manager_based/isaaclab_scene/mdp/robot_registry.py)
swaps USD, joint names, wheel radius, base-link name, and the FORWARD_AXIS
convention in one place.

```bash
ROBOT=turtlebot   # default. body +X forward, wheel radius 0.033 m
ROBOT=ictbot      # body -Y forward, wheel radius 0.05 m, link named `link_base`
```

---

## 2. Tasks

| Task ID | Scene | Used for |
|---|---|---|
| `IsaaclabScene-Stage1-v0`            | Open arena, four outer walls only             | Warm-up / PPO baseline |
| `IsaaclabScene-Stage3-v0`            | Pinwheel rotating obstacles                   | Rotating-obstacle dodge |
| `IsaaclabScene-Stage31-v0`           | Stage 4 maze, static obstacles (no motion)    | Curriculum bridge between Stage 3 and 4 |
| `IsaaclabScene-Stage4-v0`            | 4 inner walls + 2 moving obstacles, fixed spawn | Stage 4 baseline |
| `IsaaclabScene-Stage4-RandSpawn-v0`  | Stage 4 with random robot spawn               | Robustness training |
| `IsaaclabScene-Stage4-MixedSpawn-v0` | ~50% fixed / ~50% random spawn                | Recommended Stage 4 finetune target |
| `IsaaclabScene-Stage4-ThreePillar-v0`| Stage 4 + a third moving pillar               | Generalization test (no retrain) |
| `IsaaclabScene-Stage5-v0`            | 4 inner walls + 3 moving obstacles, fixed spawn | Stage 5 baseline / eval |
| `IsaaclabScene-Stage5-MixedSpawn-v0` | Stage 5 with mixed spawn                      | Recommended Stage 5 finetune target |
| `IsaaclabScene-Stage5-MixedSpawn-Orbit-v0`  | Stage 5 mixed-spawn + orbit-penalty reward    | Fix orbit-dodge behavior |
| `IsaaclabScene-Stage5-MixedSpawn-Smooth-v0` | Stage 5 mixed-spawn + orbit + gap-bonus       | Smooth-weave finetune |
| `IsaaclabScene-Stage6-v0`            | Stage 5 arena + 6 obstacles (4 moving + 2 static blockers) | Planner stress-test |
| `IsaaclabScene-Stage6a-v0`           | No outer walls; corridor entry + open obstacle field | Planner eval, open world |

---

## 3. Training

### 3.1 Common scaffolding (any stage)

```bash
ROBOT=<turtlebot|ictbot> LIDAR_STACK_FRAMES=6 OBSTACLE_SPEED_SCALE=<scale> \
python scripts/train_td3.py \
  --task <task-id> \
  --num_envs 128 --headless \
  --total_steps <budget> \
  --start_steps <warmup> \
  --batch_size 512 --buffer_size 2000000 \
  --hidden_dim 512 --use_conv --privileged_critic \
  --actor_lr <lr> --critic_lr <lr> \
  --gamma 0.99 --tau 0.005 \
  --policy_noise 0.2 --noise_clip 0.5 --policy_delay 4 --expl_noise 0.15 \
  --save_interval 250000 \
  --run_name <date>/<descriptive_name>
```

Add `--load_checkpoint <path>` to resume / finetune. Add `--reset_critic`
only when the reward or the privileged-obs dimension changed (most finetunes
should NOT use it).

### 3.2 Stage 1 — open arena warm-up

```bash
ROBOT=turtlebot python scripts/train_td3.py \
  --task IsaaclabScene-Stage1-v0 \
  --num_envs 128 --headless \
  --total_steps 1000000 \
  --start_steps 25000 \
  --hidden_dim 512 --use_conv --privileged_critic \
  --actor_lr 1e-4 --critic_lr 1e-3 \
  --batch_size 512 --buffer_size 2000000 \
  --policy_noise 0.2 --noise_clip 0.5 --policy_delay 4 --expl_noise 0.15 \
  --save_interval 100000 \
  --run_name stage1_warmup
```

### 3.3 Stage 4 — moving obstacles, mixed spawn

```bash
OBSTACLE_SPEED_SCALE=2.0 ROBOT=turtlebot \
python scripts/train_td3.py \
  --task IsaaclabScene-Stage4-MixedSpawn-v0 \
  --num_envs 128 --headless \
  --total_steps 5000000 \
  --start_steps 25000 \
  --hidden_dim 512 --use_conv --privileged_critic \
  --actor_lr 1e-4 --critic_lr 1e-3 \
  --batch_size 512 --buffer_size 2000000 \
  --policy_noise 0.2 --noise_clip 0.5 --policy_delay 4 --expl_noise 0.15 \
  --save_interval 250000 \
  --load_checkpoint <stage1_best.pt> \
  --run_name stage4_mixedspawn
```

### 3.4 Stage 5 — 3 obstacles, full arena

Train from-scratch (longer) OR continue from a Stage 4 ckpt:

```bash
OBSTACLE_SPEED_SCALE=1.0 ROBOT=turtlebot \
python scripts/train_td3.py \
  --task IsaaclabScene-Stage5-MixedSpawn-v0 \
  --num_envs 128 --headless \
  --total_steps 10000000 \
  --start_steps 2000 \
  --hidden_dim 512 --use_conv --privileged_critic \
  --actor_lr 5e-5 --critic_lr 1e-3 \
  --batch_size 512 --buffer_size 2000000 \
  --policy_noise 0.2 --noise_clip 0.5 --policy_delay 4 --expl_noise 0.15 \
  --save_interval 500000 \
  --load_checkpoint <stage4_best.pt> \
  --run_name stage5_mixedspawn
```

`OBSTACLE_SPEED_SCALE` semantics: keyframe times are *multiplied* by the
scale, so **higher = slower obstacles** (easier). Recommended curriculum
order — train at the easiest scale until SR plateaus, then step down:

- `4.0` = quarter speed (easiest — curriculum start)
- `3.0` = one-third speed
- `2.0` = half speed (intermediate)
- `1.5` = ~33% slower than baseline (stress test / final curriculum step)
- `1.0` = fastest (training default, hardest)

Concretely: start at `OBSTACLE_SPEED_SCALE=4.0` for ~1M steps to learn the
basic dodge behavior, save the best ckpt, then finetune at `3.0` for
another ~1M, then `2.0`, etc. Each step reuses the previous ckpt via
`--load_checkpoint` and a smaller actor LR to keep the policy stable as
obstacles get faster.

### 3.5 ict_bot adaptation — finetune a TurtleBot policy

When porting a TurtleBot-trained policy to ict_bot, do a short warmup so the
actor adapts to the new wheel radius / inertia. Keep LRs very low to preserve
the loaded behavior; do NOT reset the critic.

```bash
ROBOT=ictbot LIDAR_STACK_FRAMES=6 OBSTACLE_SPEED_SCALE=1.0 \
python scripts/train_td3.py \
  --task IsaaclabScene-Stage5-MixedSpawn-v0 \
  --num_envs 128 --headless \
  --load_checkpoint <turtlebot_best.pt> \
  --use_conv --hidden_dim 512 --privileged_critic \
  --total_steps 500000 \
  --start_steps 5000 \
  --actor_lr 5e-6 --critic_lr 5e-5 \
  --policy_delay 2 --expl_noise 0.03 \
  --save_interval 50000 \
  --run_name ict_bot/stage5_ictbot_adapt
```

Eval the early checkpoints (50k, 100k, …) on `Stage5-v0` fixed-spawn — usually
50k or 100k is the new best ict_bot ckpt.

### 3.6 Behavior-shaping finetune (orbit-penalty, then gap-bonus)

If the policy spins in front of obstacles instead of weaving past them, finetune
on the orbit-penalty variant:

```bash
ROBOT=ictbot LIDAR_STACK_FRAMES=6 OBSTACLE_SPEED_SCALE=1.0 \
python scripts/train_td3.py \
  --task IsaaclabScene-Stage5-MixedSpawn-Orbit-v0 \
  --num_envs 128 --headless \
  --load_checkpoint <ictbot_adapt_best.pt> \
  --use_conv --hidden_dim 512 --privileged_critic \
  --total_steps 200000 \
  --start_steps 5000 \
  --actor_lr 1e-6 --critic_lr 1e-5 \
  --policy_delay 4 --expl_noise 0.02 \
  --save_interval 25000 \
  --run_name ict_bot/stage5_orbit_finetune
```

For a second pass that also rewards side-weaving, use the Smooth task
(`Stage5-MixedSpawn-Smooth-v0`) and resume from the orbit-best ckpt. The
gap-bonus is more disruptive to the value function — be ready for the
finetune to NaN; smaller LRs (`5e-7` / `5e-6`) and shorter budget (`100k`)
help. See §5 for the reward-curriculum guide.

---

## 4. Evaluation

### 4.1 Pure policy on Stage 5 (sanity check, no planner)

Fast batch eval; reproducible SR number for the checkpoint.

```bash
ROBOT=ictbot LIDAR_STACK_FRAMES=6 OBSTACLE_SPEED_SCALE=1.0 STAGE5_OBSTACLE_PHASE_RESET=0 \
python scripts/play_td3.py \
  --task IsaaclabScene-Stage5-v0 \
  --num_envs 16 --headless \
  --checkpoint <path>.pt \
  --use_conv --hidden_dim 512 \
  --eval_episodes 1000
```

Notes:
- Use `Stage5-v0` (fixed-spawn) for reproducibility, not `Stage5-MixedSpawn-v0`.
- `STAGE5_OBSTACLE_PHASE_RESET=0` is required at eval — without it, obstacle
  phases re-randomize on every reset and SR isn't comparable across runs.
  The same flag exists per stage (`STAGE4_…`, `STAGE6_…`); set to 0 at eval.

### 4.2 Planner + policy on Stage 6/6a (deployment scene)

```bash
ROBOT=ictbot LIDAR_STACK_FRAMES=6 OBSTACLE_SPEED_SCALE=1.5 STAGE6_OBSTACLE_PHASE_RESET=0 \
python scripts/play_td3_planner.py \
  --task IsaaclabScene-Stage6a-v0 --num_envs 32 --headless \
  --checkpoint <path>.pt \
  --use_conv --hidden_dim 512 --eval_episodes 100 \
  --inflate 0.30 --block_radius 0.40 --cluster_dist 0.30 \
  --replan_every 2 --replan_lookahead_m 2.5 --proximity_replan_m 1.2 \
  --switch_radius 0.25 --waypoint_spacing 0.10 \
  --episode_length_s 90.0 --no_progress_timeout_s 25.0 \
  --enable_backward_recovery
```

**Planner flags worth knowing:**

| Flag | Default | What it controls |
|---|---|---|
| `--inflate`               | 0.30  | Static-wall buffer in the A* grid (m) |
| `--block_radius`          | 0.20  | Dynamic-obstacle disk size; ≥ `0.40` recommended for safety |
| `--cluster_dist`          | 0.30  | Coarse-grid cell size used to dedupe lidar hits into one disk |
| `--replan_every`          | 3     | Step period between replan checks; lower = more reactive, but `1` thrashes |
| `--replan_lookahead_m`    | 1.5   | Only check path-blockage within this distance from the robot |
| `--proximity_replan_m`    | 0.8   | Trigger replan whenever any obstacle is within this radius |
| `--switch_radius`         | 0.15  | Advance to next waypoint within this distance |
| `--waypoint_spacing`      | 0.1   | Dense waypoint spacing along the planned path (m) |
| `--episode_length_s`      | 75    | Per-episode time budget |
| `--no_progress_timeout_s` | 15    | Declare STALLED if no forward progress for this long |
| `--enable_backward_recovery` | off | When pinned, override action with reverse if rear is clear, else wait |

### 4.3 Visual debug run (single env, viewer on)

Drop `--headless`, set `--num_envs 1`, and add `--show_path` to draw the
planned route:

```bash
ROBOT=ictbot OBSTACLE_SPEED_SCALE=1.5 STAGE6_OBSTACLE_PHASE_RESET=0 \
python scripts/play_td3_planner.py \
  --task IsaaclabScene-Stage6a-v0 --num_envs 1 \
  --checkpoint <path>.pt \
  --use_conv --hidden_dim 512 --eval_episodes 20 \
  --show_path --enable_backward_recovery
```
---

## 5. Reward curriculum — adding a new variant

Reward shaping lives in [`mdp/rewards.py`](source/isaaclab_scene/isaaclab_scene/tasks/manager_based/isaaclab_scene/mdp/rewards.py).
The current production reward is `Stage4RewardsCfgV2` (used by Stages 4, 5, 6).
The two variants for behavior shaping are:

- `Stage4RewardsCfgV2Orbit` — adds an orbit-penalty (penalize spin-in-place near obstacles).
- `Stage4RewardsCfgV2Smooth` — orbit-penalty + gap-through bonus (pays the policy to side-weave).

### 5.1 To add a new reward variant (the safe pattern)

1. **Write a new reward function in `rewards.py`** that wraps an existing one.
   Wrapping (`base = navigation_reward_stage4_v2(env)` then `return base + your_term`)
   means future changes to the base reward propagate automatically.

   ```python
   def navigation_reward_stage4_v2_myvariant(env):
       base = navigation_reward_stage4_v2(env)
       # … compute your new term using helpers in this file …
       return base + r_my_term


   @configclass
   class Stage4RewardsCfgV2MyVariant:
       navigation = RewTerm(func=navigation_reward_stage4_v2_myvariant, weight=1.0)
   ```

2. **Add a Stage env-cfg** in [`stage5_env_cfg.py`](source/isaaclab_scene/isaaclab_scene/tasks/manager_based/isaaclab_scene/stage5_env_cfg.py)
   that inherits the relevant base env and only overrides `rewards`:

   ```python
   @configclass
   class Stage5MixedSpawnMyVariantEnvCfg(Stage5MixedSpawnEnvCfg):
       rewards: Stage4RewardsCfgV2MyVariant = Stage4RewardsCfgV2MyVariant()
   ```

3. **Register the task** in [`__init__.py`](source/isaaclab_scene/isaaclab_scene/tasks/manager_based/isaaclab_scene/__init__.py):

   ```python
   gym.register(
       id="IsaaclabScene-Stage5-MixedSpawn-MyVariant-v0",
       entry_point="isaaclab.envs:ManagerBasedRLEnv",
       disable_env_checker=True,
       kwargs={"env_cfg_entry_point": f"{__name__}.stage5_env_cfg:Stage5MixedSpawnMyVariantEnvCfg"},
   )
   ```

4. **Finetune from your current best on the new task.** Reward shifts disturb
   the value function — use a small actor LR (`1e-6` to `5e-6`), a small critic
   LR (`1e-5` to `5e-5`), short total_steps (100k–500k), and `--save_interval 25_000`
   so you can grab the peak before regression.

### 5.2 Why reward changes are risky on a finetune

The loaded critic has Q-values calibrated to the *old* reward. A new term —
even a small `+0.30/step` bonus on a frequent condition — shifts the Bellman
targets. If LRs are normal, critic-loss spikes, the critic chases the new
targets noisily, and the actor follows bad gradients → NaN.

Two recipes that work:

- **Conservative wrap** (recommended): tiny new term + tiny LRs. Lets the
  critic absorb the shift without destabilizing.
- **Reset critic** (`--reset_critic`): throw away the loaded critic. Actor
  stays anchored, critic relearns from scratch. Use a higher critic LR
  (`1e-4`) and a longer budget (3M+) since the critic is starting from zero.

When in doubt, prefer the conservative wrap. Only reset_critic if the
distribution shift is huge (e.g., scale 2.0 → scale 1.0).

---

## 6. Environment variables reference

These can all be combined freely on a single command line.

### Robot selection

| Variable | Values | Effect |
|---|---|---|
| `ROBOT`         | `turtlebot` (default), `ictbot` | USD path, joint names, wheel radius, base link, forward axis |
| `FORWARD_AXIS`  | `X`, `-X`, `Y`, `-Y`            | Override the robot's natural forward axis (rare) |

### Observation

| Variable | Default | Effect |
|---|---|---|
| `LIDAR_STACK_FRAMES` | `6` | Number of stacked lidar frames in the policy observation |

### Curriculum / difficulty

| Variable | Default | Effect |
|---|---|---|
| `OBSTACLE_SPEED_SCALE`     | `2.0` (stage 6), see file for others | Keyframe-time multiplier; higher = slower obstacles. Curriculum: `4.0 → 3.0 → 2.0 → 1.5 → 1.0` |
| `STAGE4_OBSTACLE_PHASE_RESET` | `1` | Set to `0` at eval time so SR is reproducible |
| `STAGE5_OBSTACLE_PHASE_RESET` | `1` | Same as above for Stage 5 |
| `STAGE6_OBSTACLE_PHASE_RESET` | `1` | Same as above for Stage 6 / 6a |

### Deployment tuning (post-training, no retrain)

| Variable | Default | Effect |
|---|---|---|
| `LINEAR_GAIN`  | `1.0` | Scales `MAX_LINEAR_SPEED` at deployment |
| `ANGULAR_GAIN` | `1.0` | Scales `MAX_ANGULAR_SPEED` at deployment |

---

## 7. Troubleshooting

- **`ValueError: Invalid value type for the argument 'width'`** — `--width` /
  `--height` are reserved by `AppLauncher` and can't be added to the script
  parser. Use the renderer instead: `--rendering_mode performance`.
- **NaN losses early in training** — usually a critic mismatch when finetuning
  with a different reward/obs dim. Add `--reset_critic`, lower critic LR, and
  raise `--policy_delay` to 4.
- **Eval SR drops after finetune** — likely actor LR too high or critic reset
  without enough budget. Eval *early* checkpoints (10–25% into the run); the
  peak usually arrives long before the final ckpt.
- **`Failed to find a prim at path expression`** — base link name mismatch.
  Check `robot_registry.py`'s `base_link` field matches the USD. TurtleBot
  uses `base_link`; ict_bot uses `link_base`.
- **Visual FPS pinned at 15** — the renderer is the bottleneck. Switch to
  Storm via the viewport dropdown, or pass `--rendering_mode performance`.
- **Pylance / VSCode indexing crashes** — too many Omniverse packages indexed.
  Edit `.vscode/settings.json` and exclude unused `extscache/omni.*` paths.

---
