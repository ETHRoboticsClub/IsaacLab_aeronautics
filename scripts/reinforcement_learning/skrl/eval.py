"""Evaluate a trained skrl PPO agent on Isaac-Racing-Quadcopter-Direct-v0.

Unlike the generic play.py, this script is hardcoded for our racing quadcopter
task and does NOT use Hydra — what you see is what runs.

Usage:
    # auto-resolve the latest checkpoint under logs/skrl/racing_quadcopter/
    python scripts/reinforcement_learning/skrl/eval.py

    # or point to a specific checkpoint
    python scripts/reinforcement_learning/skrl/eval.py \
        --checkpoint logs/skrl/racing_quadcopter/<run>/checkpoints/best_agent.pt

Optional flags:
    --num_envs 16   Override number of parallel environments (default: 16)
    --real_time     Sleep between steps to match real-time speed
    --headless      Run without the GUI window
"""

import argparse
import math
import os

from isaaclab.app import AppLauncher

# ---------------------------------------------------------------------------
# 1. Parse CLI args BEFORE launching Isaac Sim (AppLauncher reads from them)
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser(description="Evaluate a trained racing-quadcopter skrl agent.")
parser.add_argument("--checkpoint", type=str, default=None, help="Path to the skrl .pt checkpoint. If omitted, the latest checkpoint under logs/skrl/racing_quadcopter/ is used.")
parser.add_argument("--num_envs", type=int, default=16, help="Number of parallel environments. Recommended to be square.")
parser.add_argument("--real_time", action="store_true", default=False, help="Run at real-time speed.")
parser.add_argument("--episodes", type=int, default=3, help="Episodes per env to collect before printing a summary.")
parser.add_argument("--video", action="store_true", default=False, help="Record a video of the evaluation.")
parser.add_argument("--video_length", type=float, default=30.0, help="Duration of the recorded video in seconds.")

# AppLauncher adds --headless, --device, --cpu, etc.
AppLauncher.add_app_launcher_args(parser)
args, _ = parser.parse_known_args()

# Video recording requires the camera renderer even in headless mode.
if args.video:
    args.enable_cameras = True

# AppLauncher must be created before any omni/isaacsim imports.
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# ---------------------------------------------------------------------------
# 2. All Isaac / skrl imports happen AFTER the sim is running
# ---------------------------------------------------------------------------

import datetime
import importlib
import importlib.resources
import time
import torch
import yaml
import gymnasium as gym

from skrl.utils.runner.torch import Runner

import isaaclab.sim as sim_utils
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.utils.math import quat_from_angle_axis

from isaaclab_rl.skrl import SkrlVecEnvWrapper

import isaaclab_tasks  # noqa: F401  registers built-in tasks
import isaaclab_tasks.direct.racing_quadcopter  # noqa: F401  registers our task

from isaaclab_tasks.direct.racing_quadcopter.quadcopter_env_cfg import RacingQuadcopterEnvCfg
from isaaclab_tasks.direct.racing_quadcopter.eval_metrics import EvalMetricsTracker
from isaaclab_tasks.utils import get_checkpoint_path

# ---------------------------------------------------------------------------
# 3. Build the environment config directly (no Hydra, no YAML lookup)
# ---------------------------------------------------------------------------

env_cfg = RacingQuadcopterEnvCfg()
env_cfg.scene.num_envs = args.num_envs
env_cfg.sim.device = args.device if args.device is not None else env_cfg.sim.device
env_cfg.randomize_velocity_limit = False  # keep things deterministic during eval
env_cfg.eval_mode = True  # truncate episodes on lap completion

# Match the training env: square grid with max_init_terrain_level=None so
# rows are distributed randomly across all tiles (not pinned to row 0).
grid = math.ceil(math.sqrt(args.num_envs))
env_cfg.terrain.terrain_generator.num_rows = grid
env_cfg.terrain.terrain_generator.num_cols = grid
env_cfg.terrain.max_init_terrain_level = None

# ---------------------------------------------------------------------------
# 4. Create and wrap the environment
# ---------------------------------------------------------------------------

gym_env = gym.make(
    "Isaac-Racing-Quadcopter-Direct-v0",
    cfg=env_cfg,
    render_mode="rgb_array" if args.video else None,
)
base_env = gym_env.unwrapped  # RacingQuadcopterEnv — used for direct state access

if args.video:
    step_dt = env_cfg.sim.dt * env_cfg.decimation
    video_steps = int(args.video_length / step_dt)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    video_dir = os.path.join("logs", "videos", "eval", timestamp)
    gym_env = gym.wrappers.RecordVideo(
        gym_env,
        video_folder=video_dir,
        step_trigger=lambda step: step == 0,
        video_length=video_steps,
        disable_logger=True,
    )
    print(f"[eval] Recording {args.video_length:.0f}s ({video_steps} steps) to: {video_dir}")

env = SkrlVecEnvWrapper(gym_env, ml_framework="torch")

# ---------------------------------------------------------------------------
# 5. Agent config — loaded directly from agents/skrl_ppo_cfg.yaml so it
#    always stays in sync with any training changes.
# ---------------------------------------------------------------------------

_yaml_path = str(importlib.resources.files("isaaclab_tasks.direct.racing_quadcopter.agents").joinpath("skrl_ppo_cfg.yaml"))
with open(_yaml_path) as _f:
    experiment_cfg = yaml.safe_load(_f)

# Disable logging and checkpointing during eval; keep the env alive after exit.
experiment_cfg["agent"]["experiment"]["write_interval"] = 0
experiment_cfg["agent"]["experiment"]["checkpoint_interval"] = 0
experiment_cfg["trainer"]["close_environment_at_exit"] = False

LOG_DIR = experiment_cfg["agent"]["experiment"]["directory"]

# ---------------------------------------------------------------------------
# 6. Resolve checkpoint path (explicit or auto-discover latest)
# ---------------------------------------------------------------------------

if args.checkpoint:
    checkpoint_path = args.checkpoint
else:
    log_root_path = os.path.abspath(os.path.join("logs", "skrl", LOG_DIR))
    checkpoint_path = get_checkpoint_path(
        log_root_path, run_dir=".*_ppo_torch", other_dirs=["checkpoints"]
    )
    print(f"[eval] Auto-resolved checkpoint: {checkpoint_path}")

# ---------------------------------------------------------------------------
# 7. Build the runner, load the checkpoint, switch to eval mode
# ---------------------------------------------------------------------------

runner = Runner(env, experiment_cfg)

print(f"[eval] Loading checkpoint: {checkpoint_path}")
runner.agent.load(checkpoint_path)
runner.agent.set_running_mode("eval")

# ---------------------------------------------------------------------------
# 8. Evaluation loop
# ---------------------------------------------------------------------------

try:
    dt = env.step_dt  # some skrl wrapper versions expose this directly
except AttributeError:
    dt = env.unwrapped.step_dt  # fall back through the wrapper chain

obs, _ = env.reset()
batch_count  = 0
report_every = args.num_envs * args.episodes  # print after this many total episodes

tracker = EvalMetricsTracker(
    num_envs  = args.num_envs,
    num_gates = base_env._num_gates,
    device    = base_env.device,
)

# Line markers: 4 thin capsules per drone, one to each corner of the target gate.
line_cfg = VisualizationMarkersCfg(
    prim_path="/Visuals/DroneToGate",
    markers={
        "line": sim_utils.CapsuleCfg(
            radius=0.02,
            height=1.0,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
        )
    },
)
line_marker = VisualizationMarkers(line_cfg)


def update_line():
    """Draw 4 lines(/capsules) from each drone to the 4 corners of its current target gate."""
    drone_pos = base_env._robot.data.root_pos_w           # [N, 3]
    gate_pos, gate_ori = base_env._gather_current_gate()  # [N, 3], [N, 4]
    N = drone_pos.shape[0]

    half = base_env._gate_size / 2.0
    right = base_env._gate_right(gate_ori)  # [N, 3]
    up    = base_env._gate_up(gate_ori)     # [N, 3]

    # 4 corners: top-right, top-left, bottom-right, bottom-left  [N, 4, 3]
    corners = torch.stack([
        gate_pos + right * half + up * half,
        gate_pos - right * half + up * half,
        gate_pos + right * half - up * half,
        gate_pos - right * half - up * half,
    ], dim=1)

    drone_exp = drone_pos.unsqueeze(1).expand(-1, 4, -1)  # [N, 4, 3]
    to_corner = corners - drone_exp                        # [N, 4, 3]
    dist      = to_corner.norm(dim=-1, keepdim=True).clamp(min=1e-6)  # [N, 4, 1]
    direction = to_corner / dist                           # [N, 4, 3]
    midpoint  = (drone_exp + corners) * 0.5               # [N, 4, 3]

    # Flatten to [4N, *]
    M         = N * 4
    mid_flat  = midpoint.reshape(M, 3)
    dir_flat  = direction.reshape(M, 3)
    dist_flat = dist.reshape(M)

    # Rotate Z=[0,0,1] onto each direction
    dx, dy, dz = dir_flat[:, 0], dir_flat[:, 1], dir_flat[:, 2]
    axis     = torch.stack([-dy, dx, torch.zeros(M, device=drone_pos.device)], dim=-1)
    axis_len = axis.norm(dim=-1, keepdim=True)
    safe_axis = torch.where(axis_len > 1e-6, axis / axis_len,
                            torch.tensor([[1., 0., 0.]], device=drone_pos.device).expand(M, -1))
    angle        = torch.acos(dz.clamp(-1.0, 1.0))
    orientations = quat_from_angle_axis(angle, safe_axis)

    scales         = torch.ones(M, 3, device=drone_pos.device)
    scales[:, 2]   = dist_flat

    line_marker.visualize(translations=mid_flat, orientations=orientations, scales=scales)


print("[eval] Starting evaluation. Close the simulator window to stop.")
while simulation_app.is_running():
    step_start = time.time()

    with torch.inference_mode():
        # act() returns (sampled_actions, log_prob, outputs_dict)
        # Use mean_actions for deterministic evaluation (no noise)
        gates_before = base_env._gates_passed.clone()

        outputs = runner.agent.act(obs, timestep=0, timesteps=0)
        actions = outputs[-1].get("mean_actions", outputs[0])
        obs, _, terminated, truncated, _ = env.step(actions)

        speed = base_env._robot.data.root_lin_vel_w.norm(dim=-1)
        tracker.on_step(speed)

        # Detect new gate passages this step
        new_passages = base_env._gates_passed > gates_before
        if new_passages.any():
            ids      = new_passages.nonzero(as_tuple=False).view(-1)
            gate_idx = (base_env._gates_passed[ids] - 1) % base_env._num_gates # modulo shouldn't be necessary
            tracker.on_gate_passage(ids, gate_idx, speed[ids])

        done = (terminated | truncated).view(-1)  # flatten: SKRL may add trailing dim
        if done.any():
            ids = done.nonzero(as_tuple=False).view(-1)  # [K] env indices
            # _gates_passed is already reset to 0 inside env.step(); use the
            # pre-reset snapshot the env saves in _last_episode_gates_passed
            tracker.on_episode_end(ids, base_env._last_episode_gates_passed[ids].long(), base_env.step_dt)

            if tracker.total_episodes >= report_every:
                batch_count += 1
                tracker.print_summary(batch_id=batch_count)
                tracker.reset_accumulators()

    update_line()

    # Optional: slow down to real time
    if args.real_time:
        sleep_time = dt - (time.time() - step_start)
        if sleep_time > 0:
            time.sleep(sleep_time)

env.close()
simulation_app.close()
