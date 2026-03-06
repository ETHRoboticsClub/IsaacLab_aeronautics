# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for racing quadcopter environment."""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.envs.common import ViewerCfg
from isaaclab.envs.ui import BaseEnvWindow
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensor, ContactSensorCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.terrains.config.racing import create_racing_gates_terrain
from isaaclab.utils import configclass

from isaaclab_assets import CRAZYFLIE_CFG
from .reward_manager import RewardWeights


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------


class QuadcopterEnvWindow(BaseEnvWindow):
    """Window manager for the Quadcopter environment."""

    def __init__(self, env: "RacingQuadcopterEnv", window_name: str = "IsaacLab"):
        super().__init__(env, window_name)
        with self.ui_window_elements["main_vstack"]:
            with self.ui_window_elements["debug_frame"]:
                with self.ui_window_elements["debug_vstack"]:
                    self._create_debug_vis_ui_element("targets", self.env)


# ---------------------------------------------------------------------------
# Sub-configs
# ---------------------------------------------------------------------------


@configclass
class GateRewardConfig:
    """Configuration for gate-based rewards.
    
    Primary rewards encourage gate passage and proper trajectory.
    Penalties discourage unsafe or inefficient behavior.
    """

    # Primary rewards (positive)
    gate_passage_reward: float = 20.0   # Large reward for passing through gate
    gate_progress_scale: float = 2.0     # Reward for moving toward next gate
    gate_centering_scale: float = 0.5    # Reward for staying aligned with gate center
    velocity_toward_gate_scale: float = 0.1  # Reward for velocity toward gate
    
    # Penalties (negative)
    velocity_limit_penalty_scale: float = -0.05
    orientation_penalty_scale: float = -0.005
    ang_vel_penalty_scale: float = -0.001
    action_smoothness_scale: float = -0.0001
    crash_penalty: float = -50.0
    wrong_side_penalty_scale: float = -0.1

    reward_scale: float = 1/100.0 # Overall scaling to keep rewards in a reasonable range
    
    def to_reward_weights(self) -> RewardWeights:
        """Convert to RewardWeights dataclass."""
        return RewardWeights(
            gate_passage=self.gate_passage_reward,
            gate_progress=self.gate_progress_scale,
            gate_centering=self.gate_centering_scale,
            velocity_toward_gate=self.velocity_toward_gate_scale,
            velocity_limit=self.velocity_limit_penalty_scale,
            orientation=self.orientation_penalty_scale,
            angular_velocity=self.ang_vel_penalty_scale,
            action_smoothness=self.action_smoothness_scale,
            crash=self.crash_penalty,
            wrong_side=self.wrong_side_penalty_scale,
            reward_scale=self.reward_scale,
        )


@configclass
class GateObservationConfig:
    """Configuration for gate-based observations."""

    num_next_gates: int = 3  # Increased from 3 to 5 for longer trajectories
    history_length: int = 0
    include_velocity_limit: bool = True


# ---------------------------------------------------------------------------
# Main environment config
# ---------------------------------------------------------------------------


@configclass
class RacingQuadcopterEnvCfg(DirectRLEnvCfg):
    """Configuration for racing quadcopter environment with terrain-generated gates."""

    seed: int = 42
    
    # Environment settings
    episode_length_s: float = 20.0
    decimation: int = 2
    action_space: int = 4
    state_space: int = 0
    debug_vis: bool = False
    eval_mode: bool = False  # When True, truncate episodes on lap completion

    # Velocity settings
    velocity_limit: float = 15.0
    randomize_velocity_limit: bool = False
    velocity_limit_range: tuple[float, float] = (8.0, 15.0)

    # Reward / observation sub-configs
    reward: GateRewardConfig = GateRewardConfig()
    observation: GateObservationConfig = GateObservationConfig()

    # Observation space (recalculated in __post_init__)
    # lin_vel(3) + ang_vel(3) + gravity(3) + vel_limit(1) + gates(N * 6)
    # Each gate: rel_pos(3) + forward_dir(3)
    observation_space: int = 10 + 5 * 6  # Updated for 5 gates

    # Termination thresholds
    upside_down_threshold: float = 0.7
    max_velocity: float = 50.0
    collision_force_threshold: float = 0.001  # Force threshold for collision detection (N)

    # UI
    ui_window_class_type = QuadcopterEnvWindow
    
    # curriculum
    curriculum_min_gates_passed = 10
    curriculum_min_mean_velocity = 2.5
    curriculum_max_center_offset = 0.25

    # Simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 50,
        render_interval=2,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )

    # Terrain with racing gates (configured in __post_init__ based on observation config)
    terrain: TerrainImporterCfg = None

    # Scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=1024,
        env_spacing=5.0,  # Increased spacing for larger terrain (15x15m)
        replicate_physics=True,
    )

    camera_on_drone: bool = False

    if camera_on_drone:
        viewer: ViewerCfg = ViewerCfg(
            eye=(-0.25, 0.0, 0.0), lookat=(5.0, 0.0, -0.5), origin_type="asset_body", env_index=0, asset_name="robot", body_name="body")
    else:
        # Position camera to see the first environment's terrain (15x15m terrain)
        viewer: ViewerCfg = ViewerCfg(
            eye=(15.0, 15.0, 7.0), lookat=(3.0, 3.0, 0.0), origin_type="world")


    # Robot
    robot: ArticulationCfg = CRAZYFLIE_CFG.replace(prim_path="/World/envs/env_.*/Robot").\
                                           replace(spawn = CRAZYFLIE_CFG.spawn.replace(activate_contact_sensors=True))
    
    # Contact sensor for collision detection
    contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/.*", 
        history_length=2, 
        update_period=0.01,
        track_air_time=False
    )
    
    thrust_to_weight: float = 1.9
    moment_scale: float = 0.01

    def __post_init__(self):
        """Post initialization."""
        base = 10  # lin_vel(3) + ang_vel(3) + gravity(3) + vel_limit(1)
        gates = self.observation.num_next_gates * 6  # position(3) + forward_direction(3) for each gate
        history = self.observation.history_length * 10 if self.observation.history_length > 0 else 0
        self.observation_space = base + gates + history
        self.sim.render_interval = self.decimation
        
        # Create enhanced configuration with longer, more challenging trajectories
        terrain_generator = create_racing_gates_terrain(self.seed)
        
        self.terrain = TerrainImporterCfg(
            prim_path="/World/ground",
            terrain_type="generator",
            terrain_generator=terrain_generator,
            max_init_terrain_level=0,
            collision_group=-1,
            physics_material=sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode="multiply",
                restitution_combine_mode="multiply",
                static_friction=1.0,
                dynamic_friction=1.0,
            ),
            debug_vis=True,
        )
