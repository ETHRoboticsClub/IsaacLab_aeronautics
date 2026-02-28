from __future__ import annotations

import gymnasium as gym
import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.envs.common import ViewerCfg
from isaaclab.envs.ui import BaseEnvWindow
from isaaclab.markers import CUBOID_MARKER_CFG, VisualizationMarkers
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensor, ContactSensorCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.terrains.config.racing import create_racing_gates_terrain
from isaaclab.terrains.trimesh.racing_gates import get_gate_registry
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply, quat_apply_inverse

from isaaclab_assets import CRAZYFLIE_CFG
from .reward_manager import GateRewardManager, RewardWeights

from .reward_manager import GateRewardManager, RewardWeights



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
    gate_passage_reward: float = 50.0   # Large reward for passing through gate
    gate_progress_scale: float = 2.0     # Reward for moving toward next gate
    gate_centering_scale: float = 0.5    # Reward for staying aligned with gate center
    velocity_toward_gate_scale: float = 0.1  # Reward for velocity toward gate
    
    # Penalties (negative)
    velocity_limit_penalty_scale: float = -0.00005
    orientation_penalty_scale: float = -0.005
    ang_vel_penalty_scale: float = -0.001
    action_smoothness_scale: float = -0.0001
    action_magnitude_scale: float = -0.00002  # Penalty for large actions (keep actions near 0)
    crash_penalty: float = -20.0
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
            action_magnitude=self.action_magnitude_scale,
            crash=self.crash_penalty,
            wrong_side=self.wrong_side_penalty_scale,
            reward_scale=self.reward_scale,
        )


@configclass
class GateObservationConfig:
    """Configuration for gate-based observations."""

    num_next_gates: int = 3
    history_length: int = 1
    include_velocity_limit: bool = True


# ---------------------------------------------------------------------------
# Main environment config
# ---------------------------------------------------------------------------

@configclass
class RacingQuadcopterEnvCfg(DirectRLEnvCfg):
    """Configuration for racing quadcopter environment with terrain-generated gates."""

    # Environment settings
    episode_length_s: float = 20.0
    decimation: int = 2
    action_space: int = 4
    state_space: int = 0
    debug_vis: bool = False

    # Observation space
    obs_n_next_gates: int = 3
    obs_history_length: int = 0
    obs_include_velocity_limit: bool = True
    base: int = 10  # lin_vel(3) + ang_vel(3) + gravity(3) + vel_limit(1)
    observation_space: int = base
    if obs_n_next_gates > 0:
        observation_space += obs_n_next_gates * 6  # gate_pos(3) + gate_orient(3) per gate
    if obs_history_length > 0:
        observation_space += obs_history_length * 10 # action (4) + lin_vel(3) + ang_vel(3) history

    # Velocity settings
    velocity_limit: float = 7.0
    randomize_velocity_limit: bool = False
    velocity_limit_range: tuple[float, float] = (8.0, 15.0)

    # Reward / observation sub-configs
    reward: GateRewardConfig = GateRewardConfig()
    observation: GateObservationConfig = GateObservationConfig(
        num_next_gates=obs_n_next_gates,
        history_length=obs_history_length,
        include_velocity_limit=obs_include_velocity_limit,
    )

    # Termination thresholds
    upside_down_threshold: float = 0.7
    # min_height: float = 0.2
    max_velocity: float = 50.0
    collision_force_threshold: float = 0.2  # Force threshold for collision detection (N)

    # UI
    # ui_window_class_type = QuadcopterEnvWindow

    # Simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 50,
        render_interval=decimation,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )
    
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=create_racing_gates_terrain(),
        max_init_terrain_level=0,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        debug_vis=False,
    )

    # Scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=1024,
        env_spacing=5.0,
        replicate_physics=True,
    )

    # Robot
    # robot: ArticulationCfg = CRAZYFLIE_CFG.replace(
    #     prim_path="/World/envs/env_.*/Robot",
    # )

    robot: ArticulationCfg = CRAZYFLIE_CFG.replace(prim_path="/World/envs/env_.*/Robot").\
                                           replace(spawn = CRAZYFLIE_CFG.spawn.replace(activate_contact_sensors=True))
    
    # Contact sensor for collision detection
    contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/.*", 
        history_length=2, 
        update_period=sim.dt/2,
        track_air_time=False
    )

    camera_on_drone: bool = True
    if camera_on_drone:
        # Camera behind and slightly above drone, looking forward
        # In drone body frame: x=forward, y=left, z=up
        viewer: ViewerCfg = ViewerCfg(
            eye=(-0.5, 0.0, 0.15), lookat=(1.0, 0.0, 0.0), origin_type="asset_body", env_index=0, asset_name="robot", body_name="body")
    else:
        # Position camera to see the first environment's terrain (15x15m terrain)
        viewer: ViewerCfg = ViewerCfg(
            eye=(15.0, 15.0, 7.0), lookat=(0.0, 0.0, 2.5), origin_type="world")
    
    thrust_to_weight: float = 1.9
    moment_scale: float = 0.01