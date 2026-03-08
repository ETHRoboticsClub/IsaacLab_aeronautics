# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import gymnasium as gym
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.envs.ui import BaseEnvWindow
from isaaclab.markers import VisualizationMarkers
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import subtract_frame_transforms

##
# Pre-defined configs
##
from isaaclab_assets import CRAZYFLIE_CFG  # isort: skip
from isaaclab.markers import CUBOID_MARKER_CFG  # isort: skip

from .domain_randomization import EventCfg, make_event_cfg



_SIM_DT = 1 / 100


class HoveringQuadcopterEnvWindow(BaseEnvWindow):
    """Window manager for the HoveringQuadcopter environment."""

    def __init__(self, env: HoveringQuadcopterEnv, window_name: str = "IsaacLab"):
        """Initialize the window.

        Args:
            env: The environment object.
            window_name: The name of the window. Defaults to "IsaacLab".
        """
        # initialize base window
        super().__init__(env, window_name)
        # add custom UI elements
        with self.ui_window_elements["main_vstack"]:
            with self.ui_window_elements["debug_frame"]:
                with self.ui_window_elements["debug_vstack"]:
                    # add command manager visualization
                    self._create_debug_vis_ui_element("targets", self.env)


@configclass
class HoveringQuadcopterEnvCfg(DirectRLEnvCfg):
    eval_mode = False
    # env
    episode_length_s = 5.0
    decimation = 2
    action_space = 4
    observation_space = 12
    state_space = 0
    debug_vis = True
    dt = _SIM_DT

    # Offset ranges when generating the next goal (randomly)
    goal_offset_range_x: tuple[float, float] = (-0.5, 0.5)
    goal_offset_range_y: tuple[float, float] = (-0.5, 0.5)
    goal_offset_range_z: tuple[float, float] = (-0.3, 0.3)

    ui_window_class_type = HoveringQuadcopterEnvWindow

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=dt,
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
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        debug_vis=False,
    )

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=4096, env_spacing=2.5, replicate_physics=True, clone_in_fabric=True
    )

    # Randomization
    events: EventCfg | None = make_event_cfg()

    # robot
    robot: ArticulationCfg = CRAZYFLIE_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    thrust_to_weight = 1.9
    moment_scale = 0.01

    # Persistent disturbance applied every step alongside thrust (sampled once per episode at reset)
    disturbance_force_range: tuple[float, float] = (-2.0, 2.0)   # N per axis
    disturbance_torque_range: tuple[float, float] = (-0.2, 0.2)  # N·m per axis


@configclass
class HoveringQuadcopterEnvCfg_PLAY(HoveringQuadcopterEnvCfg):
    eval_mode = True
    episode_length_s = 10.0

    # Randomized. 0.5 = goal swaps every 2 seconds, 0.25 = 4 sec etc.
    goal_swap_prob_seconds = 0.05
    goal_swap_prob_step = _SIM_DT * goal_swap_prob_seconds

    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=64, env_spacing=2.5, replicate_physics=True, clone_in_fabric=True
    )


@configclass
class HoveringQuadcopterEnvCfg_PLAY_TRAIN(HoveringQuadcopterEnvCfg_PLAY):
    """Evaluate under exact training conditions (same randomization as train)."""
    pass


@configclass
class HoveringQuadcopterEnvCfg_PLAY_IDEAL(HoveringQuadcopterEnvCfg_PLAY):
    """No randomization — ideal, undisturbed conditions."""
    events = None
    disturbance_force_range: tuple[float, float] = (0.0, 0.0)
    disturbance_torque_range: tuple[float, float] = (0.0, 0.0)


@configclass
class HoveringQuadcopterEnvCfg_PLAY_EASY(HoveringQuadcopterEnvCfg_PLAY):
    """Light randomization — mild disturbances, small mass variation, infrequent pushes."""
    events: EventCfg = make_event_cfg(
        mass_scale_range=(0.8, 1.2),
        push_velocity_range=0.3,
        push_interval_range_s=(0.1, 1.0),
    )
    disturbance_force_range: tuple[float, float] = (-1.0, 1.0)
    disturbance_torque_range: tuple[float, float] = (-0.1, 0.1)


@configclass
class HoveringQuadcopterEnvCfg_PLAY_HARD(HoveringQuadcopterEnvCfg_PLAY):
    """Heavy randomization — large disturbances, wide mass variation, frequent pushes."""
    events: EventCfg = make_event_cfg(
        mass_scale_range=(0.5, 2.0),
        push_velocity_range=1.0,
        push_interval_range_s=(0.1, 0.3),
    )
    disturbance_force_range: tuple[float, float] = (-4.0, 4.0)
    disturbance_torque_range: tuple[float, float] = (-0.4, 0.4)



