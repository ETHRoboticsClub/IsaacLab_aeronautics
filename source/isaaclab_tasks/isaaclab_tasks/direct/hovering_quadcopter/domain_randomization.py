from __future__ import annotations

import isaaclab.envs.mdp as mdp
from isaaclab.managers import EventTermCfg as EventTerm, SceneEntityCfg
from isaaclab.utils import configclass


@configclass
class EventCfg:
    randomize_mass: EventTerm | None = None
    push_robot: EventTerm | None = None


def make_event_cfg(
    mass_scale_range: tuple[float, float] = (0.75, 1.33),
    push_velocity_range: float = 0.75,
    push_interval_range_s: tuple[float, float] = (0.1, 0.3),
) -> EventCfg:
    """Factory that builds an EventCfg with the given DR parameters."""
    v = push_velocity_range
    return EventCfg(
        randomize_mass=EventTerm(
            func=mdp.randomize_rigid_body_mass,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
                "mass_distribution_params": mass_scale_range,
                "operation": "scale",
                "distribution": "uniform",
                "recompute_inertia": True,
            },
        ),
        push_robot=EventTerm(
            func=mdp.push_by_setting_velocity,
            mode="interval",
            interval_range_s=push_interval_range_s,
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "velocity_range": {
                    "x": (-v, v),
                    "y": (-v, v),
                    "z": (-0.3, 0.3),
                    "roll": (-0.2, 0.2),
                    "pitch": (-0.2, 0.2),
                    "yaw": (-0.2, 0.2),
                },
            },
        ),
    )
