# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for racing gate terrains."""

import isaaclab.terrains as terrain_gen

from ..terrain_generator_cfg import TerrainGeneratorCfg

__all__ = ["RACING_GATES_CFG", "RACING_GATES_SIMPLE_CFG", "create_racing_gates_terrain"]

RACING_GATES_CFG = TerrainGeneratorCfg(
    size=(10.0, 10.0),
    border_width=0.5,
    num_rows=8,
    num_cols=8,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    curriculum=True,  # Enable curriculum for difficulty progression
    difficulty_range=(0.0, 1.0),
    sub_terrains={
        "racing_gates_figure8": terrain_gen.MeshRacingGatesTerrainCfg(
            proportion=0.4,
            trajectory_type="figure8",
            num_waypoints=100,
            gate_spacing=3.0,
            gate_size_range=(0.4, 0.8),
            gate_height_range=(1.0, 2.0),
            bar_thickness=0.05,
            gate_color=(1.0, 0.3, 0.0),  # Orange
        ),
        "racing_gates_oval": terrain_gen.MeshRacingGatesTerrainCfg(
            proportion=0.3,
            trajectory_type="oval",
            num_waypoints=80,
            gate_spacing=3.5,
            gate_size_range=(0.4, 0.8),
            gate_height_range=(1.0, 2.0),
            bar_thickness=0.05,
            gate_color=(0.0, 0.8, 0.2),  # Green
        ),
        "racing_gates_random": terrain_gen.MeshRacingGatesTerrainCfg(
            proportion=0.3,
            trajectory_type="random",
            num_waypoints=120,
            gate_spacing=2.5,
            gate_size_range=(0.3, 0.7),
            gate_height_range=(1.0, 2.5),
            bar_thickness=0.05,
            gate_color=(0.2, 0.4, 1.0),  # Blue
        ),
    },
)
"""Racing gates terrain configuration with multiple track types."""


RACING_GATES_SIMPLE_CFG = TerrainGeneratorCfg(
    size=(10.0, 10.0),
    border_width=0.5,
    num_rows=4,
    num_cols=4,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    curriculum=False,
    sub_terrains={
        "racing_gates": terrain_gen.MeshRacingGatesTerrainCfg(
            proportion=1.0,
            trajectory_type="figure8",
            num_waypoints=200,  # Enough waypoints for minimum gate count
            gate_spacing=3.0,  # Base spacing
            gate_spacing_range=(2.0, 4.0),  # Random spacing range for variety
            min_gates=3,  # Must match num_next_gates in GateObservationConfig
            gate_size_range=(0.6, 0.6),  # Slightly larger gates
            gate_height_range=(1.5, 1.5),  # Fixed height
            bar_thickness=0.05,
            gate_color=(1.0, 0.3, 0.0),
        ),
    },
)
"""Simple racing gates configuration for basic training."""

RACING_GATES_RANDOM_CFG = TerrainGeneratorCfg(
    size=(15.0, 15.0),
    difficulty_range=(0.7, 1.0),  # Force high difficulty for complex trajectories
    border_width=0.5,
    num_rows=4,
    num_cols=4,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    curriculum=False,
    sub_terrains={
        "racing_gates": terrain_gen.MeshRacingGatesTerrainCfg(
            proportion=1.0,
            trajectory_type="random",  # More complex than figure8
            num_waypoints=300,  # More waypoints for smoother, longer curves
            gate_spacing=2.0,  # Closer spacing for more gates
            gate_spacing_range=(1.5, 2.5),  # Tighter range for more consistent density
            min_gates=10,  # Ensure at least 10 gates (more than doubled from original)
            gate_size_range=(0.5, 0.7),  # Slightly smaller gates for more challenge
            gate_height_range=(0.8, 3.5),  # Much wider height range for vertical variety
            bar_thickness=0.05,
            gate_color=(1.0, 0.3, 0.0),
        ),
    },
)
"""Simple racing gates configuration for basic training."""


def create_racing_gates_terrain(
    size: tuple[float, float] = (10.0, 10.0),
    min_gates: int = 3,
    curriculum: bool = False,
    gate_spacing_range: tuple[float, float] | None = (2.0, 4.0),
    gate_size: float = 0.6,
    gate_height: float = 1.5,
    trajectory_type: str = "figure8",
    num_rows: int = 4,
    num_cols: int = 4,
) -> TerrainGeneratorCfg:
    """Factory function to create a racing gates terrain configuration.
    
    Args:
        size: Size of each terrain cell (width, length) in meters.
        min_gates: Minimum number of gates per environment. Should match num_next_gates in observation config.
        curriculum: Whether to enable curriculum learning with multiple terrain types.
        gate_spacing_range: Range of gate spacing (min, max) in meters. If None, uses fixed spacing of 3.0m.
        gate_size: Size of the gate opening in meters.
        gate_height: Height of gate centers above ground in meters.
        trajectory_type: Type of trajectory ("figure8", "oval", "random").
        num_rows: Number of terrain rows in the grid.
        num_cols: Number of terrain columns in the grid.
        
    Returns:
        TerrainGeneratorCfg configured with the specified parameters.
    """

    return RACING_GATES_RANDOM_CFG


    # Calculate num_waypoints to ensure enough trajectory for min_gates
    # Assume average spacing of 3.0m and add buffer
    avg_spacing = 3.0 if gate_spacing_range is None else (gate_spacing_range[0] + gate_spacing_range[1]) / 2.0
    num_waypoints = max(200, int(min_gates * avg_spacing * 1.5))
    
    if curriculum:
        return TerrainGeneratorCfg(
            size=size,
            border_width=0.5,
            num_rows=num_rows,
            num_cols=num_cols,
            horizontal_scale=0.1,
            vertical_scale=0.005,
            slope_threshold=0.75,
            use_cache=False,
            curriculum=True,
            difficulty_range=(0.0, 1.0),
            sub_terrains={
                "racing_gates_figure8": terrain_gen.MeshRacingGatesTerrainCfg(
                    proportion=0.4,
                    trajectory_type="figure8",
                    num_waypoints=num_waypoints,
                    gate_spacing=3.0,
                    gate_spacing_range=gate_spacing_range,
                    min_gates=min_gates,
                    gate_size_range=(0.4, 0.8),
                    gate_height_range=(1.0, 2.0),
                    bar_thickness=0.05,
                    gate_color=(1.0, 0.3, 0.0),
                ),
                "racing_gates_oval": terrain_gen.MeshRacingGatesTerrainCfg(
                    proportion=0.3,
                    trajectory_type="oval",
                    num_waypoints=num_waypoints,
                    gate_spacing=3.5,
                    gate_spacing_range=gate_spacing_range,
                    min_gates=min_gates,
                    gate_size_range=(0.4, 0.8),
                    gate_height_range=(1.0, 2.0),
                    bar_thickness=0.05,
                    gate_color=(0.0, 0.8, 0.2),
                ),
                "racing_gates_random": terrain_gen.MeshRacingGatesTerrainCfg(
                    proportion=0.3,
                    trajectory_type="random",
                    num_waypoints=num_waypoints,
                    gate_spacing=2.5,
                    gate_spacing_range=gate_spacing_range,
                    min_gates=min_gates,
                    gate_size_range=(0.3, 0.7),
                    gate_height_range=(1.0, 2.5),
                    bar_thickness=0.05,
                    gate_color=(0.2, 0.4, 1.0),
                ),
            },
        )
    else:
        return TerrainGeneratorCfg(
            size=size,
            border_width=0.5,
            num_rows=num_rows,
            num_cols=num_cols,
            horizontal_scale=0.1,
            vertical_scale=0.005,
            slope_threshold=0.75,
            use_cache=False,
            curriculum=False,
            sub_terrains={
                "racing_gates": terrain_gen.MeshRacingGatesTerrainCfg(
                    proportion=1.0,
                    trajectory_type=trajectory_type,
                    num_waypoints=num_waypoints,
                    gate_spacing=3.0,
                    gate_spacing_range=gate_spacing_range,
                    min_gates=min_gates,
                    gate_size_range=(gate_size, gate_size),
                    gate_height_range=(gate_height, gate_height+3.0),
                    bar_thickness=0.05,
                    gate_color=(1.0, 0.3, 0.0),
                ),
            },
        )

