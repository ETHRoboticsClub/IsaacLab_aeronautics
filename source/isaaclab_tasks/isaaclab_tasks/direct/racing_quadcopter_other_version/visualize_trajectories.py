#!/usr/bin/env python3
"""Trajectory visualization tool for debugging and analysis.

This script generates and visualizes racing trajectories using matplotlib.
Useful for debugging trajectory generation parameters and understanding
difficulty progression in the curriculum learning system.

Usage:
    python visualize_trajectories.py --type figure8 --difficulty 0.5
    python visualize_trajectories.py --all-types
    python visualize_trajectories.py --library 0 --show-difficulty-range

# run as
#   ./_isaac_sim/python.sh /workspace/isaaclab/source/isaaclab_tasks/isaaclab_tasks/direct/racing_quadcopter/visualize_trajectories.py
"""

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for headless environments
import argparse
try:
    import matplotlib.pyplot as plt
except ImportError:
    raise ImportError("matplotlib is required for visualization. Please install it inside the container.")
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import torch

# Add isaaclab to path
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2] / "source"))

from trajectory_generator import (
    TrajectoryLibrary,
    TrajectoryGeneratorCfg,
)

# Get script directory for saving plots
SCRIPT_DIR = Path(__file__).resolve().parent

PLOT_DIR = SCRIPT_DIR / "trajectory_viz_plots"


def visualize_single_trajectory(traj_type: str, difficulty: float = 0.5, seed: int = 42):
    """Visualize a single trajectory of specified type and difficulty.
    
    Args:
        traj_type: Trajectory type (banked_loop, figure8, cloverleaf, racing_circuit, spline)
        difficulty: Difficulty level [0, 1]
        seed: Random seed for reproducibility
    """
    # Create generator with single library of specified type
    cfg = TrajectoryGeneratorCfg(
        num_libraries=1,
        trajectories_per_library=100,
    )
    generator = TrajectoryLibrary(cfg=cfg, device="cpu", seed=seed)
    
    # Generate library
    library_type = traj_type
    waypoints, tangents, difficulties = generator._generate_single_library(0, library_type)
    
    # Find trajectory closest to desired difficulty
    diff_errors = torch.abs(difficulties - difficulty)
    selected_idx = torch.argmin(diff_errors)
    actual_difficulty = difficulties[selected_idx].item()
    
    # Get waypoint data
    selected_waypoints = waypoints[selected_idx].numpy()
    selected_tangents = tangents[selected_idx].numpy()
    
    # Calculate trajectory arc length and generate gates
    segment_lengths = np.linalg.norm(selected_waypoints[1:] - selected_waypoints[:-1], axis=1)
    last_segment = np.linalg.norm(selected_waypoints[0] - selected_waypoints[-1])
    segment_lengths = np.append(segment_lengths, last_segment)
    cumulative_arc = np.zeros(len(selected_waypoints))
    cumulative_arc[1:] = np.cumsum(segment_lengths[:-1])
    total_arc_length = cumulative_arc[-1] + segment_lengths[-1]
    
    # Generate gate positions (every 10m)
    gate_spacing = 10.0
    num_gates = int(total_arc_length / gate_spacing)
    gate_positions = []
    for i in range(num_gates):
        target_arc = i * gate_spacing
        idx = np.searchsorted(cumulative_arc, target_arc)
        idx = np.clip(idx, 0, len(selected_waypoints) - 1)
        gate_positions.append(selected_waypoints[idx])
    gate_positions = np.array(gate_positions)
    
    # Create 3D plot
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot trajectory
    ax.plot(
        selected_waypoints[:, 0],
        selected_waypoints[:, 1],
        selected_waypoints[:, 2],
        'b-', linewidth=2, label='Trajectory'
    )
    
    # Plot waypoints
    ax.scatter(
        selected_waypoints[:, 0],
        selected_waypoints[:, 1],
        selected_waypoints[:, 2],
        c='red', s=10, alpha=0.2, label='Waypoints'
    )
    
    # Plot gates
    if len(gate_positions) > 0:
        ax.scatter(
            gate_positions[:, 0],
            gate_positions[:, 1],
            gate_positions[:, 2],
            c='orange', s=150, marker='s', alpha=0.8, label=f'Gates (n={num_gates})', edgecolors='black', linewidths=2
        )
    
    # Plot tangent vectors at intervals
    step = max(1, len(selected_waypoints) // 20)
    for i in range(0, len(selected_waypoints), step):
        pos = selected_waypoints[i]
        tang = selected_tangents[i] * 0.3  # Scale for visibility
        ax.quiver(
            pos[0], pos[1], pos[2],
            tang[0], tang[1], tang[2],
            color='green', alpha=0.6, arrow_length_ratio=0.3
        )
    
    # Mark start point
    ax.scatter(
        [selected_waypoints[0, 0]],
        [selected_waypoints[0, 1]],
        [selected_waypoints[0, 2]],
        c='lime', s=200, marker='*', label='Start', edgecolors='black', linewidths=2
    )
    
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title(f'{traj_type.replace("_", " ").title()}\\nDifficulty: {actual_difficulty:.3f} (requested: {difficulty:.3f})\\nTotal Length: {total_arc_length:.1f}m')
    ax.legend()
    ax.set_box_aspect([1, 1, 0.5])
    
    plt.tight_layout()
    plot_path = PLOT_DIR / f"trajectory_{traj_type}_difficulty_{difficulty:.2f}.png"
    plt.savefig(str(plot_path))
    print(f"Saved plot to {plot_path}")


def visualize_all_types(difficulty: float = 0.5, seed: int = 42):
    """Visualize all trajectory types at specified difficulty.
    
    Args:
        difficulty: Difficulty level [0, 1]
        seed: Random seed for reproducibility
    """
    trajectory_types = ["banked_loop", "figure8", "cloverleaf", "racing_circuit", "spline"]
    
    fig = plt.figure(figsize=(20, 12))
    
    # Create generator
    cfg = TrajectoryGeneratorCfg(
        num_libraries=len(trajectory_types),
        trajectories_per_library=100,
    )
    generator = TrajectoryLibrary(cfg=cfg, device="cpu", seed=seed)
    
    for idx, traj_type in enumerate(trajectory_types):
        # Generate library
        waypoints, tangents, difficulties = generator._generate_single_library(idx, traj_type)
        
        # Find trajectory closest to desired difficulty
        diff_errors = torch.abs(difficulties - difficulty)
        selected_idx = torch.argmin(diff_errors)
        actual_difficulty = difficulties[selected_idx].item()
        
        # Get waypoint data
        selected_waypoints = waypoints[selected_idx].numpy()
        
        # Create subplot
        ax = fig.add_subplot(2, 3, idx + 1, projection='3d')
        
        # Plot trajectory
        ax.plot(
            selected_waypoints[:, 0],
            selected_waypoints[:, 1],
            selected_waypoints[:, 2],
            'b-', linewidth=2
        )
        
        # Plot waypoints
        ax.scatter(
            selected_waypoints[:, 0],
            selected_waypoints[:, 1],
            selected_waypoints[:, 2],
            c='red', s=10, alpha=0.2
        )
        
        # Mark start point
        ax.scatter(
            [selected_waypoints[0, 0]],
            [selected_waypoints[0, 1]],
            [selected_waypoints[0, 2]],
            c='lime', s=1000, marker='*', edgecolors='black', linewidths=1
        )
        
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_title(f'{traj_type.replace("_", " ").title()}\nDifficulty: {actual_difficulty:.3f}')
        ax.set_box_aspect([1, 1, 0.5])
    
    plt.tight_layout()
    plot_path = PLOT_DIR / f"all_trajectory_types_difficulty_{difficulty:.2f}.png"
    plt.savefig(str(plot_path))
    print(f"Saved plot to {plot_path}")


def visualize_difficulty_range(traj_type: str, library_idx: int = 0, seed: int = 42):
    """Visualize difficulty range within a library.
    
    Shows trajectories at different difficulty levels from the same library.
    
    Args:
        traj_type: Trajectory type
        library_idx: Library index to visualize
        seed: Random seed for reproducibility
    """
    # Create generator
    cfg = TrajectoryGeneratorCfg(
        num_libraries=1,
        trajectories_per_library=100,
    )
    generator = TrajectoryLibrary(cfg=cfg, device="cpu", seed=seed)
    
    # Generate library
    waypoints, tangents, difficulties = generator._generate_single_library(0, traj_type)
    
    # Select 5 difficulty levels
    difficulty_levels = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    
    fig = plt.figure(figsize=(20, 12))
    
    for idx, target_diff in enumerate(difficulty_levels):
        # Find trajectory closest to target difficulty
        diff_errors = torch.abs(difficulties - target_diff)
        selected_idx = torch.argmin(diff_errors)
        actual_difficulty = difficulties[selected_idx].item()
        
        # Get waypoint data
        selected_waypoints = waypoints[selected_idx].numpy()
        
        # Create subplot
        ax = fig.add_subplot(2, 3, idx + 1, projection='3d')
        
        # Plot trajectory
        ax.plot(
            selected_waypoints[:, 0],
            selected_waypoints[:, 1],
            selected_waypoints[:, 2],
            'b-', linewidth=2
        )
        
        # Plot waypoints
        ax.scatter(
            selected_waypoints[:, 0],
            selected_waypoints[:, 1],
            selected_waypoints[:, 2],
            c='red', s=10, alpha=0.2
        )
        
        # Mark start point
        ax.scatter(
            [selected_waypoints[0, 0]],
            [selected_waypoints[0, 1]],
            [selected_waypoints[0, 2]],
            c='lime', s=100, marker='*', edgecolors='black', linewidths=1
        )
        
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_title(f'Difficulty: {actual_difficulty:.3f}')
        ax.set_box_aspect([1, 1, 0.5])
    
    # Add overall title
    fig.suptitle(f'{traj_type.replace("_", " ").title()} - Difficulty Progression', fontsize=16, y=0.995)
    
    plt.tight_layout()
    plot_path = PLOT_DIR / f"difficulty_range_{traj_type}.png"
    plt.savefig(str(plot_path))
    print(f"Saved plot to {plot_path}")


def visualize_all_types_difficulty_range(seed: int = 42):
    """Visualize difficulty range for all trajectory types.
    
    Shows 5 different difficulty levels for each trajectory type.
    Generates one figure per trajectory type.
    
    Args:
        seed: Random seed for reproducibility
    """
    trajectory_types = ["banked_loop", "figure8", "cloverleaf", "racing_circuit", "spline"]
    
    # Create generator with one library per type
    cfg = TrajectoryGeneratorCfg(
        num_libraries=len(trajectory_types),
        trajectories_per_library=100,
    )
    generator = TrajectoryLibrary(cfg=cfg, device="cpu", seed=seed)
    
    # Create one figure per trajectory type
    for traj_idx, traj_type in enumerate(trajectory_types):
        fig = plt.figure(figsize=(20, 12))
        
        # Generate library for this type
        waypoints, tangents, difficulties = generator._generate_single_library(traj_idx, traj_type)
        
        # Get actual difficulty range and sample evenly from it
        min_diff = difficulties.min().item()
        max_diff = difficulties.max().item()
        print(f"{traj_type}: difficulty range [{min_diff:.3f}, {max_diff:.3f}]")
        
        # Create 6 evenly-spaced difficulty levels from actual range
        difficulty_levels = [min_diff + i * (max_diff - min_diff) / 5 for i in range(6)]
        
        for idx, target_diff in enumerate(difficulty_levels):
            # Find trajectory closest to target difficulty
            diff_errors = torch.abs(difficulties - target_diff)
            selected_idx = torch.argmin(diff_errors)
            actual_difficulty = difficulties[selected_idx].item()
            
            # Get waypoint data
            selected_waypoints = waypoints[selected_idx].numpy()
            selected_tangents = tangents[selected_idx].numpy()
            
            # Compute gates
            gate_spacing = 10.0  # meters
            segment_lengths = np.linalg.norm(np.diff(selected_waypoints, axis=0), axis=1)
            cumulative_arc = np.concatenate([[0], np.cumsum(segment_lengths)])
            total_arc = cumulative_arc[-1]
            num_gates = int(total_arc / gate_spacing)
            
            gate_positions = []
            for g in range(num_gates):
                target_arc = g * gate_spacing
                gate_idx = np.searchsorted(cumulative_arc, target_arc)
                gate_idx = np.clip(gate_idx, 0, len(selected_waypoints) - 1)
                gate_positions.append(selected_waypoints[gate_idx])
            gate_positions = np.array(gate_positions)
            
            # Create subplot
            ax = fig.add_subplot(2, 3, idx + 1, projection='3d')
            
            # Plot trajectory
            ax.plot(
                selected_waypoints[:, 0],
                selected_waypoints[:, 1],
                selected_waypoints[:, 2],
                'b-', linewidth=2
            )
            
            # Plot waypoints
            ax.scatter(
                selected_waypoints[:, 0],
                selected_waypoints[:, 1],
                selected_waypoints[:, 2],
                c='red', s=10, alpha=0.2
            )
            
            # Plot gates
            if len(gate_positions) > 0:
                ax.scatter(
                    gate_positions[:, 0],
                    gate_positions[:, 1],
                    gate_positions[:, 2],
                    c='orange', s=100, marker='s', alpha=0.7, edgecolors='black', linewidths=1.5
                )
            
            # Mark start point
            ax.scatter(
                [selected_waypoints[0, 0]],
                [selected_waypoints[0, 1]],
                [selected_waypoints[0, 2]],
                c='lime', s=100, marker='*', edgecolors='black', linewidths=1
            )
            
            ax.set_xlabel('X (m)')
            ax.set_ylabel('Y (m)')
            ax.set_zlabel('Z (m)')
            ax.set_title(f'Difficulty: {actual_difficulty:.3f}')
            ax.set_box_aspect([1, 1, 0.5])
        
        # Add overall title
        fig.suptitle(f'{traj_type.replace("_", " ").title()} - Difficulty Progression', fontsize=16, y=0.995)
        
        plt.tight_layout()
        plot_path = PLOT_DIR / f"all_types_difficulty_range_{traj_type}.png"
        plt.savefig(str(plot_path))
        print(f"Saved plot to {plot_path}")
        plt.close(fig)


def visualize_library_statistics(seed: int = 42):
    """Show statistics about trajectory generation and difficulty distribution.
    
    Args:
        seed: Random seed for reproducibility
    """
    # Create generator with default configuration
    cfg = TrajectoryGeneratorCfg(
        num_libraries=30,
        trajectories_per_library=100,
    )
    generator = TrajectoryLibrary(cfg=cfg, device="cpu", seed=seed)
    
    # Collect statistics
    trajectory_types = TrajectoryLibrary.TRAJECTORY_TYPES
    type_counts = {t: 0 for t in trajectory_types}
    difficulty_distributions = {t: [] for t in trajectory_types}
    
    for lib_idx in range(cfg.num_libraries):
        lib_type = generator.get_library_type(lib_idx)
        type_counts[lib_type] += 1
        
        # Sample a few trajectories to get difficulty distribution
        waypoints, tangents, difficulties = generator._generate_single_library(lib_idx, lib_type)
        difficulty_distributions[lib_type].extend(difficulties.numpy().tolist())
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot 1: Library type distribution
    ax1 = axes[0]
    types = list(type_counts.keys())
    counts = [type_counts[t] for t in types]
    colors = plt.cm.Set3(np.linspace(0, 1, len(types)))
    
    bars = ax1.bar(range(len(types)), counts, color=colors, edgecolor='black')
    ax1.set_xticks(range(len(types)))
    ax1.set_xticklabels([t.replace('_', ' ').title() for t in types], rotation=45, ha='right')
    ax1.set_ylabel('Number of Libraries')
    ax1.set_title(f'Library Distribution Across {cfg.num_libraries} Libraries')
    ax1.grid(axis='y', alpha=0.3)
    
    # Add count labels on bars
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{count}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Plot 2: Difficulty distribution by type
    ax2 = axes[1]
    positions = []
    data = []
    labels = []
    
    for idx, traj_type in enumerate(types):
        if difficulty_distributions[traj_type]:
            positions.append(idx)
            data.append(difficulty_distributions[traj_type])
            labels.append(traj_type.replace('_', ' ').title())
    
    bp = ax2.boxplot(data, positions=positions, widths=0.6, patch_artist=True,
                     labels=labels, showfliers=False)
    
    # Color boxes to match bar chart
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax2.set_ylabel('Difficulty Level')
    ax2.set_title('Difficulty Distribution by Trajectory Type')
    ax2.set_ylim(-0.05, 1.05)
    ax2.grid(axis='y', alpha=0.3)
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    plot_path = PLOT_DIR / "library_statistics.png"
    plt.savefig(str(plot_path))
    print(f"Saved plot to {plot_path}")
    
    # Print summary statistics
    print("\n" + "="*60)
    print("TRAJECTORY LIBRARY STATISTICS")
    print("="*60)
    print(f"Total Libraries: {cfg.num_libraries}")
    print(f"Trajectories per Library: {cfg.trajectories_per_library}")
    print(f"Total Trajectories: {cfg.num_libraries * cfg.trajectories_per_library}")
    print("\nLibrary Type Distribution:")
    for traj_type in types:
        count = type_counts[traj_type]
        percentage = (count / cfg.num_libraries) * 100
        print(f"  {traj_type.replace('_', ' ').title():<20} {count:>3} libraries ({percentage:>5.1f}%)")
    
    print("\nDifficulty Ranges by Type:")
    for traj_type in types:
        if difficulty_distributions[traj_type]:
            diffs = difficulty_distributions[traj_type]
            print(f"  {traj_type.replace('_', ' ').title():<20} "
                  f"min={min(diffs):.3f}, max={max(diffs):.3f}, "
                  f"mean={np.mean(diffs):.3f}, std={np.std(diffs):.3f}")
    print("="*60 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize racing trajectories for debugging and analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Visualize a single figure-8 trajectory at medium difficulty
  python visualize_trajectories.py --type figure8 --difficulty 0.5
  
  # Show all trajectory types at high difficulty
  python visualize_trajectories.py --all-types --difficulty 0.8
  
  # Show difficulty progression for spline trajectories
  python visualize_trajectories.py --type spline --show-difficulty-range
  
  # Show library statistics and distributions
  python visualize_trajectories.py --statistics
        """
    )
    
    parser.add_argument(
        '--type',
        type=str,
        choices=['banked_loop', 'figure8', 'cloverleaf', 'racing_circuit', 'spline'],
        help='Trajectory type to visualize'
    )
    parser.add_argument(
        '--difficulty',
        type=float,
        default=0.5,
        help='Target difficulty level [0, 1] (default: 0.5)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    parser.add_argument(
        '--all-types',
        action='store_true',
        help='Visualize all trajectory types'
    )
    parser.add_argument(
        '--show-difficulty-range',
        action='store_true',
        help='Show trajectories at different difficulty levels'
    )
    parser.add_argument(
        '--library',
        type=int,
        default=0,
        help='Library index to visualize (default: 0)'
    )
    parser.add_argument(
        '--statistics',
        action='store_true',
        help='Show library statistics and distributions'
    )
    
    args = parser.parse_args()

    PLOT_DIR.mkdir(exist_ok=True)
    
    # Validate arguments
    if args.statistics:
        visualize_library_statistics(args.seed)
    elif args.all_types and args.show_difficulty_range:
        # Show difficulty range for all trajectory types
        visualize_all_types_difficulty_range(args.seed)
    elif args.all_types:
        visualize_all_types(args.difficulty, args.seed)
    elif args.show_difficulty_range:
        if not args.type:
            parser.error("--show-difficulty-range requires --type")
        visualize_difficulty_range(args.type, args.library, args.seed)
    elif args.type:
        visualize_single_trajectory(args.type, args.difficulty, args.seed)
    else:
        parser.print_help()
        print("\nNo visualization option specified. Use --help for usage examples.")


if __name__ == "__main__":
    main()
