# Curriculum Learning and Parametrized Trajectories Guide

## Overview

This guide explains the newly added curriculum learning system and parametrized trajectory generation for the quadcopter racing environment.

## What Changed

### 1. Parametrized Trajectory Generation

**Before:**
```python
# Hardcoded values scattered throughout trajectory generation
radius_x = 1.5 + torch.rand(1).item() * 2.0  # 1.5 to 3.5
height_base = 1.0 + torch.rand(1).item() * 1.0  # 1.0 to 2.0
num_oscillations = torch.randint(1, 4, (1,)).item()  # 1 to 3
```

**After:**
```python
# Centralized configuration
@dataclass
class TrajectoryGeneratorCfg:
    num_libraries: int = 30  # Number of independent libraries
    trajectories_per_library: int = 100  # Tracks per library
    radius_range: tuple[float, float] = (1.5, 3.5)
    height_range: tuple[float, float] = (1.0, 2.0)
    num_oscillations_range: tuple[int, int] = (1, 3)
    # ... more parameters
```

**Benefits:**
- All trajectory parameters in one place
- Easy to tune for different difficulty ranges
- No magic numbers in code
- Clear documentation of parameter ranges
- Multiple independent libraries for diversity across environments

### 2. Difficulty Scoring

Each trajectory now has a difficulty score (0.0 to 1.0) based on:
- **Banked loops**: Banking angle + oscillations + height variation
- **Figure-8**: Scale + height variation + twist
- **Cloverleaf**: Number of lobes + height variation
- **Racing circuits**: Corner tightness + banking
- **Spline tracks**: Number of control points + height variation

Trajectories are sorted by difficulty within each library for efficient curriculum selection.

### 3. Multi-Library System for Diversity

**Key Innovation**: Instead of a single shared library, the system creates **multiple independent libraries** (default: 30), each specialized in **one trajectory type**.

**Library Type Specialization**:
- Each library contains only one type of trajectory (e.g., all banked loops, or all figure-8s)
- Configurable distribution ratios (default: 20% each type)
- Makes difficulty progression more consistent within each library
- Same type → comparable difficulty metrics

**Environment Assignment**:
- Each environment is assigned to a library using `env_idx % num_libraries`
- With 2048 environments and 30 libraries:
  - ~68 environments share each library
  - Example: 6 libraries with banked loops, 6 with figure-8s, etc.
  - Total: 30 libraries × 100 trajectories = 3,000 unique tracks
  - Memory efficient: Only 3,000 tracks vs 204,800 if per-environment

**Benefits**:
- **Consistent Difficulty**: Within a library, all trajectories are same type → more comparable difficulty
- **Diversity**: Different environments train on different trajectory types
- **Scalability**: Works efficiently with thousands of environments
- **Memory Efficient**: Fixed memory footprint regardless of environment count
- **Curriculum Compatible**: Each library has full difficulty range (0.0-1.0) within its type

**Example Distribution** (30 libraries, default ratios):
```python
# Library type distribution:
banked_loop: 6 libraries (20%)        # Envs: 0, 30, 60, 90, ...
figure8: 6 libraries (20%)            # Envs: 1, 31, 61, 91, ...
cloverleaf: 6 libraries (20%)         # Envs: 2, 32, 62, 92, ...
racing_circuit: 6 libraries (20%)     # Envs: 3, 33, 63, 93, ...
spline: 6 libraries (20%)             # Envs: 4, 34, 64, 94, ...
```

### 3. Curriculum Learning System

**New Class: `TrajectoryCurriculumManager`**
- Tracks per-environment difficulty levels
- Monitors episode performance (contour error, progress velocity)
- Automatically advances/demotes difficulty based on success/failure streaks
- Provides detailed statistics for logging

**Flow:**
1. All environments start at `initial_difficulty` (default: 0.0)
2. Each environment is assigned to a library: `library_idx = env_idx % num_libraries`
3. During episodes, metrics are accumulated
4. At episode end, curriculum evaluates performance
5. Success → increment streak → advance difficulty when threshold reached
6. Failure → increment failure streak → demote difficulty (optional)
7. Next episode: trajectory selected from environment's library at current difficulty
8. **Key**: Same difficulty, different environments → different tracks (from their respective libraries)

## Configuration

### Trajectory Generation Parameters

```python
from trajectory_generator import TrajectoryGeneratorCfg

traj_cfg = TrajectoryGeneratorCfg(
    # Library structure
    num_libraries=30,                     # Number of independent libraries
    trajectories_per_library=100,         # Tracks per library
    
    # Library type distribution (must sum to 1.0)
    # Each library specializes in one type for consistent difficulty
    library_type_ratios={
        "banked_loop": 0.20,              # 20% of libraries = simple loops
        "figure8": 0.20,                  # 20% = figure-8 patterns
        "cloverleaf": 0.20,               # 20% = cloverleaf patterns
        "racing_circuit": 0.20,           # 20% = racing circuits
        "spline": 0.20,                   # 20% = complex spline tracks
    },
    # Or use None for default equal distribution
    
    # Size parameters
    radius_range=(1.5, 3.5),              # Min/max radius (meters)
    height_range=(1.0, 2.0),              # Min/max base height (meters)
    height_amplitude_range=(0.2, 0.7),    # Vertical variation (meters)
    
    # Complexity parameters
    banking_angle_range=(-0.2, 0.2),      # Banking angle (radians)
    num_oscillations_range=(1, 3),        # Vertical waves
    num_lobes_range=(3, 4),               # Cloverleaf lobes
    num_control_points_range=(5, 8),      # Spline control points
    
    # Circuit parameters
    straight_length_range=(2.0, 4.0),     # Straight sections (meters)
    corner_radius_range=(0.5, 0.8),       # Corner tightness factor
    banking_range=(0.1, 0.3),             # Corner banking
    
    # Smoothing
    smooth_window=5,                       # Smoothing window size
)
```

**Customizing Type Distribution:**
```python
# Example: Focus more on complex tracks
traj_cfg = TrajectoryGeneratorCfg(
    num_libraries=30,
    library_type_ratios={
        "banked_loop": 0.10,    # 10% = 3 libraries
        "figure8": 0.10,        # 10% = 3 libraries
        "cloverleaf": 0.20,     # 20% = 6 libraries
        "racing_circuit": 0.30, # 30% = 9 libraries
        "spline": 0.30,         # 30% = 9 libraries
    },
)

# Example: Only specific types
traj_cfg = TrajectoryGeneratorCfg(
    num_libraries=20,
    library_type_ratios={
        "racing_circuit": 0.50,  # 50% = 10 libraries
        "spline": 0.50,          # 50% = 10 libraries
        # Other types excluded (ratio = 0)
    },
)
```

### Curriculum Learning Parameters

```python
from curriculum import CurriculumCfg

curriculum_cfg = CurriculumCfg(
    # Success criteria
    contour_error_threshold=0.15,         # Max error to count as success (meters)
    progress_velocity_threshold=0.8,      # Min velocity to count as success (m/s)
    success_streak_required=50,           # Episodes needed to advance
    
    # Difficulty progression
    initial_difficulty=0.0,               # Starting difficulty (0.0 = easiest)
    difficulty_increment=0.05,            # Amount to increase on advancement
    max_difficulty=1.0,                   # Maximum difficulty (1.0 = hardest)
    
    # Demotion (optional)
    enable_demotion=True,                 # Allow difficulty decrease
    failure_streak_threshold=100,         # Episodes needed to demote
    difficulty_decrement=0.1,             # Amount to decrease on demotion
)
```

### Environment Configuration

```python
from quadcopter_env import QuadcopterEnvCfgCopy

cfg = QuadcopterEnvCfgCopy()

# Enable curriculum with custom config
cfg.enable_curriculum = True
cfg.curriculum = curriculum_cfg

# Pass trajectory generator config
cfg.trajectory.trajectory_generator_cfg = traj_cfg
```

## Difficulty Levels

### Level 0.0 - 0.2 (Easiest)
- Simple banked loops
- Small radius, minimal height variation
- No oscillations or gentle banking
- Perfect for initial learning

### Level 0.2 - 0.4 (Easy)
- Figure-8 patterns with minimal twist
- Moderate size, gentle height changes
- Good for learning trajectory following

### Level 0.4 - 0.6 (Medium)
- Cloverleaf patterns with 3 lobes
- More height variation
- Requires anticipation of turns

### Level 0.6 - 0.8 (Hard)
- Racing circuits with tight corners
- Significant banking
- Straights and hairpins
- Tests speed control

### Level 0.8 - 1.0 (Hardest)
- Complex spline-based tracks
- 7-8 control points
- Maximum height variation
- Unpredictable curves
- Requires mastery

## Monitoring Progress

### TensorBoard Metrics

The curriculum system logs several metrics to help monitor training:

```
curriculum/mean_difficulty          # Average difficulty across envs
curriculum/min_difficulty           # Lowest current difficulty
curriculum/max_difficulty           # Highest current difficulty
curriculum/mean_success_streak      # Average success streak length
curriculum/mean_failure_streak      # Average failure streak length
curriculum/total_advancements       # Total difficulty increases
curriculum/total_demotions          # Total difficulty decreases
```

### Expected Progression

Typical training progression:
1. **0-500k steps**: Difficulty 0.0-0.2 (learning basic flight)
2. **500k-2M steps**: Difficulty 0.2-0.5 (improving trajectory following)
3. **2M-5M steps**: Difficulty 0.5-0.8 (mastering complex tracks)
4. **5M+ steps**: Difficulty 0.8-1.0 (perfecting racing skills)

## Tuning Tips

### If agent struggles to advance:
- **Decrease** `contour_error_threshold` (be more lenient)
- **Decrease** `progress_velocity_threshold` (accept slower speeds)
- **Decrease** `success_streak_required` (advance faster)
- **Increase** `difficulty_increment` (smaller steps)
- **Adjust library ratios**: Increase easier types (banked_loop, figure8)

### If agent advances too quickly:
- **Increase** `contour_error_threshold` (require better performance)
- **Increase** `progress_velocity_threshold` (require faster speeds)
- **Increase** `success_streak_required` (require consistency)
- **Decrease** `difficulty_increment` (larger steps)
- **Adjust library ratios**: Increase harder types (spline, racing_circuit)

### If agent gets stuck at certain difficulty:
- **Enable** `enable_demotion=True` (allow backtracking)
- **Adjust** trajectory parameters to create smoother difficulty gradient within each type
- **Increase** `failure_streak_threshold` (give more attempts before demotion)
- **Rebalance library types**: May indicate certain types are too hard/easy

### If you want to focus on specific trajectory types:
- **Modify** `library_type_ratios` to emphasize certain types
- Example: More racing circuits for sim2real transfer
- Example: More splines for maximum challenge

## Disabling Curriculum

To train without curriculum (random trajectories):

```python
cfg = QuadcopterEnvCfgCopy()
cfg.enable_curriculum = False
cfg.trajectory.randomize_track_per_env = True
```

This will randomly select from all 100 trajectories regardless of difficulty.

## Implementation Details

### Type-Specialized Library Architecture

```python
# System creates 30 libraries, each specialized in one type
num_libraries = 30
trajectories_per_library = 100
total_trajectories = 30 * 100 = 3,000

# Default distribution (20% each)
banked_loop libraries: 6 (indices 0, 5, 10, 15, 20, 25)
figure8 libraries: 6 (indices 1, 6, 11, 16, 21, 26)
cloverleaf libraries: 6 (indices 2, 7, 12, 17, 22, 27)
racing_circuit libraries: 6 (indices 3, 8, 13, 18, 23, 28)
spline libraries: 6 (indices 4, 9, 14, 19, 24, 29)

# Environment assignment (round-robin)
env_0 → library_0 (banked_loop)
env_1 → library_1 (figure8)
env_2 → library_2 (cloverleaf)
env_3 → library_3 (racing_circuit)
env_4 → library_4 (spline)
env_5 → library_5 (banked_loop)
...
env_30 → library_0 (banked_loop)  # Wraps around

# Memory usage
Shape: [30, 100, 100, 3]  # [libraries, trajectories, waypoints, xyz]
Memory: ~3.6 MB (float32)  # Very efficient!
```

### Trajectory Selection

```python
# Curriculum mode (difficulty-based, type-specialized)
library_idx = env_idx % num_libraries
library_type = library_types[library_idx]  # e.g., "figure8"
difficulty = curriculum_manager.get_difficulty(env_idx)

# Get trajectory of correct type at correct difficulty
waypoints, tangents, traj_idx = trajectory_library.get_trajectory_by_difficulty(
    env_idx, difficulty
)

# Result: Environment gets trajectory matching its difficulty 
# from its assigned library (which has only one type)
# Difficulty 0.5 figure8 vs difficulty 0.5 banked_loop → 
# different tracks, but comparable within their types!
```

### Performance Evaluation

At each step:
```python
episode_contour_errors += contour_error
episode_progress_velocities += progress_velocity
episode_steps += 1
```

At episode end:
```python
mean_contour_error = episode_contour_errors / episode_steps
mean_progress_velocity = episode_progress_velocities / episode_steps
curriculum_manager.update(mean_contour_error, mean_progress_velocity, reset_ids)
```

### Streak Tracking

Each environment has independent success/failure streaks:
- **Success**: `contour_error < threshold AND progress_velocity > threshold`
- **Failure**: Not success
- Streaks reset when condition changes
- Advancement/demotion occurs when streak threshold reached

## Benefits

### 1. Improved Learning Efficiency
- Start simple, gradually increase complexity
- Reduces catastrophic forgetting
- Better sample efficiency

### 2. Better Final Performance
- Smooth progression builds robust policies
- Handles full range of track difficulties
- Generalizes better to unseen tracks

### 3. Training Stability
- Avoids overwhelming agent early in training
- Automatic adjustment to agent capability
- Recovers from performance drops via demotion

### 4. Interpretability
- Clear difficulty progression metrics
- Easy to diagnose training issues
- Understand agent capabilities at each stage

## Example Training Script

```python
from isaaclab_tasks.direct.quadcopter_copy import QuadcopterEnvCfgCopy
from isaaclab_tasks.direct.quadcopter_copy.curriculum import CurriculumCfg
from isaaclab_tasks.direct.quadcopter_copy.trajectory_generator import TrajectoryGeneratorCfg

# Configure environment with curriculum
cfg = QuadcopterEnvCfgCopy()
cfg.enable_curriculum = True
cfg.curriculum = CurriculumCfg(
    initial_difficulty=0.0,
    difficulty_increment=0.05,
    success_streak_required=50,
)

# Configure trajectory generation
cfg.trajectory.trajectory_generator_cfg = TrajectoryGeneratorCfg(
    radius_range=(1.5, 3.5),
    height_range=(1.0, 2.0),
)

# Create environment
env = gym.make("Isaac-Quadcopter-Direct-v0", cfg=cfg)

# Train with your favorite RL algorithm
# The curriculum will automatically adjust difficulty
```

## Troubleshooting

### Issue: All environments stuck at low difficulty
**Solution:** Check if success criteria are too strict. Lower thresholds or reduce streak requirement.

### Issue: Difficulty increases too fast
**Solution:** Increase `success_streak_required` to require more consistency before advancing.

### Issue: Agent performs well at high difficulty then fails
**Solution:** Enable demotion or reduce `difficulty_increment` for smaller steps.

### Issue: Curriculum statistics not appearing in logs
**Solution:** Ensure `cfg.enable_curriculum = True` and check TensorBoard for metrics starting with `curriculum/`.

## References

This curriculum implementation is inspired by:
- Isaac Lab's `DifficultyScheduler` pattern from manipulation tasks
- Curriculum learning principles from "Automatic Curriculum Learning" (Portelas et al., 2020)
- Progressive difficulty in "Learning Dexterity" (OpenAI, 2019)
