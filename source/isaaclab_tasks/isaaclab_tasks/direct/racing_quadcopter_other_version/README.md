# Quadcopter Racing Environment - Code Structure

## Overview
This quadcopter racing environment has been refactored into a clean, modular architecture with separate manager classes for different responsibilities.

## File Structure

### 1. `trajectory.py` - Trajectory Management
**Class: `RacingTrajectory`**
- Generates and manages racing trajectories (circle, oval, figure-8)
- Tracks drone progress along trajectory (arc-length based)
- Computes closest points and contour errors
- Provides look-ahead points for anticipation
- Handles velocity limits with randomization

**Class: `TrajectoryConfig`**
- Configuration dataclass for trajectory parameters

### 2. `managers.py` - Component Managers
**Class: `RewardManager`**
- Computes all reward components:
  - Progress reward (forward motion along trajectory)
  - Contour error (distance from racing line)
  - Velocity alignment (direction matching)
  - Speed tracking (maintain target speed)
  - Velocity limit penalty (don't exceed limits)
  - Orientation penalty (stay upright)
  - Angular velocity penalty (smooth flight)
  - Action smoothness (no jerky controls)
- Tracks episode statistics for logging
- Handles per-environment reward state

**Class: `ObservationManager`**
- Generates observation vectors for the policy network:
  - Current velocities (linear & angular)
  - Gravity direction (orientation info)
  - Velocity limit
  - Look-ahead trajectory points (in body frame)
  - Historical velocities and actions
- Maintains observation history buffers
- Handles coordinate transformations

**Class: `DisturbanceManager`**
- Applies random force and torque disturbances
- Configurable scales and enable/disable flag
- Improves sim2real transfer robustness

**Configuration Dataclasses:**
- `RewardConfig` - Reward component scales
- `ObservationConfig` - Observation parameters

### 3. `trajectory_generator.py` - Racing Track Generation
**Class: `TrajectoryLibrary`**
- Generates 100 diverse racing trajectories using procedural generation
- 5 different track types with varying complexity:
  - Banked loops (easiest)
  - Figure-8 patterns (medium-easy)
  - Cloverleaf patterns (medium)
  - Racing circuits with straights and corners (medium-hard)
  - Complex spline-based tracks (hardest)
- Uses Catmull-Rom splines for smooth interpolation
- Computes difficulty scores for curriculum learning
- Parametrized generation with configurable ranges

**Class: `TrajectoryGeneratorCfg`**
- Configuration for trajectory generation parameters
- Defines ranges for radius, height, banking, complexity, etc.
- Eliminates hardcoded values for better tuning

### 4. `curriculum.py` - Curriculum Learning
**Class: `TrajectoryCurriculumManager`**
- Manages progressive difficulty for trajectory selection
- Per-environment difficulty tracking
- Automatic advancement based on performance:
  - Success criteria: low contour error + high progress velocity
  - Requires consistent streak of successes to advance
  - Optional demotion on prolonged failure
- Inspired by Isaac Lab's DifficultyScheduler pattern
- Provides statistics for monitoring progress

**Class: `CurriculumCfg`**
- Configuration for curriculum behavior:
  - Success/failure thresholds
  - Streak requirements
  - Difficulty increment/decrement rates
  - Initial and maximum difficulty levels

### 5. `quadcopter_env.py` - Main Environment
**Class: `QuadcopterEnvCopy`**
- Inherits from `DirectRLEnv`
- Coordinates all managers including curriculum
- Handles physics simulation
- Manages episode resets with difficulty-based trajectory selection
- Tracks episode metrics for curriculum updates
- Provides debug visualization

**Class: `QuadcopterEnvCfgCopy`**
- Environment configuration using manager configs
- All parameters in one place
- Curriculum learning settings

## Design Benefits

### 1. **Separation of Concerns**
- Each manager handles one responsibility
- Easy to test components independently
- Clear interfaces between modules

### 2. **Maintainability**
- Reward functions cleanly separated
- Easy to add/remove reward components
- Observation generation isolated
- Configuration centralized

### 3. **Extensibility**
- Add new reward terms by extending `RewardManager`
- Add new observation features in `ObservationManager`
- Create new trajectory types in `RacingTrajectory`
- Swap out managers without changing main environment

### 4. **Code Quality**
- Comprehensive type hints
- Docstrings for all public methods
- Clear variable naming
- Logical organization

### 5. **Reusability**
- Managers can be used in other environments
- Trajectory system is environment-agnostic
- Configuration dataclasses are self-documenting

## Key Features

### Racing Trajectory Library
- 100 precomputed racing tracks with varying difficulty
- Procedurally generated using mathematical functions and splines
- Smooth, continuous 3D trajectories optimized for drone racing
- Automatic difficulty scoring for curriculum learning
- Configurable generation parameters (no hardcoded values)

### Curriculum Learning
- Progressive difficulty for improved training
- Starts with simple tracks, advances to complex ones
- Per-environment difficulty tracking
- Performance-based advancement:
  - Advances when agent consistently follows trajectory well
  - Can demote on prolonged failure (optional)
- Logged statistics for monitoring progress
- Configurable thresholds and progression rates

### Time-Independent Tracking
- Progress based on arc-length, not time
- Drone can adjust speed naturally
- No penalties for going slow around corners

### Look-Ahead Observations
- 6 future trajectory points at fixed distances
- Enables anticipation of upcoming turns
- All in body frame for learning efficiency

### Robustness Features
- Random velocity limits per episode
- Dynamic force/torque disturbances
- Observation history for temporal reasoning
- Smoothness penalties for stable control

### Comprehensive Rewards
- 8 different reward components
- Configurable scales for tuning
- Automatic episode statistics logging
- Progress and contour error tracking

## Usage Example

```python
from quadcopter_env import QuadcopterEnvCfgCopy, QuadcopterEnvCopy
from trajectory_generator import TrajectoryGeneratorCfg
from curriculum import CurriculumCfg

# Create config
cfg = QuadcopterEnvCfgCopy()

# Customize trajectory library
cfg.trajectory.use_trajectory_library = True
cfg.trajectory.num_library_trajectories = 100
cfg.trajectory.trajectory_generator_cfg = TrajectoryGeneratorCfg(
    radius_range=(1.5, 3.5),
    height_range=(1.0, 2.0),
    banking_angle_range=(-0.2, 0.2),
)

# Enable and configure curriculum learning
cfg.enable_curriculum = True
cfg.curriculum = CurriculumCfg(
    contour_error_threshold=0.15,
    progress_velocity_threshold=0.8,
    success_streak_required=50,
    initial_difficulty=0.0,
    difficulty_increment=0.05,
)

# Customize rewards
cfg.reward.progress_scale = 15.0
cfg.reward.contour_error_scale = -3.0

# Enable/disable disturbances
cfg.enable_disturbances = True

# Create environment
env = QuadcopterEnvCopy(cfg)
```

## Performance Considerations

- Managers operate on batched tensors (all envs at once)
- Minimal data copying between managers
- Efficient trajectory lookups using vectorized operations
- History updates use `torch.roll` for speed
- Trajectory library precomputed once at initialization
- Difficulty-based selection uses sorted indexing (O(1))

## Curriculum Learning Details

### How It Works
1. **Initialization**: All environments start at `initial_difficulty` (default: 0.0 = easiest)
2. **Episode Tracking**: During episodes, contour error and progress velocity are accumulated
3. **Performance Evaluation**: At episode end, mean metrics are computed
4. **Advancement**: If metrics meet thresholds for `success_streak_required` consecutive episodes, difficulty increases
5. **Trajectory Selection**: Higher difficulty → more complex tracks (splines, tight corners, banking)
6. **Optional Demotion**: If agent fails for many consecutive episodes, difficulty can decrease

### Logged Metrics
- `curriculum/mean_difficulty`: Average difficulty across all environments
- `curriculum/min_difficulty`: Easiest current difficulty
- `curriculum/max_difficulty`: Hardest current difficulty
- `curriculum/mean_success_streak`: Average success streak length
- `curriculum/mean_failure_streak`: Average failure streak length
- `curriculum/total_advancements`: Total difficulty increases
- `curriculum/total_demotions`: Total difficulty decreases

## Future Improvements

Potential areas for enhancement:
1. ~~Add more trajectory types (race tracks from files)~~ ✅ Done
2. Implement dynamic velocity limits that change along trajectory
3. Add wind disturbances with spatial variation
4. Support multiple drones with collision avoidance
5. ~~Add curriculum learning for progressive difficulty~~ ✅ Done
6. Implement learned reward functions
