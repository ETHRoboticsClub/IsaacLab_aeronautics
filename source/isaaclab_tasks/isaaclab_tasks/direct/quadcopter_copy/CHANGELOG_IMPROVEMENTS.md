# Quadcopter Training Improvements - Changelog

## Overview
This document summarizes three major improvements to the quadcopter racing training system:
1. **Continuous Difficulty Progression** - Performance-based curriculum advancement
2. **Trajectory Visualization Tool** - Standalone script for debugging trajectories
3. **Virtual Gate System** - Racing gates with collision detection and enhanced penalties

---

## 1. Continuous Difficulty Progression

### Motivation
The previous system used fixed difficulty increments (0.05) and decrements (0.1), which meant:
- Only discrete difficulty levels were accessed (0.0, 0.05, 0.10, ...)
- Many trajectories in libraries were unused
- Progression didn't adapt to actual performance quality

### Implementation

**Modified Files:**
- `curriculum.py` - Core curriculum learning logic
- `quadcopter_env.py` - Updated configuration

**Key Changes:**

1. **New Configuration Parameters** (`CurriculumCfg`):
```python
use_continuous_progression: bool = True  # Enable performance-based progression
base_increment: float = 0.05             # Minimum advancement step
max_increment: float = 0.15              # Maximum advancement step
base_decrement: float = 0.05             # Minimum demotion step
max_decrement: float = 0.15              # Maximum demotion step
```

2. **Performance-Based Advancement**:
- **Better performance → Faster progression**
  - Calculates `error_quality` = 1.0 - (contour_error / threshold)
  - Calculates `velocity_quality` = progress_velocity / (2 × threshold)
  - Combined performance quality scales increment: `base + quality × (max - base)`

3. **Failure-Based Demotion**:
- **Worse performance → Larger demotion**
  - Calculates `error_penalty` based on how much error exceeds threshold
  - Calculates `velocity_penalty` based on how much below required speed
  - Combined failure severity scales decrement

**Benefits:**
- ✅ All trajectories in libraries can be accessed (continuous difficulty 0.0-1.0)
- ✅ Agents that excel advance faster
- ✅ Agents that struggle get gentler progression
- ✅ Smoother learning curve with adaptive pacing

---

## 2. Trajectory Visualization Tool

### Motivation
Need a standalone tool to:
- Debug trajectory generation parameters
- Visualize difficulty distributions
- Understand library type distributions
- Test trajectory generator without full simulation

### Implementation

**New File:**
- `scripts/tools/visualize_trajectories.py` - Standalone visualization script

**Features:**

1. **Single Trajectory Visualization**:
```bash
python visualize_trajectories.py --type figure8 --difficulty 0.5
```
- Shows waypoints, tangents, gates, and start position
- Displays actual vs requested difficulty
- Shows total trajectory arc length
- **Gate visualization included** (orange squares every 10m)

2. **All Types Comparison**:
```bash
python visualize_trajectories.py --all-types --difficulty 0.8
```
- Side-by-side view of all 5 trajectory types
- Compare shapes and characteristics at same difficulty

3. **Difficulty Progression**:
```bash
python visualize_trajectories.py --type spline --show-difficulty-range
```
- Shows 5 difficulty levels: 0.0, 0.25, 0.5, 0.75, 1.0
- Visualizes how difficulty affects trajectory parameters

4. **Library Statistics**:
```bash
python visualize_trajectories.py --statistics
```
- Bar chart of library type distribution
- Box plots of difficulty ranges per type
- Printed summary statistics

**Technical Details:**
- Uses matplotlib for 3D visualization
- Imports trajectory generation modules directly
- CPU-only operation (no GPU required)
- Comprehensive command-line interface

---

## 3. Virtual Gate System

### Motivation
Racing drones must pass through gates precisely. This system:
- Adds realism to racing scenarios
- Increases difficulty near precision checkpoints
- Provides clear failure conditions (gate collision)

### Implementation

**Modified Files:**
- `trajectory.py` - Gate generation and collision detection
- `managers.py` - Gate-aware contour error penalties
- `quadcopter_env.py` - Gate collision termination + visualization

### Gate System Architecture

#### A. Configuration (`TrajectoryConfig`)

```python
enable_gates: bool = True                    # Enable virtual gates
gate_spacing: float = 10.0                   # Distance between gates (meters)
gate_size: float = 2.0                       # Gate opening size (meters)
gate_penalty_multiplier: float = 5.0         # Contour error amplification near gates
gate_collision_threshold: float = 1.0        # Collision detection radius (meters)
```

#### B. Gate Generation (`RacingTrajectory._generate_gates()`)

**Process:**
1. Calculate `num_gates = total_arc_length / gate_spacing`
2. For each gate:
   - Place at target arc-length position
   - Interpolate between waypoints for precise positioning
   - Orient perpendicular to trajectory tangent
3. Store gate positions and orientations

**Example:** 50m trajectory with 10m spacing → 5 gates

#### C. Collision Detection (`RacingTrajectory.check_gate_collision()`)

**Algorithm:**
```python
for each environment:
    1. Find distance to nearest gate
    2. Check if near_gate: distance < collision_threshold (1.0m)
    3. Check if outside_gate: contour_error > gate_size/2 (1.0m)
    4. collision = near_gate AND outside_gate
```

**Result:** Boolean mask indicating which environments hit a gate

#### D. Penalty Multiplier (`RacingTrajectory.get_gate_penalty_multiplier()`)

**Purpose:** Amplify contour error penalty near gates

**Formula:**
```python
multiplier = 1.0 + (gate_penalty_multiplier - 1.0) × exp(-distance / decay_distance)
```

**Effect:**
- At gate center: multiplier ≈ 5.0× (configured value)
- Far from gates: multiplier → 1.0× (baseline)
- Smooth exponential decay between

**Applied in `RewardManager._compute_contour_reward()`:**
```python
contour_reward = exp(-2.0 × contour_error × gate_multiplier)
```

#### E. Environment Integration

**Termination Conditions** (`quadcopter_env._get_dones()`):
```python
altitude_died = altitude < 0.1m OR altitude > 2.0m
gate_collision = check_gate_collision(position, contour_error)
died = altitude_died OR gate_collision
```

**Visualization** (`quadcopter_env._debug_vis_callback()`):
- Orange square markers at gate positions
- Gate size matches configured opening (2m × 2m)
- Visible during simulation with `debug_vis=True`

### Gate System Behavior

| Scenario | Contour Error | Near Gate? | Result |
|----------|---------------|------------|--------|
| Perfect flight through center | 0.0m | Yes | ✅ High reward, no collision |
| Slight deviation through gate | 0.5m | Yes | ⚠️ Reduced reward (5× penalty), no collision |
| Major deviation at gate | 1.2m | Yes | ❌ Collision → Episode terminates |
| Deviation away from gate | 0.8m | No | ✅ Normal penalty (1× multiplier) |

### Benefits

- ✅ **Realistic Racing**: Mimics real-world drone racing gates
- ✅ **Progressive Challenge**: Gates add precision requirements
- ✅ **Clear Feedback**: Visual gates show targets, collisions provide explicit failure signal
- ✅ **Curriculum Compatible**: Works with difficulty-based trajectory selection
- ✅ **Configurable**: All parameters tunable without code changes

---

## Usage Examples

### Training with New Features

```python
# In your training script or config
cfg = QuadcopterEnvCfgCopy(
    # Enable curriculum with continuous progression
    enable_curriculum=True,
    curriculum=CurriculumCfg(
        use_continuous_progression=True,
        base_increment=0.05,
        max_increment=0.15,
        base_decrement=0.05,
        max_decrement=0.15,
    ),
    
    # Configure gates
    trajectory=TrajectoryConfig(
        enable_gates=True,
        gate_spacing=10.0,
        gate_size=2.0,
        gate_penalty_multiplier=5.0,
        gate_collision_threshold=1.0,
    ),
)
```

### Debugging Trajectories

```bash
# Visualize a specific trajectory with gates
python scripts/tools/visualize_trajectories.py --type racing_circuit --difficulty 0.7

# Compare all types
python scripts/tools/visualize_trajectories.py --all-types

# Check library statistics
python scripts/tools/visualize_trajectories.py --statistics
```

### Monitoring Training

Watch for these new metrics in TensorBoard:

**Curriculum (already existed, now continuous):**
- `curriculum/mean_difficulty` - Should progress smoothly 0.0 → 1.0
- `curriculum/total_advancements` - Should increase steadily
- `curriculum/total_demotions` - Should be minimal if well-tuned

**Episode Termination (enhanced):**
- `Episode_Termination/died` - Now includes gate collisions
- Watch for spike if gates too challenging

**Rewards (enhanced):**
- `Episode_Reward/contour_error` - Should adapt to gate locations
- Near gates, expect more variation due to 5× multiplier

---

## Configuration Recommendations

### For Beginners (Easier)
```python
curriculum=CurriculumCfg(
    use_continuous_progression=True,
    base_increment=0.08,           # Faster advancement
    max_increment=0.20,            # Allow big jumps
    success_streak_required=30,    # Advance sooner
),
trajectory=TrajectoryConfig(
    enable_gates=True,
    gate_spacing=15.0,             # Fewer gates
    gate_size=2.5,                 # Larger openings
    gate_penalty_multiplier=3.0,   # Gentler penalties
)
```

### For Advanced Training (Harder)
```python
curriculum=CurriculumCfg(
    use_continuous_progression=True,
    base_increment=0.03,           # Slower, more gradual
    max_increment=0.10,            # Smaller steps
    success_streak_required=80,    # Require consistency
),
trajectory=TrajectoryConfig(
    enable_gates=True,
    gate_spacing=8.0,              # More gates
    gate_size=1.5,                 # Tighter openings
    gate_penalty_multiplier=7.0,   # Stricter penalties
)
```

### Disabling Features
```python
# Disable gates (back to original contour tracking)
trajectory=TrajectoryConfig(
    enable_gates=False,
)

# Disable continuous progression (use fixed steps)
curriculum=CurriculumCfg(
    use_continuous_progression=False,
    base_increment=0.05,  # Fixed step size
)
```

---

## Testing & Validation

### 1. Verify Continuous Progression
```python
# Run short training session, monitor difficulty spread
# Should see non-discrete values: 0.123, 0.287, etc. (not just 0.0, 0.05, 0.10)
```

### 2. Verify Gates Generated
```bash
# Check console output during environment creation:
# "Generated N virtual gates along trajectory (spacing: 10.0m)"
```

### 3. Verify Gate Collisions
```python
# Enable debug visualization
cfg.debug_vis = True
# Run simulation, fly drone outside gates
# Should see orange gate markers and episode terminations
```

### 4. Verify Visualization Tool
```bash
python scripts/tools/visualize_trajectories.py --all-types
# Should display 5 trajectory types with orange gate squares
```

---

## Technical Notes

### Performance Impact
- **Gate collision checking**: O(num_envs × num_gates) per step
  - Typical: 4096 envs × 5 gates = 20k distance calculations
  - Vectorized PyTorch operations, negligible overhead
- **Penalty multiplier**: O(num_envs × num_gates) per step
  - Same as collision checking
- **Continuous progression**: No performance impact (same computation, just variable increment)

### Memory Impact
- Gates stored per trajectory: `num_gates × (3 pos + 4 quat) × 4 bytes`
- Typical: 5 gates × 7 floats × 4 bytes = 140 bytes per trajectory
- Negligible compared to waypoint storage

### Compatibility
- ✅ Works with existing trajectory library system
- ✅ Compatible with all 5 trajectory types
- ✅ Does not affect observation or action spaces
- ✅ Backward compatible (can disable all new features)

---

## Future Improvements

### Potential Enhancements
1. **Dynamic Gate Sizes**: Scale gate openings with difficulty
2. **Gate Passage Detection**: Reward specifically for passing through gates
3. **Sequential Gates**: Enforce gate order (must pass gate N before gate N+1)
4. **Gate Rewards**: Bonus reward for clean gate passages
5. **Curved Gates**: Non-perpendicular gate orientations for banking challenges

### Implementation Suggestions
```python
# Example: Dynamic gate sizes based on difficulty
gate_size = base_gate_size * (1.0 + (1.0 - difficulty) * 0.5)
# Easy (0.0): 1.5 × base_size
# Hard (1.0): 1.0 × base_size

# Example: Gate passage reward
if distance_to_gate < threshold and contour_error < gate_size/2:
    gate_passage_reward = 10.0
```

---

## Summary

These three improvements work together to create a more sophisticated training system:

1. **Continuous Progression** → Smoother learning with adaptive pacing
2. **Visualization Tool** → Better debugging and trajectory understanding
3. **Virtual Gates** → Realistic racing challenges with clear objectives

All features are configurable and can be enabled/disabled independently, maintaining backward compatibility while enabling advanced training scenarios.

**Result:** A more robust, debuggable, and realistic quadcopter racing training environment.
