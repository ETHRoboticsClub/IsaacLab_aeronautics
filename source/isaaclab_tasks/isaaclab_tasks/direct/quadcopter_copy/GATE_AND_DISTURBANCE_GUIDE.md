# Gate Visualization & Difficulty-Dependent Disturbances

## Overview
This guide covers two new advanced features:
1. **4-Sided Gate Visualization** - Realistic gate rendering with proper geometry
2. **Difficulty-Dependent Disturbances** - Adaptive robustness training that scales with curriculum

---

## 1. 4-Sided Gate Visualization

### Motivation
Previous gate visualization used a single flat square marker. For realistic racing simulation, gates should look like actual racing gates with 4 distinct bars forming a frame.

### Implementation

**Visual Structure:**
```
        ┌────────────────┐  ← Top bar (horizontal)
        │                │
  Left  │                │  Right bar
  bar   │    CENTER      │  (vertical)
(vert)  │                │
        │                │
        └────────────────┘  ← Bottom bar (horizontal)
```

**Technical Details:**

Each gate consists of 4 separate cuboid markers:
- **Top Bar**: Horizontal (gate_size × 0.08 × 0.08), offset +Z by gate_size/2
- **Bottom Bar**: Horizontal (gate_size × 0.08 × 0.08), offset -Z by gate_size/2  
- **Left Bar**: Vertical (0.08 × 0.08 × gate_size), offset -Y by gate_size/2
- **Right Bar**: Vertical (0.08 × 0.08 × gate_size), offset +Y by gate_size/2

**Code Changes:**

1. **Marker Creation** (`quadcopter_env._set_debug_vis_impl()`):
```python
# Create 4 separate visualizers
self.gate_top_visualizer    # Horizontal top bar
self.gate_bottom_visualizer # Horizontal bottom bar
self.gate_left_visualizer   # Vertical left bar
self.gate_right_visualizer  # Vertical right bar
```

2. **Position Calculation** (`quadcopter_env._debug_vis_callback()`):
```python
# Transform local offsets to world coordinates using gate orientation
top_offset = [0, 0, +gate_size/2]    # Up
bottom_offset = [0, 0, -gate_size/2] # Down
left_offset = [0, -gate_size/2, 0]   # Left
right_offset = [0, +gate_size/2, 0]  # Right

# Rotate by gate orientation and add to gate center
top_pos = gate_center + quat_rotate(gate_quat, top_offset)
# ... (same for bottom, left, right)
```

3. **Orientation Handling**:
- All 4 bars share the same orientation quaternion as the gate
- Orientation aligns gate perpendicular to trajectory
- Bars automatically follow trajectory curvature

**Visual Properties:**
- **Color**: Orange (1.0, 0.3, 0.0) - High visibility against environment
- **Bar Thickness**: 0.08m - Visible but not obstructive
- **Gate Size**: Configurable (default 2.0m)
- **Separation**: Clear opening in center for passage

### Usage

Gates automatically visualize when `debug_vis=True`:
```python
cfg = QuadcopterEnvCfgCopy(
    debug_vis=True,
    trajectory=TrajectoryConfig(
        enable_gates=True,
        gate_spacing=10.0,
        gate_size=2.0,
    )
)
```

**During Simulation:**
- Orange rectangular frames appear along trajectory
- Gates orient perpendicular to flight path
- Visible from all angles
- Clear opening indicates valid passage zone

---

## 2. Difficulty-Dependent Disturbances

### Motivation

Training progression should gradually increase disturbance magnitude to build robustness:
- **Early training** (low difficulty): Minimal disturbances, focus on basic control
- **Mid training** (medium difficulty): Moderate disturbances, develop robustness
- **Late training** (high difficulty): Strong disturbances, prepare for real-world deployment

### Design Philosophy

**Key Insight**: Don't just scale disturbance magnitude with difficulty. Instead, scale the **range from which episode-specific disturbance std dev is sampled**.

**Why?**
- Creates diversity even at same difficulty level
- Each episode has consistent disturbance characteristics
- More realistic: real world has varying wind conditions per flight
- Better generalization: agent experiences full spectrum of disturbances

### Implementation Architecture

#### A. Configuration Parameters

```python
class DisturbanceManager:
    def __init__(
        self,
        force_scale=0.05,                      # Base force scale (below threshold)
        torque_scale=0.02,                     # Base torque scale (below threshold)
        difficulty_dependent=True,             # Enable difficulty scaling
        difficulty_threshold=0.33,             # Difficulty below which minimal disturbance
        max_force_scale_at_max_difficulty=0.15,   # Maximum force scale at difficulty=1.0
        max_torque_scale_at_max_difficulty=0.06,  # Maximum torque scale at difficulty=1.0
    )
```

#### B. Difficulty Mapping

**Three-Phase Response:**

```
Disturbance
Magnitude
    ↑
    │                                      ┌─────────────
    │                                    ╱
    │                                  ╱
    │                                ╱
    │                              ╱
    │                            ╱
    │  ────────────────────────┘
    │  (minimal disturbance)
    └──────────────────────────────────────────────────→
    0.0        0.33                                    1.0
          threshold              Difficulty
```

**Phase 1: Below Threshold (difficulty < 0.33)**
- Purpose: Let agent learn basic control without complications
- Behavior: Minimal disturbance (base scales)
- Force: 0.05 × robot_weight
- Torque: 0.02

**Phase 2: Transition (difficulty 0.33 → 1.0)**
- Purpose: Gradually increase robustness requirements
- Behavior: Linear interpolation from base to max scales
- Formula:
  ```python
  normalized_diff = (difficulty - 0.33) / (1.0 - 0.33)
  max_std = base_scale + normalized_diff × (max_scale - base_scale)
  ```

**Phase 3: Maximum Difficulty (difficulty = 1.0)**
- Purpose: Full sim2real robustness
- Force: 0.15 × robot_weight (3× base)
- Torque: 0.06 (3× base)

#### C. Per-Episode Random Sampling

**Critical Innovation**: At episode start, sample random std dev for that episode.

```python
def reset_episode_disturbances(env_ids, difficulties):
    # Compute max allowed std dev for this difficulty
    max_force_std = compute_max_for_difficulty(difficulties)
    max_torque_std = compute_max_for_difficulty(difficulties)
    
    # Sample random std dev uniformly from [0, max]
    episode_force_std = rand(num_envs) * max_force_std
    episode_torque_std = rand(num_envs) * max_torque_std
    
    # Use this std dev for entire episode
    store_for_episode(episode_force_std, episode_torque_std)
```

**During Episode**: Disturbances sampled using episode-specific std dev:
```python
def sample_disturbances():
    force = randn(num_envs, 3) * episode_force_std * robot_weight
    torque = randn(num_envs, 3) * episode_torque_std
```

**Benefits:**
- ✅ Diversity: Even at same difficulty, episodes vary
- ✅ Consistency: Within episode, disturbance characteristics stable
- ✅ Realism: Mimics varying environmental conditions
- ✅ Generalization: Agent learns to adapt to different disturbance levels

#### D. Integration with Curriculum

Disturbances automatically adapt as curriculum advances:

```python
# In environment reset
if curriculum_enabled:
    difficulties = curriculum_manager.get_difficulty(env_ids)
    trajectory.reset_with_difficulty(env_ids, difficulties)
    disturbance_manager.reset_episode_disturbances(env_ids, difficulties)
```

**Progression Example:**

| Episode | Difficulty | Max Force Std | Sampled Force Std | Max Torque Std | Sampled Torque Std |
|---------|-----------|---------------|-------------------|----------------|-------------------|
| 100     | 0.15      | 0.05          | 0.023             | 0.02           | 0.011             |
| 500     | 0.30      | 0.05          | 0.041             | 0.02           | 0.017             |
| 1000    | 0.45      | 0.073         | 0.038             | 0.029          | 0.015             |
| 2000    | 0.65      | 0.098         | 0.072             | 0.039          | 0.028             |
| 5000    | 0.85      | 0.123         | 0.098             | 0.049          | 0.037             |
| 10000   | 1.00      | 0.150         | 0.127             | 0.060          | 0.051             |

**Note**: Sampled values vary randomly each episode within max range.

---

## Configuration Examples

### Conservative (Gentle Progression)
```python
cfg = QuadcopterEnvCfgCopy(
    enable_disturbances=True,
    force_disturbance_scale=0.03,        # Lower base
    torque_disturbance_scale=0.015,      # Lower base
    
    enable_curriculum=True,
    curriculum=CurriculumCfg(
        # ... curriculum settings
    ),
    
    # Disturbance manager will use:
    # - difficulty_threshold=0.33
    # - max_force_scale=0.15
    # - max_torque_scale=0.06
)
```

### Aggressive (Fast Progression)
```python
# Modify DisturbanceManager initialization in environment:
self._disturbance_manager = DisturbanceManager(
    # ... standard args ...
    difficulty_threshold=0.2,            # Start scaling earlier
    max_force_scale_at_max_difficulty=0.20,   # Higher maximum
    max_torque_scale_at_max_difficulty=0.08,  # Higher maximum
)
```

### Sim2Real Transfer Focus
```python
# Maximum disturbances throughout training
self._disturbance_manager = DisturbanceManager(
    # ... standard args ...
    difficulty_dependent=False,          # Disable curriculum scaling
    force_scale=0.15,                    # Use max always
    torque_scale=0.06,                   # Use max always
)
```

### Debugging / Observation
```python
# Minimal disturbances for debugging
cfg = QuadcopterEnvCfgCopy(
    enable_disturbances=True,
    force_disturbance_scale=0.01,        # Very low
    torque_disturbance_scale=0.005,      # Very low
)
```

---

## Monitoring & Analysis

### TensorBoard Metrics

**Existing Metrics Enhanced:**
- `Episode_Reward/progress` - Should remain relatively stable as disturbances increase
- `Episode_Reward/contour_error` - May decrease slightly at higher difficulties
- `curriculum/mean_difficulty` - Track disturbance scaling progression

**Recommended Custom Logging:**

Add to environment's `_reset_idx()`:
```python
if self._curriculum_manager is not None:
    # Log disturbance statistics
    difficulties = self._curriculum_manager.difficulty_levels[env_ids]
    mean_force_std = self._disturbance_manager._episode_force_std[env_ids].mean()
    mean_torque_std = self._disturbance_manager._episode_torque_std[env_ids].mean()
    
    extras["log"]["Disturbances/mean_force_std"] = mean_force_std.item()
    extras["log"]["Disturbances/mean_torque_std"] = mean_torque_std.item()
    extras["log"]["Disturbances/mean_difficulty"] = difficulties.mean().item()
```

### Expected Behavior

**Healthy Training Progression:**
1. Episodes 0-1000: Difficulty rises smoothly, disturbances minimal
2. Episodes 1000-3000: Disturbances start increasing, performance dips slightly
3. Episodes 3000-5000: Agent adapts, performance recovers despite higher disturbances
4. Episodes 5000+: High disturbances, agent maintains good performance

**Warning Signs:**
- Performance collapse when disturbances increase → Threshold too low
- Agent never reaches high difficulty → Disturbances too strong
- Training unstable → Consider gentler max scales

---

## Testing & Validation

### 1. Verify Gate Visualization

```python
# Run environment with debug visualization
cfg.debug_vis = True

# Check console output:
# "Generated N virtual gates along trajectory (spacing: 10.0m)"

# Visual verification:
# - Should see orange rectangular frames along trajectory
# - Each gate has 4 distinct bars
# - Gates oriented perpendicular to path
```

### 2. Verify Disturbance Scaling

```python
# Add breakpoint in reset_episode_disturbances()
# Check values at different difficulties:

difficulty = 0.0
# Expected: force_std ≈ 0.05, torque_std ≈ 0.02

difficulty = 0.33
# Expected: force_std ≈ 0.05, torque_std ≈ 0.02 (at threshold)

difficulty = 0.5
# Expected: force_std ≈ 0.075-0.10, torque_std ≈ 0.03-0.04

difficulty = 1.0
# Expected: force_std ≈ 0.15, torque_std ≈ 0.06
```

### 3. Verify Per-Episode Variation

```python
# Log episode force_std for same difficulty
# Should see variation:
# Env 0 @ diff=0.5: force_std = 0.082
# Env 1 @ diff=0.5: force_std = 0.091
# Env 2 @ diff=0.5: force_std = 0.067
# ... etc (all different but within range)
```

---

## Implementation Details

### Gate Visualization Math

**Coordinate System:**
- Gate orientation stored as quaternion
- Local coordinates: X=forward (trajectory tangent), Y=left, Z=up
- Bar offsets in local frame, transformed to world

**Transformation:**
```python
# Local offset (gate-relative coordinates)
top_offset_local = [0, 0, gate_size/2]

# Transform to world coordinates
top_offset_world = quat_rotate(gate_orientation, top_offset_local)

# Final position
top_position = gate_center + top_offset_world
```

**Why 4 Bars Instead of 1?**
- Single flat square appears as thin line from certain angles
- 4 bars create 3D structure visible from all viewpoints
- Realistic representation of physical racing gates
- Clear visual feedback for collision zone

### Disturbance Sampling Details

**Force Disturbance:**
```python
# Magnitude scaled by robot weight for realistic physics
force = randn(3) × episode_force_std × robot_weight

# Applied to robot center of mass
# Direction: random 3D vector (isotropic)
```

**Torque Disturbance:**
```python
# Angular disturbance independent of weight
torque = randn(3) × episode_torque_std

# Applied as external torque
# Direction: random 3D angular perturbation
```

**Sampling Frequency:**
- Force & torque sampled **every timestep**
- Episode std dev sampled **once per episode**
- Result: High-frequency noise within episode-consistent envelope

---

## Advanced Tuning

### Adjusting Difficulty Threshold

**Lower threshold (e.g., 0.2):**
- Disturbances increase earlier
- Faster robustness development
- Risk: May destabilize early learning

**Higher threshold (e.g., 0.5):**
- Longer period with minimal disturbances
- Agent masters basic control first
- Risk: May not reach high disturbances in training time

**Recommendation**: Start with 0.33, adjust based on training curves.

### Adjusting Max Scales

**Higher max scales (e.g., 0.20 force, 0.08 torque):**
- Better sim2real transfer
- More robust final policy
- Risk: Training may not converge

**Lower max scales (e.g., 0.10 force, 0.04 torque):**
- Easier convergence
- Smoother training
- Risk: Less robust to real-world disturbances

**Calibration**: Test real drone response, adjust to match.

### Custom Difficulty Curves

For non-linear scaling, modify `reset_episode_disturbances()`:

```python
# Example: Exponential scaling
normalized_diff = (difficulty - threshold) / (1.0 - threshold)
exponential_factor = normalized_diff ** 2  # Quadratic
max_std = base + exponential_factor × (max - base)

# Example: Sigmoid scaling
import torch.nn.functional as F
sigmoid_factor = torch.sigmoid(10 * (normalized_diff - 0.5))
max_std = base + sigmoid_factor × (max - base)
```

---

## Troubleshooting

### Problem: Gates not visible in simulation

**Solution:**
1. Verify `debug_vis=True` in config
2. Check `enable_gates=True` in trajectory config
3. Ensure trajectory long enough for gates (>10m arc length)
4. Check console for "Generated N virtual gates" message

### Problem: Gates appear disconnected or rotated wrong

**Solution:**
- Gate orientation computed from trajectory tangent
- If trajectory has sharp turns, gates may look tilted (this is correct)
- Verify trajectory smoothness with visualization tool

### Problem: Training unstable with disturbances

**Solution:**
1. Lower `max_force_scale_at_max_difficulty` (try 0.10)
2. Raise `difficulty_threshold` (try 0.5)
3. Slow down curriculum progression (increase `success_streak_required`)
4. Check robot mass/inertia parameters

### Problem: Agent doesn't improve with disturbances

**Solution:**
1. Verify disturbances actually increasing (add logging)
2. Check if agent stuck at low difficulty (curriculum issue)
3. Consider reward shaping to encourage robustness
4. May need longer training time for adaptation

---

## Summary

### Gate Visualization
- ✅ Realistic 4-bar gate structure
- ✅ Proper 3D geometry and orientation
- ✅ High visibility orange coloring
- ✅ Automatic generation along trajectory

### Difficulty-Dependent Disturbances
- ✅ Three-phase scaling (minimal → transition → maximum)
- ✅ Threshold-based activation at difficulty 0.33
- ✅ Per-episode random std dev sampling for diversity
- ✅ Automatic integration with curriculum learning
- ✅ Configurable parameters for different training strategies

**Result**: A more realistic, visually clear, and progressively challenging training environment that better prepares policies for real-world deployment.
