# Diagnostic Test Instructions

## Quick Test Run (50k steps)

The training script is already configured for a 50k step test. To run:

```bash
# Activate your conda environment first
conda activate mario_a3c

# Run the training
python a3c_mario6.py
```

## What to Look For

### 1. Diagnostic Output Format
Every 10 updates per worker, you'll see lines like:
```
[Worker 0] Update 10 | Rollout rewards: mean=-0.250, min=-2.500, max=0.100 | Grad norm: 12.3456 | Policy loss: -1234.5678, Value loss: 5678.9012
```

### 2. Typical Reward Range
- **Expected**: Mean rewards typically between -2.0 and 0.5 per step
- **Min/Max**: Can vary widely, but min usually -5.0 to -2.0, max usually 0.0 to 2.0
- **Note**: These are per-step rewards in a rollout, not episode returns

### 3. Gradient Norm Stability
- **Good**: < 50 (stable training)
- **Acceptable**: 50-100 (may need tuning)
- **Warning**: > 100 (gradients may be exploding)
- **Critical**: > 1000 (definitely exploding)

### 4. NaN/Inf Detection
- Look for `[NaN DETECTED!]` or `[Inf DETECTED!]` in diagnostic lines
- If present, training is broken and needs fixing
- Should NOT appear in normal training

### 5. Loss Values
- **Policy loss**: Usually negative (can be large in magnitude, e.g., -1000 to -100)
- **Value loss**: Usually positive (typically 1000-20000 early in training)
- Both should decrease over time (policy loss becomes less negative, value loss decreases)

## Expected Output Sample

```
[Worker 0] Update 10 | Rollout rewards: mean=-0.180, min=-2.100, max=0.050 | Grad norm: 8.2341 | Policy loss: -1234.56, Value loss: 4567.89
[Worker 1] Update 10 | Rollout rewards: mean=-0.220, min=-2.500, max=0.100 | Grad norm: 9.1234 | Policy loss: -1456.78, Value loss: 5123.45
[Worker 0] Ep 1 | Global steps 523 | Ep return -50.1 | Avg20 -50.1 | last policy_loss: -2132.71, last value_loss: 9340.10
...
```

## After Running

Report back:
1. **Typical reward range**: min/mean/max from diagnostic lines
2. **Gradient stability**: Are norms staying < 100?
3. **NaN/Inf**: Did any appear?

## To Disable Diagnostics

Edit `a3c_mario6.py` line 34:
```python
ENABLE_DIAGNOSTICS = False  # Disable diagnostic logging
```

## To Change Diagnostic Frequency

Edit `a3c_mario6.py` line 35:
```python
DIAGNOSTICS_INTERVAL = 20  # Print every 20 updates instead of 10
```

