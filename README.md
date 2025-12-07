# A3C Super Mario Bros (PyTorch + gym-super-mario-bros)

This project trains an A3C (Asynchronous Advantage Actor-Critic) agent to play **Super Mario Bros** using:

- PyTorch (Actor–Critic neural network)
- gym_super_mario_bros + nes_py (NES environment)
- Multi-process workers (true A3C)
- Custom wrappers for frame preprocessing & reward shaping

The focus is **correct, stable A3C implementation** — no NaNs, no broken gradients, correct episode tracking, correct bootstrap behavior, and valid training logs.

---

## File Structure

| File | Purpose |
|------|---------|
| `a3c_mario6.py` | Main training script – spawns workers, handles rollouts, updates global model |
| `atari_wrapper.py` | Observation + reward wrappers for Mario (resize, grayscale, frame stacking, reward shaping) |
| `eval_mario.py` | Evaluation script to test trained models |

---

## Setup Instructions

### 1. Create Conda Environment

```bash
conda create -n mario_a3c python=3.10 -y
conda activate mario_a3c
```

### 2. Install PyTorch

**CPU version (simplest):**
```bash
conda install pytorch torchvision torchaudio cpuonly -c pytorch -y
```

**GPU version (if you have CUDA support):**
```bash
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y
```

### 3. Install Dependencies

```bash
pip install gym==0.26.2 gym-super-mario-bros==7.3.0 nes-py==8.2.1 numpy opencv-python matplotlib
```

---

## Usage

### Training

Run the training script:

```bash
# Default training (2M steps, 4 workers)
python a3c_mario6.py

# Quick test run (50k steps for testing)
python a3c_mario6.py --max-steps 50000

# Custom configuration
python a3c_mario6.py --max-steps 1000000 --workers 8 --save-path ./my_model.pth

# See all options
python a3c_mario6.py --help
```

**Command-line arguments:**
- `--max-steps N`: Maximum global training steps (default: 2,000,000)
- `--workers N`: Number of worker processes (default: 4)
- `--save-path PATH`: Path to save trained model (default: ./a3c_mario.pth)
- `--env-id ID`: Environment ID (default: SuperMarioBros-1-1-v0)

The script will:
- Spawn worker processes (default: 4)
- Train for specified number of steps (default: 2,000,000)
- Save the model to specified path (default: ./a3c_mario.pth)

**Training logs look like:**
```
[Worker 0] Ep 1 | Global steps 523 | Ep return -50.1 | Avg20 -50.1 | last policy_loss: -2132.7134, last value_loss: 9340.0996
[Worker 1] Ep 2 | Global steps 1,761 | Ep return -42.8 | Avg20 -46.5 | last policy_loss: -1860.7898, last value_loss: 8213.8486
```

**Note:** Early episodes will have negative returns (around -40 to -50) because the agent hasn't learned yet. As training progresses, returns should gradually increase.

### Evaluation

Test a trained model:

```bash
# Basic evaluation (5 episodes, no rendering)
python eval_mario.py --model-path a3c_mario.pth

# Evaluation with rendering (watch Mario play)
python eval_mario.py --model-path a3c_mario.pth --num-episodes 5 --render
```

**Evaluation output:**
```
Episode  1 | Return:   -50.2 | Distance:   40.0 | Steps:  523
Episode  2 | Return:   -45.1 | Distance:  120.5 | Steps:  612
...
Average return:  -47.6 ± 3.2
Average distance: 85.3 ± 45.1
```

---

## Hyperparameters

Current settings in `a3c_mario6.py`:

```python
GAMMA            = 0.99      # Discount factor
ENTROPY_BETA     = 0.01       # Entropy regularization coefficient
VALUE_LOSS_COEF  = 0.5        # Value loss weight
LR               = 1e-4       # Learning rate
T_MAX            = 20         # N-step return horizon
NUM_WORKERS      = 4          # Number of parallel workers
MAX_GLOBAL_STEPS = 2_000_000  # Total training steps
SAVE_PATH        = "./a3c_mario.pth"
```

---

## How A3C Training Works

1. **Global Model**: A shared ActorCritic model sits in shared memory
2. **Workers**: Each worker process:
   - Creates its own Mario environment via `create_mario_env()`
   - Copies global model weights locally
   - Runs up to `T_MAX` rollout steps
   - Collects `{values, log_probs, rewards, entropies}`
   - Computes n-step returns & advantages
   - Backpropagates locally and **replaces** global gradients (not accumulates)
   - Global optimizer steps; workers sync updated weights

3. **Training stops** when `MAX_GLOBAL_STEPS` is reached

---

## Critical Implementation Details

### Episode Return Tracking
- ✅ Track actual episode return: `episode_return += reward`
- ✅ Reset `episode_return = 0.0` only when `done=True`
- ❌ Do NOT sum rollout rewards (rollouts can span multiple episodes)

### Episode Counter
- ✅ Increment `episode += 1` only when environment ends (`done=True`)
- ❌ Do NOT increment per rollout

### Value Bootstrapping
- ✅ Bootstrap **BEFORE** resetting environment:
  ```python
  if done:
      R = 0  # terminal, no bootstrap
  else:
      R = value(next_state)  # bootstrap from current rollout's final state
  # THEN reset environment
  ```
- ❌ Do NOT bootstrap using the next episode's reset state

### Gradient Synchronization
- ✅ Each worker **replaces** global gradients: `global_param.grad.copy_(local_grad)`
- ❌ Do NOT accumulate: `global_param.grad += local_grad` (wrong!)

### Loss Computation
- ✅ Initialize losses as tensors: `policy_loss = torch.zeros(1, device=DEVICE)`
- ❌ Do NOT use floats: `policy_loss = 0.0` (causes type issues)

---

## Reward Shaping

The reward shaping in `atari_wrapper.py` includes:

- **Distance reward**: `min(max(distance - prev_distance, 0.0), 2.0)` - encourages forward progress (max +2.0 per step)
- **Time penalty**: `(prev_time - time_left) * -0.01` - discourages standing still (soft penalty, -0.01 per time unit)
- **Status reward**: `(status - prev_status) * 5.0` - rewards power-ups (e.g., small -> big = +5.0)
- **Score reward**: `(score - prev_score) * 0.025` - small reward for score increases
- **Terminal bonus/penalty**: `+15.0` if distance > 3225 (level complete), else `-15.0` (reduced from ±50.0 for stability)

**Recent fixes:**
- Terminal penalty reduced from ±50.0 to ±15.0 to prevent overwhelming the learning signal
- Time penalty kept at -0.01 to maintain soft discouragement without dominating rewards

---

## Troubleshooting

### Warnings
- **RuntimeWarning: overflow encountered in ubyte_scalars**: These are harmless warnings from `gym_super_mario_bros` when Mario wraps around the screen. They're automatically suppressed in the code.

### Negative Returns
- Early training episodes will have negative returns (around -15 to -25) because:
  - The agent hasn't learned to make progress
  - The terminal penalty (-15.0) is applied when episodes end early
  - Time penalties accumulate during episodes
- As training progresses, returns should gradually increase as the agent learns to make forward progress.
- With the reduced terminal penalty (±15.0 instead of ±50.0), returns should be less negative and learning should be more stable.

### Training Not Learning
If returns stay extremely negative after many episodes, check:
1. Gradient synchronization (should use `copy_`, not `+=`)
2. Value bootstrapping (should happen before reset)
3. Episode return tracking (should track actual episodes, not rollouts)
4. Loss initialization (should be tensors, not floats)

---

## License

This project is for educational/research purposes.
