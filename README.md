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
python a3c_mario6.py
```

The script will:
- Spawn 4 worker processes (configurable via `NUM_WORKERS`)
- Train for 2,000,000 steps (configurable via `MAX_GLOBAL_STEPS`)
- Save the model to `./a3c_mario.pth` (configurable via `SAVE_PATH`)

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

- **Distance reward**: `min(max(distance - prev_distance, 0.0), 2.0)` - encourages forward progress
- **Time penalty**: `(prev_time - time_left) * -0.01` - discourages standing still (soft penalty)
- **Status reward**: `(status - prev_status) * 5.0` - rewards power-ups
- **Score reward**: `(score - prev_score) * 0.025` - small reward for score increases
- **Terminal bonus/penalty**: `+50.0` if distance > 3225 (level complete), else `-50.0`

**Note:** The time penalty was reduced from `-0.1` to `-0.01` to prevent it from overwhelming learning signals.

---

## Troubleshooting

### Warnings
- **RuntimeWarning: overflow encountered in ubyte_scalars**: These are harmless warnings from `gym_super_mario_bros` when Mario wraps around the screen. They're automatically suppressed in the code.

### Negative Returns
- Early training episodes will have negative returns (around -40 to -50) because:
  - The agent hasn't learned to make progress
  - The terminal penalty (-50) is applied when episodes end early
- As training progresses, returns should gradually increase as the agent learns to make forward progress.

### Training Not Learning
If returns stay extremely negative after many episodes, check:
1. Gradient synchronization (should use `copy_`, not `+=`)
2. Value bootstrapping (should happen before reset)
3. Episode return tracking (should track actual episodes, not rollouts)
4. Loss initialization (should be tensors, not floats)

---

## License

This project is for educational/research purposes.
