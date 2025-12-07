"""
Modern A3C trainer for Super Mario Bros using Gym / Gymnasium + PyTorch.
Compatible with gym-super-mario-bros and nes-py via our compatibility wrappers.

Usage:
    python a3c_mario6.py [--max-steps N] [--workers N] [--save-path PATH]
"""

import os
import argparse
import warnings
from collections import deque
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp

# Suppress overflow warnings from gym_super_mario_bros (harmless, occurs during screen wrap)
warnings.filterwarnings("ignore", category=RuntimeWarning, module="gym_super_mario_bros")

from atari_wrapper import create_mario_env  # make sure atari_wrapper.py is in same folder

# ================================================================
# Hyperparameters (can be overridden via command-line)
# ================================================================
ENV_ID = "SuperMarioBros-1-1-v0"
GAMMA = 0.99
ENTROPY_BETA = 0.01
VALUE_LOSS_COEF = 0.5
LR = 1e-4
T_MAX = 20
NUM_WORKERS = 4
MAX_GLOBAL_STEPS = 2_000_000  # Set to 50_000 for quick testing
SAVE_PATH = "./a3c_mario.pth"

# Diagnostics settings
ENABLE_DIAGNOSTICS = False  # Set to False to disable diagnostic logging
DIAGNOSTICS_INTERVAL = 10  # Print diagnostics every N updates (per worker)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)


# ================================================================
# Model: Shared ConvNet for policy and value
# ================================================================
class ActorCritic(nn.Module):
    def __init__(self, num_actions):
        super().__init__()
        # Input from env is (4, 84, 84)
        self.conv1 = nn.Conv2d(4, 32, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 64, 3, stride=2, padding=1)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(64 * 11 * 11, 512)
        self.policy = nn.Linear(512, num_actions)
        self.value = nn.Linear(512, 1)

    def forward(self, x):
        # x: (B, 4, 84, 84), uint8
        x = x / 255.0
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.flatten(x)
        x = F.relu(self.fc(x))
        policy_logits = self.policy(x)
        value = self.value(x)
        return policy_logits, value


# ================================================================
# Shared optimizer (Hogwild)
# ================================================================
class SharedAdam(torch.optim.Adam):
    def __init__(self, params, lr=1e-4):
        super().__init__(params, lr=lr)
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                state["step"] = torch.zeros(1)
                state["exp_avg"] = torch.zeros_like(p.data)
                state["exp_avg_sq"] = torch.zeros_like(p.data)
                state["exp_avg"].share_memory_()
                state["exp_avg_sq"].share_memory_()


# ================================================================
# Worker process
# ================================================================
def worker_fn(worker_id, global_model, optimizer, global_counter, env_id, max_global_steps):
    env = create_mario_env(env_id)
    local_model = ActorCritic(env.action_space.n).to(DEVICE)
    local_model.load_state_dict(global_model.state_dict())
    local_model.train()

    episode = 0
    returns_window = deque(maxlen=20)
    episode_return = 0.0  # Track actual episode return
    
    # Diagnostics tracking
    update_count = 0  # Track number of updates for periodic diagnostics

    # Initialize state before the main loop
    obs = env.reset()
    state = torch.from_numpy(obs).float().unsqueeze(0).to(DEVICE)

    while True:
        # Stop if we've hit the global step limit
        with global_counter.get_lock():
            if global_counter.value >= max_global_steps:
                break

        values, log_probs, rewards, entropies = [], [], [], []

        t = 0
        done = False
        while t < T_MAX and not done:
            t += 1

            # Forward pass WITH gradients so we can backprop later
            logits, value = local_model(state)
            probs = F.softmax(logits, dim=-1)
            log_probs_all = F.log_softmax(logits, dim=-1)

            dist = torch.distributions.Categorical(probs)
            action = dist.sample()

            log_prob = log_probs_all[0, action]
            entropy = -(probs * log_probs_all).sum()

            # --- robust step() with error handling ---
            try:
                next_obs, reward, done, info = env.step(int(action.item()))
            except ValueError as e:
                print(f"[Worker {worker_id}] Caught env error in step(): {e}. Forcing done/reset.")
                done = True
                reward = 0.0
                info = {"env_error": str(e)}
                next_obs = state.cpu().numpy()[0]
            # -----------------------------------------

            next_state = torch.from_numpy(next_obs).float().unsqueeze(0).to(DEVICE)

            values.append(value)
            log_probs.append(log_prob)
            # Store rewards as tensors on correct device for consistent computation
            rewards.append(torch.tensor(reward, dtype=torch.float32, device=DEVICE))
            entropies.append(entropy)
            episode_return += float(reward)  # Track actual episode return (for logging only)

            state = next_state

            with global_counter.get_lock():
                global_counter.value += 1
                if global_counter.value >= max_global_steps:
                    done = True

        # Skip if no steps collected (shouldn't happen, but safety check)
        if len(rewards) == 0:
            continue

        # Compute rollout reward statistics for diagnostics
        if ENABLE_DIAGNOSTICS:
            rollout_rewards = [r.item() for r in rewards]  # Convert tensors to floats
            mean_reward = np.mean(rollout_rewards) if rollout_rewards else 0.0
            min_reward = np.min(rollout_rewards) if rollout_rewards else 0.0
            max_reward = np.max(rollout_rewards) if rollout_rewards else 0.0

        # Bootstrap value BEFORE resetting (critical fix!)
        # If done=True: terminal state, bootstrap with 0 (no future value)
        # If done=False: bootstrap with V(s_T) where s_T is the last state of the rollout
        # FIX: Bootstrap must use the state AFTER the last action, which is 'state' (next_state from last step)
        if done:
            # Terminal state: no future value, bootstrap with 0
            # Use scalar tensor for consistent shape with rewards and values
            R = torch.tensor(0.0, device=DEVICE, dtype=torch.float32)
        else:
            # Non-terminal: bootstrap with value of last state in rollout
            # 'state' is the state after the last action, which is correct for bootstrapping
            with torch.no_grad():
                _, v = local_model(state)  # Use state from current rollout (before reset)
                # v is shape (1, 1), squeeze to scalar for consistent computation
                R = v.squeeze().detach()  # Shape: scalar tensor on DEVICE

        # Initialize losses as tensors (not floats) for proper gradient computation
        policy_loss = torch.zeros(1, device=DEVICE, dtype=torch.float32)
        value_loss = torch.zeros(1, device=DEVICE, dtype=torch.float32)

        # Backward pass through time - compute n-step returns
        # R_t = r_t + gamma * r_{t+1} + ... + gamma^{n-1} * r_{t+n-1} + gamma^n * V(s_{t+n})
        # advantage_t = R_t - V(s_t)
        for i in reversed(range(len(rewards))):
            # Compute n-step return: R = r_i + gamma * R (R starts as bootstrap value)
            # rewards[i] is tensor shape (), R is scalar tensor, result is scalar tensor
            R = rewards[i] + GAMMA * R
            
            # values[i] is shape (1, 1) from model, squeeze to scalar for subtraction
            value_i = values[i].squeeze()  # Shape: scalar tensor
            advantage = R - value_i  # Both scalars, result is scalar
            
            value_loss = value_loss + 0.5 * advantage.pow(2)
            
            # Policy loss: -log_prob * advantage (detached) - entropy_beta * entropy
            # Use advantage directly (simpler and more stable than GAE here)
            policy_loss = policy_loss - log_probs[i] * advantage.detach() - ENTROPY_BETA * entropies[i]

        # Clear local gradients before backward pass
        local_model.zero_grad()
        optimizer.zero_grad()
        
        total_loss = policy_loss + VALUE_LOSS_COEF * value_loss
        total_loss.backward()

        # Clip gradients for stability
        torch.nn.utils.clip_grad_norm_(local_model.parameters(), 40.0)
        
        # Compute gradient norm for diagnostics (before copying to global)
        grad_norm = None
        if ENABLE_DIAGNOSTICS:
            try:
                # Compute gradient norm from local model before copying to global
                total_norm = 0.0
                param_count = 0
                for param in local_model.parameters():
                    if param.grad is not None:
                        param_norm = param.grad.data.norm(2)
                        total_norm += param_norm.item() ** 2
                        param_count += 1
                if param_count > 0:
                    grad_norm = total_norm ** (1. / 2)
            except Exception:
                grad_norm = None  # Skip if computation fails

        # Copy local gradients to global model (REPLACE, not accumulate)
        # In A3C, each worker replaces global grads with its own, then optimizer steps
        for local_param, global_param in zip(local_model.parameters(), global_model.parameters()):
            if local_param.grad is None:
                continue
            # Detach and copy to avoid keeping references to local computation graph
            local_grad = local_param.grad.detach()
            if global_param.grad is None:
                global_param.grad = local_grad.clone()
            else:
                global_param.grad.copy_(local_grad)  # REPLACE, not accumulate

        optimizer.step()
        local_model.load_state_dict(global_model.state_dict())
        
        # Increment update counter for diagnostics
        update_count += 1
        
        # Check for NaN/inf in losses
        has_nan = False
        has_inf = False
        if ENABLE_DIAGNOSTICS:
            policy_loss_val = policy_loss.item()
            value_loss_val = value_loss.item()
            has_nan = torch.isnan(policy_loss) or torch.isnan(value_loss) or \
                     (np.isnan(policy_loss_val) or np.isnan(value_loss_val))
            has_inf = torch.isinf(policy_loss) or torch.isinf(value_loss) or \
                     (np.isinf(policy_loss_val) or np.isinf(value_loss_val))
        
        # Periodic diagnostics logging
        if ENABLE_DIAGNOSTICS and update_count % DIAGNOSTICS_INTERVAL == 0:
            diagnostics_msg = (
                f"[Worker {worker_id}] Update {update_count} | "
                f"Rollout rewards: mean={mean_reward:.3f}, min={min_reward:.3f}, max={max_reward:.3f} | "
            )
            if grad_norm is not None:
                diagnostics_msg += f"Grad norm: {grad_norm:.4f} | "
            else:
                diagnostics_msg += "Grad norm: N/A | "
            diagnostics_msg += (
                f"Policy loss: {policy_loss.item():.4f}, Value loss: {value_loss.item():.4f}"
            )
            if has_nan:
                diagnostics_msg += " [NaN DETECTED!]"
            if has_inf:
                diagnostics_msg += " [Inf DETECTED!]"
            print(diagnostics_msg)

        # Logging - only when episode actually ends
        # FIX: episode_return tracks the FULL episode reward accumulated across
        # all rollouts until done=True. It is reset immediately when done=True to prevent
        # mixing returns from different episodes if a rollout somehow spans episodes.
        # The episode counter increments only when done=True (not per rollout).
        if done:
            episode += 1  # Increment episode counter only when episode ends
            returns_window.append(episode_return)  # Log full episode return
            avg20 = np.mean(returns_window) if len(returns_window) > 0 else 0.0
            print(
                f"[Worker {worker_id}] Ep {episode} | "
                f"Global steps {global_counter.value:,} | "
                f"Ep return {episode_return:.1f} | Avg20 {avg20:.1f} | "
                f"last policy_loss: {policy_loss.item():.4f}, "
                f"last value_loss: {value_loss.item():.4f}"
            )
            # FIX: Reset episode_return BEFORE resetting environment to ensure clean separation
            episode_return = 0.0  # Reset for next episode
            
            # Reset environment AFTER logging and resetting episode_return
            # (critical: bootstrap happens before reset, episode_return reset before env reset)
            obs = env.reset()
            state = torch.from_numpy(obs).float().unsqueeze(0).to(DEVICE)

        if global_counter.value >= max_global_steps:
            break



# ================================================================
# Main training launcher
# ================================================================
def main():
    parser = argparse.ArgumentParser(description="Train A3C agent on Super Mario Bros")
    parser.add_argument(
        "--max-steps",
        type=int,
        default=MAX_GLOBAL_STEPS,
        help=f"Maximum global training steps (default: {MAX_GLOBAL_STEPS})",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=NUM_WORKERS,
        help=f"Number of worker processes (default: {NUM_WORKERS})",
    )
    parser.add_argument(
        "--save-path",
        type=str,
        default=SAVE_PATH,
        help=f"Path to save trained model (default: {SAVE_PATH})",
    )
    parser.add_argument(
        "--env-id",
        type=str,
        default=ENV_ID,
        help=f"Environment ID (default: {ENV_ID})",
    )
    args = parser.parse_args()
    
    # Use command-line arguments
    max_steps = args.max_steps
    num_workers = args.workers
    save_path = args.save_path
    env_id = args.env_id
    
    print("=" * 60)
    print("A3C Training Configuration:")
    print(f"  Environment: {env_id}")
    print(f"  Max steps: {max_steps:,}")
    print(f"  Workers: {num_workers}")
    print(f"  Save path: {save_path}")
    print(f"  Device: {DEVICE}")
    print("=" * 60)
    
    os.environ["OMP_NUM_THREADS"] = "1"

    # On Windows, make sure we use 'spawn'
    try:
        mp.set_start_method("spawn")
    except RuntimeError:
        # Already set
        pass

    env = create_mario_env(env_id)
    num_actions = env.action_space.n
    env.close()

    global_model = ActorCritic(num_actions).to(DEVICE)
    global_model.share_memory()

    optimizer = SharedAdam(global_model.parameters(), lr=LR)
    global_counter = mp.Value("i", 0)

    print(f"\nStarting training with {num_workers} workers...")
    print("Press Ctrl+C to stop early (model will still be saved)\n")

    workers = []
    for worker_id in range(num_workers):
        p = mp.Process(
            target=worker_fn,
            args=(worker_id, global_model, optimizer, global_counter, env_id, max_steps),
        )
        p.start()
        workers.append(p)

    try:
        for p in workers:
            p.join()
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user. Stopping workers...")
        for p in workers:
            p.terminate()
            p.join()

    torch.save(global_model.state_dict(), save_path)
    print(f"\nTraining finished. Model saved to {save_path}")


if __name__ == "__main__":
    main()
