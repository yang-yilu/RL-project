import time
from collections import deque

import cv2
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp

import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, RIGHT_ONLY
from nes_py.wrappers import JoypadSpace

# =========================
#  Hyperparameters
# =========================
GAMMA               = 0.99
ENTROPY_BETA_START  = 0.02
ENTROPY_BETA_END    = 0.001
VALUE_LOSS_COEF     = 0.5
LR                  = 7e-5          # Adam LR
T_MAX               = 20            # rollout length
NUM_WORKERS         = 4
MAX_GLOBAL_STEPS    = 3_000_000
SAVE_PATH           = "./a3c_mario_fixed.pth"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)

# =========================
#  Env wrappers
# =========================

class SkipFrame(gym.Wrapper):
    """Return every `skip`-th frame and repeat action in between."""
    def __init__(self, env, skip=4):
        super().__init__(env)
        self._skip = skip

    def step(self, action):
        total_reward = 0.0
        info = {}
        for _ in range(self._skip):
            out = self.env.step(action)
            if len(out) == 4:
                obs, reward, done, info = out
            else:
                obs, reward, terminated, truncated, info = out
                done = terminated or truncated
            total_reward += reward
            if done:
                break
        return obs, total_reward, done, info

class PreprocessFrame(gym.ObservationWrapper):
    """Gray-scale, resize and transpose to CHW for PyTorch."""
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(84, 84), dtype=np.uint8
        )

    def observation(self, obs):
        # obs is HWC RGB (240x256x3)
        obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        obs = cv2.resize(obs, (84, 84), interpolation=cv2.INTER_AREA)
        return obs

class FrameStack(gym.Wrapper):
    """Stack the last k frames along the channel dimension."""
    def __init__(self, env, k=4):
        super().__init__(env)
        self.k = k
        self.frames = deque([], maxlen=k)
        shp = env.observation_space.shape
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(k, shp[0], shp[1]),
            dtype=np.uint8
        )

    def reset(self, **kwargs):
        out = self.env.reset(**kwargs)
        if isinstance(out, tuple):
            obs = out[0]
        else:
            obs = out
        for _ in range(self.k):
            self.frames.append(obs)
        return self._get_obs()

    def step(self, action):
        out = self.env.step(action)
        if len(out) == 4:
            obs, reward, done, info = out
        else:
            obs, reward, terminated, truncated, info = out
            done = terminated or truncated
        self.frames.append(obs)
        return self._get_obs(), reward, done, info

    def _get_obs(self):
        assert len(self.frames) == self.k
        return np.stack(self.frames, axis=0)

def make_env(use_simple_movement=False):
    """
    Create the Mario environment with wrappers.
    For stability we use RIGHT_ONLY by default (no left) because it
    makes the early-learning problem much easier.
    """
    env = gym_super_mario_bros.make("SuperMarioBros-1-1-v0")
    if use_simple_movement:
        env = JoypadSpace(env, SIMPLE_MOVEMENT)
        action_meaning = SIMPLE_MOVEMENT
    else:
        env = JoypadSpace(env, RIGHT_ONLY)
        action_meaning = RIGHT_ONLY
    env = SkipFrame(env, skip=4)
    env = PreprocessFrame(env)
    env = FrameStack(env, k=4)
    return env, action_meaning

# =========================
#  Network
# =========================

class ActorCriticNet(nn.Module):
    def __init__(self, num_actions: int):
        super().__init__()
        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        self.fc = nn.Linear(64 * 7 * 7, 512)
        self.policy_head = nn.Linear(512, num_actions)
        self.value_head = nn.Linear(512, 1)

    def forward(self, x):
        # x: (B, 4, 84, 84) uint8 -> float32 [0,1]
        x = x.float() / 255.0
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc(x))
        policy_logits = self.policy_head(x)
        value = self.value_head(x)
        return policy_logits, value

# =========================
#  Reward shaping
# =========================

def shape_reward(env_reward, info, done, x_pos, prev_x, best_x):
    """
    Distance-based reward with extra bonuses for reaching new regions
    and strong penalties for dying early.
    """
    # Base: reward for positive movement to the right
    delta_x = max(0, x_pos - prev_x)
    r = 0.05 * delta_x

    # Bonus every 100 pixels we move forward for the first time
    bucket_prev = prev_x // 100
    bucket_now = x_pos // 100
    if bucket_now > bucket_prev:
        r += 1.0 * (bucket_now - bucket_prev)

    # Small living reward to prefer staying alive over instant death
    r += 0.01

    # Additional bonus for reaching a new personal best within the episode
    if x_pos > best_x:
        r += 1.0
        best_x = x_pos

    # Large bonus for reaching the flag
    if info.get("flag_get", False):
        r += 100.0

    # Penalty on death
    if done and not info.get("flag_get", False):
        # Stronger penalty if we die very early in the level
        if x_pos < 600:
            r -= 50.0
        else:
            r -= 25.0

    return r, best_x

# =========================
#  Worker
# =========================

def worker(rank, global_model, optimizer, global_counter, lock):
    env, action_meaning = make_env(use_simple_movement=False)
    num_actions = len(action_meaning)

    local_model = ActorCriticNet(num_actions).to(DEVICE)
    local_model.load_state_dict(global_model.state_dict())

    while True:
        with global_counter.get_lock():
            if global_counter.value >= MAX_GLOBAL_STEPS:
                break

        # Episode init
        reset_out = env.reset()
        if isinstance(reset_out, tuple):
            state = reset_out[0]
        else:
            state = reset_out

        state = torch.from_numpy(state).unsqueeze(0).to(DEVICE)

        episode_env_return = 0.0
        episode_shaped_return = 0.0
        done = False

        prev_x = 40
        best_x = 40
        step_in_episode = 0

        while not done:
            log_probs = []
            values = []
            rewards = []
            entropies = []

            # sync local params with global
            local_model.load_state_dict(global_model.state_dict())

            t = 0
            while t < T_MAX and not done:
                with global_counter.get_lock():
                    global_step = global_counter.value
                    global_counter.value += 1

                # Anneal entropy bonus over training
                progress = min(1.0, global_step / float(MAX_GLOBAL_STEPS))
                entropy_beta = (ENTROPY_BETA_START +
                                (ENTROPY_BETA_END - ENTROPY_BETA_START) * progress)

                logits, value = local_model(state)
                policy = F.softmax(logits, dim=-1)
                dist = torch.distributions.Categorical(policy)

                action = dist.sample()
                log_prob = dist.log_prob(action)
                entropy = dist.entropy().mean()

                # Step env
                out = env.step(action.item())
                if len(out) == 4:
                    next_state, env_reward, done, info = out
                else:
                    next_state, env_reward, terminated, truncated, info = out
                    done = terminated or truncated

                x_pos = info.get("x_pos", prev_x)
                shaped_r, best_x = shape_reward(env_reward, info, done, x_pos, prev_x, best_x)
                prev_x = x_pos

                episode_env_return += env_reward
                episode_shaped_return += shaped_r

                rewards.append(shaped_r)
                values.append(value.squeeze(0))
                log_probs.append(log_prob)
                entropies.append(entropy)

                next_state_t = torch.from_numpy(next_state).unsqueeze(0).to(DEVICE)
                state = next_state_t

                t += 1
                step_in_episode += 1

                if done:
                    break

                with global_counter.get_lock():
                    if global_counter.value >= MAX_GLOBAL_STEPS:
                        done = True
                        break

            # Bootstrap value
            if done:
                R = torch.zeros(1, 1, device=DEVICE)
            else:
                with torch.no_grad():
                    _, v = local_model(state)
                    R = v.detach()

            policy_loss = 0.0
            value_loss = 0.0

            for r, value, log_prob, entropy in reversed(
                list(zip(rewards, values, log_probs, entropies))
            ):
                R = torch.tensor([[r]], device=DEVICE) + GAMMA * R
                advantage = R - value

                value_loss += advantage.pow(2)
                policy_loss -= log_prob * advantage.detach()
                policy_loss -= entropy_beta * entropy

            loss = policy_loss + VALUE_LOSS_COEF * value_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(local_model.parameters(), 40.0)

            # Push local grads to global
            with lock:
                for global_param, local_param in zip(global_model.parameters(), local_model.parameters()):
                    if local_param.grad is not None:
                        global_param._grad = local_param.grad
                optimizer.step()

            # Logging (only one worker to avoid spam)
            if rank == 0 and done:
                print(
                    f"[Worker {rank}] Ep done | Global steps {global_step} "
                    f"| Env return {episode_env_return:.1f} "
                    f"| Shaped return {episode_shaped_return:.1f} "
                    f"| x_pos {x_pos}"
                )

        # end episode loop

# =========================
#  Evaluation
# =========================

def evaluate(model_path=SAVE_PATH, num_episodes=5, render=False):
    env, action_meaning = make_env(use_simple_movement=False)
    num_actions = len(action_meaning)

    model = ActorCriticNet(num_actions).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()

    print("Action set:")
    for i, a in enumerate(action_meaning):
        print(f"{i}: {a}")

    for ep in range(1, num_episodes + 1):
        reset_out = env.reset()
        if isinstance(reset_out, tuple):
            state = reset_out[0]
        else:
            state = reset_out
        state = torch.from_numpy(state).unsqueeze(0).to(DEVICE)

        done = False
        env_return = 0.0
        x_pos = 40

        while not done:
            if render:
                env.render()

            with torch.no_grad():
                logits, _ = model(state)
                probs = F.softmax(logits, dim=-1)
                action = probs.argmax(dim=-1).item()

            out = env.step(action)
            if len(out) == 4:
                next_state, r, done, info = out
            else:
                next_state, r, terminated, truncated, info = out
                done = terminated or truncated

            env_return += r
            x_pos = info.get("x_pos", x_pos)

            state = torch.from_numpy(next_state).unsqueeze(0).to(DEVICE)

        print(f"[EVAL] Episode {ep}: env return = {env_return:.1f}, x_pos = {x_pos}")

    env.close()

# =========================
#  Main
# =========================

def main():
    mp.set_start_method("spawn", force=True)

    env, action_meaning = make_env(use_simple_movement=False)
    num_actions = len(action_meaning)
    env.close()

    global_model = ActorCriticNet(num_actions).to(DEVICE)
    global_model.share_memory()

    optimizer = torch.optim.Adam(global_model.parameters(), lr=LR)

    global_counter = mp.Value('i', 0)
    lock = mp.Lock()

    processes = []
    for rank in range(NUM_WORKERS):
        p = mp.Process(target=worker, args=(rank, global_model, optimizer, global_counter, lock))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    torch.save(global_model.state_dict(), SAVE_PATH)
    print(f"Training finished. Model saved to {SAVE_PATH}")

    evaluate(num_episodes=5, render=False)

if __name__ == "__main__":
    main()
