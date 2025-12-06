"""
A3C for Super Mario Bros (clean fixed setup)
- Macro actions with short/long jump (hold A) to clear pipes
- Simplified action space (fewer, more useful actions)
- Reward shaping: forward progress, milestone bonuses, trap-zone penalty, stall penalty, death penalty, flag bonus
- Entropy decay to reduce exploration over time (A3C-style)
- Frame preprocessing (grayscale 84x84) + 4-frame stacking
- Multi-worker with torch.multiprocessing (spawn-safe)

Run:
    python a3c_mario_fixed.py
"""

import os
import math
import time
from collections import deque
from dataclasses import dataclass

import cv2
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.multiprocessing as mp
import torch.nn.functional as F

import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace

# =========================
#  Hyperparameters
# =========================
GAMMA               = 0.99
ENTROPY_BETA_START  = 0.02
ENTROPY_BETA_END    = 0.001
ENTROPY_DECAY_STEPS = 1_000_000  # global steps to decay entropy bonus
VALUE_LOSS_COEF     = 0.5
LR                  = 2.5e-4
T_MAX               = 20          # rollout length
NUM_WORKERS         = 4
MAX_GLOBAL_STEPS    = 3_000_000
SAVE_PATH           = "./a3c_mario_fixed.pth"
LOG_INTERVAL        = 5000

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================
#  Macro action config
#  We build JoypadSpace with these combos, then define MACRO actions
#  that repeat certain combos to "hold" A for longer jumps.
# =========================
# Base combos used by JoypadSpace
MOVEMENT = [
    ['NOOP'],          # 0
    ['right'],         # 1
    ['right', 'A'],    # 2  (jump while moving right)
    ['A'],             # 3  (vertical jump)
    ['right', 'B'],    # 4  (run)
]

@dataclass
class Macro:
    base_id: int
    repeat: int
    name: str

# Define a compact macro action space that the agent actually chooses from
MACROS = [
    Macro(base_id=0, repeat=1,  name="NOOP"),
    Macro(base_id=1, repeat=2,  name="RIGHT"),
    Macro(base_id=2, repeat=6,  name="RIGHT_JUMP_S"),   # short right jump
    Macro(base_id=2, repeat=12, name="RIGHT_JUMP_L"),   # long right jump (hold A longer)
    Macro(base_id=3, repeat=6,  name="JUMP_S"),         # short vertical jump
    Macro(base_id=4, repeat=4,  name="RIGHT_RUN"),      # run forward a bit
]
N_MACROS = len(MACROS)

# =========================
#  Wrappers: Preprocess, FrameStack, MacroAction, RewardShaping
# =========================

class Preprocess84Gray(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(84, 84), dtype=np.float32)

    def observation(self, obs):
        # obs: HxWxC (RGB)
        # Resize -> grayscale -> normalize [0,1]
        img = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        img = cv2.resize(img, (84, 84), interpolation=cv2.INTER_AREA)
        img = img.astype(np.float32) / 255.0
        return img


class FrameStack(gym.Wrapper):
    def __init__(self, env, k=4):
        super().__init__(env)
        self.k = k
        self.frames = deque(maxlen=k)
        shp = env.observation_space.shape
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(k, shp[0], shp[1]), dtype=np.float32)

    def reset(self, **kwargs):
        res = self.env.reset(**kwargs)
        obs = res[0] if isinstance(res, tuple) else res
        self.frames.clear()
        for _ in range(self.k):
            self.frames.append(obs)
        return self._get_obs(), {} if isinstance(res, tuple) else self._get_obs()

    def step(self, action):
        res = self.env.step(action)
        if len(res) == 5:
            obs, reward, terminated, truncated, info = res
            done = terminated or truncated
        else:
            obs, reward, done, info = res
        self.frames.append(obs)
        return self._get_obs(), reward, done, info

    def _get_obs(self):
        assert len(self.frames) == self.k
        return np.stack(list(self.frames), axis=0)


class MacroActionWrapper(gym.Wrapper):
    """
    Maps macro action index -> (base JoypadSpace action id, repeat count).
    Repeats the env.step(base_id) for 'repeat' frames, accumulating reward and respecting done.
    """
    def __init__(self, env, macros):
        super().__init__(env)
        self.macros = macros
        self.action_space = gym.spaces.Discrete(len(macros))

    def step(self, macro_idx):
        macro = self.macros[macro_idx]
        total_r = 0.0
        done = False
        info_acc = {}
        obs = None
        for _ in range(macro.repeat):
            res = self.env.step(macro.base_id)
            if len(res) == 5:
                o, r, terminated, truncated, info = res
                d = terminated or truncated
            else:
                o, r, d, info = res
            obs = o
            total_r += r
            done = d
            info_acc = info
            if done:
                break
        return obs, total_r, done, info_acc


class ShapedRewardWrapper(gym.Wrapper):
    """
    Adds shaping to native reward:
    + forward progress
    + milestone bonuses at new zones (400,800,1200,1600)
    + trap-zone penalty around x=[280..340]
    + stall penalty (no forward movement for N steps)
    + death penalty at end, big flag bonus
    """
    def __init__(self, env):
        super().__init__(env)
        self.last_x = 0
        self.stall_steps = 0
        self.milestones_hit = set()
        self.trap_zone_streak = 0

    def reset(self, **kwargs):
        res = self.env.reset(**kwargs)
        obs = res[0] if isinstance(res, tuple) else res
        self.last_x = 0
        self.stall_steps = 0
        self.trap_zone_streak = 0
        self.milestones_hit = set()
        return (obs, {}) if isinstance(res, tuple) else obs

    def step(self, action):
        res = self.env.step(action)
        if len(res) == 5:
            obs, env_r, terminated, truncated, info = res
            done = terminated or truncated
        else:
            obs, env_r, done, info = res

        x = int(info.get('x_pos', 0))
        shaped = 0.0

        # Forward progress shaping
        dx = x - self.last_x
        shaped += 0.01 * dx  # gentle push forward

        # Stall penalty
        if dx <= 0:
            self.stall_steps += 1
        else:
            self.stall_steps = 0
        if self.stall_steps >= 18:  # around ~0.3 seconds with default skips
            shaped -= 0.5

        # Trap-zone penalty: x roughly around 300 where agent often gets stuck
        if 280 <= x <= 340:
            self.trap_zone_streak += 1
            shaped -= 0.10  # per step in trap
            if self.trap_zone_streak % 30 == 0:
                shaped -= 1.0  # extra slap every ~30 frames if camping
        else:
            self.trap_zone_streak = 0

        # Milestone bonuses (once per episode)
        for m in (400, 800, 1200, 1600, 2000):
            if x >= m and m not in self.milestones_hit:
                shaped += 5.0
                self.milestones_hit.add(m)

        # Episode end shaping
        if done:
            if info.get('flag_get', False):
                shaped += 50.0
            else:
                shaped -= 5.0  # death/timeout penalty

        info = dict(info)  # copy for safety
        info['env_reward'] = env_r
        info['shaped_reward'] = shaped

        self.last_x = x
        return obs, env_r + shaped, done, info


def make_env(seed=None):
    env = gym_super_mario_bros.make("SuperMarioBros-1-1-v0", apply_api_compatibility=True)  # gym<=0.26 compat flag
    env = JoypadSpace(env, MOVEMENT)
    env = MacroActionWrapper(env, MACROS)
    env = Preprocess84Gray(env)
    env = FrameStack(env, k=4)
    env = ShapedRewardWrapper(env)
    if seed is not None:
        try:
            env.reset(seed=seed)
        except TypeError:
            # Older gym
            env.seed(seed)
    return env


# =========================
#  Model (Conv -> LSTM -> Policy/Value)
# =========================
class A3CNet(nn.Module):
    def __init__(self, n_actions=N_MACROS):
        super().__init__()
        # Input: (B, 4, 84, 84)
        self.conv = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4),  # -> 32x20x20
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2), # -> 64x9x9
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1), # -> 64x7x7
            nn.ReLU(inplace=True),
        )
        self.fc = nn.Linear(64*7*7, 512)
        self.lstm = nn.LSTMCell(512, 256)
        self.policy = nn.Linear(256, n_actions)
        self.value  = nn.Linear(256, 1)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x, hx, cx):
        # x: (B=1, 4, 84, 84)
        z = self.conv(x)
        z = z.view(z.size(0), -1)
        z = F.relu(self.fc(z))
        hx, cx = self.lstm(z, (hx, cx))
        logits = self.policy(hx)
        value  = self.value(hx)
        return logits, value, hx, cx

    def initial_state(self, batch_size=1, device=DEVICE):
        hx = torch.zeros(batch_size, 256, device=device)
        cx = torch.zeros(batch_size, 256, device=device)
        return hx, cx


# =========================
#  Shared Adam (A3C-style)
# =========================
class SharedAdam(torch.optim.Adam):
    def __init__(self, params, lr=1e-4, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
        super().__init__(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = torch.zeros(1)
                state['exp_avg'] = torch.zeros_like(p.data)
                state['exp_avg_sq'] = torch.zeros_like(p.data)

        # move to shared memory
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['exp_avg'].share_memory_()
                state['exp_avg_sq'].share_memory_()
                state['step'].share_memory_()


def compute_returns(rewards, values, dones, gamma=GAMMA, next_value=0.0):
    R = next_value
    returns = []
    for r, d in zip(reversed(rewards), reversed(dones)):
        if d:
            R = 0.0
        R = r + gamma * R
        returns.insert(0, R)
    return returns


def global_entropy_beta(global_step):
    # Exponential decay from START to END
    frac = min(1.0, global_step / ENTROPY_DECAY_STEPS)
    return ENTROPY_BETA_START * (ENTROPY_BETA_END / ENTROPY_BETA_START) ** frac


def worker_fn(rank, global_model, optimizer, global_step, global_ep, done_flag):
    local_model = A3CNet().to(DEVICE)
    local_model.load_state_dict(global_model.state_dict())

    env = make_env(seed=123 + rank)
    obs = env.reset()
    if isinstance(obs, tuple):
        obs, _info = obs
    obs_np = np.array(obs, dtype=np.float32)
    episode_env_return = 0.0
    episode_shaped_return = 0.0
    episode_x_pos = 0
    ep_count = 0

    hx, cx = global_model.initial_state(device=DEVICE)

    while True:
        if done_flag.value:
            break
        values, log_probs, rewards, dones = [], [], [], []
        hx_det, cx_det = hx.detach(), cx.detach()

        # rollout
        for t in range(T_MAX):
            state_t = torch.from_numpy(obs_np).unsqueeze(0).to(DEVICE)  # (1,4,84,84)
            logits, value, hx_det, cx_det = local_model(state_t, hx_det, cx_det)
            prob = F.softmax(logits, dim=-1)
            # Sample action (A3C exploration); entropy term decays separately
            action = prob.multinomial(num_samples=1).detach()
            log_prob = F.log_softmax(logits, dim=-1).gather(1, action)

            action_idx = int(action.item())
            next_obs, reward, done, info = env.step(action_idx)

            # bookkeeping
            env_r = float(info.get('env_reward', 0.0))
            shaped_r = float(info.get('shaped_reward', 0.0))
            episode_env_return += env_r
            episode_shaped_return += (reward - env_r) + env_r  # equals 'reward', but keep variable clear
            episode_x_pos = int(info.get('x_pos', 0))

            values.append(value)
            log_probs.append(log_prob)
            rewards.append(float(reward))
            dones.append(done)

            obs_np = np.array(next_obs, dtype=np.float32)

            with global_step.get_lock():
                global_step.value += 1
                gs = global_step.value

            if done:
                ep_count += 1
                print(f"[Worker {rank}] Ep {ep_count} | Global steps {gs} | "
                      f"Env return {episode_env_return:.1f} | Shaped return {episode_shaped_return:.1f} | x_pos {episode_x_pos}")
                # reset episode
                res = env.reset()
                if isinstance(res, tuple):
                    next_obs, _info = res
                else:
                    next_obs = res
                obs_np = np.array(next_obs, dtype=np.float32)
                episode_env_return = 0.0
                episode_shaped_return = 0.0
                episode_x_pos = 0
                hx_det, cx_det = global_model.initial_state(device=DEVICE)
                break

        # bootstrap value
        with torch.no_grad():
            state_t = torch.from_numpy(obs_np).unsqueeze(0).to(DEVICE)
            _, next_value, _, _ = local_model(state_t, hx_det, cx_det)
            next_value = next_value.detach().squeeze(0).cpu().item()

        returns = compute_returns(rewards, [v.item() for v in values], dones, gamma=GAMMA, next_value=next_value)

        # convert to tensors
        returns_t = torch.tensor(returns, dtype=torch.float32, device=DEVICE).unsqueeze(1)
        log_probs_t = torch.cat(log_probs, dim=0)  # (T,1)
        values_t = torch.cat(values, dim=0)        # (T,1)

        advantage = returns_t - values_t

        # losses
        policy_loss = -(log_probs_t * advantage.detach()).mean()
        value_loss = VALUE_LOSS_COEF * advantage.pow(2).mean()

        # entropy (encourage exploration early; decay over time)
        logits, _, _, _ = local_model(state_t, hx_det, cx_det)  # entropy at last state (ok approximation)
        probs = F.softmax(logits, dim=-1)
        entropy = -(F.log_softmax(logits, dim=-1) * probs).sum(dim=1).mean()

        with global_step.get_lock():
            gs_now = global_step.value
        ent_beta = global_entropy_beta(gs_now)
        loss = policy_loss + value_loss - ent_beta * entropy

        optimizer.zero_grad()
        loss.backward()
        # gradient clipping helps stability
        torch.nn.utils.clip_grad_norm_(local_model.parameters(), 40.0)

        # push local grads to global model
        for local_param, global_param in zip(local_model.parameters(), global_model.parameters()):
            if global_param.grad is None:
                global_param.grad = local_param.grad.clone()
            else:
                global_param.grad.copy_(local_param.grad)

        optimizer.step()
        # sync local params
        local_model.load_state_dict(global_model.state_dict())

        # periodic save + stop
        if gs_now >= MAX_GLOBAL_STEPS:
            with done_flag.get_lock():
                done_flag.value = True

        if rank == 0 and gs_now % LOG_INTERVAL < T_MAX:
            torch.save(global_model.state_dict(), SAVE_PATH)

    env.close()


def main():
    mp.set_start_method("spawn", force=True)

    global_model = A3CNet().to(DEVICE)
    global_model.share_memory()

    optimizer = SharedAdam(global_model.parameters(), lr=LR)
    optimizer.share_memory()

    global_step = mp.Value('i', 0)
    global_ep = mp.Value('i', 0)
    done_flag = mp.Value('b', False)

    procs = []
    for rank in range(NUM_WORKERS):
        p = mp.Process(target=worker_fn, args=(rank, global_model, optimizer, global_step, global_ep, done_flag))
        p.start()
        procs.append(p)

    for p in procs:
        p.join()

    torch.save(global_model.state_dict(), SAVE_PATH)
    print(f"Training finished. Saved: {SAVE_PATH}")


if __name__ == "__main__":
    main()
