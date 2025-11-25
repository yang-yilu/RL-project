# a3c_mario5.py

import time
from collections import deque

import cv2
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
from torch.distributions import Categorical

import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace


# =========================
#  Hyperparameters
# =========================
GAMMA             = 0.99
BASE_ENTROPY_BETA = 0.02      # will be decayed as steps increase
VALUE_LOSS_COEF   = 0.5
LR                = 5e-5      # smaller LR for stability
T_MAX             = 20        # longer unrolls
NUM_WORKERS       = 4
MAX_GLOBAL_STEPS  = 3_000_000
SAVE_PATH         = "./a3c_mario3.pth"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)


# =========================
#  Actions & jump helper
# =========================

# Only treat RIGHT+JUMP actions as "big jumps"
JUMP_ACTIONS = [
    i for i, a in enumerate(SIMPLE_MOVEMENT)
    if ("A" in a and "right" in a)
]
print("JUMP_ACTIONS indices:", JUMP_ACTIONS,
      "->", [SIMPLE_MOVEMENT[i] for i in JUMP_ACTIONS])


class StickyJumpEnv(gym.Wrapper):
    """
    If the agent chooses a jump-related action, keep repeating it
    for 'hold_frames' steps so a single choice creates a full jump.
    """
    def __init__(self, env, hold_frames=5, jump_actions=None):
        super().__init__(env)
        self.hold_frames = hold_frames
        self.jump_actions = set(jump_actions or [])
        self._current_action = None
        self._hold = 0

    def reset(self, **kwargs):
        self._current_action = None
        self._hold = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        # If we're still holding a jump action, ignore new action
        if self._hold > 0 and self._current_action is not None:
            act = self._current_action
            self._hold -= 1
        else:
            act = action
            # Start holding if this is a jump action
            if act in self.jump_actions:
                self._current_action = act
                self._hold = self.hold_frames - 1  # this step + N-1 more
            else:
                self._current_action = None
                self._hold = 0

        return self.env.step(act)


class GrayResizeObs(gym.ObservationWrapper):
    """Convert RGB frames to 84x84 grayscale."""
    def __init__(self, env, w=84, h=84):
        super().__init__(env)
        self.w = w
        self.h = h
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(h, w, 1), dtype=np.uint8
        )

    def observation(self, obs):
        gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        gray = cv2.resize(gray, (self.w, self.h), interpolation=cv2.INTER_AREA)
        return gray[:, :, None]


class FrameStack(gym.Wrapper):
    """Stack last k grayscale frames (channel-first)."""
    def __init__(self, env, k=4):
        super().__init__(env)
        self.k = k
        self.frames = deque(maxlen=k)
        h, w, _ = env.observation_space.shape
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(k, h, w), dtype=np.uint8
        )

    def reset(self):
        obs = self.env.reset()
        obs = self._process_obs(obs)
        self.frames.clear()
        for _ in range(self.k):
            self.frames.append(obs)
        return self._get_obs()

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        obs = self._process_obs(obs)
        self.frames.append(obs)
        return self._get_obs(), reward, done, info

    def _process_obs(self, obs):
        # incoming obs is H x W x 1 grayscale
        return obs[:, :, 0]

    def _get_obs(self):
        return np.stack(self.frames, axis=0)


class BaseRewardInfoWrapper(gym.Wrapper):
    """
    Simple wrapper that just ensures info["base_reward"] is always set to
    the true environment reward, without changing it.
    Used for evaluation (no shaping).
    """
    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        info = dict(info)
        info.setdefault("base_reward", float(reward))
        return obs, reward, done, info


class ProgressRewardWrapper(gym.Wrapper):
    """
    Shaped reward (for training):

      shaped = + dx_scale * max(0, Δx)
               - step_cost each step
               + survival_bonus every `survival_every` steps IF enough progress
               - death_penalty if done without flag
               - enemy_penalty if we detect life loss on the death step
               + flag_bonus if flag reached

    We also:
      - keep base_reward (raw env reward) in info["base_reward"]
      - store un-clipped shaped in info["shaped_raw"]
      - optionally clip per-step shaped reward (non-terminal) into [clip_min, clip_max]
        and return that clipped value as the reward.
    """
    def __init__(self,
                 env,
                 dx_scale=0.05,
                 step_cost=0.0005,
                 death_penalty=8.0,
                 enemy_penalty=5.0,
                 flag_bonus=50.0,
                 survival_every=60,
                 survival_bonus=0.5,
                 min_dx_for_bonus=3.0,
                 clip_rewards=True,
                 clip_min=-20.0,
                 clip_max=20.0):
        super().__init__(env)
        self.dx_scale = dx_scale
        self.step_cost = step_cost
        self.death_penalty = death_penalty
        self.enemy_penalty = enemy_penalty
        self.flag_bonus = flag_bonus
        self.survival_every = survival_every
        self.survival_bonus = survival_bonus
        self.min_dx_for_bonus = min_dx_for_bonus

        self.clip_rewards = clip_rewards
        self.clip_min = clip_min
        self.clip_max = clip_max

        self._reset_internal_state()

    def _reset_internal_state(self):
        self.last_x = 0
        self.steps_alive = 0
        self.x_at_last_bonus = 0
        self.last_life = None

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        self._reset_internal_state()
        return obs

    def step(self, action):
        obs, env_r, done, info = self.env.step(action)
        info = dict(info)

        # Always expose the raw env reward
        info.setdefault("base_reward", float(env_r))

        # Forward progress
        x = info.get("x_pos", 0)
        dx = max(0, x - self.last_x)
        self.last_x = x

        shaped = self.dx_scale * dx - self.step_cost

        # Survival bonus: only if alive AND we've moved enough since last bonus
        self.steps_alive += 1
        if (not done) and (self.steps_alive % self.survival_every == 0):
            if (self.last_x - self.x_at_last_bonus) >= self.min_dx_for_bonus:
                shaped += self.survival_bonus
                self.x_at_last_bonus = self.last_x

        flag_get = info.get("flag_get", False)
        life = info.get("life", None)
        enemy_hit = False

        # Track life loss (proxy for enemy / bad hit)
        if life is not None:
            if self.last_life is None:
                self.last_life = life
            elif life < self.last_life:
                enemy_hit = True
                self.last_life = life

        # Terminal adjustments
        if done:
            if flag_get:
                shaped += self.flag_bonus
            else:
                shaped -= self.death_penalty
                if enemy_hit:
                    shaped -= self.enemy_penalty

        # Diagnostics
        info["shaped_raw"] = float(shaped)

        # Clip only NON-terminal steps to keep variance under control
        if self.clip_rewards and not done:
            shaped_clipped = float(np.clip(shaped, self.clip_min, self.clip_max))
            info["shaped_clipped"] = shaped_clipped
            reward = shaped_clipped
        else:
            reward = float(shaped)

        if done:
            # reset counters for next episode
            self._reset_internal_state()

        return obs, reward, done, info


def make_mario_env(train: bool = True, use_shaped_env: bool = True):
    """
    Unified env constructor.

    - train=True,  use_shaped_env=True  -> A3C training w/ ProgressRewardWrapper
    - train=False, use_shaped_env=True  -> evaluation with shaped reward
    - train=False, use_shaped_env=False -> evaluation with raw env reward
    """
    env = gym_super_mario_bros.make("SuperMarioBros-1-1-v0")
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    env = StickyJumpEnv(env, hold_frames=5, jump_actions=JUMP_ACTIONS)

    if train and use_shaped_env:
        env = ProgressRewardWrapper(env)
    else:
        # Just tag base_reward and leave reward unchanged
        env = BaseRewardInfoWrapper(env)

    env = GrayResizeObs(env)
    env = FrameStack(env, k=4)
    return env


# =========================
#  A3C Network
# =========================

class ActorCriticNet(nn.Module):
    def __init__(self, num_actions):
        super().__init__()
        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc    = nn.Linear(7 * 7 * 64, 512)

        self.policy_head = nn.Linear(512, num_actions)
        self.value_head  = nn.Linear(512, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc(x))
        logits = self.policy_head(x)
        value  = self.value_head(x)
        return logits, value.squeeze(-1)


def preprocess_state(state: np.ndarray) -> torch.Tensor:
    """
    state: np array (4, 84, 84), uint8
    return: torch tensor (1, 4, 84, 84), float32 in [0,1]
    """
    x = torch.from_numpy(state).float() / 255.0
    return x.unsqueeze(0)


def get_entropy_beta(global_steps: int) -> float:
    """
    Simple linear decay of entropy coefficient:
    starts at BASE_ENTROPY_BETA and decays to ~0.1 * BASE_ENTROPY_BETA
    by 80% of MAX_GLOBAL_STEPS, then stays there.
    """
    frac = min(global_steps / (0.8 * MAX_GLOBAL_STEPS), 1.0)
    return BASE_ENTROPY_BETA * (1.0 - 0.9 * frac)


# =========================
#  Worker process
# =========================

def worker_process(rank, global_model, optimizer,
                   global_counter, global_episode, print_lock):
    print(f"Worker {rank} starting on device {DEVICE}")
    env = make_mario_env(train=True, use_shaped_env=True)

    # Local worker model lives on DEVICE
    local_model = ActorCriticNet(num_actions=env.action_space.n).to(DEVICE)
    local_model.load_state_dict(global_model.state_dict())
    local_model.train()

    recent_rewards = deque(maxlen=10)

    while True:
        # Check global step limit before starting a new episode
        with global_counter.get_lock():
            if global_counter.value >= MAX_GLOBAL_STEPS:
                break

        state = env.reset()
        done = False
        episode_shaped = 0.0
        episode_base = 0.0

        while not done:
            log_probs = []
            values    = []
            rewards   = []
            entropies = []
            steps_this_batch = 0

            # Collect up to T_MAX steps
            while steps_this_batch < T_MAX and not done:
                s = preprocess_state(state).to(DEVICE)
                logits, value = local_model(s)

                probs = F.softmax(logits, dim=-1)
                log_probs_all = F.log_softmax(logits, dim=-1)
                dist = Categorical(probs)
                action = dist.sample()

                log_prob = log_probs_all[0, action]
                entropy  = -(probs * log_probs_all).sum()

                next_state, reward, done, info = env.step(action.item())

                base_r = info.get("base_reward", reward)
                shaped_raw = info.get("shaped_raw", reward)

                episode_base += float(base_r)
                episode_shaped += float(shaped_raw)

                log_probs.append(log_prob)
                values.append(value.squeeze(0))
                rewards.append(torch.tensor(reward, dtype=torch.float32, device=DEVICE))
                entropies.append(entropy)

                state = next_state
                steps_this_batch += 1

            # Update global step counter once per rollout
            with global_counter.get_lock():
                global_counter.value += steps_this_batch
                current_steps = global_counter.value
                reached_limit = (global_counter.value >= MAX_GLOBAL_STEPS)

            # Bootstrap value
            if done or reached_limit:
                R = torch.zeros(1, device=DEVICE)
            else:
                s = preprocess_state(state).to(DEVICE)
                _, value = local_model(s)
                R = value.detach().squeeze(0)

            policy_loss = torch.zeros(1, device=DEVICE)
            value_loss  = torch.zeros(1, device=DEVICE)

            entropy_beta = get_entropy_beta(current_steps)

            for i in reversed(range(len(rewards))):
                R = rewards[i] + GAMMA * R
                advantage = R - values[i]

                value_loss = value_loss + advantage.pow(2)
                policy_loss = policy_loss - log_probs[i] * advantage.detach() - entropy_beta * entropies[i]

            loss = policy_loss + VALUE_LOSS_COEF * value_loss

            # ---- clear grads ----
            local_model.zero_grad()
            optimizer.zero_grad()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(local_model.parameters(), 40.0)

            # copy local grads (DEVICE) into shared global model (CPU)
            for global_param, local_param in zip(global_model.parameters(), local_model.parameters()):
                if local_param.grad is None:
                    continue
                if global_param.grad is None:
                    global_param.grad = local_param.grad.detach().cpu().clone()
                else:
                    global_param.grad.copy_(local_param.grad.detach().cpu())

            optimizer.step()

            # sync local weights from updated global model
            local_model.load_state_dict(global_model.state_dict())

            if done or reached_limit:
                recent_rewards.append(episode_shaped)
                with global_episode.get_lock():
                    global_episode.value += 1
                    ep = global_episode.value
                avg10 = float(np.mean(recent_rewards)) if len(recent_rewards) > 0 else 0.0
                with print_lock:
                    print(
                        f"[Worker {rank}] Ep {ep} | "
                        f"Global steps {current_steps} | "
                        f"Base {episode_base:.1f} | "
                        f"Shaped {episode_shaped:.1f} | "
                        f"Avg10_shaped {avg10:.1f}"
                    )
                break

    env.close()
    print(f"Worker {rank} finished.")


# =========================
#  Main
# =========================

def main():
    mp.set_start_method("spawn", force=True)

    # Figure out action space
    tmp_env = make_mario_env(train=True, use_shaped_env=True)
    n_actions = tmp_env.action_space.n
    tmp_env.close()

    # Global model on CPU (shared params)
    global_model = ActorCriticNet(num_actions=n_actions)
    global_model.share_memory()

    optimizer = torch.optim.Adam(global_model.parameters(), lr=LR)

    global_counter = mp.Value("i", 0)
    global_episode = mp.Value("i", 0)
    print_lock     = mp.Lock()

    processes = []
    for rank in range(NUM_WORKERS):
        p = mp.Process(
            target=worker_process,
            args=(rank, global_model, optimizer,
                  global_counter, global_episode, print_lock)
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    torch.save(global_model.state_dict(), SAVE_PATH)
    print(f"Training finished. Model saved to {SAVE_PATH}")


if __name__ == "__main__":
    main()
