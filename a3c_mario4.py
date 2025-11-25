# a3c_mario.py
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
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace


# =========================
#  Hyperparameters
# =========================
GAMMA = 0.99

# Entropy schedule: start more random, decay to smaller exploration
ENTROPY_BETA_START = 0.02
ENTROPY_BETA_END   = 0.005

VALUE_LOSS_COEF = 0.5
LR              = 5e-5        # smaller LR for stability
T_MAX           = 20          # longer unrolls for better credit assignment
NUM_WORKERS     = 4
MAX_GLOBAL_STEPS = 300_000

SAVE_PATH = "./a3c_mario.pth"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)


def get_entropy_beta(current_steps: int) -> float:
    """Linearly decay entropy coefficient from START to END."""
    frac = min(1.0, current_steps / float(MAX_GLOBAL_STEPS))
    return ENTROPY_BETA_START + frac * (ENTROPY_BETA_END - ENTROPY_BETA_START)


# =========================
#  Mario env + preprocessing
# =========================

# Only treat RIGHT+JUMP actions as "big jumps"
JUMP_ACTIONS = [
    i for i, a in enumerate(SIMPLE_MOVEMENT)
    if ("A" in a and "right" in a)
]
print("JUMP_ACTIONS indices:", JUMP_ACTIONS, "->", [SIMPLE_MOVEMENT[i] for i in JUMP_ACTIONS])


class StickyJumpEnv(gym.Wrapper):
    """
    If the agent chooses a jump-related action, keep repeating it
    for 'hold_frames' steps so a single choice creates a full jump.
    """
    def __init__(self, env, hold_frames=4, jump_actions=None):
        # NOTE: hold_frames was 8 before (too long, locked Mario into attacks)
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
            if act in self.jump_actions:
                # start holding this jump for a few frames
                self._current_action = act
                self._hold = self.hold_frames - 1
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


class ProgressRewardWrapper(gym.Wrapper):
    """
    Shaped reward:

      * Forward progress in x (reward)
      * Optional small step cost when completely stuck
      * Survival bonus over time
      * Enemy hit penalty when life drops
      * Death penalty when episode ends without flag
      * Large bonus when reaching flag

    Also exposes:
      info["base_reward"]    : original env reward
      info["shaped_raw"]     : unclipped shaped reward
      info["shaped_clipped"] : clipped shaped reward actually returned
      info["enemy_hit"]      : True if life decreased this step
    """
    def __init__(
        self,
        env,
        dx_scale=0.02,
        step_cost=0.001,
        enemy_penalty=15.0,
        death_penalty=40.0,
        flag_bonus=300.0,
        survival_every=60,        # add survival bonus every N frames alive
        survival_bonus=0.1,
        clip_range=(-10.0, 10.0), # clip only non-terminal rewards
    ):
        super().__init__(env)
        self.dx_scale = dx_scale
        self.step_cost = step_cost
        self.enemy_penalty = enemy_penalty
        self.death_penalty = death_penalty
        self.flag_bonus = flag_bonus
        self.survival_every = survival_every
        self.survival_bonus = survival_bonus
        self.clip_min, self.clip_max = clip_range

        self.last_x = 0
        self.last_life = None
        self.steps_alive = 0

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        self.last_x = 0
        self.last_life = None
        self.steps_alive = 0
        return obs

    def step(self, action):
        # Get original reward from game
        obs, base_reward, done, info = self.env.step(action)

        x = info.get("x_pos", 0)
        life = info.get("life", None)
        status = info.get("status", "")

        # Forward progress
        dx = x - self.last_x
        self.last_x = x

        shaped = 0.0
        if dx > 0:
            shaped += self.dx_scale * dx

        # Small cost if completely stuck (no progress)
        if dx == 0:
            shaped -= self.step_cost

        # Survival bonus
        self.steps_alive += 1
        if self.steps_alive % self.survival_every == 0:
            shaped += self.survival_bonus

        flag_get = info.get("flag_get", False)

        # Detect enemy hit via life drop (single trigger per step)
        enemy_hit = False
        if life is not None:
            if self.last_life is not None and life < self.last_life:
                enemy_hit = True
            self.last_life = life

        if enemy_hit:
            shaped -= self.enemy_penalty

        # Terminal penalties / bonus
        if done and not flag_get:
            shaped -= self.death_penalty
        if flag_get:
            shaped += self.flag_bonus

        # --------- reward clipping policy ----------
        # We only clip *intermediate* shaped rewards so that the relative
        # scale between "beat level" vs "die to first goomba" is preserved.
        if done:
            reward_out = shaped
            shaped_clipped = shaped
        else:
            shaped_clipped = float(np.clip(shaped, self.clip_min, self.clip_max))
            reward_out = shaped_clipped

        # Attach diagnostics
        info["base_reward"] = float(base_reward)
        info["shaped_raw"] = float(shaped)
        info["shaped_clipped"] = float(shaped_clipped)
        info["enemy_hit"] = enemy_hit

        return obs, reward_out, done, info


def make_mario_env(train: bool = True):
    """
    train=True  -> use shaped rewards (ProgressRewardWrapper)
    train=False -> use original env reward (no shaping), still grayscale+framestack
    """
    env = gym_super_mario_bros.make("SuperMarioBros-1-1-v0")
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    env = StickyJumpEnv(env, hold_frames=4, jump_actions=JUMP_ACTIONS)

    if train:
        env = ProgressRewardWrapper(env)

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


# =========================
#  Worker process
# =========================

def worker_process(rank, global_model, optimizer,
                   global_counter, global_episode, print_lock):
    print(f"Worker {rank} starting on device {DEVICE}")
    env = make_mario_env(train=True)

    # Local worker model lives on DEVICE
    local_model = ActorCriticNet(num_actions=env.action_space.n).to(DEVICE)
    local_model.load_state_dict(global_model.state_dict())
    local_model.train()

    recent_shaped_rewards = deque(maxlen=10)

    while True:
        # Fast check before starting an episode
        with global_counter.get_lock():
            if global_counter.value >= MAX_GLOBAL_STEPS:
                break

        state = env.reset()
        episode_shaped = 0.0
        episode_base = 0.0
        done = False

        while not done:
            log_probs = []
            values = []
            rewards = []   # shaped rewards used for training
            entropies = []
            t = 0
            current_steps = 0  # will get set inside loop

            while t < T_MAX and not done:
                s = preprocess_state(state).to(DEVICE)
                logits, value = local_model(s)

                probs = F.softmax(logits, dim=-1)
                log_probs_all = F.log_softmax(logits, dim=-1)
                dist = torch.distributions.Categorical(probs)
                action = dist.sample()

                log_prob = log_probs_all[0, action]
                entropy = -(probs * log_probs_all).sum()

                next_state, reward, done, info = env.step(action.item())

                # shaped reward used for training
                episode_shaped += info.get("shaped_raw", float(reward))
                episode_base += info.get("base_reward", 0.0)

                log_probs.append(log_prob)
                values.append(value.squeeze(0))
                rewards.append(torch.tensor(reward, dtype=torch.float32, device=DEVICE))
                entropies.append(entropy)

                state = next_state
                t += 1

                # ---- global step counter (single lock) ----
                with global_counter.get_lock():
                    global_counter.value += 1
                    current_steps = global_counter.value

                if current_steps >= MAX_GLOBAL_STEPS:
                    done = True
                    break

            # If no steps collected (rare), just break
            if len(rewards) == 0:
                break

            # Bootstrap value
            if done:
                R = torch.zeros(1, device=DEVICE)
            else:
                s = preprocess_state(state).to(DEVICE)
                _, value = local_model(s)
                R = value.detach().squeeze(0)

            policy_loss = torch.zeros(1, device=DEVICE)
            value_loss  = torch.zeros(1, device=DEVICE)

            # Entropy coefficient based on last current_steps
            entropy_beta = get_entropy_beta(current_steps)

            # n-step returns (backwards)
            for i in reversed(range(len(rewards))):
                R = rewards[i] + GAMMA * R
                advantage = R - values[i]

                value_loss = value_loss + advantage.pow(2)
                policy_loss = policy_loss - log_probs[i] * advantage.detach() \
                              - entropy_beta * entropies[i]

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
                g = local_param.grad.detach().cpu()
                if global_param.grad is None:
                    global_param.grad = g.clone()
                else:
                    global_param.grad.copy_(g)

            optimizer.step()

            # sync local weights from updated global model
            local_model.load_state_dict(global_model.state_dict())

        # End of episode
        if done:
            recent_shaped_rewards.append(episode_shaped)
            with global_episode.get_lock():
                global_episode.value += 1
                ep = global_episode.value
            avg10 = np.mean(recent_shaped_rewards) if recent_shaped_rewards else 0.0
            with print_lock:
                print(
                    f"[Worker {rank}] Ep {ep} | "
                    f"Global steps {global_counter.value} | "
                    f"Shaped {episode_shaped:.1f} | "
                    f"Base {episode_base:.1f} | "
                    f"Avg10 (shaped) {avg10:.1f}"
                )

    env.close()
    print(f"Worker {rank} finished.")


# =========================
#  Evaluation (greedy, original env reward)
# =========================

EVAL_USE_SAMPLING = False  # False = greedy argmax, True = sample from policy


def evaluate(num_episodes=35, render=False, use_shaped_env=False):
    """
    After training, load the saved model and run episodes with
    either sampling or greedy action selection.

    use_shaped_env = False -> eval on original env reward
    use_shaped_env = True  -> eval on shaped reward env
    """
    env = make_mario_env(train=use_shaped_env)
    n_actions = env.action_space.n

    model = ActorCriticNet(num_actions=n_actions).to(DEVICE)
    model.load_state_dict(torch.load(SAVE_PATH, map_location=DEVICE))
    model.eval()

    for ep in range(1, num_episodes + 1):
        state = env.reset()
        done = False
        ep_reward = 0.0

        while not done:
            s = preprocess_state(state).to(DEVICE)
            with torch.no_grad():
                logits, _ = model(s)
                if EVAL_USE_SAMPLING:
                    probs = F.softmax(logits, dim=-1)
                    action = torch.multinomial(probs, num_samples=1).item()
                else:
                    action = torch.argmax(logits, dim=-1).item()

            next_state, reward, done, info = env.step(action)
            ep_reward += reward
            state = next_state

            if render:
                env.render()

        print(f"Episode {ep}: reward = {ep_reward:.1f}")

    env.close()


# =========================
#  Main
# =========================

def main():
    mp.set_start_method("spawn", force=True)

    # Figure out action space
    tmp_env = make_mario_env(train=True)
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

    # Run eval after training on original env reward
    evaluate(num_episodes=35, render=False, use_shaped_env=False)


if __name__ == "__main__":
    main()
