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
GAMMA            = 0.99
ENTROPY_BETA     = 0.01        # less forced randomness than 0.02
VALUE_LOSS_COEF  = 0.5
LR               = 1e-4        # slightly higher LR than 5e-5
T_MAX            = 5           # shorter rollouts for easier credit assignment
NUM_WORKERS      = 4
MAX_GLOBAL_STEPS = 3_000_000   # more total experience
SAVE_PATH        = "./a3c_mario.pth"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)


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
    def __init__(self, env, hold_frames=8, jump_actions=None):
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
    R1-style reward:
      + dx_scale * max(0, Δx)        (forward progress)
      - step_cost each step          (living cost)
      - extra idle_penalty if stuck  (no progress for too long)
      - death_penalty if episode ends without flag
      + flag_bonus if flag reached
    """
    def __init__(self, env,
                 dx_scale=0.05,      # more reward for forward progress
                 step_cost=0.003,    # slightly smaller living cost
                 death_penalty=20.0,
                 flag_bonus=100.0,
                 idle_penalty=0.05,  # penalty when stuck too long
                 max_idle_steps=20   # how many steps of no progress before we penalize
                 ):
        super().__init__(env)
        self.dx_scale = dx_scale
        self.step_cost = step_cost
        self.death_penalty = death_penalty
        self.flag_bonus = flag_bonus
        self.idle_penalty = idle_penalty
        self.max_idle_steps = max_idle_steps

        self.last_x = 0
        self.idle_steps = 0

    def reset(self):
        obs = self.env.reset()
        self.last_x = 0
        self.idle_steps = 0
        return obs

    def step(self, action):
        obs, _, done, info = self.env.step(action)

        # Forward progress
        x = info.get("x_pos", 0)
        dx = max(0, x - self.last_x)
        self.last_x = x

        shaped = self.dx_scale * dx - self.step_cost

        # Track how long we've gone without forward progress
        if dx > 0:
            self.idle_steps = 0
        else:
            self.idle_steps += 1
            if self.idle_steps >= self.max_idle_steps:
                # Extra penalty for being stuck / wasting time
                shaped -= self.idle_penalty

        flag_get = info.get("flag_get", False)

        # Penalize any termination that is not success
        if done and not flag_get:
            shaped -= self.death_penalty

        # Big bonus for reaching the flag
        if flag_get:
            shaped += self.flag_bonus

        return obs, shaped, done, info


def make_mario_env():
    env = gym_super_mario_bros.make("SuperMarioBros-1-1-v0")
    env = JoypadSpace(env, SIMPLE_MOVEMENT)

    # Sticky big jump for right+jump
    env = StickyJumpEnv(env, hold_frames=8, jump_actions=JUMP_ACTIONS)

    # Stronger incentive for progress, more punishment for dying,
    # plus explicit idle penalty if Mario stalls for too long
    env = ProgressRewardWrapper(
        env,
        dx_scale=0.05,
        step_cost=0.003,
        death_penalty=20.0,
        flag_bonus=100.0,
        idle_penalty=0.05,
        max_idle_steps=20,
    )

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
    env = make_mario_env()

    # Local worker model lives on GPU/CPU (DEVICE)
    local_model = ActorCriticNet(num_actions=env.action_space.n).to(DEVICE)
    local_model.load_state_dict(global_model.state_dict())
    local_model.train()

    recent_rewards = deque(maxlen=10)

    while True:
        with global_counter.get_lock():
            if global_counter.value >= MAX_GLOBAL_STEPS:
                break

        state = env.reset()
        episode_reward = 0.0
        done = False

        while not done:
            log_probs = []
            values    = []
            rewards   = []
            entropies = []

            t = 0
            while t < T_MAX and not done:
                s = preprocess_state(state).to(DEVICE)
                logits, value = local_model(s)

                probs = F.softmax(logits, dim=-1)
                log_probs_all = F.log_softmax(logits, dim=-1)
                dist = torch.distributions.Categorical(probs)
                action = dist.sample()

                log_prob = log_probs_all[0, action]
                entropy  = -(probs * log_probs_all).sum()

                next_state, reward, done, info = env.step(action.item())
                episode_reward += reward

                log_probs.append(log_prob)
                values.append(value.squeeze(0))          # tensor on DEVICE
                rewards.append(torch.tensor(reward, dtype=torch.float32, device=DEVICE))
                entropies.append(entropy)

                state = next_state
                t += 1

                with global_counter.get_lock():
                    global_counter.value += 1
                    if global_counter.value >= MAX_GLOBAL_STEPS:
                        done = True
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

            for i in reversed(range(len(rewards))):
                R = rewards[i] + GAMMA * R
                advantage = R - values[i]

                value_loss = value_loss + advantage.pow(2)
                policy_loss = policy_loss - log_probs[i] * advantage.detach() - ENTROPY_BETA * entropies[i]

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

            if done:
                recent_rewards.append(episode_reward)
                with global_episode.get_lock():
                    global_episode.value += 1
                    ep = global_episode.value
                avg10 = np.mean(recent_rewards)
                with print_lock:
                    print(f"[Worker {rank}] Ep {ep} | "
                          f"Global steps {global_counter.value} | "
                          f"Reward {episode_reward:.1f} | "
                          f"Avg10 {avg10:.1f}")
                break

    env.close()
    print(f"Worker {rank} finished.")


# =========================
#  Evaluation (sampling vs greedy)
# =========================

EVAL_USE_SAMPLING = True  # True = sample from policy; False = greedy argmax


def evaluate(num_episodes=35, render=False):
    """
    After training, load the saved model and run episodes with
    either sampling or greedy action selection.
    """
    env = make_mario_env()
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
    tmp_env = make_mario_env()
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

    # Run eval after training (you can set render=True if you want visuals)
    evaluate(num_episodes=35, render=False)


if __name__ == "__main__":
    main()
