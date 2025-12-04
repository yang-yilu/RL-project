import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import cv2
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace

MODEL_PATH = "./a3c_mario_new.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)


# -----------------------------
# Network (must match training)
# -----------------------------
class ActorCritic(nn.Module):
    def __init__(self, num_actions):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(4, 32, 8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1), nn.ReLU()
        )

        self.fc = nn.Sequential(
            nn.Linear(3136, 512),
            nn.ReLU()
        )

        self.policy = nn.Linear(512, num_actions)
        self.value = nn.Linear(512, 1)

    def forward(self, x):
        x = x / 255.0
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        logits = self.policy(x)
        value = self.value(x)
        return logits, value


# -----------------------------
# Preprocessing helpers
# -----------------------------
def preprocess(obs):
    """Turn RGB obs into 84x84 grayscale."""
    obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
    obs = cv2.resize(obs, (84, 84))
    return obs


def stack_frames(frames, new_frame):
    frames[:-1] = frames[1:]
    frames[-1] = new_frame
    return frames


# -----------------------------
# Evaluation loop
# -----------------------------
def evaluate(num_episodes=5, render=False):
    # Show the action mapping so we know what index = what
    print("Evaluating with SIMPLE_MOVEMENT:")
    for i, a in enumerate(SIMPLE_MOVEMENT):
        print(f"{i}: {a}")

    env = gym_super_mario_bros.make("SuperMarioBros-1-1-v0")
    env = JoypadSpace(env, SIMPLE_MOVEMENT)

    model = ActorCritic(len(SIMPLE_MOVEMENT)).to(DEVICE)

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model file '{MODEL_PATH}' not found. "
            "Make sure you trained and saved with a3c_mario5.py."
        )

    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.eval()

    for ep in range(1, num_episodes + 1):
        obs = env.reset()
        frame = preprocess(obs)
        frames = np.stack([frame] * 4, axis=0).astype(np.uint8)

        done = False
        env_return = 0.0
        final_x = 40
        max_x = 40

        while not done:
            if render:
                env.render()

            s = torch.tensor(frames, dtype=torch.float32, device=DEVICE).unsqueeze(0)

            with torch.no_grad():
                logits, _ = model(s)
                probs = F.softmax(logits, dim=1)
                action = probs.argmax(dim=1).item()  # greedy

            obs, r, done, info = env.step(action)
            env_return += r

            x_pos = info.get("x_pos", final_x)
            final_x = x_pos
            if x_pos > max_x:
                max_x = x_pos

            new_frame = preprocess(obs)
            frames = stack_frames(frames, new_frame)

        print(
            f"[EVAL] Episode {ep}: "
            f"env return = {env_return:.1f}, "
            f"final x_pos = {final_x}, max x_pos = {max_x}"
        )

    env.close()


if __name__ == "__main__":
    # Set render=True if you actually want to watch Mario
    evaluate(num_episodes=5, render=True)
