"""
Evaluation script for trained A3C Mario agent.

Usage:
    python eval_mario.py --model-path a3c_mario.pth --num-episodes 5 --render
"""

import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from atari_wrapper import create_mario_env

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ================================================================
# Model: Must match training architecture exactly
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


def evaluate(model_path, num_episodes=5, render=False):
    """Evaluate a trained A3C model on Super Mario Bros."""
    
    # Create environment with same wrappers as training
    env = create_mario_env("SuperMarioBros-1-1-v0")
    num_actions = env.action_space.n
    
    # Load model
    model = ActorCritic(num_actions).to(DEVICE)
    
    print(f"Loading model from {model_path}...")
    try:
        state_dict = torch.load(model_path, map_location=DEVICE)
        model.load_state_dict(state_dict)
        print("Model loaded successfully!")
    except FileNotFoundError:
        print(f"ERROR: Model file '{model_path}' not found.")
        print("Train a model first using: python a3c_mario6.py")
        return
    except Exception as e:
        print(f"ERROR loading model: {e}")
        return
    
    model.eval()
    
    episode_returns = []
    episode_distances = []
    
    print(f"\nEvaluating {num_episodes} episodes...")
    print("=" * 60)
    
    for ep in range(1, num_episodes + 1):
        obs = env.reset()
        state = torch.from_numpy(obs).float().unsqueeze(0).to(DEVICE)
        
        done = False
        episode_return = 0.0
        episode_distance = 0.0
        steps = 0
        
        while not done:
            if render:
                env.render()
            
            # Greedy action selection (argmax, not sampling)
            with torch.no_grad():
                logits, _ = model(state)
                probs = F.softmax(logits, dim=-1)
                action = probs.argmax(dim=1).item()
            
            # Step environment
            next_obs, reward, done, info = env.step(action)
            episode_return += reward
            steps += 1
            
            # Track distance if available
            if info and "distance" in info:
                episode_distance = max(episode_distance, info.get("distance", 0.0))
            
            # Update state
            state = torch.from_numpy(next_obs).float().unsqueeze(0).to(DEVICE)
        
        episode_returns.append(episode_return)
        episode_distances.append(episode_distance)
        
        print(
            f"Episode {ep:2d} | "
            f"Return: {episode_return:7.1f} | "
            f"Distance: {episode_distance:6.1f} | "
            f"Steps: {steps:4d}"
        )
    
    env.close()
    
    # Summary statistics
    print("=" * 60)
    print(f"Average return:  {np.mean(episode_returns):.1f} ± {np.std(episode_returns):.1f}")
    print(f"Average distance: {np.mean(episode_distances):.1f} ± {np.std(episode_distances):.1f}")
    print(f"Best return:     {np.max(episode_returns):.1f}")
    print(f"Best distance:   {np.max(episode_distances):.1f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate trained A3C Mario agent")
    parser.add_argument(
        "--model-path",
        type=str,
        default="./a3c_mario.pth",
        help="Path to trained model file (default: ./a3c_mario.pth)",
    )
    parser.add_argument(
        "--num-episodes",
        type=int,
        default=5,
        help="Number of evaluation episodes (default: 5)",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Render the environment (watch Mario play)",
    )
    
    args = parser.parse_args()
    
    evaluate(args.model_path, args.num_episodes, args.render)

