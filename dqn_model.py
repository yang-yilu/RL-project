# dqn_model.py
import torch
import torch.nn as nn
import numpy as np

class DQN(nn.Module):
    def __init__(self, n_actions):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ReLU(),
            nn.Flatten(),
        )
        # compute conv output size
        with torch.no_grad():
            dummy = torch.zeros(1, 4, 84, 84)
            conv_out = self.net(dummy).shape[1]
        self.head = nn.Sequential(
            nn.Linear(conv_out, 512), nn.ReLU(),
            nn.Linear(512, n_actions)
        )

    def forward(self, x):
        # x: (B,84,84,4) from gym -> convert to (B,4,84,84)
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float32)
        x = x.permute(0, 3, 1, 2) / 255.0
        return self.head(self.net(x))
