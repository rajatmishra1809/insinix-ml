
import torch

import torch.nn as nn


class RoverPolicy(nn.Module):
    """
    INSINIX ML Steering & Obstacle Avoidance Neural Network
    State vector (22 features):
      - 16 LiDAR ray distances (normalized 0-1)
      - goal_angle (normalized -1 to 1)
      - goal_distance (normalized 0-1)
      - speed (normalized 0-1)
      - heading (normalized -1 to 1)
      - payload_weight (normalized 0-1)
      - friction (normalized 0-1)
    Outputs:
      - steering: [-1, 1] (Tanh)
      - target_speed: [0, 1] (Sigmoid)
    """

    def __init__(self, input_size=22):
        super().__init__()

        self.backbone = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
        )

        self.steering_head = nn.Sequential(
            nn.Linear(32, 1),
            nn.Tanh()
        )

        self.speed_head = nn.Sequential(
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        h = self.backbone(x)
        steering = self.steering_head(h)
        target_speed = self.speed_head(h)
        return torch.cat([steering, target_speed], dim=1)
