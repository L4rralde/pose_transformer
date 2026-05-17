from typing import Dict

import torch
import torch.nn as nn


class CameraPoseHead(nn.Module):
    def __init__(self, dim: int):
        super().__init__()

        self.dim = dim
        self.backbone = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
            nn.ReLU()
        )

        self.fc_t = nn.Linear(dim, 3)
        self.fc_qvec = nn.Linear(dim, 4)
        self.fc_fov = nn.Sequential(
            nn.Linear(dim, 2),
            nn.ReLU()
        )

    def forward(self, cam_token: torch.Tensor) -> Dict[str, torch.Tensor]:
        B, S, cam_token_len, D = cam_token.shape
        assert cam_token_len == 1, "Must be an individual token"
        assert D == self.dim, "Feature dimension mismatch"

        cam_token = cam_token.view(B*S, D)

        cam_token = self.backbone(cam_token)
        t = self.fc_t(cam_token).view(B, S, 3)
        qvec = self.fc_qvec(cam_token).view(B, S, 4)
        fov = self.fc_fov(cam_token).view(B, S, 2)

        return {
            't': t,
            'qvec': qvec,
            'fov': fov
        }
