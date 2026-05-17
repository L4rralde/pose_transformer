import torch
import torch.nn as nn

from .aggregator import Aggregator
from .heads import CameraPoseHead as Head


class Former(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        depth: int,
    ):
        super().__init__()

        self.patch_size = backbone.patch_size
        self.embed_dim = backbone.embed_dim
        self.num_heads = backbone.num_heads

        self.backbone = backbone.eval()
        self.aggregator = Aggregator(
            self.embed_dim,
            self.embed_dim,
            depth,
            backbone.n_storage_tokens
        )
        self.head = Head(self.embed_dim)

    def train(self, mode: bool=True):
        super().train(mode)
        self.backbone.eval()

    def get_trainable_parameters(self) -> list:
        backbone_params = set(self.backbone.parameters())
        trainable_params = [
            p for p in self.parameters()
            if p not in backbone_params
        ]
        return trainable_params
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, C, H, W = x.shape
        if H % self.patch_size or W % self.patch_size:
            raise ValueError("Input image size is not multiple of patch size")
        
        x = x.view(B*S, C, H, W)
        with torch.no_grad():
            x = self.backbone.forward_features(x)
        if isinstance(x, dict):
            x = x['x_prenorm']

        BS, num_tokens, d = x.shape
        x = x.view(B, S, num_tokens, self.embed_dim)

        x = self.aggregator(x)
        x = self.head(x['cam'])

        return x

