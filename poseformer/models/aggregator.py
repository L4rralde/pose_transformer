from typing import List

import torch.nn as nn
import torch

from poseformer.eupe.layers import SelfAttentionBlock


class FrameBlock(SelfAttentionBlock):
    def forward(self, x, rope=None) -> List:
        B, S, N, D = x.shape
        x = x.view(B*S, N, D)
        #TODO. Pass rope
        x = super().forward(x)
        x = x.view(B, S, N, D)
        return x
    

class GlobalBlock(SelfAttentionBlock):
    def forward(self, x, rope=None) -> List:
        B, S, N, D = x.shape
        x = x.view(B*S, N, D)
        #TODO. Pass rope
        x = super().forward(x)
        x = x.view(B, S, N, D)
        return x


class Aggregator(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        depth: int,
        num_registers: int,
    ):
        super().__init__()

        alternate_blocks = []
        for _ in range(depth):
            alternate_blocks.append(GlobalBlock(embed_dim, num_heads))
            alternate_blocks.append(FrameBlock(embed_dim, num_heads))

        self.alternate_blocks = nn.ModuleList(alternate_blocks)
        self.norm = nn.LayerNorm(embed_dim)

        self.ref_camera_token = nn.Parameter(
            torch.zeros(1, 1, 1, embed_dim)
        )
        self.shared_camera_token = nn.Parameter(
            torch.zeros(1, 1, 1, embed_dim)
        )
        self.num_registers = num_registers
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, _, d = x.shape
        ref_camera_token = self.ref_camera_token.expand(B, 1, 1, d)
        shared_camera_token = self.shared_camera_token.expand(B, S-1, 1, d)
        camera_tokens = torch.cat(
            [ref_camera_token, shared_camera_token],
            dim=1
        ) # (B, S, 1, D)

        x[:, :, 0:, ...] = camera_tokens

        for _, blk in enumerate(self.alternate_blocks):
            x = blk(x)
        x = self.norm(x)
        return {
            'cam': x[:, :, 0:1, ...], #(B, S, 1, S)
            'registers': x[:, :, 1:1+self.num_registers, ...], #(B, S, num_regs, d)
            'patch_tokens': x[:, :, 1+self.num_registers:, ...]
        }
