from typing import Dict

import torch
import torch.nn as nn
import numpy as np



class PatchEmbed(nn.Module):
    """
    Conv2d para convertir imagen (B, S, C, H, W) en tokens (B, S, N, D),
    con kernel y stride = patch_size.
    """

    def __init__(self,
                 img_size: tuple,
                 patch_size: int,
                 in_chans: int,
                 embed_dim: int):
        super().__init__()
        self.img_h, self.img_w = img_size
        self.patch_size = patch_size
        self.grid_w = self.img_w // patch_size
        self.grid_h = self.img_h // patch_size
        self.num_patches = self.grid_h * self.grid_w

        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, C, H, W = x.shape
        assert H == self.img_h and W == self.img_w, \
            f"Se espera imagen {self.img_size}x{self.img_size}, recibida {H}x{W}"
        x = x.view(B*S, C, H, W)
        x = self.proj(x)  # (B*S, D, Gh, Gw)
        x = x.flatten(2).transpose(1, 2)  # (B*S, N, D)
        x = x.view(B, S, self.num_patches, -1) # (B, S, N, D)
        return x


# -------------------------------------------------------------------------
# Positional embeddings 2D
# -------------------------------------------------------------------------

def _get_1d_sincos_pos_embed_from_grid(embed_dim: int,
                                       pos: np.ndarray) -> np.ndarray:
    """
    embed_dim debe ser par. pos es (M,) o (M,).
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega = 1.0 / (10000 ** (omega / (embed_dim / 2.0)))
    pos = pos.reshape(-1)
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2)
    emb_sin = np.sin(out)
    emb_cos = np.cos(out)
    return np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)


def get_2d_sincos_pos_embed(
        embed_dim: int,
        h_len: int,
        w_len: int,
    ) -> np.ndarray:
    """
    Returns 2D positional embedding of shape (h_len*w_len, embed_dim).
    """
    assert embed_dim % 2 == 0, "Embed dimension must be divisible by 2"
    
    # Generate grids
    grid_h = np.arange(h_len, dtype=np.float32)
    grid_w = np.arange(w_len, dtype=np.float32)
    
    # meshgrid order: (w, h) -> returns (X-coords, Y-coords)
    # Output shapes will be (h_len, w_len)
    grid_x, grid_y = np.meshgrid(grid_w, grid_h) 
    
    # Flatten grids to (N,)
    grid_x = grid_x.reshape(-1)
    grid_y = grid_y.reshape(-1)

    # Generate embeddings
    # We map Half the dimension to Width (X) and Half to Height (Y)
    emb_w = _get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid_x)
    emb_h = _get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid_y)
    
    # Concatenate [Height, Width] (Standard convention, though [W, H] works too)
    emb = np.concatenate([emb_h, emb_w], axis=1)
    
    return emb  # (h_len * w_len, D)


class Attention(nn.Module):
    def __init__(self, dim: int, num_heads: int) -> None:
        super().__init__()
        assert dim%num_heads==0, "dim must be divisible by number of heads"
        self.dim = dim
        self.num_heads = num_heads

        self.head_dim = dim//num_heads #Or query size
        self.qkv = nn.Linear(dim, 3*dim) #query, key, value embedding

        #dim to dim projection
        self.project = nn.Linear(dim, dim)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        #Tokens are of shape B x n_tokens x dim
        batch_size, n_tokens, dim = tokens.shape
        assert dim == self.dim, "Dimension mismatch"

        #Embed qkv tokens
        qkv = self.qkv(tokens) #B x n_tokens x (3*dim)
        qkv = qkv.view(batch_size, n_tokens, 3, self.dim) # B x n x 3 x dim
        #Now num_heads-reshaping awareness
        qkv = qkv.view(batch_size, n_tokens, 3, self.num_heads, self.head_dim) #B x n x 3 x heads x head_d
        qkv = qkv.transpose(1, 3) #B x heads x 3 x n x heads_d
        q, k, v = torch.unbind(qkv, 2) #3 tensors of shape B x heads x n x heads_d
        #softmax((Q * K)/scale) * value. Attention
        heads_tokens = nn.functional.scaled_dot_product_attention(q, k, v) #B x heads x n x heads_d

        #Now reshaping back to token dimension
        tokens = heads_tokens.transpose(1, 2) #B x n x heads x heads_dim
        tokens = tokens.view(batch_size, n_tokens, self.dim) #B x n x (heads*heads_dim) = B x n x dim

        #Final projection
        tokens = self.project(tokens)

        return tokens


class Block(nn.Module):
    def __init__(
            self,
            hidden_size: int,
            num_heads: int,
            mlp_ratio: float = 4.0,
            ):

        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads)
        self.norm2 = nn.LayerNorm(hidden_size, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim),
            nn.GELU(approximate='tanh'),
            nn.Linear(mlp_hidden_dim, hidden_size)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert len(x.shape) == 3, "(B, N, D) Tensor expected"
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class FrameBlock(Block):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, N, D = x.shape
        x = x.view(B*S, N, D)
        
        x = super().forward(x)
        x = x.view(B, S, N, D)
        return x


class GlobalBlock(Block):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, N, D = x.shape
        x = x.view(B, S*N, D)
        x = super().forward(x)
        x = x.view(B, S, N, D)
        return x


class FinalLayer(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size, eps=1e-6)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x)


class Encoder(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        depth: int = 2,
        mlp_ratio: float = 4.0,
    ):
        super().__init__()
        self.blocks = nn.ModuleList([
            FrameBlock(hidden_size, num_heads, mlp_ratio)
            for _ in range(depth)
        ])

    def forward(self, x: torch.Tensor) -> None:
        for block in self.blocks:
            x = block(x)
        return x
        

class TransformerEncoder(nn.Module):
    def __init__(
        self,
        input_size: tuple = (120, 160),
        patch_size: int = 16,
        hidden_size: int = 512,
        num_heads: int = 8,
    ):
        super().__init__()

        self.in_channels = 3
        self.out_channels = 3
        self.patch_size = patch_size
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_registers = 4


        self.x_embedder = PatchEmbed(
            img_size=input_size,
            patch_size=patch_size,
            in_chans=3,
            embed_dim=hidden_size,
        )

        num_patches = self.x_embedder.num_patches
        self.pos_embed = nn.Parameter(
            torch.zeros(1, 1, num_patches, hidden_size),
            requires_grad=False,
        )

        self.encoder = Encoder(hidden_size, num_heads, depth=2)

        alternate_blocks = []
        for _ in range(2):
            alternate_blocks.append(GlobalBlock(hidden_size, num_heads))
            alternate_blocks.append(FrameBlock(hidden_size, num_heads))
        self.alternate_blocks = nn.ModuleList(alternate_blocks)

        self.final_layer = FinalLayer(hidden_size)

        self.ref_camera_token = nn.Parameter(torch.randn(1, 1, 1, hidden_size))
        self.shared_camera_token = nn.Parameter(torch.randn(1, 1, 1, hidden_size))

        self.ref_register_token = nn.Parameter(torch.randn(1, 1, self.num_registers, hidden_size))
        self.shared_register_token = nn.Parameter(torch.randn(1, 1, self.num_registers, hidden_size))

        self._initialize()

    def _initialize(self) -> None:
        pos = get_2d_sincos_pos_embed(
            self.hidden_size,
            self.x_embedder.grid_h,
            self.x_embedder.grid_w
        )

        self.pos_embed.data.copy_(
            torch.from_numpy(pos).float()[None, None]
        )#[None, None] is double unsqueezes


    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        B, S, C, H, W = x.shape

        tokens = self.x_embedder(x) + self.pos_embed #(B, S, N, D)

        B, S, n_feat_tokens, D = tokens.shape

        ref_camera_token = self.ref_camera_token.expand(B, 1, 1, D)
        shared_camera_token = self.shared_camera_token.expand(B, S-1, 1, D)
        camera_tokens = torch.cat(
            [ref_camera_token, shared_camera_token],
            dim=1
        ) # (B, S, 1, D)

        ref_register_token = self.ref_register_token.expand(B, 1, self.num_registers, D)
        shared_register_token = self.shared_register_token.expand(B, S-1, self.num_registers, D)
        register_tokens = torch.cat(
            [ref_register_token, shared_register_token],
            dim=1
        ) # (B, S, num_registers, D)

        tokens = torch.cat(
            [camera_tokens, register_tokens, tokens],
            dim=2
        ) # (B, S, 1+num_registers+N, D)


        B, S, N, D = tokens.shape
        tokens = self.encoder(tokens) #(B, S, N, D)

        for block in self.alternate_blocks:
            tokens = block(tokens)

        tokens = self.final_layer(tokens)

        return {
            'cam': tokens[:, :, 0:1, ...], #(B, S, 1, D)
            'register': tokens[:, :, 1: 1+self.num_registers, ...], #(B, S, num_regs, D)
            'feats': tokens[:, :, 1+self.num_registers:, ...] #(B, S, n_tokens, D)
        }


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


class PoseTransformer(nn.Module):
    def __init__(
        self,
        input_size: tuple = (120, 160),
        patch_size: int = 16,
        hidden_size: int = 512,
        num_heads: int = 8,
    ):
        super().__init__()
        self.backbone = TransformerEncoder(
            input_size,
            patch_size,
            hidden_size,
            num_heads
        )
        self.cam_head = CameraPoseHead(hidden_size)

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        tokens = self.backbone(img)
        cam = self.cam_head(tokens['cam'])

        return cam
