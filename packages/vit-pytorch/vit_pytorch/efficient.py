import torch
from einops import rearrange, repeat
from torch import nn


class ViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, transformer, channels = 3):
        super().__init__()
        assert image_size % patch_size == 0, 'image dimensions must be divisible by the patch size'
        num_patches = (image_size // patch_size) ** 2
        patch_dim = channels * patch_size ** 2

        self.patch_size = patch_size

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.patch_to_embedding = nn.Linear(patch_dim, dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.transformer = transformer

        self.to_cls_token = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, num_classes)
        )

    def forward(self, img):
        p = self.patch_size

        x = rearrange(img, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = p, p2 = p)
        x = self.patch_to_embedding(x)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.transformer(x)

        x = self.to_cls_token(x[:, 0])
        return self.mlp_head(x)


class FeatureTransform(nn.Module):
    def __init__(self, *, width, height, channels, dim, transformer):
        super().__init__()

        num_vec = channels
        vec_dim = width * height
        self.vec_dim = vec_dim
        self.num_vec = num_vec
        self.pos_embedding = nn.Parameter(torch.randn(1, num_vec, dim))
        self.patch_to_embedding = nn.Linear(vec_dim, dim)

        self.transformer = transformer

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, vec_dim)
        )

    def forward(self, img):
        b, c, h, w = img.shape
        assert w * h == self.vec_dim
        assert c == self.num_vec

        x = img.reshape((b,c,w*h))

        x = self.patch_to_embedding(x)
        x += self.pos_embedding
        x = self.transformer(x)
        x = self.mlp_head(x)
        out = x.reshape((b,c,w,h))
        return out
