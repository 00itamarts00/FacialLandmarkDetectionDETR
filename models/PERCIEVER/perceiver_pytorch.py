from functools import wraps
from math import pi, log

import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import nn, einsum


# helpers
from main.detr.models.misc_nets import MLP


def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


def cache_fn(f):
    cache = None

    @wraps(f)
    def cached_fn(*args, _cache=True, **kwargs):
        if not _cache:
            return f(*args, **kwargs)
        nonlocal cache
        if cache is not None:
            return cache
        cache = f(*args, **kwargs)
        return cache

    return cached_fn


def fourier_encode(x, max_freq, num_bands=4, base=2):
    x = x.unsqueeze(-1)
    device, dtype, orig_x = x.device, x.dtype, x

    scales = torch.logspace(0., log(max_freq / 2) / log(base), num_bands, base=base, device=device, dtype=dtype)
    scales = scales[(*((None,) * (len(x.shape) - 1)), Ellipsis)]

    x = x * scales * pi
    x = torch.cat([x.sin(), x.cos()], dim=-1)
    x = torch.cat((x, orig_x), dim=-1)
    return x


# helper classes

class PreNorm(nn.Module):
    def __init__(self, dim, fn, context_dim=None):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)
        self.norm_context = nn.LayerNorm(context_dim) if exists(context_dim) else None

    def forward(self, x, **kwargs):
        x = self.norm(x)

        if exists(self.norm_context):
            context = kwargs['context']
            normed_context = self.norm_context(context)
            kwargs.update(context=normed_context)

        return self.fn(x, **kwargs)


class GEGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim=-1)
        return x * F.gelu(gates)


class FeedForward(nn.Module):
    def __init__(self, dim, mult=4, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult * 2),
            GEGLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mult, dim)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(context_dim, inner_dim * 2, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, context=None, mask=None):
        h = self.heads

        q = self.to_q(x)
        context = default(context, x)
        k, v = self.to_kv(context).chunk(2, dim=-1)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

        if exists(mask):
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h=h)
            sim.masked_fill_(~mask, max_neg_value)

        # attention, what we cannot get enough of
        attn = sim.softmax(dim=-1)

        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
        return self.to_out(out)


# main class

class Perceiver(nn.Module):
    def __init__(self, args):
        """The shape of the final attention mechanism will be:
        depth * (cross attention -> self_per_cross_attn * self attention)

        Args:
             num_freq_bands: Number of freq bands, with original value (2 * K + 1)
             depth: Depth of net.
             max_freq: Maximum frequency, hyperparameter depending on how fine the data is.
             freq_base: Base for the frequency
             input_channels: Number of channels for each token of the input.
             input_axis: Number of axes for input data (2 for images, 3 for video)
             num_latents: Number of latents, or induced set points, or centroids. Different papers giving it different names.
             latent_dim: Latent dimension.
             cross_heads: Number of heads for cross attention. Paper said 1.
             latent_heads: Number of heads for latent self attention, 8.
             cross_dim_head: Number of dimensions per cross attention head.
             latent_dim_head: Number of dimensions per latent self attention head.
             num_classes: Output number of classes.
             attn_dropout: Attention dropout
             ff_dropout: Feedforward dropout
             weight_tie_layers: Whether to weight tie layers (optional).
             fourier_encode_data: Whether to auto-fourier encode the data, using the input_axis given. defaults to True, but can be turned off
                                  if you are fourier encoding the data yourself.
             self_per_cross_attn: Number of self attention blocks per cross attn.
        """
        super().__init__()
        self.input_axis = args.input_axis
        self.max_freq = args.max_freq
        self.num_freq_bands = args.num_freq_bands
        self.freq_base = args.freq_base

        self.fourier_encode_data = args.fourier_encode_data
        fourier_channels = (args.input_axis * ((args.num_freq_bands * 2) + 1)) if args.fourier_encode_data else 0
        input_dim = fourier_channels + args.input_channels

        self.latents = nn.Parameter(torch.randn(args.num_latents, args.latent_dim))

        get_cross_attn = lambda: PreNorm(args.latent_dim,
                                         Attention(args.latent_dim, input_dim, heads=args.cross_heads,
                                                   dim_head=args.cross_dim_head, dropout=args.attn_dropout),
                                         context_dim=input_dim)
        get_cross_ff = lambda: PreNorm(args.latent_dim, FeedForward(args.latent_dim, dropout=args.ff_dropout))
        get_latent_attn = lambda: PreNorm(args.latent_dim,
                                          Attention(args.latent_dim, heads=args.latent_heads, dim_head=args.latent_dim_head,
                                                    dropout=args.attn_dropout))
        get_latent_ff = lambda: PreNorm(args.latent_dim, FeedForward(args.latent_dim, dropout=args.ff_dropout))

        get_cross_attn, get_cross_ff, get_latent_attn, get_latent_ff = map(cache_fn, (
            get_cross_attn, get_cross_ff, get_latent_attn, get_latent_ff))

        self.layers = nn.ModuleList([])
        for i in range(args.depth):
            should_cache = i > 0 and args.weight_tie_layers
            cache_args = {'_cache': should_cache}

            self_attns = nn.ModuleList([])

            for _ in range(args.self_per_cross_attn):
                self_attns.append(nn.ModuleList([
                    get_latent_attn(**cache_args),
                    get_latent_ff(**cache_args)
                ]))

            self.layers.append(nn.ModuleList([
                get_cross_attn(**cache_args),
                get_cross_ff(**cache_args),
                self_attns
            ]))

        self.to_logits = nn.Sequential(
            nn.LayerNorm(args.latent_dim),
            nn.Linear(args.latent_dim, 2)
        )

    def forward(self, data, mask=None):
        # data = data.flatten(2)
        data = data.permute(0, 2, 3, 1)
        b, *axis, _, device = *data.shape, data.device
        assert len(axis) == self.input_axis, 'input data must have the right number of axis'

        if self.fourier_encode_data:
            # calculate fourier encoded positions in the range of [-1, 1], for all axis
            axis_pos = list(map(lambda size: torch.linspace(-1., 1., steps=size, device=device), axis))
            pos = torch.stack(torch.meshgrid(*axis_pos), dim=-1)
            enc_pos = fourier_encode(pos, self.max_freq, self.num_freq_bands, base=self.freq_base)
            enc_pos = rearrange(enc_pos, '... n d -> ... (n d)')
            enc_pos = repeat(enc_pos, '... -> b ...', b=b)

            context = torch.cat((data, enc_pos), dim=-1)

        # concat to channels of data and flatten axis
        context = rearrange(context, 'b ... d -> b (...) d')

        x = repeat(self.latents, 'n d -> b n d', b=b)

        # layers
        for cross_attn, cross_ff, self_attns in self.layers:
            x = cross_attn(x, context=context, mask=mask) + x
            x = cross_ff(x) + x

            for self_attn, self_ff in self_attns:
                x = self_attn(x) + x
                x = self_ff(x) + x

        # x = x.mean(dim=-2)
        return self.to_logits(x).sigmoid() * 256 + 0.5
