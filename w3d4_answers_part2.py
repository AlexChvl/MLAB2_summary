#%%

import matplotlib.pyplot as plt
import torch as t
from torch import functional as F
from einops import rearrange, repeat
from fancy_einsum import einsum
from torch import nn

# from torch.nn.modules.conv import Conv2d
 
from w3d4_answers import DiffusionModel

MAIN = __name__ == "__main__"
device = "cuda" if t.cuda.is_available() else "cpu"

# %%


class GroupNorm(nn.Module):
    def __init__(self, num_groups: int, num_channels: int, eps=1e-05, affine=True, device=None, dtype=None):
        super().__init__()
        assert num_channels % num_groups == 0

        self.num_groups = num_groups
        self.num_channels = num_channels
        self.eps = eps
        self.affine = affine
        self.device = device
        self.dtype = dtype
        if self.affine:
            self.weight = nn.Parameter(t.ones((self.num_channels,), device=self.device, dtype=self.dtype))
            self.bias = nn.Parameter(t.zeros((self.num_channels,), device=self.device, dtype=self.dtype))
        else:
            self.weight = None
            self.bias = None

    def reset_parameters(self) -> None:
        """Initialize the weight and bias, if applicable."""
        if self.affine:
            nn.init.ones_(self.weight)
            nn.init.zeros_(self.bias)

    def forward(self, x: t.Tensor):
        """Apply normalization to each group of channels.

        x: shape (batch, channels, height, width)
        out: shape (batch, channels, height, width)
        """

        groups = t.split(x, self.num_channels // self.num_groups, dim=1)
        tensor_groups = t.stack(groups, dim=0)
        # (groups, batch, smaller channels, height, widgth)
        group_means = t.mean(tensor_groups, (-3, -2, -1), keepdim=True)
        group_vars = t.var(tensor_groups, (-3, -2, -1), keepdim=True, unbiased=False)
        normalised_groups = (tensor_groups - group_means) / ((group_vars + self.eps) ** 0.5)
        reshaped_groups = rearrange(
            normalised_groups, "groups batch chan height width -> batch (groups chan) height width"
        )

        if self.affine:
            weight = rearrange(self.weight, "chan ->1 chan 1 1")
            bias = rearrange(self.bias, "chan ->1 chan 1 1")
            return reshaped_groups * weight + bias
        else:
            return reshaped_groups

 

# %%
class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, embedding_size: int):
        super().__init__()
        self.embedding_size = embedding_size
        assert self.embedding_size % 2 == 0

        k = t.arange(0, self.embedding_size // 2) + 1
        self.omega_vec = 1 / (10000 ** (2 * k / self.embedding_size))

    def forward(self, x: t.Tensor) -> t.Tensor:
        """
        x: shape (batch, ) - for each batch element, the number of noise steps
        Out: shape (batch, embedding_size)
        """

        rep_omega = repeat(self.omega_vec, "d -> b d", b=x.shape[0]).to(x.device)
        omega_t = t.einsum("b ..., b d-> b d ...", x, rep_omega)

        sin_x = t.sin(omega_t)
        cos_x = t.cos(omega_t)

        combined = t.stack((sin_x, cos_x), dim=0)
        # sin/cos, batch, embedding size
        threaded = rearrange(combined, "f b d ... -> b (f d) ...")
        assert threaded.device == x.device
        return threaded


if MAIN:
    pass
    # emb = SinusoidalPositionEmbeddings(128)
    # out = emb(t.arange(50))

    # fig, ax = plt.subplots(figsize=(15, 5))
    # ax.set(xlabel="Embedding Dimension", ylabel="Num Steps", title="Position Embeddings")
    # im = ax.imshow(out, vmin=-1, vmax=1)
    # fig.colorbar(im)

    # fig, ax = plt.subplots(figsize=(9, 9))
    # im = ax.imshow(out @ out.T)
    # fig.colorbar(im)
    # ax.set(xlabel="Num Steps", ylabel="Num Steps", title="Dot product of position embeddings")

# %%


def swish(x: t.Tensor) -> t.Tensor:
    return x * nn.Sigmoid()(x)


class SiLU(nn.Module):
    def forward(self, x: t.Tensor) -> t.Tensor:
        return swish(x)


if MAIN:
    fig, ax = plt.subplots()
    x = t.linspace(-5, 5, 100)
    ax.plot(x, swish(x))
    ax.set(xlabel="x", ylabel="swish(x)")


# %%


class SelfAttention(nn.Module):
    def __init__(self, channels: int, num_heads=4):
        """Self-Attention with two spatial dimensions.

        channels: the number of channels. Should be divisible by the number of heads.
        """
        super().__init__()
        self.num_heads = num_heads
        self.channels = channels
        assert self.channels % self.num_heads == 0

        self.head_size = self.channels // num_heads
        self.project_query = nn.Linear(self.channels, num_heads * self.head_size)
        self.project_key = nn.Linear(self.channels, num_heads * self.head_size)
        self.project_value = nn.Linear(self.channels, num_heads * self.head_size)
        self.project_output = nn.Linear(self.num_heads * self.head_size, self.channels)

    def attention_pattern_pre_softmax(self, x: t.Tensor) -> t.Tensor:
        """x: shape (batch, channels, height, width)"""

        # B, C, H, W = x.shape
        # x = rearrange(x, "b c h w -> b (h w) c")

        B, C, D = x.shape

        Q = self.project_query(x)
        Q = rearrange(Q, "b seq (head head_size) -> b head seq head_size", head=self.num_heads)
        K = self.project_key(x)
        K = rearrange(K, "b seq (head head_size) -> b head seq head_size", head=self.num_heads)
        out = einsum("b head seq_q head_size, b head seq_k head_size -> b head seq_q seq_k", Q, K)
        out = out / (self.head_size**0.5)
        assert out.shape == (B, self.num_heads, C, C)
        return out

    def forward(self, x: t.Tensor) -> t.Tensor:
        """
        x: shape(batch, channels, height, width)

        Return: (batch, channels, height, width)
        """
        B, C, H, W = x.shape
        x = rearrange(x, "b c h w -> b (h w) c")

        attention_pattern = self.attention_pattern_pre_softmax(x)
        softmaxed_attention = attention_pattern.softmax(dim=-1)
        V = self.project_value(x)
        V = rearrange(V, "b seq (head head_size) -> b head seq head_size", head=self.num_heads)
        combined_values = einsum(
            "b head seq_k head_size, b head seq_q seq_k -> b head seq_q head_size",
            V,
            softmaxed_attention,
        )
        out = self.project_output(rearrange(combined_values, "b head seq head_size -> b seq (head head_size)"))
        assert out.shape == (B, H * W, C)

        return rearrange(out, "b (h w) c -> b c h w", h=H, w=W)

 

#%%
ConvTranspose2d = nn.ConvTranspose2d

# %%
class AttentionBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.channels = channels
        self.attention_path = nn.Sequential(GroupNorm(1, self.channels), SelfAttention(self.channels, 4))

    def forward(self, x):
        return self.attention_path(x) + x
 

# %%


class ResidualBlock(nn.Module):
    def __init__(self, input_channels: int, output_channels: int, step_dim: int, groups: int):
        """
        input_channels: number of channels in the input to foward
        output_channels: number of channels in the returned output
        step_dim: embedding dimension size for the number of steps
        groups: number of groups in the GroupNorms

        Note that the conv in the left branch is needed if c_in != c_out.
        """
        super().__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.step_dim = step_dim
        self.groups = groups

        self.emb_seq = nn.Sequential(SiLU(), nn.Linear(self.step_dim, self.output_channels))
        self.conv_group_1 = nn.Sequential(
            nn.Conv2d(self.input_channels, self.output_channels, (3, 3), padding=1),
            GroupNorm(self.groups, self.output_channels),
            SiLU(),
        )
        self.conv_group_2 = nn.Sequential(
            nn.Conv2d(self.output_channels, self.output_channels, (3, 3), padding=1),
            GroupNorm(self.groups, self.output_channels),
            SiLU(),
        )
        self.optional_conv = nn.Conv2d(self.input_channels, self.output_channels, (1, 1))

    def forward(self, x, time_emb):
        """
        Note that the output of the (silu, linear) block should be of shape (batch, c_out). Since we would like to add this to the output of the first (conv, norm, silu) block, which will have a different shape, we need to first add extra dimensions to the output of the (silu, linear) block.
        """
        c = self.conv_group_2(self.conv_group_1(x) + rearrange(self.emb_seq(time_emb), "b c -> b c 1 1"))
        return self.optional_conv(x) + c

 

# %%


class DownBlock(nn.Module):
    def __init__(self, channels_in: int, channels_out: int, time_emb_dim: int, groups: int, downsample: bool):
        super().__init__()
        self.channels_in = channels_in
        self.channels_out = channels_out
        self.time_emb_dim = time_emb_dim
        self.groups = groups
        self.downsample = downsample
        self.resblock1 = ResidualBlock(self.channels_in, self.channels_out, self.time_emb_dim, self.groups)
        self.resblock2 = ResidualBlock(self.channels_out, self.channels_out, self.time_emb_dim, self.groups)
        self.attn = SelfAttention(self.channels_out)
        if self.downsample:
            self.conv2d = nn.Conv2d(self.channels_out, self.channels_out, (4, 4), stride=2, padding=1)
        else:
            self.conv2d = nn.Identity()

    def forward(self, x: t.Tensor, step_emb: t.Tensor) -> tuple[t.Tensor, t.Tensor]:
        """
        x: shape (batch, channels, height, width)
        step_emb: shape (batch, emb)
        Return: (downsampled output, full size output to skip to matching UpBlock)
        """
        r1 = self.resblock1(x, step_emb)
        r2 = self.resblock2(r1, step_emb)
        attention = self.attn(r2)
        return (self.conv2d(attention), attention) 
#%%
class UpBlock(nn.Module):
    def __init__(self, dim_in: int, dim_out: int, time_emb_dim: int, groups: int, upsample: bool):
        """
        IMPORTANT: arguments are with respect to the matching DownBlock.

        """
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.time_emb_dim = time_emb_dim
        self.groups = groups
        self.upsample = upsample

        self.residual1 = ResidualBlock(2 * self.dim_out, self.dim_in, self.time_emb_dim, self.groups)
        self.residual2 = ResidualBlock(self.dim_in, self.dim_in, self.time_emb_dim, self.groups)
        self.attn = SelfAttention(self.dim_in)
        if upsample:
            self.conv2d = ConvTranspose2d(self.dim_in, self.dim_in, (4, 4), stride=2, padding=1)
        else:
            self.conv2d = nn.Identity()

    def forward(self, x: t.Tensor, step_emb: t.Tensor, skip: t.Tensor) -> t.Tensor:
        """
        x = (batch, channels, im_shape)
        skip = (batch, channels, im_shape)"""
        concat = t.cat((x, skip), dim=1)
        r1 = self.residual1(concat, step_emb)
        r2 = self.residual2(r1, step_emb)

        return self.conv2d(self.attn(r2))

 

#%%
class MidBlock(nn.Module):
    def __init__(self, mid_dim: int, time_emb_dim: int, groups: int):
        super().__init__()
        self.mid_dim = mid_dim
        self.time_emb_dim = time_emb_dim
        self.groups = groups
        self.residual1 = ResidualBlock(self.mid_dim, self.mid_dim, self.time_emb_dim, self.groups)
        self.attn = SelfAttention(self.mid_dim)
        self.residual2 = ResidualBlock(self.mid_dim, self.mid_dim, self.time_emb_dim, self.groups)

    def forward(self, x: t.Tensor, step_emb: t.Tensor):
        return self.residual2(self.attn(self.residual1(x, step_emb)), step_emb)
 

#%%
class Unet(DiffusionModel):
    def __init__(
        self,
        image_shape: tuple[int, int, int],
        channels: int = 128,
        dim_mults=(1, 2, 4, 8),
        groups: int = 4,
        max_steps: int = 1000,
    ):
        """
        image_shape: the input and output image shape, a tuple of (C, H, W)
        channels: the number of channels after the first convolution.
        dim_mults: the number of output channels for downblock i is dim_mults[i] * channels. Note that the default arg of (1, 2, 4, 8) will contain one more DownBlock and UpBlock than the DDPM image above.
        groups: number of groups in the group normalization of each ResnetBlock (doesn't apply to attention block)
        """
        super().__init__()

        self.im_shape = image_shape
        self.noise_schedule = None
        self.channels = channels
        self.dim_mults = dim_mults
        self.groups = groups
        self.max_steps = max_steps
        self.device = device

        self.embedding = nn.Sequential(
            SinusoidalPositionEmbeddings(self.max_steps),
            nn.Linear(self.max_steps, 4 * self.channels),
            nn.GELU(),
            nn.Linear(4 * self.channels, 4 * self.channels),
        )

        self.conv1 = nn.Conv2d(self.im_shape[0], self.channels, (7, 7), padding=3)
        self.down0 = DownBlock(self.channels, self.dim_mults[0] * self.channels, 4 * self.channels, self.groups, True)
        self.down1 = DownBlock(self.channels, self.dim_mults[1] * self.channels, 4 * self.channels, self.groups, True)
        self.down2 = DownBlock(
            self.dim_mults[1] * self.channels, self.dim_mults[2] * self.channels, 4 * self.channels, self.groups, False
        )

        self.mid = MidBlock(self.dim_mults[2] * self.channels, 4 * self.channels, self.groups)

        self.up0 = UpBlock(
            self.dim_mults[1] * self.channels,
            self.dim_mults[2] * self.channels,
            4 * self.channels,
            self.groups,
            upsample=True,
        )
        self.up1 = UpBlock(
            self.dim_mults[0] * self.channels,
            self.dim_mults[1] * self.channels,
            4 * self.channels,
            self.groups,
            upsample=True,
        )

        self.residual = ResidualBlock(self.channels, self.channels, 4 * self.channels, self.groups)
        self.conv2 = nn.Conv2d(self.channels, self.im_shape[0], (1, 1))

    def forward(self, x, num_steps):
        """
        x: shape (batch, channels, height, width)
        num_steps: shape (batch, )

        out: shape (batch, channels, height, width)
        """
        assert x.device == num_steps.device


        emb = self.embedding(num_steps)
        a = self.conv1(x)
        b, _ = self.down0(a, emb)
        c, skip1 = self.down1(b, emb)
        d, skip2 = self.down2(c, emb)
        e = self.mid(d, emb)
        f = self.up0(e, emb, skip2)
        g = self.up1(f, emb, skip1)
        h = self.residual(g, emb)
        return self.conv2(h)

 
# %%
