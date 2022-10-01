# %ache()
import math
from typing import Tuple, Optional
import os
from dataclasses import dataclass
import torch as t
from torch._C import Block
import transformers
from einops import rearrange, repeat
from fancy_einsum import einsum
from torch import nn
from torch.nn import functional as F
from tqdm.auto import tqdm
import utils

os.environ["TOKENIZERS_PARALLELISM"] = "false"
MAIN = __name__ == "__main__"


@dataclass(frozen=True)
class GPTConfig:
    """Constants used throughout the GPT2 model."""

    num_layers: int = 12
    num_heads: int = 12
    vocab_size: int = 50257
    hidden_size: int = 768
    max_position_embeddings: int = 1024
    dropout: float = 0.1
    layer_norm_epsilon: float = 1e-05


config = GPTConfig()
pretrained_gpt = utils.load_pretrained_gpt()


class BlockCache:
    def __init__(self, device):
        self.device = device
        self.KQVs = t.Tensor([]).to(device)
        self.outputs = t.Tensor([]).to(device)

    def insert(self, newKQV, newoutput):
        self.KQVs = t.stack(self.KQVs, newKQV, dim=0)
        self.outputs = t.stack(self.outputs, newoutput)

    def to(self, device):
        self.device = device
        self.KQVS = self.KQVS.to(self.device)
        self.outputs = self.outputs.to(self.device)

    def clone(self):
        newblock = BlockCache(self.device)
        newblock.KQVs = t.clone(self.KQVs)
        newblock.outputs = t.clone(self.outputs)
        return newblock

class Cache:
    def __init__(self, hidden_size: int, vocab_size: int, device: str):

        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.device = device

        self.num_blocks = 0
        self.blocks: list[BlockCache] = []

    def append_block(self, blockcache : BlockCache):
        """Insert a new block at the end of current list of blocks"""
        self.blocks.append(blockcache)
        self.num_blocks += 1


    def clone(self):
        """returns a clone of the cache, stored in different memory. 
        This is useful because new caches are computed by in-place computations"""
        newblock = Cache(self.hidden_size, self.vocab_size, self.device)
        for block in self.blocks:
            newblock.append_block(block.clone())



# %%


class UnidirectionalAttention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int):
        super().__init__()
        self.hidden_size = hidden_size
        assert hidden_size % num_heads == 0
        self.c_attn = nn.Linear(hidden_size, hidden_size * 3)
        self.c_proj = nn.Linear(hidden_size, hidden_size)
        self.head_size = hidden_size // num_heads

    def forward(self, x: t.Tensor, blockcache: Optional[BlockCache] = None) -> Tuple[t.Tensor, BlockCache]:
        """
        x: shape (batch, seq, hidden_size)

        Return: shape (batch, seq, hidden_size)
        """

        if blockcache is None:
            new_blockcache = BlockCache(x.device)
            out = self.c_attn(x)

            new_blockcache.KQVs = out

            qout, kout, vout = t.split(out, self.hidden_size, dim=-1)
            qout = rearrange(qout, "b s (h n) -> b h s n", n=self.head_size)
            kout = rearrange(kout, "b s (h n) -> b h s n", n=self.head_size)
            vout = rearrange(vout, "b s (h n) -> b h s n", n=self.head_size)
            dotted = einsum("b h q_seq n, b h k_seq n -> b h q_seq k_seq", qout, kout)
            dotted = dotted / math.sqrt(self.head_size)
            mask = t.triu(t.ones_like(dotted), diagonal=1)
            masked = dotted * (1 - mask) - 10000 * mask
            attention_weights = t.softmax(masked, dim=-1)
            weighted_sum = einsum(
                "b h q_seq k_seq, b h k_seq head_size -> b h q_seq head_size", attention_weights, vout
            )
            rearrange_heads = rearrange(weighted_sum, "b h q_seq head_size -> b q_seq (h head_size)")
            new_blockcache.outputs = self.c_proj(rearrange_heads)

            return new_blockcache.outputs, new_blockcache

        else:

            new_blockcache = BlockCache(x.device)
            out_old = blockcache.KQVs
            out_new = self.c_attn(x[:, -1, :])  # (batch, 1, hidd)
            out_new = rearrange(out_new, "a b -> a 1 b")
            out = t.cat((out_old, out_new), dim=1)
            new_blockcache.KQVs = out

            qout, kout, vout = t.split(out, self.hidden_size, dim=-1)
            qout = rearrange(qout, "b s (h n) -> b h s n", n=self.head_size)
            kout = rearrange(kout, "b s (h n) -> b h s n", n=self.head_size)
            vout = rearrange(vout, "b s (h n) -> b h s n", n=self.head_size)

            dotted = einsum("b h n, b h s n -> b h s", qout[:, :, -1, :], kout) / math.sqrt(self.head_size)

            weights = t.softmax(dotted, dim=-1)
            weighted_vec = einsum("b h s, b h s n ->b h n", weights, vout)  # new vector for new token
            weighted_vec = rearrange(weighted_vec, "b h n -> b (h n)")
            new_out = rearrange(self.c_proj(weighted_vec), "b hid ->b 1 hid")

            new_blockcache.outputs = t.cat((blockcache.outputs, new_out), dim=1)
            return new_blockcache.outputs, new_blockcache
 

# %%


class GPT2MLP(nn.Module):
    def __init__(self, hidden_size: int, dropout: float):
        super().__init__()
        self.hidden_size = hidden_size

        self.linear1 = nn.Linear(hidden_size, hidden_size * 4)
        self.linear2 = nn.Linear(hidden_size * 4, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.linear1(x)
        x = self.linear2(F.gelu(x))
        return self.dropout(x)


class GPT2Block(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, dropout: float, layer_norm_epsilon: float):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.dropout = dropout
        self.layer_norm_epsilon = layer_norm_epsilon
        self.ln_1 = nn.LayerNorm([hidden_size], self.layer_norm_epsilon)
        self.attn = UnidirectionalAttention(self.hidden_size, self.num_heads)
        self.ln_2 = nn.LayerNorm([hidden_size], self.layer_norm_epsilon)
        self.mlp = GPT2MLP(hidden_size, dropout)

    def forward(self, x: t.Tensor, blockcache: Optional[BlockCache] = None) -> Tuple[t.Tensor, BlockCache]:
        """
        x: shape (batch, seq, hidden_size)

        Return: shape (batch, seq, hidden_size)
        """
        y = self.ln_1(x)
        y, new_blockcache = self.attn(y, blockcache)
        y += x
        return self.mlp(self.ln_2(y)) + y, new_blockcache


# %%
class GPT2(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.wte = nn.Embedding(config.vocab_size, config.hidden_size)
        self.wpe = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.drop = nn.Dropout(config.dropout)
        self.h = nn.ModuleList(
            [
                GPT2Block(config.hidden_size, config.num_heads, config.dropout, config.layer_norm_epsilon)
                for _ in range(config.num_layers)
            ]
        )
        self.ln_f = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)

    def forward(self, x: t.Tensor, cache=None) -> Tuple[t.Tensor, Cache]:
        """
        x: shape (batch, seq), dtype t.int64 - the token ids

        Return: shape (batch, seq, vocab_size), dtype t.float23 - the output logits
        """
        if cache == None:
            cache = Cache(config.hidden_size, config.vocab_size, x.device)
        positions = t.arange(x.shape[1]).to(x.device)
        y = self.wte(x) + self.wpe(positions)
        y = self.drop(y)
        has_no_blocks = cache.blocks == []
        for i, module in enumerate(self.h):  # for module, blockcache in zip(self.h, cache.blocks)
            if has_no_blocks:
                y, block = module(y)  # y = module(y, blockcache)
                cache.blocks.append(block)
            else:
                y, block = module(y, cache.blocks[i])
                cache.blocks[i] = block
        y = self.ln_f(y)

        return (einsum("vocab hidden, batch seq hidden -> batch seq vocab", self.wte.weight, y), cache)
 
# %%
def _copy_weight_bias(mine, theirs, transpose=False):
    mine.weight.copy_(theirs.weight.T if transpose else theirs.weight)
    if mine.bias is not None:
        mine.bias.copy_(theirs.bias)


def load_pretrained_weights():
    pretrained_gpt = utils.load_pretrained_gpt()
    my_gpt = GPT2(config)
    for p in my_gpt.parameters():
        p.requires_grad = False
    my_gpt.wte.weight.copy_(pretrained_gpt.transformer.wte.weight)
    my_gpt.wpe.weight.copy_(pretrained_gpt.transformer.wpe.weight)
    _copy_weight_bias(my_gpt.ln_f, pretrained_gpt.transformer.ln_f)

    from transformers.models.gpt2.modeling_gpt2 import GPT2Block as HFGPT2Block

    my_block: GPT2Block
    hf_block: HFGPT2Block
    for (my_block, hf_block) in zip(my_gpt.h, pretrained_gpt.transformer.h):
        _copy_weight_bias(my_block.ln_1, hf_block.ln_1)
        _copy_weight_bias(my_block.attn.c_attn, hf_block.attn.c_attn, transpose=True)
        _copy_weight_bias(my_block.attn.c_proj, hf_block.attn.c_proj, transpose=True)
        _copy_weight_bias(my_block.ln_2, hf_block.ln_2)
        _copy_weight_bias(my_block.mlp.linear1, hf_block.mlp.c_fc, transpose=True)
        _copy_weight_bias(my_block.mlp.linear2, hf_block.mlp.c_proj, transpose=True)
    for p in my_gpt.parameters():
        p.requires_grad_(True)
    return my_gpt


if MAIN:
    my_gpt = load_pretrained_weights()
    my_gpt.eval()
    tokenizer = transformers.AutoTokenizer.from_pretrained("gpt2")

# %%


# %%
