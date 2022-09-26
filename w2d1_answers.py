#%%

from dataclasses import dataclass
from lib2to3.pgen2 import token
from ssl import OPENSSL_VERSION_NUMBER
from typing import KeysView, List, Optional, Union
import torch as t
from torch.nn.modules import dropout
import transformers
from einops import rearrange, repeat
from fancy_einsum import einsum
from torch import add, nn
from torch.nn import functional as F
import utils
import w2d1_test_selfmade

import math

MAIN = __name__ == "__main__"


@dataclass(frozen=True)
class BertConfig:
    """Constants used throughout the Bert model. Most are self-explanatory.

    intermediate_size is the number of hidden neurons in the MLP (see schematic)
    type_vocab_size is only used for pretraining on "next sentence prediction", which we aren't doing.
    """

    vocab_size: int = 28996
    intermediate_size: int = 3072
    hidden_size: int = 768
    num_layers: int = 12
    num_heads: int = 12
    max_position_embeddings: int = 512
    dropout: float = 0.1
    type_vocab_size: int = 2
    layer_norm_epsilon: float = 1e-12


if MAIN:
    config = BertConfig()
# %%
class BertSelfAttention(nn.Module):
    def __init__(self, config: BertConfig):
        super().__init__()
        self.config = config
        self.head_size = round(self.config.hidden_size / self.config.num_heads)
        self.Q_maps = nn.Linear(
            self.config.hidden_size, self.config.hidden_size
        )  # will rearrange to get individual attention heads
        self.K_maps = nn.Linear(self.config.hidden_size, self.config.hidden_size)
        self.V_maps = nn.Linear(self.config.hidden_size, self.config.hidden_size)
        self.O_maps = nn.Linear(self.config.hidden_size, self.config.hidden_size)

    def attention_pattern_pre_softmax(self, x: t.Tensor) -> t.Tensor:
        """Return the attention pattern after scaling but before softmax.

        pattern[batch, head, q, k] should be the match between a query at sequence position q and a key at sequence position k.
        """

        Queries = rearrange(self.Q_maps(x), "b  s (numh headsz)-> b numh s headsz", numh=self.config.num_heads)
        Keys = rearrange(self.K_maps(x), "b  s (numh headsz)-> b numh s headsz", numh=self.config.num_heads)
        assert Queries.shape == Keys.shape
        return t.einsum("bnqi, bnki -> bnqk", Queries, Keys) / math.sqrt(self.head_size)

    def forward(self, x: t.Tensor, additive_attention_mask: Optional[t.Tensor] = None) -> t.Tensor:
        """
        x: shape (batch, seq, hidden_size)
        additive_attention_mask: shape (batch, head=1, seq_q=1, seq_k) — used in training to prevent copying data from padding tokens. Contains 0 for a real input token and a large negative number for a padding token. If provided, add this to the attention pattern (pre softmax).

        Return: (batch, seq, hidden_size)
        """
        pre_soft_pattern = self.attention_pattern_pre_softmax(x)
        if additive_attention_mask is not None:
            pre_soft_pattern = pre_soft_pattern + additive_attention_mask
        soft_pattern = F.softmax(pre_soft_pattern, dim=3)
        Vals = rearrange(self.V_maps(x), "b  s (numh headsz)-> b numh s headsz", numh=self.config.num_heads)

        weighted = rearrange(soft_pattern @ Vals, "b numh s headsz -> b s (numh headsz)")
        return self.O_maps(weighted)



class LayerNorm(nn.Module):
    def __init__(
        self, normalized_shape: Union[int, tuple, t.Size], eps=1e-05, elementwise_affine=True, device=None, dtype=None
    ):
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        self.device = device
        self.dtype = dtype
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize the weight and bias, if applicable."""
        if self.elementwise_affine:

            self.weights = nn.Parameter(t.ones(self.normalized_shape)).to(self.device)
            self.biases = nn.Parameter(t.zeros(self.normalized_shape)).to(self.device)

    def forward(self, x: t.Tensor):
        """x and the output should both have shape (N, *)."""
        dims = tuple([len(x.size()) - 1 - i for i in range(len(self.normalized_shape))])
        ret = (x - t.mean(x, dims, keepdim=True)) / t.sqrt(t.var(x, dims, unbiased=False, keepdim=True) + self.eps)
        if self.elementwise_affine:
            ret = ret * self.weights + self.biases
        return ret



if MAIN:
    w2d1_test_selfmade.test_layer_norm(LayerNorm)

# %%
class Embedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__()
        self.weight = nn.Parameter(t.randn((num_embeddings, embedding_dim)) * 0.02)

    def forward(self, x: t.LongTensor) -> t.Tensor:
        """For each integer in the input, return that row of the embedding.

        Don't convert x to one-hot vectors — this works but is too slow.
        """
        return self.weight[x]
 

# %%


class BertMLP(nn.Module):
    def __init__(self, config: BertConfig):
        super().__init__()
        self.linear1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.linear2 = nn.Linear(config.intermediate_size, config.hidden_size)
        self.layer_norm = LayerNorm(config.hidden_size, config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: t.Tensor) -> t.Tensor:
        y = self.dropout((self.linear2(F.gelu(self.linear1(x)))))
        return self.layer_norm(x + y)

 
 

class BertAttention(nn.Module):
    def __init__(self, config: BertConfig):
        super().__init__()
        self.self_attention = BertSelfAttention(config)
        self.dropout = nn.Dropout(config.dropout)
        self.layernorm = LayerNorm(config.hidden_size, config.layer_norm_epsilon)

    def forward(self, x: t.Tensor, additive_attention_mask: Optional[t.Tensor] = None) -> t.Tensor:

        return self.layernorm(x + self.dropout((self.self_attention(x, additive_attention_mask))))


class BertBlock(nn.Module):
    def __init__(self, config: BertConfig):
        super().__init__()
        self.bertattention = BertAttention(config)
        self.MLP = BertMLP(config)

    def forward(self, x: t.Tensor, additive_attention_mask: Optional[t.Tensor] = None) -> t.Tensor:

        return self.MLP(self.bertattention(x, additive_attention_mask))
 

# %%
def make_additive_attention_mask(boolean_attention_mask: t.Tensor, big_negative_number=-10000) -> t.Tensor:
    """
    boolean_attention_mask: shape (batch, seq). Contains 1 if this is a valid token and 0 if it is a padding token.
    big_negative_number: Any number large enough that in exp(big_negative_number) is 0.0 for the floating point precision used.

    Out: shape (batch, heads, seq, seq). Contains 0 if attention is allowed, and big_negative_number if it is not allowed.
    """

    return t.where(boolean_attention_mask, t.Tensor([0.0]), t.Tensor([big_negative_number]))


class BertCommon(nn.Module):
    def __init__(self, config: BertConfig):
        super().__init__()
        self.config = config
        self.token_embedding = Embedding(config.vocab_size, config.hidden_size)
        self.positional_embedding = Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embedding = Embedding(config.type_vocab_size, config.hidden_size)
        self.layer_norm = LayerNorm(config.hidden_size, config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout)
        self.bert_blocks = nn.ModuleList([BertBlock(config) for i in range(config.num_layers)])

    def forward(
        self,
        input_ids: t.Tensor,
        token_type_ids: Optional[t.Tensor] = None,
        boolean_attention_mask: Optional[t.Tensor] = None,
    ) -> t.Tensor:
        """
        input_ids: (batch, seq) - the token ids
        token_type_ids: (batch, seq) - only used for next sentence prediction.
        boolean_attention_mask: (batch, seq) - only used in training. See make_additive_attention_mask.
        """
        if boolean_attention_mask is not None:
            additive_mask = make_additive_attention_mask(boolean_attention_mask)
        else:
            additive_mask = None
        if token_type_ids is None:
            token_type_ids = t.zeros_like(input_ids)
        embedded = (
            self.token_embedding(input_ids)
            + self.positional_embedding(t.arange(input_ids.size(-1)))
            + self.token_type_embedding(token_type_ids)
        )
        normalized = self.layer_norm(embedded)
        dropped = self.dropout(normalized)
        for module in self.bert_blocks:
            dropped = module(dropped, additive_mask)
        return dropped


class BertLanguageModel(nn.Module):
    def __init__(self, config: BertConfig):
        super().__init__()
        self.bertcommon = BertCommon(config)
        self.linear = nn.Linear(config.hidden_size, config.hidden_size)
        self.layernorm = LayerNorm(config.hidden_size)
        self.unembed = nn.Linear(config.hidden_size, config.vocab_size)
        self.unembed.weight = self.bertcommon.token_embedding.weight
        self.unembed.bias = nn.Parameter(t.zeros(config.vocab_size))

    def forward(
        self,
        input_ids: t.Tensor,
        token_type_ids: Optional[t.Tensor] = None,
        boolean_attention_mask: Optional[t.Tensor] = None,
    ) -> t.Tensor:
        """Compute logits for each token in the vocabulary.

        Return: shape (batch, seq, vocab_size)
        """
        x = self.bertcommon(input_ids, token_type_ids, boolean_attention_mask)
        x = F.gelu(self.linear(x))
        x = self.layernorm(x)
        return self.unembed(x)

 

# %%


def load_pretrained_weights(config: BertConfig) -> BertLanguageModel:
    """Load a pretrained Bert using the load_pretrained_bert() function from utils.py and assign its layers to the layers of a BertLanguageModel as constructed here. Return the new model."""
    with t.inference_mode():
        hf_bert = utils.load_pretrained_bert()
        ourbert = BertLanguageModel(config)
        ourbert.bertcommon.token_embedding.weight[:] = hf_bert.bert.embeddings.word_embeddings.weight
        ourbert.bertcommon.positional_embedding.weight[:] = hf_bert.bert.embeddings.position_embeddings.weight
        ourbert.bertcommon.token_type_embedding.weight[:] = hf_bert.bert.embeddings.token_type_embeddings.weight
        ourbert.bertcommon.layer_norm.weights = hf_bert.bert.embeddings.LayerNorm.weight
        ourbert.bertcommon.layer_norm.biases = hf_bert.bert.embeddings.LayerNorm.bias
        for ourblock, theirblock in zip(ourbert.bertcommon.bert_blocks, hf_bert.bert.encoder.layer):
            ourblock.bertattention.self_attention.Q_maps.load_state_dict(theirblock.attention.self.query.state_dict())
            ourblock.bertattention.self_attention.K_maps.load_state_dict(theirblock.attention.self.key.state_dict())
            ourblock.bertattention.self_attention.V_maps.load_state_dict(theirblock.attention.self.value.state_dict())
            print(
                ourblock.bertattention.self_attention.O_maps.load_state_dict(
                    theirblock.attention.output.dense.state_dict()
                )
            )
            ourblock.bertattention.layernorm.weights = theirblock.attention.output.LayerNorm.weight
            ourblock.bertattention.layernorm.biases = theirblock.attention.output.LayerNorm.bias
            ourblock.MLP.linear1.load_state_dict(theirblock.intermediate.dense.state_dict())
            print(ourblock.MLP.linear2.load_state_dict(theirblock.output.dense.state_dict()))
            ourblock.MLP.layer_norm.weights = theirblock.output.LayerNorm.weight
            ourblock.MLP.layer_norm.biases = theirblock.output.LayerNorm.bias
        ourbert.linear.load_state_dict(hf_bert.cls.predictions.transform.dense.state_dict())
        ourbert.layernorm.weights = hf_bert.cls.predictions.transform.LayerNorm.weight
        ourbert.layernorm.biases = hf_bert.cls.predictions.transform.LayerNorm.bias
        ourbert.unembed.bias = hf_bert.cls.predictions.decoder.bias
    return ourbert


if MAIN:
    my_bert = load_pretrained_weights(config)
    for (name, p) in my_bert.named_parameters():
        assert (
            p.is_leaf
        ), "Parameter {name} is not a leaf node, which will cause problems in training. Try adding detach() somewhere."




# %%


def predict(model: BertLanguageModel, tokenizer, text: str, k=15) -> List[List[str]]:
    """
    Return a list of k strings for each [MASK] in the input.
    """
    # [MASK] coded as 103
    input_ids = tokenizer(text)["input_ids"]
    token_type_ids = tokenizer(text)["token_type_ids"]
    attention_mask = tokenizer(text)["attention_mask"]

    input_ids = t.Tensor(input_ids).long()
    token_type_ids = t.Tensor(token_type_ids).long()
    positions = (input_ids == 103).nonzero().squeeze(1)

    with t.inference_mode():
        output = model(input_ids.unsqueeze(0), token_type_ids.unsqueeze(0), t.Tensor(attention_mask).bool()).squeeze()
    output = t.topk(output[positions], k)[1]
    return [tokenizer.convert_ids_to_tokens(list(preds)) for preds in list(output)]


if MAIN:
    tokenizer = transformers.AutoTokenizer.from_pretrained("bert-base-cased")

    # your_text = "The Answer to the Ultimate Question of Life, The Universe, and Everything is [MASK]."
    your_text = "The city councilmen refused the protestors a permit because the [MASK] advocated violence"

    predictions = predict(my_bert, tokenizer, your_text)
    print("Model predicted: \n", "\n".join(map(str, predictions)))

# %%
def long_predict(model, tokenizer, text, length, buffer = 3):
    input_ids = tokenizer(text)["input_ids"][:-1]
    
    model.eval()
    for i in range(length):
        input_ids_tensorfied = t.Tensor(input_ids + [103 for i in range(buffer)] + [102]).long().unsqueeze(0)
        with t.inference_mode():
            out = model(input_ids_tensorfied).squeeze()
        best = t.argmax(out[-buffer-1])
        input_ids.append(best.item())

    return " ".join(tokenizer.convert_ids_to_tokens(input_ids))

if MAIN:
    print(long_predict(my_bert, tokenizer, "Good night my", 8, buffer = 5))


# %%

