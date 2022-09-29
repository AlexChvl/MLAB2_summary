#%%

import hashlib
import os
import zipfile
import torch as t
import transformers
from einops import rearrange, repeat
from torch import nn
from torch.nn import functional as F
from tqdm.auto import tqdm
from w2d2_part1_answers import maybe_download

MAIN = __name__ == "__main__"
device = t.device("cuda" if t.cuda.is_available() else "cpu")
DATA_FOLDER = "./data/w2d2"
DATASET = "2"
BASE_URL = "https://s3.amazonaws.com/research.metamind.io/wikitext/"
DATASETS = {"103": "wikitext-103-raw-v1.zip", "2": "wikitext-2-raw-v1.zip"}
TOKENS_FILENAME = os.path.join(DATA_FOLDER, f"wikitext_tokens_{DATASET}.pt")

if MAIN:
    path = os.path.join(DATA_FOLDER, DATASETS[DATASET])
    maybe_download(BASE_URL + DATASETS[DATASET], path)
    expected_hexdigest = {"103": "0ca3512bd7a238be4a63ce7b434f8935", "2": "f407a2d53283fc4a49bcff21bc5f3770"}
    with open(path, "rb") as f:
        actual_hexdigest = hashlib.md5(f.read()).hexdigest()
        assert actual_hexdigest == expected_hexdigest[DATASET]
if MAIN:
    print(f"Using dataset WikiText-{DATASET} - options are 2 and 103")
    tokenizer = transformers.AutoTokenizer.from_pretrained("bert-base-cased")
    z = zipfile.ZipFile(path)

    def decompress(split: str) -> str:
        return z.read(f"wikitext-{DATASET}-raw/wiki.{split}.raw").decode("utf-8")

    train_text = decompress("train").splitlines()
    val_text = decompress("valid").splitlines()
    test_text = decompress("test").splitlines()

# %%
def tokenize_1d(tokenizer, lines: list[str], max_seq: int) -> t.Tensor:
    """Tokenize text and rearrange into chunks of the maximum length.

    Return (batch, seq) and an integer dtype
    """
    token_lines = tokenizer(lines, truncation=False)["input_ids"]

    token_tensor = t.Tensor([i for j in token_lines for i in j])

    length = (token_tensor.shape[0] // max_seq) * max_seq
    token_tensor = rearrange(token_tensor[:length], "(batch seq) -> batch seq", seq=max_seq)
    return token_tensor


max_seq = 128
if MAIN:
    print("Tokenizing training text...")
    train_data = tokenize_1d(tokenizer, train_text, max_seq)
    print("Training data shape is: ", train_data.shape)
    print("Tokenizing validation text...")
    val_data = tokenize_1d(tokenizer, val_text, max_seq)
    print("Tokenizing test text...")
    test_data = tokenize_1d(tokenizer, test_text, max_seq)
    print("Saving tokens to: ", TOKENS_FILENAME)
    t.save((train_data, val_data, test_data), TOKENS_FILENAME)

# %%


def flat(x: t.Tensor) -> t.Tensor:
    """Helper function for combining batch and sequence dimensions."""
    return rearrange(x, "b s ... -> (b s) ...")


def unflat(x: t.Tensor, max_seq: int) -> t.Tensor:
    """Helper function for separating batch and sequence dimensions."""
    return rearrange(x, "(b s) ... -> b s ...", s=max_seq)


def random_mask(
    input_ids: t.Tensor, mask_token_id: int, vocab_size: int, select_frac=0.15, mask_frac=0.8, random_frac=0.1
) -> tuple[t.Tensor, t.Tensor]:
    """Given a batch of tokens, return a copy with tokens replaced according to Section 3.1 of the paper.

    input_ids: (batch, seq)

    Return: (model_input, was_selected) where:

    model_input: (batch, seq) - a new Tensor with the replacements made, suitable for passing to the BertLanguageModel. Don't modify the original tensor!

    was_selected: (batch, seq) - 1 if the token at this index will contribute to the MLM loss, 0 otherwise
    """
    input_ids = input_ids.long()
    model_input = flat(t.clone(input_ids))  # one row vector
    was_selected = t.zeros_like(model_input)
    permutation = t.randperm(model_input.shape[0])

    selected = permutation[: int(select_frac * model_input.shape[0])]
    was_selected[selected] = 1
    was_selected = unflat(was_selected, max_seq)

    mask_stop = int(mask_frac * selected.shape[0])
    random_stop = int(mask_stop + random_frac * selected.shape[0])

    masked = selected[:mask_stop]  # one row vector
    randed = selected[mask_stop:random_stop]
    model_input[masked] = mask_token_id
    model_input[randed] = t.randint(0, vocab_size, randed.shape).to(model_input.device)
    model_input = unflat(model_input, max_seq)
    return (model_input, was_selected)
 

# %%
def cross_entropy_selected(pred: t.Tensor, target: t.Tensor, was_selected: t.Tensor) -> t.Tensor:
    """
    pred: (batch, seq, vocab_size) - predictions from the model
    target: (batch, seq, ) - the original (not masked) input ids
    was_selected: (batch, seq) - 1 if the token at this index will contribute to the MLM loss, 0 otherwise

    Out: the mean loss per predicted token
    """
    selected_bools = was_selected.bool()
    return nn.CrossEntropyLoss()(pred[selected_bools], target[selected_bools].long())


if MAIN:
    batch_size = 8
    seq_length = 128
    batch = t.randint(0, tokenizer.vocab_size, (batch_size, seq_length))
    pred = t.rand((batch_size, seq_length, tokenizer.vocab_size))
    (masked, was_selected) = random_mask(batch, tokenizer.mask_token_id, tokenizer.vocab_size)
    loss = cross_entropy_selected(pred, batch, was_selected).item()
    print(f"Random MLM loss on random tokens - does this make sense? {loss:.2f}")
# %%
