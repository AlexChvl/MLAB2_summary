# %%
import collections
from typing import Tuple, Optional
from dataclasses import dataclass
import os
import torch as t
import transformers
from einops import rearrange, repeat
from tqdm.auto import tqdm
import utils
from w2d3_part1_answers import load_pretrained_weights, GPT2, GPT2Block, Cache, BlockCache
import sys

os.environ["TOKENIZERS_PARALLELISM"] = "false"
MAIN = __name__ == "__main__"
device = t.device("cuda" if t.cuda.is_available() else "cpu")
if MAIN:
    my_gpt = load_pretrained_weights().eval()
    tokenizer = transformers.AutoTokenizer.from_pretrained("gpt2")




# %%
def sample_next_token(
    model: GPT2, input_ids: t.Tensor, temperature=1.0, freq_penalty=0.0, top_k=0, top_p=0.0, cache=None
) -> Tuple[int, Cache]:
    """Return the next token, sampled from the model's probability distribution with modifiers.

    input_ids: shape (seq,)
    """
    assert input_ids.ndim == 1, "input_ids should be a 1D sequence of token ids"
    assert temperature >= 0
    assert 0 <= top_p <= 1.0, "Top-p must be a probability"
    assert 0 <= top_k, "Top-k must be non-negative"
    assert not (top_p != 0 and top_k != 0), "At most one of top-p and top-k supported"
    model.eval()
    with t.inference_mode():
        all_logits, cache = model(input_ids.unsqueeze(0), cache=cache)
    (B, S, E) = all_logits.shape
    assert B == 1
    assert S == len(input_ids)
    logits = all_logits[0, -1]
    if temperature == 0:
        return greedy_search(logits), cache
    logits = apply_temperature(logits, temperature)
    logits = apply_freq_penalty(input_ids, logits, freq_penalty)
    if top_k > 0:
        return sample_top_k(logits, top_k), cache
    if top_p > 0:
        return sample_top_p(logits, top_p), cache
    return sample_basic(logits), cache


def sample_tokens(
    model: GPT2,
    tokenizer,
    initial_text: str,
    max_tokens_generated=30,
    temperature=1.0,
    freq_penalty=0.0,
    stop_at_eos=True,
    top_k=0,
    top_p=0.0,
    cache=None,
) -> str:
    """Sample tokens using sample_next_token until the model outputs `tokenizer.eos_token_id` or the specified token limit is reached.

    Return: the prompt and continuation concatenated
    """
    input_ids: list = tokenizer(initial_text).input_ids
    generated = []
    for _ in tqdm(range(max_tokens_generated)):
        new_token, _ = sample_next_token(
            model,
            t.LongTensor(input_ids + generated),
            temperature=temperature,
            freq_penalty=freq_penalty,
            top_k=top_k,
            top_p=top_p,
            cache=cache,
        )
        generated.append(new_token)
        if stop_at_eos and new_token == tokenizer.eos_token_id:
            break
    return tokenizer.decode(input_ids + generated)


# %%
def greedy_search(logits: t.Tensor) -> int:
    """
    logits: shape (vocab_size, )

    Return: the most likely token
    """
    return t.argmax(logits).item()


if MAIN:
    logits = t.ones(100)
    logits[5] = 10
    logits[8] = 10
    assert greedy_search(logits) == 5
# %%


def apply_temperature(logits: t.Tensor, temperature: float) -> t.Tensor:
    """
    logits: shape (vocab_size, )

    Return: shape (vocab_size, )
    """
    assert temperature > 0
    return logits / temperature


if MAIN:
    logits = t.tensor([1,2,3])
    assert all(apply_temperature(logits, 10) == t.Tensor([0.1, 0.2, 0.3]))

 
# %%
def apply_freq_penalty(input_ids: t.Tensor, logits: t.Tensor, freq_penalty: float) -> t.Tensor:
    """
    input_ids: shape (seq, )
    logits: shape (vocab_size, )

    Return: shape (vocab_size, )
    """
    bins = t.bincount(input_ids) # a tensor of shape (max(input_ids) + 1,) counting the occurrences of each int
    counts_per_input = bins[input_ids].float()
    penalty = t.zeros_like(logits)
    penalty[input_ids] = counts_per_input * freq_penalty
    return logits - penalty


if MAIN:
    bieber_prompt = "And I was like Baby, baby, baby, oh Like, Baby, baby, baby, no Like, Baby, baby, baby, oh I thought you'd always be mine, mine"
    input_ids = tokenizer(bieber_prompt, return_tensors="pt")["input_ids"][0]
    logits = t.ones(tokenizer.vocab_size)
    penalized_logits = apply_freq_penalty(input_ids, logits, 2.0)
    assert penalized_logits[5156].item() == -11, "Expected 6 occurrences of ' baby' with leading space"
    assert penalized_logits[14801].item() == -5, "Expected 3 occurrences of ' Baby' with leading space"
# %%
def sample_basic(logits: t.Tensor) -> int:
    """
    logits: shape (vocab_size, ) - unnormalized log-probabilities

    Return: a sampled token
    """
    distrib = t.distributions.categorical.Categorical(logits=logits)
    return distrib.sample().item()

 

# %%

if MAIN:
    N_RUNS = 1
    my_prompt = "Happy birth day to you, happy birthday to you, happy birthday dear"
    cases = [
        ("High freq penalty", dict(freq_penalty=100.0)),
        ("Negative freq penalty", dict(freq_penalty=-1.0)),
        ("Too hot!", dict(temperature=2.0)),
        ("Pleasantly cool", dict(temperature=0.7)),
        ("Pleasantly warm", dict(temperature=0.9)),
    ]
    for (name, kwargs) in cases:
        for i in range(N_RUNS):
            output = sample_tokens(my_gpt, tokenizer, my_prompt, max_tokens_generated=24, **kwargs)
            print(f"Sample {i} with: {name} ({kwargs}):")
            print(f"my model said: {repr(output)}")




# %%
def sample_top_k(logits: t.Tensor, top_k: int) -> int:
    """
    logits: shape (vocab_size, ) - unnormalized log-probabilities
    top_k: only consider this many of the most likely tokens for sampling

    Return: a sampled token
    """
    assert len(logits.shape) > 0
    topk = t.topk(logits, top_k)
    value = sample_basic(topk.values)
    return topk.indices[value]


# if MAIN:
#     k = 3
#     N = 20000
#     probs = t.linspace(0, 0.4, 5)
#     unnormalized_logits = probs.log() + 1.2345
#     samples = t.tensor([sample_top_k(unnormalized_logits, k) for _ in range(N)])
#     counts = t.bincount(samples, minlength=len(probs)) / N
#     expected = probs.clone()
#     expected[:-k] = 0
#     expected /= expected.sum()
#     print("Checking empirical frequencies (try to increase N if this test fails): ", counts)
#     utils.allclose_atol(counts, expected, atol=0.01)





# %%
if MAIN:
    my_prompt = "In a shocking finding, scientist discovered a herd of unicorns living in a remote, previously unexplored valley, in the Andes Mountains. Even more surprising to the researchers was the fact that the unicorns spoke perfect English."
    output = sample_tokens(my_gpt, tokenizer, my_prompt, temperature=0.65, top_k=60, max_tokens_generated=64)
    print(f"my model said: {repr(output)}")

# %%
def sample_top_p(logits: t.Tensor, top_p: float) -> int:
    """
    Nucleus sampling: top_p is a cut-off point; choose the highest probabilities which add up to top_p and discard the rest.

    logits: shape (vocab_size, ) - unnormalized log-probabilities

    Return: a sampled token
    """
    probs = t.softmax(logits, 0)
    sorted_probs, sorted_values = t.sort(probs, descending=True)
    cumsum = t.cumsum(sorted_probs, 0)
    mask = (cumsum <= top_p)
    selected = sorted_probs * mask
    selected = selected / t.sum(selected)
    m = t.distributions.categorical.Categorical(probs = selected)
    val = m.sample()
    return sorted_values[val].item()


if MAIN:
    N = 20000
    top_p = 0.71
    probs = t.linspace(0, 0.4, 5)
    unnormalized_logits = probs.log() + 1.2345
    samples = t.tensor([sample_top_p(unnormalized_logits, top_p) for _ in range(N)])
    counts = t.bincount(samples, minlength=len(probs)) / N
    expected = probs.clone()
    expected[0:3] = 0
    expected /= expected.sum()
    print("Checking empirical frequencies (try to increase N if this test fails): ", counts)
    utils.allclose_atol(counts, expected, atol=0.01)
# %%



def sample_tokens_with_cache(
    model: GPT2,
    tokenizer,
    initial_text: str,
    max_tokens_generated=30,
    temperature=1.0,
    freq_penalty=2.0,
    stop_at_eos=True,
    top_k=0,
    top_p=0.0,
    cache=None,
) -> str:
    """Does the exact same thing as sample_tokens, but using cache to be faster."""
    input_ids: list = tokenizer(initial_text).input_ids
    generated = []
    for _ in tqdm(range(max_tokens_generated)):
        new_token, cache = sample_next_token(
            model,
            t.LongTensor(input_ids + generated),
            temperature=temperature,
            freq_penalty=freq_penalty,
            top_k=top_k,
            top_p=top_p,
            cache=cache,
        )
        generated.append(new_token)
        if stop_at_eos and new_token == tokenizer.eos_token_id:
            break
    return tokenizer.decode(input_ids + generated)


if MAIN:
    text =  "It is pitch black and pouring with rain. The shadow of a black umbrella appears down the street. I can hear footsteps through the rain."
    print(sample_tokens_with_cache(my_gpt, tokenizer, text, 64, temperature = 0.3, freq_penalty = 3, stop_at_eos = True, top_k = 40))




# %%


def beam_search(
    model, input_ids: t.Tensor, num_return_sequences: int, num_beams: int, max_new_tokens: int, tokenizer, verbose=False, cache = None
) -> list[tuple[float, t.Tensor]]:
    """
    input_ids: (seq, ) - the prompt

    max_new_tokens: stop after this many new tokens are generated, even if no EOS is generated. In this case, the best incomplete sequences should also be returned.
    verbose: if True, print the current (unfinished) completions after each iteration for debugging purposes

    Return list of length num_return_sequences. Each element is a tuple of (logprob, tokens) where the tokens include both prompt and completion, sorted by descending logprob.
    """

    assert num_return_sequences <= num_beams
    model.eval()

    with t.inference_mode():
        distrib, cache = model(input_ids.unsqueeze(0), cache)  # distrib is (batch, seq, vocab_size)
        best = t.topk(distrib[0][-1], num_beams)    #best is (top_k vals, top_k indices)
        probs = t.softmax(best[0], 0)
        continue_list = list(
            zip(
                [t.cat((input_ids, best[1][i].unsqueeze(0))) for i in range(num_beams)],
                list(t.log(probs)),
                [cache for i in range(num_beams)],
            )
        )  # each element in continue_list is [path, log-prob, cache]

        assert len(continue_list) == num_beams
        return_list = []
        while len(return_list) < num_return_sequences:

            combined = []
            for candidate in continue_list:
                newcache = candidate[2].clone()
                distrib, newcache = model(candidate[0].unsqueeze(0), newcache)
                best = t.topk(distrib[0][-1], num_beams)
                probs = t.softmax(best[0], 0)
                combined += [
                    [t.cat((candidate[0], best[1][i].unsqueeze(0))), candidate[1] + t.log(probs)[i], newcache]
                    for i in range(best[0].shape[0])
                ]
            sort_combined = [
                candidate
                for _, candidate in sorted(zip([candidate[1] for candidate in combined], combined), reverse=True)
            ]

            continue_list = []
            for candidate in sort_combined:
                if (
                    candidate[0][-1].item() == tokenizer.eos_token_id
                    or candidate[0].shape[0] >= max_new_tokens + input_ids.shape[0]
                ):
                    return_list.append(candidate)
                else:
                    continue_list.append(candidate)
                    if len(continue_list) >= num_beams:
                        break
        return_list = return_list[:num_return_sequences]
        return [(candidate[1], candidate[0]) for candidate in return_list]


if MAIN:

    my_prompt = "I don't want to rule the universe. I just think"
    beam_out = beam_search(
        model=my_gpt,
        input_ids=tokenizer(my_prompt, return_tensors="pt", return_attention_mask=False)["input_ids"][0],
        num_return_sequences=3,
        num_beams=6,
        max_new_tokens=20,
        tokenizer=tokenizer,
        verbose=True,
    )

    print("\n\n Final answer: \n\n ")
    for (score, tokens) in beam_out:
        print(tokenizer.decode(tokens))
        print(f"{score:.4f}: {repr(tokenizer.decode(tokens))}")

 
# %%
