#%%

import torch as t
import transformers
from matplotlib import pyplot as plt
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.data.dataset import TensorDataset
from tqdm.auto import tqdm
import wandb
import time


from w2d1_answers import BertConfig, BertLanguageModel, predict
from w2d2_part3_answers import cross_entropy_selected, random_mask

MAIN = __name__ == "__main__"
device = t.device("cuda" if t.cuda.is_available() else "cpu")

# %%
if MAIN:
    tokenizer = transformers.AutoTokenizer.from_pretrained("bert-base-cased")
    hidden_size = 512
    assert hidden_size % 64 == 0
    bert_config_tiny = BertConfig(
        max_position_embeddings=128,
        hidden_size=hidden_size,
        intermediate_size=4 * hidden_size,
        num_layers=8,
        num_heads=hidden_size // 64,
    )
    config_dict = dict(
        filename="./data/w2d2/bert_lm.pt",
        lr=0.0002,
        epochs=40,
        batch_size=128,
        weight_decay=0.01,
        mask_token_id=tokenizer.mask_token_id,
        warmup_step_frac=0.01,
        eps=1e-06,
        max_grad_norm=None,
    )
    (train_data, val_data, test_data) = t.load("./data/w2d2/wikitext_tokens_2.pt")
    print("Training data size: ", train_data.shape)
    train_loader = DataLoader(
        TensorDataset(train_data), shuffle=True, batch_size=config_dict["batch_size"], drop_last=True
    )


#%%


def lr_for_step(step: int, max_step: int, max_lr: float, warmup_step_frac: float):
    """Return the learning rate for use at this step of training."""
    if step <= warmup_step_frac * max_step:
        return 0.1 * max_lr + step * 0.9 * max_lr / (warmup_step_frac * max_step)
    else:
        return max_lr * 0.1 + (max_step - step) * max_lr * 0.9 / ((1 - warmup_step_frac) * max_step)


if MAIN:
    max_step = int(len(train_loader) * config_dict["epochs"])
    lrs = [
        lr_for_step(step, max_step, max_lr=config_dict["lr"], warmup_step_frac=config_dict["warmup_step_frac"])
        for step in range(max_step)
    ]
    (fig, ax) = plt.subplots(figsize=(12, 4))
    ax.plot(lrs)
    ax.set(xlabel="Step", ylabel="Learning Rate", title="Learning Rate Schedule")


# %%
def make_optimizer(model: BertLanguageModel, config_dict: dict) -> t.optim.AdamW:
    """
    Loop over model parameters and form two parameter groups:

    - The first group includes the weights of each Linear layer and uses the weight decay in config_dict
    - The second has all other parameters and uses weight decay of 0
    """
    linear_weights = []
    others = []
    for submodule in model.modules():  # test if submodule is nn.Linear and check if it has already occurred
        if isinstance(submodule, nn.Linear):
            if not any([other.shape == submodule.weight.shape and other is submodule.weight for other in others]):
                linear_weights.append(submodule.weight)
            others.append(submodule.bias)

        else:
            others = others + list(submodule.parameters(recurse=False))
    return t.optim.AdamW(
        [
            {
                "params": linear_weights,
                "lr": config_dict["lr"],
                "eps": config_dict["eps"],
                "weight_decay": config_dict["weight_decay"],
            },
            {"params": others, "lr": config_dict["lr"], "eps": config_dict["eps"]},
        ]
    )


if MAIN:
    test_config = BertConfig(max_position_embeddings=4, hidden_size=1, intermediate_size=4, num_layers=3, num_heads=1)
    optimizer_test_model = BertLanguageModel(test_config)
    opt = make_optimizer(optimizer_test_model, dict(weight_decay=0.1, lr=0.0001, eps=1e-06))
    expected_num_with_weight_decay = test_config.num_layers * 6 + 1
    wd_group = opt.param_groups[0]
    actual = len(wd_group["params"])
    assert (
        actual == expected_num_with_weight_decay
    ), f"Expected 6 linear weights per layer (4 attn, 2 MLP) plus the final lm_linear weight to have weight decay, got {actual}"
    all_params = set()
    for group in opt.param_groups:
        all_params.update(group["params"])
    assert all_params == set(optimizer_test_model.parameters()), "Not all parameters were passed to optimizer!"


# %%


def bert_mlm_pretrain(model: BertLanguageModel, config_dict: dict, train_loader: DataLoader) -> None:
    """Train using masked language modelling."""
    wandb.init(project="w2d2_part4", config=config_dict)
    wandb_config = wandb.config

    model.to(device)

    wandb.watch(model, criterion=cross_entropy_selected, log="all", log_freq=10, log_graph=True)

    optim = make_optimizer(model, config_dict)
    scheduler = t.optim.lr_scheduler.LambdaLR(
        optim, lambda step: lr_for_step(step, max_step, 1.0, config_dict["warmup_step_frac"])
    )
    start_time = time.time()
    examples_seen = 0
    for epoch in tqdm(range(config_dict["epochs"])):
        for (mini_batch,) in tqdm(train_loader):
            mini_batch = mini_batch.to(device)
            masked, was_selected = random_mask(mini_batch, config_dict["mask_token_id"], bert_config_tiny.vocab_size)
            predicted = model(masked)
            loss = cross_entropy_selected(predicted, mini_batch, was_selected)
            optim.zero_grad()
            loss.backward()
            optim.step()
            scheduler.step()

            wandb.log(
                dict(train_loss=loss, elapsed=time.time() - start_time, learning_rate=scheduler.get_last_lr()),
                step=examples_seen,
            )
            examples_seen += 1

    t.save(model.state_dict(), config_dict["filename"])


if MAIN:
    model = BertLanguageModel(bert_config_tiny)
    num_params = sum((p.nelement() for p in model.parameters()))
    print("Number of model parameters: ", num_params)
    bert_mlm_pretrain(model, config_dict, train_loader)
