#%%
import os
from matplotlib import pyplot as plt
import time
from dataclasses import dataclass
import torch as t
import transformers
from einops import rearrange
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm
from transformers.tokenization_utils_base import TruncationStrategy
import wandb
from utils import assert_all_equal
from w2d1_answers import (
    BertCommon,
    BertConfig,
    BertLanguageModel,
    load_pretrained_weights
)
from w2d2_part1_answers import DATA_FOLDER,SAVED_TOKENS_PATH, to_dataset


MAIN = __name__ == "__main__"
device = t.device("cuda" if t.cuda.is_available() else "cpu")

if MAIN:
    (train_data, test_data) = t.load(SAVED_TOKENS_PATH)
    bert_config = BertConfig()


    plt.hist([train_data[i][3] for i in range(len(train_data))])

# %%


@dataclass
class BertClassifierOutput:
    """The output of BertClassifier."""
    star_rating: t.Tensor


class BertClassifier(nn.Module):
    def __init__(self, config: BertConfig):
        super().__init__()
        self.config = config
        self.bertcommon = BertCommon(config)
        self.dropout = nn.Dropout(config.dropout)
        self.star_unembed = nn.Linear(config.hidden_size, 1)

    def forward(self, x: t.Tensor, boolean_attention_mask: t.Tensor) -> BertClassifierOutput:
        assert len(x.shape) == 2, f"Input lacks batch dimension? Input has {len(x.shape)} dimensions"

        x = self.bertcommon(x, boolean_attention_mask)
        x = x[:, 0]

        x = self.dropout(x)
        star_rating = self.star_unembed(x) * 5 + 5
        return BertClassifierOutput(star_rating)

# %%



def train_loss_func(output: BertClassifierOutput, star_labels):
    return F.l1_loss(output.star_rating.squeeze(1), star_labels)


def train(tokenizer, config_dict: dict, pre_tuned=None) -> BertClassifier:
    wandb.init(project="w2d2_selfmade", config=config_dict)
    wandb_config = wandb.config
    if pre_tuned is not None:
        bert_classifier = pre_tuned
    else:
        bert_classifier = BertClassifier(bert_config)
        bert_lm = load_pretrained_weights(bert_config)
        bert_classifier.bertcommon.load_state_dict(bert_lm.bertcommon.state_dict())
    for p in bert_classifier.parameters():
        p.requires_grad = True
    bert_classifier.to(device)
    optim = t.optim.AdamW(bert_classifier.parameters(), lr=config_dict["lr"], weight_decay=config_dict["weight_decay"])
    train_loader = DataLoader(train_data, config_dict["batch_size"], shuffle=True)

    wandb.watch(bert_classifier, criterion=train_loss_func, log="all", log_freq=10, log_graph=True)
    examples_seen = 0
    start_time = time.time()
    for epoch in range(config_dict["epochs"]):
        for input_ids, attention_mask, _, star_labels in tqdm(train_loader):
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            star_labels = star_labels.to(device)
            optim.zero_grad()

            output = bert_classifier(input_ids, attention_mask)
            loss = train_loss_func(output, star_labels)
            loss.backward()

            t.nn.utils.clip_grad_norm_(bert_classifier.parameters(), 1.0)
            optim.step()

            wandb.log(
                dict(train_loss=loss, elapsed=time.time() - start_time),
                step=examples_seen,
            )
            examples_seen += input_ids.shape[0]
    t.save(bert_classifier.state_dict(), config_dict["filename"])
    wandb.save(config_dict["filename"])
    bert_classifier.to("cpu")
    return bert_classifier


if MAIN:
    tokenizer = transformers.AutoTokenizer.from_pretrained("bert-base-cased")
    wandb.config = dict(
        lr=5e-07,
        batch_size=8,
        step_every=1,
        epochs=1,
        weight_decay=0,
        num_steps=9600,
        filename="./data/w2d2/bert_classifier_800revs_lr5e-7_wt1.pt",
    )
 
    # train(tokenizer, wandb.config)
    classifier = BertClassifier(bert_config)
    classifier.load_state_dict(t.load(wandb.config["filename"]))


# %%
def test_set_predictions(model: BertClassifier, test_data: TensorDataset, batch_size=256) -> tuple[t.Tensor, t.Tensor]:
    """
    Return the predicted star rating for each test set example.

    star: shape (batch, ) - star rating
    """
    model.to(device)
    model.eval()
    test_loader = DataLoader(test_data, batch_size=batch_size)
    sentiment = []
    star = []
    with t.inference_mode():
        for input_ids, attention_mask, _, _ in test_loader:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            predictions = model(input_ids, attention_mask)
            star_ratings = predictions.star_rating.squeeze(1)
            star_ratings = t.maximum(star_ratings, t.Tensor([1]).to(device))
            star_ratings = t.minimum(star_ratings, t.Tensor([10]).to(device))
            star.append(star_ratings)
    return t.cat(star, dim=0)

#%%
if MAIN:
    n = len(test_data)
    perm = t.randperm(n)[:30]
    test_subset = TensorDataset(*test_data[perm])

    pred_stars = test_set_predictions(classifier, test_subset)
    star_diff = pred_stars.cpu() - test_subset.tensors[3]       # the signed error of the predictor (positive value means overevaluated)
    star_error = star_diff.abs().mean()
    print(f"Star MAE: {star_error:.2f}")
#%%

if MAIN:
    star_diff_abs = star_diff.abs()
    worst_scores, worst_indices = t.sort(star_diff_abs, dim = 0, descending = True)
    wrong_tokens = test_subset.tensors[0][worst_indices]

    wrong_text = tokenizer.batch_decode(wrong_tokens, skip_special_tokens=True)
    print("The following review was wrongly classified: the predicted score was ", pred_stars[worst_indices[0]],". \n The review said: \n\n", wrong_text[0], "\n\nThe actual score was", test_subset.tensors[3][worst_indices[0]].item())

# %%
def predict(tokenizer, classifier, review: str):
    """ runs the classifier on the review and returns the predicted star rating"""
    tokens = tokenizer(review, truncation=True)["input_ids"]
    mask = tokenizer(review, truncation=True)["attention_mask"]
    tokens = t.Tensor(tokens).unsqueeze(0).long()
    mask = t.Tensor(mask).unsqueeze(0).long()
    classifier.eval()
    with t.inference_mode():
        pred = classifier(tokens, mask)
        return pred.star_rating.item()



if MAIN:
    custom_review_pos = "This movie was terrific. I had such a great time watching it. Highly recommend!"
    custom_review_neg= "Absolute embarrassement of a film. The director should be embarrassed to show himself in public."
    print("Prediction for positive review:", predict(tokenizer, classifier, custom_review_pos))
    print("Prediction for negative review:", predict(tokenizer, classifier, custom_review_neg))

    
# %%
