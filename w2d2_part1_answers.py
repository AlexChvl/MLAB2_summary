#%%
import hashlib
import os
import re
import tarfile
from dataclasses import dataclass
import requests
import torch as t
import transformers
from matplotlib import pyplot as plt
from torch.utils.data import TensorDataset
from tqdm.auto import tqdm
import io


MAIN = __name__ == "__main__"
IMDB_URL = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
DATA_FOLDER = "./data/w2d2/"
IMDB_PATH = os.path.join(DATA_FOLDER, "acllmdb_v1.tar.gz")
SAVED_TOKENS_PATH = os.path.join(DATA_FOLDER, "tokens.pt")
device = t.device("cuda" if t.cuda.is_available() else "cpu")



# %%

def maybe_download(url: str, path: str) -> None:
    """Download the file from url and save it to path. If path already exists, do nothing."""
    if not os.path.exists(path):
        r = requests.get(url)
        with open(path, "wb") as file:
            file.write(r.content)



if MAIN:
    os.makedirs(DATA_FOLDER, exist_ok=True)
    expected_hexdigest = "7c2ac02c03563afcf9b574c7e56c153a"
    maybe_download(IMDB_URL, IMDB_PATH)
    with open(IMDB_PATH, "rb") as f:
        actual_hexdigest = hashlib.md5(f.read()).hexdigest()
        assert actual_hexdigest == expected_hexdigest




# %%


with tarfile.open(IMDB_PATH) as archive:
    readers = [(archive.extractfile(member), member) for member in archive.getmembers()]
    for i in range(100):
        print(readers[i][1].name)
        print(readers[i][0].readline(), "\n\n\n")


@dataclass
class Review:
    split: str
    is_positive: bool
    stars: int
    text: str




def load_reviews(path: str) -> list[Review]:
    ret = []
    with tarfile.open(path) as archive:
        for member in tqdm(archive.getmembers()):
            file =archive.extractfile(member)
            name = member.name
            name_list = name.split("/")
            if file is not None and len(name_list) == 4 and (name_list[2] == "pos" or name_list[2] == 'neg'):
                file =  io.TextIOWrapper(file)
                # assert len(name_list) == 4, f"{name_list}, {file.readline()}"
    
                rating = name_list[-1].split("_")
                rating = rating[-1].split(".")[0]
                new = Review(name_list[1], (name_list[2] == "pos"), int(rating), file.readline()) 
                ret.append(new)
    return ret




reviews = []
if MAIN:
    reviews = load_reviews(IMDB_PATH)
    assert sum((r.split == "train" for r in reviews)) == 25000
    assert sum((r.split == "test" for r in reviews)) == 25000



# %%
review_lengths = [len(review.text) for review in reviews]
# %%
plt.hist(review_lengths,bins = 100)
# %%
review_lengths_pos = [len(review.text) for review in reviews if review.is_positive]
review_lengths_neg = [len(review.text) for review in reviews if not review.is_positive]
plt.hist([review_lengths_pos, review_lengths_neg], bins = 100, histtype='step', label = ["positive", "negative"])
plt.legend()
plt.show()

# %%
plt.hist([review.stars for review in reviews], bins = list(range(11)))
plt.show()
[review for review in reviews if review.stars == 6]
len(reviews)
# %%


def to_dataset(tokenizer, reviews: list[Review]) -> TensorDataset:
    """Tokenize the reviews (which should all belong to the same split) and bundle into a TensorDataset.

    The tensors in the dataset should be:

    input_ids: shape (batch, sequence length), dtype int
    attention_mask: shape (batch, sequence_length), dtype int
    sentiment_labels: shape (batch, ), dtype int
    star_labels: shape (batch, ), dtype int
    """
    
    input_ids = t.LongTensor(tokenizer([review.text for review in tqdm(reviews)], padding =  True, truncation = True)["input_ids"])
    attention_mask = t.LongTensor(tokenizer([review.text for review in reviews], padding = True, truncation = True)["attention_mask"])
    sentiment_labels = t.LongTensor([review.is_positive for review in reviews])
    star_labels = t.LongTensor([review.stars for review in reviews])
    return TensorDataset(input_ids, attention_mask, sentiment_labels, star_labels)


if MAIN:
    tokenizer = transformers.AutoTokenizer.from_pretrained("bert-base-cased")
    train_data = to_dataset(tokenizer, [r for r in reviews if r.split == "train"])
    test_data = to_dataset(tokenizer, [r for r in reviews if r.split == "test"])
    t.save((train_data, test_data), SAVED_TOKENS_PATH)



# %%
