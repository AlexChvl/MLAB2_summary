#%%
from pathlib import Path
from typing import Any
import numpy as np

import torch as t
from w3d4_answers import sample, NoiseSchedule, train
from w3d4_answers_part2 import Unet


import torchvision
from einops import rearrange, repeat
from IPython.display import display
from PIL import Image
from scipy import linalg
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
from torchvision.transforms import CenterCrop, Compose, Lambda, RandomHorizontalFlip, Resize, ToPILImage, ToTensor
from tqdm.auto import tqdm

 
 

#%%
 

 
device = "cuda" if t.cuda.is_available() else "cpu"
MAIN = __name__ == "__main__"


def make_transform(image_size=128):
    """Pipeline from PIL Image to Tensor."""
    return Compose([Resize(image_size), CenterCrop(image_size), ToTensor(), Lambda(lambda t: t * 2 - 1)])


def make_reverse_transform():
    """Pipeline from Tensor to PIL Image."""
    return Compose(
        [
            Lambda(lambda t: (t + 1) / 2),
            Lambda(lambda t: t.permute(1, 2, 0)),
            Lambda(lambda t: t * 255.0),
            Lambda(lambda t: t.clamp(0, 255)),
            Lambda(lambda t: t.numpy().astype(np.uint8)),
            ToPILImage(),
        ]
    )


# %%

if MAIN:
    transform = make_transform()
    reverse_transform = make_reverse_transform()
    image = Image.open("./clip_images/guineapig_cucumber.jpg")
    display(image)
    x = transform(image)
    y = reverse_transform(x)
    display(y)

# %%
def get_fashion_mnist(train_transform, test_transform) -> tuple[TensorDataset, TensorDataset]:
    """Return MNIST data using the provided Tensor class."""
    mnist_train = datasets.FashionMNIST("../data", train=True, download=True)
    mnist_test = datasets.FashionMNIST("../data", train=False)
    print("Preprocessing data...")
    train_tensors = TensorDataset(
        t.stack([train_transform(img) for (img, label) in tqdm(mnist_train, desc="Training data")])
    )
    test_tensors = TensorDataset(t.stack([test_transform(img) for (img, label) in tqdm(mnist_test, desc="Test data")]))
    return (train_tensors, test_tensors)


if MAIN:
    train_transform = Compose([ToTensor(), RandomHorizontalFlip(), Lambda(lambda t: t * 2 - 1)])
    data_folder = Path("data/w3d4")
    data_folder.mkdir(exist_ok=True, parents=True)
    DATASET_FILENAME = data_folder / "generative_models_dataset_fashion.pt"
    if DATASET_FILENAME.exists():
        (train_dataset, test_dataset) = t.load(str(DATASET_FILENAME))
    else:
        (train_dataset, test_dataset) = get_fashion_mnist(train_transform, train_transform)
        t.save((train_dataset, test_dataset), str(DATASET_FILENAME))
# %%
if MAIN:
    config_dict: dict[str, Any] = dict(
        project_name = "Unet FashionMNIST"
        model_channels=28,
        model_dim_mults=(1, 2, 4),
        image_shape=(1, 28, 28),
        max_steps=500,
        epochs=20,
        lr=0.0005,
        batch_size=128,
        img_log_interval=400,
        n_images_to_log=5,
        device=t.device("cuda") if t.cuda.is_available() else t.device("cpu"),
    )

    model = Unet(
        config_dict["image_shape"],
        config_dict["model_channels"],
        config_dict["model_dim_mults"],
        max_steps=config_dict["max_steps"],
    ).to(config_dict["device"])

    model = train(model, config_dict, train_dataset, test_dataset)
    t.save(model.state_dict(), "data/w3d4_unet_500steps.pt")

# %%
