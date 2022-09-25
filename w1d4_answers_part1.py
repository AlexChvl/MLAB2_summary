#%%

from typing import Iterable, List, Union, Optional
import os

import matplotlib.pyplot as plt
import matplotlib.figure
import torch as t
import torch.nn.functional as F
from einops import rearrange, repeat
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from torchvision import transforms



MAIN = __name__ == "__main__"
device = t.device("cuda" if t.cuda.is_available() else "cpu")

#%%

 

class ImageMemorizer(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim

        self.layer1 = nn.Linear(self.in_dim, self.hidden_dim)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.layer3 = nn.Linear(self.hidden_dim, self.out_dim)

    def forward(self, x: t.Tensor) -> t.Tensor:
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        return self.layer3(x)

 
 

if MAIN:
    fname = "w1d4_data/pearl_earring.jpg"
    img = Image.open(fname)
    print(f"Image size in pixels: {img.size[0]} x {img.size[1]} = {img.size[0] * img.size[1]}")
    plt.imshow(img)

# %%


class TensorDataset:
    def __init__(self, *tensors: t.Tensor):
        """Validate the sizes and store the tensors in a field named `tensors`."""
        size = tensors[0].shape[0]
        assert all([tensor.shape[0] == size for tensor in tensors]), "batch dimensions are not consistent"
        self.size = size
        self.tensors = tensors

    def __getitem__(self, index: Union[int, slice]) -> tuple[t.Tensor, ...]:
        """Return a tuple of length len(self.tensors) with the index applied to each."""
        return tuple(tensor[index] for tensor in self.tensors)

    def __len__(self):
        """Return the size in the first dimension, common to all the tensors."""
        return self.size


# %%


def all_coordinates_scaled(height: int, width: int) -> t.Tensor:
    """Return a tensor of shape (height*width, 2) where each row is a (x, y) coordinate.

    The range of x and y should be from [-1, 1] in both height and width dimensions.
    """

    height_vec = t.linspace(-1, 1, height)
    width_vec = t.linspace(-1, 1, width)

    repeat_height = repeat(height_vec, "i -> (i w)", w=width)
    repeat_width = repeat(width_vec, "i -> (h i)", h=height)

    return t.stack((repeat_height, repeat_width), dim=1)


def preprocess_image(img: Image.Image) -> TensorDataset:
    """Convert an image into a supervised learning problem predicting (R, G, B) given (x, y).

    Return: TensorDataset wrapping input and label tensors.
    input: shape (num_pixels, 2)
    label: shape (num_pixels, 3)
    """
    tensor_im = 2 * transforms.ToTensor()(img)[0:3] - 1  # scaled so entries in [-1, 1]
    im_shape = tensor_im.shape  # (3, height, width)
    reshape_im = rearrange(tensor_im, "c h w -> (h w) c")

    return TensorDataset(all_coordinates_scaled(im_shape[1], im_shape[2]), reshape_im)


# %%


def train_test_split(all_data: TensorDataset, train_frac=0.8, val_frac=0.01, test_frac=0.01) -> list[TensorDataset]:
    """Return [train, val, test] datasets containing the specified fraction of examples.

    If the fractions add up to less than 1, some of the data is not used.
    """
    len_data = len(all_data)
    total_perm = t.randperm(len_data)

    train_f = int(len_data * train_frac)
    val_f = int(len_data * val_frac)
    test_f = int(len_data * test_frac)

    train = TensorDataset(*all_data[total_perm[0:train_f]])
    test = TensorDataset(*all_data[total_perm[train_f : train_f + val_f]])
    val = TensorDataset(*all_data[total_perm[train_f + val_f : train_f + val_f + test_f]])
    return [train, test, val]


if MAIN:
    all_data = preprocess_image(img)
    (train_data, val_data, test_data) = train_test_split(all_data)
    print(f"Dataset sizes: train {len(train_data)}, val {len(val_data)} test {len(test_data)}")

# %%


def to_grid(X: t.Tensor, Y: t.Tensor, width: int, height: int) -> t.Tensor:
    """Convert preprocessed data from the format used in the Dataset back to an image tensor.

    X: shape (n_pixels, dim=2)
    Y: shape (n_pixels, channel=3)

    Return: shape (height, width, channels=3)
    """
    img = t.zeros((height, width, 3))

    new_X = X / 2 + 1 / 2
    new_Y = Y / 2 + 1 / 2

    new_X[:, 0] *= height - 1
    new_X[:, 1] *= width - 1

    # To combat floating point error in the rounding
    new_X += 0.5
    new_X = new_X.to(t.int64)

    img[new_X[:, 0], new_X[:, 1]] = new_Y

    return img


if MAIN:
    (width, height) = img.size
    (X, Y) = train_data.tensors
    plt.figure()
    plt.imshow(to_grid(X, Y, width, height))

# %%

if MAIN:
    train_loader = DataLoader(train_data, batch_size=256, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=256)
    test_loader = DataLoader(test_data, batch_size=256)

# %%


class Adam:
    def __init__(
        self,
        params: Iterable[t.nn.parameter.Parameter],
        lr: float = 0.001,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-08,
        weight_decay: float = 0.0,
    ):
        """Implements Adam.

        Like the PyTorch version, but assumes amsgrad=False and maximize=False
            https://pytorch.org/docs/stable/generated/torch.optim.Adam.html#torch.optim.Adam
        """
        self.params = [param for param in list(params)]
        self.grads_moving_avg = [t.zeros_like(param) for param in self.params]
        self.lr = lr
        self.weight_decay = weight_decay
        self.betas = betas
        self.eps = eps
        self.square_average = [t.zeros_like(param) for param in self.params]
        self.average = [t.zeros_like(param) for param in self.params]
        self.time = 1

    def zero_grad(self) -> None:
        for param in self.params:
            param.grad = None

    def step(self) -> None:
        with t.inference_mode():
            for idx, param in enumerate(self.params):
                if param.grad is not None:
                    if self.weight_decay != 0.0:
                        param.grad += self.weight_decay * param
                    self.average[idx] *= self.betas[0]
                    self.average[idx] += (1 - self.betas[0]) * (param.grad)

                    self.square_average[idx] *= self.betas[1]
                    self.square_average[idx] += (1 - self.betas[1]) * (param.grad**2)
                    m_average = self.average[idx] / (1 - self.betas[0] ** self.time)
                    v_average = self.square_average[idx] / (1 - self.betas[1] ** self.time)
                    param += -self.lr * m_average / ((t.sqrt(v_average) + self.eps))
            self.time += 1

def train_one_epoch(model: ImageMemorizer, dataloader: DataLoader, pretrained = False) -> float:
    """Show each example in the dataloader to the model once.

    Use `torch.optim.Adam` for the optimizer (you'll build your own Adam optimizer later today).
    Use `F.l1_loss(prediction, actual)` for the loss function. This just puts less weight on very bright or dark pixels, which seems to produce nicer images.

    Return: the total loss (sum of losses of all batches) divided by the total number of examples seen.
    """
    model = model.to(device)
    optimizer = Adam(model.parameters())
    tot_loss = []
    for feature, actual in dataloader:
        feature = feature.to(device)
        actual = actual.to(device)
        optimizer.zero_grad()
        output = model(feature)
        # loss = F.l1_loss(output, actual).mean()
        loss = F.mse_loss(output, actual).mean()

        loss.backward()
        optimizer.step()
        tot_loss.append(loss.item())
    return sum(tot_loss) / len(tot_loss)

 



def evaluate(model: ImageMemorizer, dataloader: DataLoader) -> float:
    """Return the total L1 loss over the provided data divided by the number of examples."""
    tot_loss = []
    with t.inference_mode():
        for feature, actual in dataloader:
            feature = feature.to(device)
            actual = actual.to(device)
            output = model(feature)
            # loss = F.l1_loss(output, actual).mean()
            loss = F.mse_loss(output, actual).mean()

            tot_loss.append(loss.item())

    return sum(tot_loss) / len(tot_loss)


#%%
if MAIN:
    train_loader = DataLoader(train_data, batch_size=256, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=256)
    test_loader = DataLoader(test_data, batch_size=256)
    train_loss_tracker = []
    val_loss_tracker = []

    my_model = ImageMemorizer(2, 400, 3)
    prefix = "PearlEarring-Adaml2"

    old_model_names = [name for name in os.listdir("./w1d4_data") if len(name.split("_")) == 3 and name.split("_")[0] == prefix and name.split("_")[1].isnumeric() and name.split("_")[-1] == ".pt"]

    if old_model_names!= []:
        most = max([int(name.split("_")[1]) for name in old_model_names])
        filename = prefix + "_" + str(most) + "_.pt"
        my_model.load_state_dict(t.load("./w1d4_data/"+filename).state_dict())

    else:
        filename = prefix + "_0_.pt"
        t.save(my_model, "./w1d4_data/"+ filename)
        

    epochs = 10
    for i in tqdm(range(epochs)):
        train_loss = train_one_epoch(my_model, train_loader)
        train_loss_tracker.append(train_loss)
        val_loss = evaluate(my_model, val_loader)
        val_loss_tracker.append(val_loss)

    new = int(filename.split("_")[1]) + epochs
    newname = prefix+ "_" + str(new) + "_.pt"
    t.save(my_model, "./w1d4_data/"+ newname)




if MAIN:
    plt.plot(train_loss_tracker, label="train")
    plt.plot(val_loss_tracker, label="val")
    plt.legend()
    plt.show()

# %%

if MAIN:
    names = [name for name in os.listdir("./w1d4_data") if len(name.split("_")) == 3 and name.split("_")[0] == prefix and name.split("_")[1].isnumeric() and name.split("_")[-1] == ".pt"]

    epochs = [int(name.split("_")[1]) for name in names]
    epochs.sort()
    names = [prefix + "_" + str(ep) + "_.pt" for ep in epochs]

    all_coords = all_coordinates_scaled(height, width)

    for name in names:
        my_model.load_state_dict(t.load("./w1d4_data/"+name).state_dict())
        with t.inference_mode():
            my_model = my_model.to("cpu")
            model_data = my_model(all_coords)
        grid_image = to_grid(all_coords, model_data, width, height)
        plt.title(name)
        plt.imshow(grid_image, )
        plt.show()




# %%


class SGD:
    def __init__(self, params: Iterable[t.nn.parameter.Parameter], lr: float, momentum: float, weight_decay: float):
        """Implements SGD with momentum.

        Like the PyTorch version, but assume nesterov=False, maximize=False, and dampening=0
            https://pytorch.org/docs/stable/generated/torch.optim.SGD.html#torch.optim.SGD

        """
        self.params = [param for param in list(params)]
        self.grads_moving_avg = [t.zeros_like(param) for param in self.params]
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay

    def zero_grad(self) -> None:
        for param in self.params:
            param.grad = None

    def step(self) -> None:
        with t.inference_mode():
            for idx, param in enumerate(self.params):
                if param.grad is not None:
                    if self.weight_decay != 0.0:
                        param.grad += self.weight_decay * param
                    if self.momentum != 0.0:
                        self.grads_moving_avg[idx].mul_(self.momentum)
                        self.grads_moving_avg[idx].add_(param.grad)
                        param.grad = self.grads_moving_avg[idx]
                    self.params[idx].add_(-self.lr * param.grad)



#%%
class RMSprop:
    def __init__(
        self,
        params: Iterable[t.nn.parameter.Parameter],
        lr: float,
        alpha: float,
        eps: float,
        weight_decay: float,
        momentum: float,
    ):
        """Implements RMSprop.

        Like the PyTorch version, but assumes centered=False
            https://pytorch.org/docs/stable/generated/torch.optim.RMSprop.html#torch.optim.RMSprop

        """
        self.params = [param for param in list(params)]
        self.grads_moving_avg = [t.zeros_like(param) for param in self.params]
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.alpha = alpha
        self.eps = eps
        self.square_average = [t.zeros_like(param) for param in self.params]

    def zero_grad(self) -> None:
        for param in self.params:
            param.grad = None

    def step(self) -> None:
        with t.inference_mode():
            for idx, param in enumerate(self.params):
                if param.grad is not None:
                    if self.weight_decay != 0.0:
                        param.grad += self.weight_decay * param
                    self.square_average[idx] *= self.alpha
                    self.square_average[idx] += (1 - self.alpha) * (param.grad**2)

                    if self.momentum > 0.0:
                        self.grads_moving_avg[idx].mul_(self.momentum)
                        self.grads_moving_avg[idx].add_(param.grad / (t.sqrt(self.square_average[idx]) + self.eps))
                        param += -self.lr * self.grads_moving_avg[idx]
                    else:
                        param += -self.lr * param.grad / (t.sqrt(self.square_average[idx]) + self.eps)

