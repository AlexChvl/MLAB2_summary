#%%

import json
import os
import sys
from collections import OrderedDict
from io import BytesIO
from typing import Optional, Union
import requests
import torch as t
import torchvision
from einops import rearrange
from IPython.display import display
from matplotlib import pyplot as plt
from PIL import Image
from torch import nn
from torch.nn.functional import conv1d as torch_conv1d
from torch.utils.data import DataLoader
from torchvision import models, transforms
from tqdm.auto import tqdm
import utils

import w1d2_test_selfmade

MAIN = __name__ == "__main__"

URLS = [
    "https://www.oregonzoo.org/sites/default/files/styles/article-full/public/animals/H_chimpanzee%20Jackson.jpg",
    "https://anipassion.com/ow_userfiles/plugins/animal/breed_image_56efffab3e169.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/f/f2/Platypus.jpg",
    "https://static5.depositphotos.com/1017950/406/i/600/depositphotos_4061551-stock-photo-hourglass.jpg",
    "https://img.nealis.fr/ptv/img/p/g/1465/1464424.jpg",
    "http://www.tudobembresil.com/wp-content/uploads/2015/11/nouvelancopacabana.jpg",
    "https://ychef.files.bbci.co.uk/976x549/p0639ffn.jpg",
    "https://www.thoughtco.com/thmb/Dk3bE4x1qKqrF6LBf2qzZM__LXE=/1333x1000/smart/filters:no_upscale()/iguana2-b554e81fc1834989a715b69d1eb18695.jpg",
    "https://i.redd.it/mbc00vg3kdr61.jpg",
    "https://static.wikia.nocookie.net/disneyfanon/images/a/af/Goofy_pulling_his_ears.jpg",
    "https://londonhuawiki.wpi.edu/images/1/10/Contra_2_Feb_20_2010.jpg",
    "https://static.independent.co.uk/s3fs-public/thumbnails/image/2016/09/29/15/hp.jpg?quality=75&width=982&height=726&auto=webp",
    "https://trek.scene7.com/is/image/TrekBicycleProducts/TK22_LAUNCH_SS_%20BrandedApparel_B2C_PLP_WomenCyclingApparel?wid=1200",
    "https://www.hollywoodreporter.com/wp-content/uploads/2016/12/the_graduate_-_h_-_2016.jpg",
]


def load_image(url: str) -> Image.Image:
    """Return the image at the specified URL, using a local cache if possible.

    Note that a robust implementation would need to take more care cleaning the image names.
    """
    os.makedirs("./w1d2_images", exist_ok=True)
    filename = os.path.join("./w1d2_images", url.rsplit("/", 1)[1].replace("%20", ""))
    if os.path.exists(filename):
        with open(filename, "rb") as f:
            data = f.read()
    else:
        response = requests.get(url)
        data = response.content
        with open(filename, "wb") as f:
            f.write(data)
    return Image.open(BytesIO(data))


if MAIN:
    images = [load_image(url) for url in tqdm(URLS)]
    display(images[0])
# %%

if MAIN: 
    preprocess = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )


    plt.imshow(rearrange(preprocess(images[0]), "c x y ->x y c"))


# %%


def prepare_data(images: list[Image.Image]) -> t.Tensor:
    """Preprocess each image and stack them into a single tensor.

    Return: shape (batch=len(images), num_channels=3, height=224, width=224)
    """
    sol = t.stack([preprocess(image) for image in images])
    assert sol.shape == (len(images), 3, 224, 224)
    return sol


# %%


with open("w1d2_imagenet_labels.json") as f:
    imagenet_labels = list(json.load(f).values())


# %%


def predict(model, images: list[Image.Image], print_topk_preds=3) -> list[int]:
    """
    Pass the images through the model and print out the top predictions.

    For each image, `display()` the image and the most likely categories according to the model.

    Return: for each image, the index of the top prediction.
    """
    ims = prepare_data(images)
    model.eval()
    with t.inference_mode():
        results = model(ims)
    top_ind = t.topk(results, print_topk_preds).indices
    for (ind, image) in zip(top_ind, images):
        display(image)
        print([(imagenet_labels[i], i) for i in ind])
    return [inds[0].item() for inds in top_ind]


if MAIN:
    model = models.resnet34(pretrained=True)
    pretrained_categories = predict(model, images)
    print(pretrained_categories)

# %%


def einsum_trace(a: t.Tensor) -> t.Tensor:
    """Compute the trace of the square 2D input using einsum."""
    assert len(a.shape) == 2
    return t.einsum("ii -> ", a)


if MAIN:
    w1d2_test_selfmade.test_einsum_trace(einsum_trace)


def as_strided_trace(a: t.Tensor) -> t.Tensor:
    """Compute the trace of the square 2D input using as_strided and sum.

    Tip: the stride argument to `as_strided` must account for the stride of the inputs `a.stride()`.
    """
    assert len(a.shape) == 2
    return t.sum(t.as_strided(a, (a.shape[0],), (a.stride(0) + a.stride(1),)))


if MAIN:
    w1d2_test_selfmade.test_einsum_trace(as_strided_trace)
# %%
def einsum_matmul(a: t.Tensor, b: t.Tensor) -> t.Tensor:
    """Matrix multiply 2D matrices a and b (same as a @ b)."""
    return t.einsum("ij,jk -> ik", a, b)


if MAIN:
    w1d2_test_selfmade.test_matmul(einsum_matmul)


def as_strided_matmul(a: t.Tensor, b: t.Tensor) -> t.Tensor:
    """Matrix multiply 2D matrices a and b (same as a @ b), but use as_strided this time.

    Use elementwise multiplication and sum.

    Tip: the stride argument to `as_strided` must account for the stride of the inputs `a.stride()` and `b.stride()`.
    """
    assert a.shape[1] == b.shape[0]
    rows_rep = t.as_strided(a, (a.shape[0], b.shape[1], a.shape[1]), (a.stride(0), 0, a.stride(1)))
    cols_rep = t.as_strided(b, (a.shape[0], b.shape[1], b.shape[0]), (0, b.stride(1), b.stride(0)))
    return t.sum(rows_rep * cols_rep, dim=2)


if MAIN:
    w1d2_test_selfmade.test_matmul(as_strided_matmul)



# %%
def conv1d_minimal(x: t.Tensor, weights: t.Tensor) -> t.Tensor:
    """Like torch's conv1d using bias=False and all other keyword arguments left at their default values.

    x: shape (batch, in_channels, width)
    weights: shape (out_channels, in_channels, kernel_width)

    Returns: shape (batch, out_channels, output_width)
    """
    out_len = x.shape[2] - weights.shape[2] + 1 
    batch, in_channels, in_len = x.shape
    out_channels, _, kernel_width = weights.shape

    strided_x = t.as_strided(
        x, (batch, in_channels, out_len, kernel_width), (x.stride(0), x.stride(1), x.stride(2), x.stride(2))
    )
    out =  t.einsum("bilk, oik -> bol", strided_x, weights)
    assert out.shape == (batch, out_channels, out_len)
    return out



if MAIN:
    w1d2_test_selfmade.test_conv1d_minimal(conv1d_minimal)

# %%



def conv2d_minimal(x: t.Tensor, weights: t.Tensor) -> t.Tensor:
    """Like torch's conv2d using bias=False and all other keyword arguments left at their default values.

    x: shape (batch, in_channels, height, width)
    weights: shape (out_channels, in_channels, kernel_height, kernel_width)

    Returns: shape (batch, out_channels, output_height, output_width)
    """
    batch, in_channels, height, width = x.shape
    out_channels, in_channels, kernel_height, kernel_width = weights.shape
    out_width = width - kernel_width + 1
    out_height = height - kernel_height + 1

    strided_x = t.as_strided(
        x,
        (batch, in_channels, out_height, out_width, kernel_height, kernel_width),
        (x.stride(0), x.stride(1), x.stride(2), x.stride(3), x.stride(2), x.stride(3)),
    )
    return t.einsum("bihwkl,oikl -> bohw", strided_x, weights)


weights = rearrange(t.tensor([0, 1, 1, 0]), "(x y) -> 1 1 x y", x=2)
x = rearrange(t.arange(9), "(x y) -> 1 1 x y", x=3)
conv2d_minimal(x, weights)

if MAIN:
    w1d2_test_selfmade.test_conv2d_minimal(conv2d_minimal)


# %%


def pad1d(x: t.Tensor, left: int, right: int, pad_value: float) -> t.Tensor:
    """Return a new tensor with padding applied to the edges.

    x: shape (batch, in_channels, width), dtype float32

    Return: shape (batch, in_channels, left + right + width)
    """
    batch, in_channels, width = x.shape
    lzeros = t.zeros(batch, in_channels, left).to(x.device) + pad_value
    lzeros = lzeros.as_strided((batch, in_channels, left), (0,0,1))

    rzeros = t.zeros(batch, in_channels, right).to(x.device) + pad_value
    rzeros = rzeros.as_strided((batch, in_channels, right), (0,0,1))

    return t.cat((lzeros, x, rzeros), dim=-1)


if MAIN:
    w1d2_test_selfmade.test_pad1d(pad1d)


def pad2d(x: t.Tensor, left: int, right: int, top: int, bottom: int, pad_value: float) -> t.Tensor:
    """Return a new tensor with padding applied to the edges.

    x: shape (batch, in_channels, height, width), dtype float32

    Return: shape (batch, in_channels, top + height + bottom, left + width + right)

    Note: this uses a quick fix to have the padded matrix always on the same device as x. Used
    strides on x to copy x with the right size, so don't call large pad values!!!
    """
    batch, in_channels, height, width = x.shape

    lzeros = pad_value * t.ones_like(t.as_strided(x, (batch, in_channels, height, left), (0, 0, 0, 0)))
    rzeros = pad_value * t.ones_like(t.as_strided(x, (batch, in_channels, height, right), (0, 0, 0, 0)))

    nwidth = left + width + right

    tzeros = pad_value * t.ones_like(t.as_strided(x, (batch, in_channels, top, nwidth), (0, 0, 0, 0)))
    bzeros = pad_value * t.ones_like(t.as_strided(x, (batch, in_channels, bottom, nwidth), (0, 0, 0, 0)))

    res = t.cat((lzeros, x, rzeros), dim=3)
    res = t.cat((tzeros, res, bzeros), dim=2)
    return res


if MAIN:
    w1d2_test_selfmade.test_pad2d(pad2d)
# %%
import math


def conv1d(x, weights, stride: int = 1, padding: int = 0) -> t.Tensor:
    """Like torch's conv1d using bias=False.

    x: shape (batch, in_channels, width)
    weights: shape (out_channels, in_channels, kernel_width)

    Returns: shape (batch, out_channels, output_width)
    """
    batch, in_channels, in_len = x.shape
    out_channels, in_channels2, kernel_width = weights.shape
    assert in_channels == in_channels2
    out_len = math.floor((in_len + 2 * padding - kernel_width) / stride) + 1

    x = pad1d(x, padding, padding, 0)
    strided_x = t.as_strided(
        x,
        (batch, in_channels, out_len, kernel_width),
        (x.stride(0), x.stride(1), x.stride(2) * stride, x.stride(2)),
    )
    return t.einsum("bilk, oik -> bol", strided_x, weights)


# if MAIN:
#     w1d2_test.test_conv1d(conv1d)
# %%
IntOrPair = Union[int, tuple[int, int]]
Pair = tuple[int, int]


def force_pair(v: IntOrPair) -> Pair:
    """Convert v to a pair of int, if it isn't already."""
    if isinstance(v, tuple):
        if len(v) != 2:
            raise ValueError(v)
        return (int(v[0]), int(v[1]))
    elif isinstance(v, int):
        return (v, v)
    raise ValueError(v)


def conv2d(x, weights, stride: IntOrPair = 1, padding: IntOrPair = 0) -> t.Tensor:
    """Like torch's conv2d using bias=False

    x: shape (batch, in_channels, height, width)
    weights: shape (out_channels, in_channels, kernel_height, kernel_width)


    Returns: shape (batch, out_channels, output_height, output_width)
    """
    vstride, hstride = force_pair(stride)
    vpadding, hpadding = force_pair(padding)

    batch, in_channels, height, width = x.shape
    out_channels, in_channels, kernel_height, kernel_width = weights.shape
    out_width = math.floor((width + 2 * hpadding - kernel_width) / hstride) + 1
    out_height = math.floor((height + 2 * vpadding - kernel_height) / vstride) + 1

    x = pad2d(x, hpadding, hpadding, vpadding, vpadding, 0)
    strided_x = t.as_strided(
        x,
        (batch, in_channels, out_height, out_width, kernel_height, kernel_width),
        (x.stride(0), x.stride(1), x.stride(2) * vstride, x.stride(3) * hstride, x.stride(2), x.stride(3)),
    )
    return t.einsum("bihwkl,oikl -> bohw", strided_x, weights)


# if MAIN:
#     w1d2_test.test_conv2d(conv2d)
# %%
def maxpool2d(
    x: t.Tensor, kernel_size: IntOrPair, stride: Optional[IntOrPair] = None, padding: IntOrPair = 0
) -> t.Tensor:
    """Like torch's maxpool2d.

    x: shape (batch, channels, height, width)
    stride: if None, should be equal to the kernel size

    Return: (batch, channels, height, width)
    """
    kernel_height, kernel_width = force_pair(kernel_size)
    if stride is None:
        stride = kernel_size
    vstride, hstride = force_pair(stride)
    vpadding, hpadding = force_pair(padding)
    batch, channels, height, width = x.shape
    out_width = math.floor((width + 2 * hpadding - kernel_width) / hstride) + 1
    out_height = math.floor((height + 2 * vpadding - kernel_height) / vstride) + 1

    x = pad2d(x, hpadding, hpadding, vpadding, vpadding, float("-inf"))

    strided_x = t.as_strided(
        x,
        (batch, channels, out_height, out_width, kernel_height, kernel_width),
        (x.stride(0), x.stride(1), x.stride(2) * vstride, x.stride(3) * hstride, x.stride(2), x.stride(3)),
    )
    return t.amax(strided_x, (4, 5))


# if MAIN:
#     w1d2_test.test_maxpool2d(maxpool2d)
# %%


class MaxPool2d(nn.Module):
    def __init__(self, kernel_size: IntOrPair, stride: Optional[IntOrPair] = None, padding: IntOrPair = 1):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def forward(self, x: t.Tensor) -> t.Tensor:
        """Call the functional version of maxpool2d."""
        return maxpool2d(x, self.kernel_size, self.kernel_size, self.padding)

    def extra_repr(self) -> str:
        """Add additional information to the string representation of this class."""
        return f"{self.kernel_size=}, {self.stride=}, {self.padding=}"


# overwrite with stdlib for testing
# MaxPool2d = t.nn.MaxPool2d

# if MAIN:
#     # w1d2_test.test_maxpool2d_module(MaxPool2d)
#     m = MaxPool2d(3, stride=2, padding=1)
#     print(f"Manually verify that this is an informative repr: {m}")


# %%
def init_tensor(shape, num_features):
    fan = math.sqrt(num_features)

    return 2 * (t.rand(shape) - 1 / 2) / fan


class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias=True):
        """A simple linear (technically, affine) transformation.

        The fields should be named `weight` and `bias` for compatibility with PyTorch.
        If `bias` is False, set `self.bias` to None.
        """
        super().__init__()
        fan = math.sqrt(in_features)
        self.weight = nn.Parameter(init_tensor([out_features, in_features], in_features))
        if bias:
            self.bias = nn.Parameter(init_tensor([out_features], in_features))
        else:
            self.bias = None

    def forward(self, x: t.Tensor) -> t.Tensor:
        """
        x: shape (*, in_features)
        Return: shape (*, out_features)
        """

        print(x.shape, self.weight.shape)
        result = t.einsum("...j,ij ->...i", x, self.weight)

        if self.bias is not None:
            result += self.bias
        return result

    def extra_repr(self) -> str:
        return f"{self.weight=}, {self.bias=}"


# Linear = t.nn.Linear

# if MAIN:
#     w1d2_test.test_linear_forward(Linear)
#     w1d2_test.test_linear_parameters(Linear)
#     w1d2_test.test_linear_no_bias(Linear)
# %%
class Conv2d(t.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: IntOrPair,
        stride: IntOrPair = 1,
        padding: IntOrPair = 0,
        bias=False,
    ):
        """Same as torch.nn.Conv2d with bias=False.

        Name your weight field `self.weight` for compatibility with the PyTorch version.
        """
        super().__init__()
        kernel_height, kernel_width = force_pair(kernel_size)
        self.stride = stride
        self.padding = padding

        self.weight = nn.Parameter(
            init_tensor(
                (out_channels, in_channels, kernel_height, kernel_width), in_channels * kernel_width * kernel_height
            )
        )

    def forward(self, x: t.Tensor) -> t.Tensor:
        """Apply the functional conv2d you wrote earlier."""
        #    weights: shape (out_channels, in_channels, kernel_height, kernel_width)

        return conv2d(x, self.weight, self.stride, self.padding)

    def extra_repr(self) -> str:
        """"""
        return f"{self.stride=}, {self.padding=}, {self.weight=}"


# Conv2d = t.nn.Conv2d

# if MAIN:
#     print(f"Manually verify that this is a useful looking repr: {Conv2d(1, 2, (3, 4), padding=5)}")
#     w1d2_test.test_conv2d_module(Conv2d)
# %%


class BatchNorm2d(nn.Module):
    running_mean: t.Tensor
    "running_mean: shape (num_features,)"
    running_var: t.Tensor
    "running_var: shape (num_features,)"
    num_batches_tracked: t.Tensor
    "num_batches_tracked: shape ()"

    def __init__(self, num_features: int, eps=1e-05, momentum=0.1, track_running_stats=True, affine=True):
        """Like nn.BatchNorm2d with track_running_stats=True and affine=True.

        Name the learnable affine parameters `weight` and `bias` in that order.
        """
        super().__init__()
        self.momentum = momentum
        self.weight = nn.Parameter(t.ones(num_features))
        self.bias = nn.Parameter(t.zeros(num_features))
        self.eps = eps
        self.num_features = num_features

        self.register_buffer("running_mean", t.zeros(num_features))
        self.register_buffer("running_var", t.ones(num_features))
        self.register_buffer("num_batches_tracked", t.zeros(()))

    def forward(self, x: t.Tensor) -> t.Tensor:
        """Normalize each channel.

        Compute the variance using `torch.var(x, unbiased=False)`

        x: shape (batch, channels, height, width)
        Return: shape (batch, channels, height, width)
        """
        if self.training:
            v = t.var(x, (0, 2, 3), unbiased=False)
            m = t.mean(x, (0, 2, 3))
            assert len(m.shape) == 1
            assert len(v.shape) == 1
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * m
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * v
            self.num_batches_tracked += 1
        else:
            v = self.running_var
            m = self.running_mean

        m = rearrange(m, "c -> c 1 1")
        v = rearrange(v, "c -> c 1 1")

        scaled = (x - m) / (t.sqrt(v + self.eps))

        wt = rearrange(self.weight, "c -> c 1 1")
        b = rearrange(self.bias, "c -> c 1 1")
        # print("m, v = ", m, v)
        # print("scaled: ", scaled)
        # print("self.training", self.training)
        # print("x = ", x)
        # print("wt = ", wt)
        # print("b = ", b)
        return scaled * wt + b

    def extra_repr(self) -> str:
        return f"{self.momentum=}, {self.weight=}, {self.bias=}, {self.eps=}, {self.running_mean=}, {self.running_var=}, {self.num_batches_tracked=}"


# overwrite with stdlib for testing
# import functools
# BatchNorm2d = functools.partial(t.nn.BatchNorm2d, track_running_stats=True, affine=True)

# if MAIN:
#     w1d2_test.test_batchnorm2d_module(BatchNorm2d)
#     w1d2_test.test_batchnorm2d_forward(BatchNorm2d)
#     w1d2_test.test_batchnorm2d_running_mean(BatchNorm2d)
# %%


class ReLU(nn.Module):
    def forward(self, x: t.Tensor) -> t.Tensor:
        return t.maximum(x, t.zeros_like(x))


# %%
class Sequential(nn.Module):
    def __init__(self, *modules: nn.Module):
        """
        Call `self.add_module` on each provided module, giving each one a unique (within this Sequential) name.
        Internally, this adds them to the dictionary `self._modules` in the base class, which means they'll be included in self.parameters() as desired.
        """
        super().__init__()
        for i, module in enumerate(modules):
            self.add_module(f"{i}", module)
        self.mymodules = modules

    def forward(self, x: t.Tensor) -> t.Tensor:
        """Chain each module together, with the output from one feeding into the next one."""
        for module in self.mymodules:
            x = module(x)
        return x


# overwrite with stdlib for testing
# Sequential = t.nn.Sequential

# if MAIN:
#     w1d2_test.test_sequential(Sequential)
#     w1d2_test.test_sequential_forward(Sequential)

# %%
class Flatten(nn.Module):
    def __init__(self, start_dim: int = 1, end_dim: int = -1) -> None:
        super().__init__()
        self.start_dim = start_dim
        self.end_dim = end_dim

    def forward(self, input: t.Tensor) -> t.Tensor:
        """Flatten out dimensions from start_dim to end_dim, inclusive of both.

        Return a view if possible, otherwise a copy.
        """
        if self.end_dim == -1:
            end_dim = len(input.shape)
        else:
            end_dim = self.end_dim
        to_delete = input.shape[self.start_dim : end_dim + 1]
        before, after = input.shape[: self.start_dim], input.shape[end_dim + 1 :]
        new = list(before) + [int(t.prod(t.tensor(to_delete)).item())] + list(after)
        return t.reshape(input, new)

    def extra_repr(self) -> str:
        pass


# if MAIN:
#     w1d2_test.test_flatten(Flatten)
#     w1d2_test.test_flatten_is_view(Flatten)
# %%
class AveragePool(nn.Module):
    def forward(self, x: t.Tensor) -> t.Tensor:
        """
        x: shape (batch, channels, height, width)
        Return: shape (batch, channels)
        """
        return t.mean(x, (2, 3), keepdim=False)


# %%
class BasicBlock(nn.Module):
    def __init__(self, in_feats: int, out_feats: int, first_stride=1):
        """A single residual block with optional downsampling.

        For compatibility with the pretrained model, declare the left side branch first using a `Sequential`.

        If first_stride is > 1, this means the optional (conv + bn) should be present on the right branch. Declare it second using another `Sequential`.
        """
        super().__init__()

        self.conv1 = Conv2d(in_feats, out_feats, (3, 3), stride=first_stride, padding=(1, 1), bias=False)
        self.bn1 = BatchNorm2d(out_feats, track_running_stats=True, affine=True)
        self.conv2 = Conv2d(out_feats, out_feats, (3, 3), padding=(1, 1), bias=False)
        self.bn2 = BatchNorm2d(out_feats, track_running_stats=True, affine=True)
        if first_stride > 1:
            self.downsample = Sequential(
                Conv2d(in_feats, out_feats, (1, 1), stride=first_stride, bias=False),
                BatchNorm2d(out_feats, track_running_stats=True, affine=True),
            )
        else:
            self.downsample = None

    def forward(self, x: t.Tensor) -> t.Tensor:
        """Compute the forward pass.

        x: shape (batch, in_feats, height, width)

        Return: shape (batch, out_feats, height / stride, width / stride)

        If no downsampling block is present, the addition should just add the left branch's output to the input.
        """
        left = self.bn2(self.conv2(ReLU()(self.bn1(self.conv1(x)))))
        if self.downsample is not None:
            right = self.downsample(x)
        else:
            right = x
        return ReLU()(left + right)


class BlockGroup(nn.Module):
    def __init__(self, n_blocks: int, in_feats: int, out_feats: int, first_stride=1):
        """An n_blocks-long sequence of BasicBlock where only the first block uses the provided stride."""
        super().__init__()
        blocks = [BasicBlock(in_feats, out_feats, first_stride)]
        blocks.extend(BasicBlock(out_feats, out_feats) for _ in range(n_blocks - 1))
        self.blocks = Sequential(*blocks)

    def forward(self, x: t.Tensor) -> t.Tensor:
        """Compute the forward pass.
        x: shape (batch, in_feats, height, width)

        Return: shape (batch, out_feats, height / first_stride, width / first_stride)
        """
        return self.blocks(x)


def block_group(n_blocks: int, in_feats: int, out_feats: int, first_stride=1):
    blocks = [BasicBlock(in_feats, out_feats, first_stride)]
    blocks.extend(BasicBlock(out_feats, out_feats) for _ in range(n_blocks - 1))
    return Sequential(*blocks)


class ResNet34(nn.Module):
    def __init__(
        self,
        n_blocks_per_group=[3, 4, 6, 3],
        out_features_per_group=[64, 128, 256, 512],
        strides_per_group=[1, 2, 2, 2],
        n_classes=1000,
    ):
        super().__init__()

        self.conv1 = Conv2d(3, 64, kernel_size=(7, 7), stride=2, padding=(3, 3), bias=False)
        self.bn1 = BatchNorm2d(64, track_running_stats=True, affine=True)
        self.relu = ReLU()
        self.pool = MaxPool2d(kernel_size=(3, 3), stride=(3, 3))
        in_features_per_group = [64] + out_features_per_group[:-1]
        for i, (n_blocks, in_feat, out_feat, stride) in enumerate(
            zip(n_blocks_per_group, in_features_per_group, out_features_per_group, strides_per_group)
        ):
            setattr(self, f"layer{i+1}", block_group(n_blocks, in_feat, out_feat, stride))
        self.number_of_blocks = len(n_blocks_per_group)

        self.avgpool = AveragePool()
        self.fc = t.nn.Linear(out_features_per_group[-1], n_classes)

    def forward(self, x: t.Tensor) -> t.Tensor:
        """
        x: shape (batch, channels, height, width)

        Return: shape (batch, n_classes)
        """
        rep = self.pool(self.relu(self.bn1(self.conv1(x))))
        for i in range(self.number_of_blocks):
            rep = getattr(self, f"layer{i+1}")(rep)
        rep = self.avgpool(rep)
        rep = Flatten()(rep)
        return self.fc(rep)


if MAIN:
    your_model = ResNet34()
    your_model.load_state_dict(models.resnet34(pretrained=True).state_dict(), strict=True)

    print("conv1.weight", dict(your_model.state_dict().items())["conv1.weight"][0])
    print("load_failures", your_model.load_state_dict(models.resnet34(pretrained=True).state_dict(), strict=True))
 


def check_nan_hook(module: nn.Module, inputs, output):
    """Example of a hook function that can be registered to a module."""
    x = inputs[0]
    if t.isnan(x).any():
        raise ValueError(module, x)
    out = output[0]
    if t.isnan(out).any():
        raise ValueError(module, x)


def add_hook(module: nn.Module) -> None:
    """Remove any existing hooks and register our hook.

    Use model.apply(add_hook) to recursively apply the hook to model and all submodules.
    """
    utils.remove_hooks(module)
    module.register_forward_hook(check_nan_hook)


if MAIN:
    your_model.apply(add_hook)
    your_model_predictions = predict(your_model, images)
    # w1d2_test.test_same_predictions(your_model_predictions)


# if MAIN:
#     diffs = [(k1, v1, k2, v2) for (k1, v1), (k2, v2) in zip(your_model.state_dict().items(), models.resnet34().state_dict().items()) if not t.all(v1 == v2)]
#     delta = [v2 - v1 for (k1, v1), (k2, v2) in zip(your_model.state_dict().items(), models.resnet34().state_dict().items()) if not t.all(v1 == v2)]
#     print("diffs")
#     for k1, _, _, _ in diffs:
#         print(k1)
#     your_model.apply(add_hook)
#     your_model_predictions = predict(your_model, images[:10])
# w1d2_test.test_same_predictions(your_model_predictions)
# %%

cifar_classes = {
    0: "airplane",
    1: "automobile",
    2: "bird",
    3: "cat",
    4: "deer",
    5: "dog",
    6: "frog",
    7: "horse",
    8: "ship",
    9: "truck",
}


def get_cifar10():
    """Download (if necessary) and return the CIFAR10 dataset."""
    "The following is a workaround for this bug: https://github.com/pytorch/vision/issues/5039"
    # if sys.platform == "win32":
    if sys.platform == "darwin":
        import ssl
        ssl._create_default_https_context = ssl._create_unverified_context

    "Magic constants taken from: https://docs.ffcv.io/ffcv_examples/cifar10.html"
    mean = t.tensor([125.307, 122.961, 113.8575]) / 255
    std = t.tensor([51.5865, 50.847, 51.255]) / 255

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
    cifar_train = torchvision.datasets.CIFAR10("w1d2_cifar10_train", transform=transform, download=True, train=True)
    cifar_test = torchvision.datasets.CIFAR10("w1d2_cifar10_train", transform=transform, download=True, train=False)

    return (cifar_train, cifar_test)


if MAIN:
    (cifar_train, cifar_test) = get_cifar10()
    trainloader = DataLoader(cifar_train, batch_size=256, shuffle=True, pin_memory=True)
    testloader = DataLoader(cifar_test, batch_size=256, pin_memory=True)
if MAIN:
    batch = next(iter(trainloader))
    print("Mean value of each channel: ", batch[0].mean((0, 2, 3)))
    print("Std value of each channel: ", batch[0].std((0, 2, 3)))
    (fig, axes) = plt.subplots(ncols=5, figsize=(15, 5))
    for (i, ax) in enumerate(axes):
        ax.imshow(rearrange(batch[0][i], "c h w -> h w c"))
        ax.set(xlabel=cifar_classes[batch[1][i].item()])

# %%


MODEL_FILENAME = "./w1d2_resnet34_cifar10.pt"
device = "cuda" if t.cuda.is_available() else "cpu"
print("device: ", device)


def train(trainloader: DataLoader, epochs: int) -> ResNet34:
    model = ResNet34(n_classes=10).to(device).train()
    # print(next(network.parameters()).device)
    optimizer = t.optim.Adam(model.parameters())
    loss_fn = t.nn.CrossEntropyLoss()
    for epoch in range(epochs):
        for (i, (x, y)) in enumerate(tqdm(trainloader)):
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            y_hat = model(x)
            loss = loss_fn(y_hat, y)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch}, train loss is {loss}")
        print(f"Saving model to: {os.path.abspath(MODEL_FILENAME)}")
        t.save(model, MODEL_FILENAME)
    return model


# if MAIN:
#     if os.path.exists(MODEL_FILENAME):
#         print("Loading model from disk: ", MODEL_FILENAME)
#         model = t.load(MODEL_FILENAME)
#     else:

if MAIN:
    print("Training model from scratch")
    model = train(trainloader, epochs=8)

# %%
