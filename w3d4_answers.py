#%%

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import reduce
from operator import mul
from typing import Any, Optional, Union
import matplotlib.pyplot as plt
import torch as t
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import utils
import wandb
import math


MAIN = __name__ == "__main__"
device = "cuda:0" if t.cuda.is_available() else "cpu"

# %%


def gradient_images(n_images: int, img_size: tuple[int, int, int]) -> t.Tensor:
    """
    Generate n_images of img_size, each a color gradient
    """
    (C, H, W) = img_size
    corners = t.randint(0, 255, (2, n_images, C))
    xs = t.linspace(0, W / (W + H), W)
    ys = t.linspace(0, H / (W + H), H)
    (x, y) = t.meshgrid(xs, ys, indexing="xy")
    grid = x + y
    grid = grid / grid[-1, -1]
    grid = repeat(grid, "h w -> b c h w", b=n_images, c=C)
    base = repeat(corners[0], "n c -> n c h w", h=H, w=W)
    ranges = repeat(corners[1] - corners[0], "n c -> n c h w", h=H, w=W)
    gradients = base + grid * ranges
    assert gradients.shape == (n_images, C, H, W)
    return gradients / 255


def plot_img(img: t.Tensor, title: Optional[str] = None) -> None:
    img = rearrange(img, "c h w -> h w c")
    plt.imshow(img.numpy())
    if title:
        plt.title(title)
    plt.show()


if MAIN:
    img_shape = (3, 16, 16)
    n_images = 5
    imgs = gradient_images(n_images, img_shape)
    for i in range(n_images):
        plot_img(imgs[i])


def normalize_img(img: t.Tensor) -> t.Tensor:
    return img * 2 - 1


def denormalize_img(img: t.Tensor) -> t.Tensor:
    return ((img + 1) / 2).clamp(0, 1)


# %%
def linear_schedule(max_steps: int, min_noise: float = 0.0001, max_noise: float = 0.02) -> t.Tensor:
    """Return the forward process variances as in the paper.

    max_steps: total number of steps of noise addition
    out: shape (step=max_steps, ) the amount of noise at each step
    """
    return t.linspace(min_noise, max_noise, max_steps)


if MAIN:
    betas = linear_schedule(max_steps=200)
# %%
def q_eq2(x: t.Tensor, num_steps: int, betas: t.Tensor) -> t.Tensor:
    """
    Equation (2) in "Denoising Diffusion Probabilistic Models"
    Return the input image with num_steps iterations of noise added according to schedule.
    x: shape (channels, height, width)
    schedule: shape (T, ) with T >= num_steps

    out: shape (channels, height, width)
    """
    for step in range(num_steps):
        x = t.normal((1 - betas[step]) ** 0.5 * x, betas[step] ** 0.5)
    return x


if MAIN:
    x = gradient_images(1, (3, 16, 16))[0]
    for n in [1, 10, 50, 200]:
        xt = q_eq2(x, n, betas)
        plot_img(xt, f"Equation 2 after {n} step(s)")

# %%
def q_eq4(x: t.Tensor, num_steps: int, betas: t.Tensor) -> t.Tensor:
    """
    Equation (4) in "Denoising Diffusion Probabilistic Models"
    Equivalent to Equation (2) but without a for loop."""
    alpha = 1 - betas[:num_steps]
    product = t.prod(alpha)
    return t.normal((product**0.5) * x, (1 - product) ** 0.5)


if MAIN:
    for n in [1, 10, 50, 200]:
        xt = q_eq4(x, n, betas)
        plot_img(xt, f"Equation 4 after {n} steps")
# %%


class NoiseSchedule(nn.Module):
    betas: t.Tensor
    alphas: t.Tensor
    alpha_bars: t.Tensor

    def __init__(
        self, max_steps: int, device: Union[t.device, str], min_noise: float = 0.0001, max_noise: float = 0.02
    ) -> None:
        super().__init__()
        self.max_steps = max_steps
        self.device = device

        self.min_noise = min_noise
        self.max_noise = max_noise

        self.betas = linear_schedule(self.max_steps, min_noise=self.min_noise, max_noise=self.max_noise).to(self.device)
        self.alphas = (1 - self.betas).to(self.device)
        self.alpha_bars = t.cumprod((1 - self.betas), dim=-1).to(self.device)

    @t.inference_mode()
    def beta(self, num_steps: Union[int, t.Tensor]) -> t.Tensor:
        """
        Returns the beta(s) corresponding to a given number of noise steps
        num_steps: int or int tensor of shape (batch_size,)
        Returns a tensor of shape (batch_size,), where batch_size is one if num_steps is an int
        """
        if isinstance(num_steps, int):
            num_steps = t.tensor([num_steps]).to(self.device)
        return self.betas[num_steps]

    @t.inference_mode()
    def alpha(self, num_steps: Union[int, t.Tensor]) -> t.Tensor:
        """
        Returns the alphas(s) corresponding to a given number of noise steps
        num_steps: int or int tensor of shape (batch_size,)
        Returns a tensor of shape (batch_size,), where batch_size is one if num_steps is an int
        """
        if isinstance(num_steps, int):
            num_steps = t.tensor([num_steps]).to(self.device)
        return self.alphas[num_steps]

    @t.inference_mode()
    def alpha_bar(self, num_steps: Union[int, t.Tensor]) -> t.Tensor:
        """
        Returns the alpha_bar(s) corresponding to a given number of noise steps
        num_steps: int or int tensor of shape (batch_size,)
        Returns a tensor of shape (batch_size,), where batch_size is one if num_steps is an int
        """
        if isinstance(num_steps, int):
            num_steps = t.tensor([num_steps]).to(self.device)
        return self.alpha_bars[num_steps]

    def __len__(self) -> int:
        return self.max_steps



if MAIN:
    noisy = NoiseSchedule(200, device)

# %%
def noise_img(
    img: t.Tensor, noise_schedule: NoiseSchedule, max_steps: Optional[int] = None
) -> tuple[t.Tensor, t.Tensor, t.Tensor]:
    """
    Adds a random number of steps of noise to each image in img.

    img: An image tensor of shape (B, C, H, W)
    noise_schedule: The NoiseSchedule to follow
    max_steps: an optional parameter to pass if you'd like to add less noise than the maximum allowed by the noise schedule

    Returns a tuple composed of:
    num_steps: an int tensor of shape (B,) of the number of steps of noise added to each image
    noise: the unscaled, standard Gaussian noise added to each image, a tensor of shape (B, C, H, W)
    noised: the final noised image, a tensor of shape (B, C, H, W)
    """
    (B, C, H, W) = img.shape

    # Set max steps
    if max_steps is None:
        max_steps = noise_schedule.max_steps

    # num_steps: sample B number of integers from 0 to max_steps
    num_steps = t.randint(0, max_steps, (B,)).to(img.device)

    # noise: generated from standard Gaussian
    noise = t.normal(t.zeros_like(img), 1).to(img.device)

    # scaled_noise
    alpha_bars = noise_schedule.alpha_bar(num_steps).reshape((B, 1, 1, 1))
    scaled_noise = (1 - alpha_bars) ** 0.5 * noise

    # add to images
    noisy_img = scaled_noise + (alpha_bars**0.5) * img

    return (num_steps, noise, noisy_img)


if MAIN:
    max_steps = 200
    noise_schedule = NoiseSchedule(max_steps, "cpu")
    img = gradient_images(1, (3, 16, 16))
    (num_steps, noise, noised) = noise_img(img, noise_schedule, max_steps=100)
    plot_img(img[0], "Gradient")
    plot_img(noise[0], "Applied Unscaled Noise")
    plot_img(noised[0], "Gradient with Noise Applied")


# %%
def reconstruct(noisy_img: t.Tensor, noise: t.Tensor, num_steps: t.Tensor, noise_schedule: NoiseSchedule) -> t.Tensor:
    """
    Subtract the scaled noise from noisy_img to partly recover the original image.

    Returns img, a tensor with shape (B, C, H, W)
    """
    (B, C, H, W) = noisy_img.shape

    alpha_bars = noise_schedule.alpha_bar(num_steps).reshape((B, 1, 1, 1))
    scaled_noise = (1 - alpha_bars) ** 0.5 * noise

    return (noisy_img - scaled_noise) / (alpha_bars**0.5)


if MAIN:
    reconstructed = reconstruct(noised, noise, num_steps, noise_schedule)
    plot_img(img[0], "Original Gradient")
    plot_img(reconstructed[0], "Reconstruction")
    utils.allclose(reconstructed, img)

# %%
class DiffusionModel(nn.Module, ABC):
    img_shape: tuple
    noise_schedule: Optional[NoiseSchedule]

    @abstractmethod
    def forward(self, images: t.Tensor, num_steps: t.Tensor) -> t.Tensor:
        ...


@dataclass
class TinyDiffuserConfig:
    img_shape: tuple[int, ...]
    hidden_size: int
    max_steps: int


class TinyDiffuser(DiffusionModel):
    def __init__(self, config: TinyDiffuserConfig):
        """
        A toy diffusion model composed of an MLP (Linear, ReLU, Linear)
        """
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.img_shape = config.img_shape
        self.noise_schedule = None
        self.max_steps = config.max_steps
        self.im_size = math.prod(self.img_shape)
        self.MLP = nn.Sequential(
            nn.Linear(self.im_size + 1, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.im_size),
        )

    def forward(self, images: t.Tensor, num_steps: t.Tensor) -> t.Tensor:
        """
        Given a batch of images and noise steps applied, attempt to predict the noise that was applied.
        images: tensor of shape (B, C, H, W)
        num_steps: tensor of shape (B,)

        Returns
        noise_pred: tensor of shape (B, C, H, W)
        """

        assert isinstance(num_steps, t.Tensor)

        (B, C, H, W) = images.shape
        scaled_num_steps = (num_steps / self.max_steps).unsqueeze(-1)
        flat_imgs = nn.Flatten(start_dim=1, end_dim=-1)(images)

        input = t.cat((flat_imgs, scaled_num_steps), dim=-1)
        assert input.shape[0] == images.shape[0]  # number of batches preserved
        out = self.MLP(input)
        return rearrange(out, "b (c h w) -> b c h w", c=C, h=H, w=W)

    def set_scheduler(self, device, min_noise: float = 0.0001, max_noise: float = 0.02):
        self.noise_schedule = NoiseSchedule(self.max_steps, device=device, min_noise=min_noise, max_noise=max_noise)


if MAIN:
    img_shape = (3, 4, 5)
    n_images = 5
    imgs = gradient_images(n_images, img_shape)
    n_steps = t.zeros(imgs.size(0))+3
    model_config = TinyDiffuserConfig(img_shape, 16, 100)
    model = TinyDiffuser(model_config)
    out = model(imgs, n_steps)
    plot_img(out[1].detach(), "Noise prediction of untrained model")

# %%
def log_images(
    img: t.Tensor, noised: t.Tensor, noise: t.Tensor, noise_pred: t.Tensor, reconstructed: t.Tensor, num_images: int = 3
) -> list[wandb.Image]:
    """
    Convert given tensors to a format amenable to logging to Weights and Biases. Returns an image with the ground truth in the upper row, and model reconstruction on the bottom row. Left is the noised image, middle is noise, and reconstructed image is in the rightmost column.
    """
    actual = t.cat((noised, noise, img), dim=-1)
    pred = t.cat((noised, noise_pred, reconstructed), dim=-1)
    log_img = t.cat((actual, pred), dim=-2)
    images = [wandb.Image(i) for i in log_img[:num_images]]
    return images


def train(
    model: DiffusionModel, config_dict: dict[str, Any], trainset: TensorDataset, testset: Optional[TensorDataset] = None
) -> DiffusionModel:

    wandb.init(project = config_dict["project_name"], config=config_dict)
    config = wandb.config
    print(f"Training with config: {config}")

    train_loader = DataLoader(trainset, config["batch_size"], shuffle=True)
    my_scheduler = NoiseSchedule(config["max_steps"], device=config["device"])

    optimizer = t.optim.Adam(model.parameters(), lr=config["lr"])
    loss_fn = nn.MSELoss()
    wandb.watch(model, criterion=loss_fn, log="all", log_freq=10, log_graph=True)
    start_time = time.time()
    examples_seen = 0
    for epoch in range(config["epochs"]):

        for i, train_images in enumerate(tqdm(train_loader)):
            train_images = train_images[0].to(device)   #train_images is a tuple of length 1
            num_steps, noise, noisy_image = noise_img(train_images, my_scheduler)
            model.train()
            predict = model(noisy_image, num_steps)
            loss = loss_fn(predict, noise)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            info: dict[str, Any] = dict(
                train_loss=loss,
                elapsed=time.time() - start_time,
                noisy_image_mean=noisy_image.mean(),
                noisy_image_var=noisy_image.var(),
            )

            examples_seen += config["batch_size"]

            if i % config["img_log_interval"] == 0:
                info["train_images"] = log_images(
                    train_images,
                    noisy_image,
                    noise,
                    predict,
                    reconstruct(noisy_image, noise, num_steps, my_scheduler).to(device),
                    config["n_images_to_log"],
                )
            wandb.log(info, step=examples_seen)

        if testset is not None:
            losses = []
            model.eval()
            with t.inference_mode():
                for i, test_images in enumerate(tqdm(DataLoader(testset, config["batch_size"]), desc = f"Test for Epoch {epoch}")):
                    test_images = test_images[0].to(device)
                    num_steps, noise, noisy_image = noise_img(test_images, my_scheduler)
                    predict = model(noisy_image, num_steps)
                    loss = loss_fn(predict, noise)
                    losses.append(loss.item())
            info: dict[str, Any] = dict(
                test_loss=sum(losses),
                test_elapsed=time.time() - start_time,
                test_noisy_image_mean=noisy_image.mean(),
                test_noisy_image_var=noisy_image.var(),
            )
            wandb.log(info, step=examples_seen)

    eval_images = gradient_images(config["n_images_to_log"], config["image_shape"]).to(device)
    eval_num_steps, eval_noise, eval_noisy_images = noise_img(eval_images, my_scheduler)
    eval_reconstruct = reconstruct(eval_noisy_images, eval_noise, eval_num_steps, my_scheduler).to(
        device
    )  

    print("doing eval")
    model.eval()
    with t.inference_mode():
        eval_predict = model(eval_noisy_images, eval_num_steps)

    info["eval_images"] = log_images(
        eval_images, eval_noisy_images, eval_noise, eval_predict, eval_reconstruct, config["n_images_to_log"]
    )
    wandb.log(info, step=examples_seen)
    # wandb.finish()
    return model


 
#%%
if MAIN:
    # config: dict[str, Any] = dict(
    #     project_name = "gradient_diffusion_models"
    #     lr=1e-3,
    #     image_shape=(3, 4, 5),
    #     hidden_size=128,
    #     epochs=20,
    #     max_steps=100,
    #     batch_size=128,
    #     img_log_interval=200,
    #     n_images_to_log=3,
    #     n_images=50000,
    #     n_eval_images=1000,
    #     device=t.device("cuda") if t.cuda.is_available() else t.device("cpu")
    # )
    config: dict[str, Any] = dict(
        project_name = "gradient_diffusion_models",
        lr=1e-3,
        image_shape=(3, 4, 5),
        hidden_size=256,
        epochs=20,
        max_steps=200,
        batch_size=128,
        img_log_interval=200,
        n_images_to_log=3,
        n_images=80000,
        n_eval_images=1000,
        device=t.device("cuda") if t.cuda.is_available() else t.device("cpu")
    )

    images = normalize_img(gradient_images(config["n_images"], config["image_shape"]))
    train_set = TensorDataset(images)
    images = normalize_img(gradient_images(config["n_eval_images"], config["image_shape"]))
    test_set = TensorDataset(images)
    model_config = TinyDiffuserConfig(config["image_shape"], config["hidden_size"], config["max_steps"])
    model = TinyDiffuser(model_config).to(config["device"])
    model = train(model, config, train_set, test_set)






# %%
def sample(model: DiffusionModel, n_samples: int, return_all_steps: bool = False) -> Union[t.Tensor, list[t.Tensor]]:
    """
    Sample, following Algorithm 2 in the DDPM paper

    model: The trained noise-predictor
    n_samples: The number of samples to generate
    denoise_steps: The number of denoising steps to apply, if different than the maximum in the noise schedule. If it is, the denoising steps taken should be evenly spaced accross the noise schedule
    return_all_steps: if true, return a list of the reconstructed tensors generated at each step, rather than just the final reconstructed image tensor.

    out: shape (B, C, H, W), the denoised images
    """
    schedule = NoiseSchedule(config["max_steps"], device=config["device"])
    assert schedule is not None

    # Sample initial Gaussian noises
    x_t = t.normal(t.zeros((n_samples, *model.img_shape)), 1).to(config["device"])

    # Setup output for all steps
    output = [x_t]

    # Loop down
    for step in range(model.max_steps - 1, -1, -1):
        if step != 0:
            z = t.normal(t.zeros((n_samples, *model.img_shape)), 1).to(config["device"])
        else:
            z = 0

        # Iterate
        beta_t = schedule.beta(step)
        alpha_t = schedule.alpha(step)
        alpha_bar_t = schedule.alpha_bar(step)

        step_tensor = t.tensor([1+step], device=config["device"])
        step_tensor = repeat(step_tensor, "a -> (rep a)", rep=x_t.shape[0])

        x_t = (beta_t**0.5) * z + (1 / (1e-6 + (alpha_t**0.5))) * (
            x_t - (1 - alpha_t) * model(x_t, step_tensor) / (1e-6 + ((1 - alpha_bar_t) ** 0.5))
        )
 
        # Add to output
        output.append(x_t)

    if return_all_steps:
        return [x_t.cpu() for x_t in output]
    else:
        return output[-1].cpu()


if MAIN:
    print("Generating multiple images")
    assert isinstance(model, DiffusionModel)
    with t.inference_mode():
        samples = sample(model, 5)
    for s in samples:
        plot_img(denormalize_img(s).cpu())

#%%
if MAIN:
    print("Printing sequential denoising")
    assert isinstance(model, DiffusionModel)
    with t.inference_mode():
        samples = sample(model, 1, return_all_steps=True)
    for (i, s) in enumerate(samples):
        if i % (len(samples) // 20) == 0:
            plot_img(denormalize_img(s[0]), f"Step {i}")

# %%
