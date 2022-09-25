from typing import Iterable, Union, Optional, Any
import time
import argparse
import torch as t
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import wandb
from w1d2_solution import ResNet34, get_cifar10
from w1d4_part1_solution import Adam


MAIN = __name__ == "__main__"
device = t.device("cuda" if t.cuda.is_available() else "cpu")


def train(config_dict: dict[str, Any]):
    wandb.init(project="w1d4", config=config_dict)
    config = wandb.config
    print(f"Training with config: {config}")
    (cifar_train, cifar_test) = get_cifar10()
    trainloader = DataLoader(cifar_train, batch_size=config.batch_size, shuffle=True, pin_memory=True)
    testloader = DataLoader(cifar_test, batch_size=1024, pin_memory=True)
    model = ResNet34(n_blocks_per_group=[1, 1, 1, 1], n_classes=10).to(device).train()
    optimizer = Adam(
        model.parameters(), lr=config.lr, betas=(config.beta_0, config.beta_1), weight_decay=config.weight_decay
    )
    train_loss_fn = t.nn.CrossEntropyLoss()
    test_loss_fn = t.nn.CrossEntropyLoss(reduction="sum")
    wandb.watch(model, criterion=train_loss_fn, log="all", log_freq=10, log_graph=True)
    start_time = time.time()
    examples_seen = 0
    for epoch in range(config.epochs):
        # optimizer.lr = optimizer.lr * (1 / 5**epoch)
        # print("using learning rate", optimizer.lr)
        if epoch == 2:
            optimizer.lr = 0

        for (i, (x, y)) in enumerate(tqdm(trainloader)):
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            y_hat = model(x)
            loss = train_loss_fn(y_hat, y)
            acc = (y_hat.argmax(dim=-1) == y).sum() / len(x)
            loss.backward()
            optimizer.step()
            wandb.log(dict(train_loss=loss, train_accuracy=acc, elapsed=time.time() - start_time), step=examples_seen)
            examples_seen += len(x)
    with t.inference_mode():
        n_correct = 0
        n_total = 0
        loss_total = 0.0
        for (i, (x, y)) in enumerate(tqdm(testloader)):
            x = x.to(device)
            y = y.to(device)
            y_hat = model(x)
            loss_total += test_loss_fn(y_hat, y).item()
            n_correct += (y_hat.argmax(dim=-1) == y).sum().item()
            n_total += len(x)
        wandb.log(dict(test_loss=loss_total / n_total, test_accuracy=n_correct / n_total, step=examples_seen))
    filename = f"{wandb.run.dir}/model_state_dict.pt"
    t.save(model.state_dict(), filename)
    wandb.save(filename)


if MAIN:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--weight_decay", type=float, default=0)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--beta_0", type=float, default=0.9)
    parser.add_argument("--beta_1", type=float, default=0.999)
    parser.add_argument("--cuda_memory_fraction", type=float, default=0.5)
    # parser.add_argument("--lr_param", type=int, default=1)
    args = parser.parse_args()
    config_dict = vars(args)
    t.cuda.set_per_process_memory_fraction(config_dict.pop("cuda_memory_fraction"))
    train(config_dict)


# on batch size of 1024, optimal learning rate is between  0.0009 and 0.003 for two epochs with constant learning rate
# with halfing learning rate after first epoch, optimal lr is around 0.001904
# dividing learning rate by 10 on second epoch does not affect final test accuracy (67.5% test accuracy)
# on 3 epochs, dividing rate by 3**epoch gives optimal learning rate 0.0013818410273131712. Accuracy 72.8%
