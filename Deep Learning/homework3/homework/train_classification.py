import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.utils.tensorboard as tb

from .models import load_model, save_model

import csv
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

LABEL_NAMES = ["background", "kart", "pickup", "nitro", "bomb", "projectile"]

from pathlib import Path

import torch
import torch.nn as nn


class ClassificationLoss(nn.Module):
    def forward(self, logits: torch.Tensor, target: torch.LongTensor) -> torch.Tensor:
        """
        Multi-class classification loss
        Hint: simple one-liner

        Args:
            logits: tensor (b, c) logits, where c is the number of classes
            target: tensor (b,) labels

        Returns:
            tensor, scalar loss
        """

        return nn.CrossEntropyLoss()(logits, target)


class SuperTuxDataset(Dataset):
    def __init__(self, dataset_path: str):
        """
        Pairs of images and labels (int) for classification
        You won't need to modify this, but all PyTorch datasets must implement these methods
        """
        to_tensor = transforms.ToTensor()

        self.data = []

        with open(Path(dataset_path, "labels.csv"), newline="") as f:
            for fname, label, _ in csv.reader(f):
                if label in LABEL_NAMES:
                    image = Image.open(Path(dataset_path, fname))
                    label_id = LABEL_NAMES.index(label)

                    self.data.append((to_tensor(image), label_id))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def load_data(dataset_path: str, num_workers: int = 0, batch_size: int = 128, shuffle: bool = False) -> DataLoader:
    dataset = SuperTuxDataset(dataset_path)

    return DataLoader(dataset, num_workers=num_workers, batch_size=batch_size, shuffle=shuffle, drop_last=True)


def compute_accuracy(outputs: torch.Tensor, labels: torch.Tensor):
    """
    Arguments:
        outputs: torch.Tensor, shape (b, num_classes) either logits or probabilities
        labels: torch.Tensor, shape (b,) with the ground truth class labels

    Returns:
        a single torch.Tensor scalar
    """
    outputs_idx = outputs.max(1)[1].type_as(labels)

    return (outputs_idx == labels).float().mean()


def train(
    exp_dir: str = "logs",
    model_name: str = "linear",
    num_epoch: int = 50,
    lr: float = 1e-3,
    batch_size: int = 128,
    seed: int = 2024,
    **kwargs,
):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        print("CUDA not available, using CPU")
        device = torch.device("cpu")

    # set random seed so each run is deterministic
    torch.manual_seed(seed)
    np.random.seed(seed)

    # directory with timestamp to save tensorboard logs and model checkpoints
    log_dir = Path(exp_dir) / f"{model_name}_{datetime.now().strftime('%m%d_%H%M%S')}"
    logger = tb.SummaryWriter(log_dir)

    # note: the grader uses default kwargs, you'll have to bake them in for the final submission
    model = load_model(model_name, **kwargs)
    model = model.to(device)
    model.train()

    train_data = load_data("classification_data/train", shuffle=True, batch_size=batch_size, num_workers=2)
    val_data = load_data("classification_data/val", shuffle=False)

    # create loss function and optimizer
    loss_func = ClassificationLoss()
    # optimizer = ...
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    global_step = 0

    # training loop
    for epoch in range(num_epoch):
        metrics = {"train_acc": [], "val_acc": []}

        for img, label in train_data:
            img, label = img.to(device), label.to(device)

            optimizer.zero_grad()
            output = model(img)
            loss = loss_func(output, label)
            loss.backward()
            optimizer.step()

            # Compute training accuracy
            _, predicted = torch.max(output, 1)
            correct = (predicted == label).sum().item()
            accuracy = correct / img.size(0)
            metrics["train_acc"].append(accuracy)

            global_step += 1

        # torch.inference_mode calls model.eval() and disables gradient computation
        with torch.inference_mode():
            for img, label in val_data:
                img, label = img.to(device), label.to(device)

                output = model(img)
                _, predicted = torch.max(output, 1)
                correct = (predicted == label).sum().item()
                accuracy = correct / img.size(0)
                metrics["val_acc"].append(accuracy)

        # log average train and val accuracy to tensorboard
        epoch_train_acc = torch.as_tensor(metrics["train_acc"]).mean()
        epoch_val_acc = torch.as_tensor(metrics["val_acc"]).mean()
        logger.add_scalar("Train Accuracy", epoch_train_acc, epoch)
        logger.add_scalar("Validation Accuracy", epoch_val_acc, epoch)
        

        # print on first, last, every 10th epoch
        if epoch == 0 or epoch == num_epoch - 1 or (epoch + 1) % 10 == 0:
            print(
                f"Epoch {epoch + 1:2d} / {num_epoch:2d}: "
                f"train_acc={epoch_train_acc:.4f} "
                f"val_acc={epoch_val_acc:.4f}"
            )

    # save and overwrite the model in the root directory for grading
    save_model(model)

    # save a copy of model weights in the log directory
    torch.save(model.state_dict(), log_dir / f"{model_name}.th")
    print(f"Model saved to {log_dir / f'{model_name}.th'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--exp_dir", type=str, default="logs")
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--num_epoch", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=2024)

    # optional: additional model hyperparamters
    # parser.add_argument("--num_layers", type=int, default=3)

    # pass all arguments to train
    train(**vars(parser.parse_args()))
