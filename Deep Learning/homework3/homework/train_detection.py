import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.utils.tensorboard as tb

from .models import load_model, save_model
from .datasets.road_dataset import load_data
from .metrics import *


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

    print(model)

    train_data = load_data("road_data/train", shuffle=True, batch_size=batch_size, num_workers=2)
    val_data = load_data("road_data/val", shuffle=False)

    # create loss function and optimizer
    classification_loss = nn.CrossEntropyLoss()
    regression_loss = nn.MSELoss()

    # optimizer = ...
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    global_step = 0

    # training loop...   
    for epoch in range(num_epoch):
        epoch_loss = 0.0
        
        for data in train_data:
            images, tracks, depths = data['image'].to(device), data['track'].to(device), data['depth'].to(device)

            optimizer.zero_grad()

            # Forward pass
            logits, predicted_depth = model(images)

            # Compute losses
            seg_loss = classification_loss(logits, tracks)
            #print(depths.shape)
            #print(predicted_depth.shape)
            depth_loss = regression_loss(predicted_depth, depths)

            # Combine losses
            loss = seg_loss + depth_loss

            # Backward pass and optimization step
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        # Print epoch statistics
        print(f"Epoch [{epoch+1}/{num_epoch}], Loss: {epoch_loss / len(train_data):.4f}")


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
