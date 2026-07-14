import os
import torch
import argparse
from models.models import LightMedSeg
from utils.dataset import ISLESDataset, OctantCropDataset
from utils.training import train_model
from torch.utils.data import DataLoader, random_split


import torch.nn as nn
from test_model import visualize_prediction

import torch.nn.functional as F
from tqdm import tqdm


def main():
    torch.manual_seed(42)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-o",
        "--output",
        help="Where to output the best and last weights.",
        default="out",
    )
    parser.add_argument(
        "-c",
        "--crop",
        help="Whether to train on cropped octants",
        default=0,
        type=int
    )
    parser.add_argument(
        "-e",
        "--epochs",
        help="Number of epochs for training/eval",
        type=int,
        default=40,
    )
    args = parser.parse_args()

    output_dir = args.output
    crop = bool(args.crop)
    epochs = args.epochs

    BATCH_SIZE = 8 if crop else 1

    dataset = (
        OctantCropDataset(range=(0, 400))
        if crop
        else ISLESDataset(range=(0, 400), random_crop=False)
    )

    print(len(dataset))

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # print(len(train_dataset))

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True)
    model = LightMedSeg(n_classes=2, num_anchors=8, downsample=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    os.mkdir(output_dir)

    train_model(
        model,
        train_loader,
        val_loader,
        num_epochs=epochs,
        device=device,
        lr=(1e-2, 1e-4),
        save_path_best=os.path.join(output_dir, "best.pth"),
        save_path_last=os.path.join(output_dir, "last.pth"),
    )

    # image = dataset[0]["image"].to(device)
    # mask = dataset[0]["mask"].to(device)

    # visualize_prediction(model, image, mask)


if __name__ == "__main__":
    main()
