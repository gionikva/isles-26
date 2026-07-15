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
        "-e",
        "--epochs",
        help="Number of epochs for training/eval.",
        type=int,
        default=40,
    )
    parser.add_argument(
        "-r",
        "--range",
        help="Range of datapoints to train on in the format start_idx:end_idx.",
        type=str,
        default=None,
    )
    parser.add_argument(
        "-a",
        "--num-anchors",
        help="num_anchors hyperparameter value.",
        type=int,
        default=8,
    )
    parser.add_argument(
        "-m",
        "--ignore-metadata",
        help="Disables the metadata FiLM functionality.",
        action="store_true",
    )
    parser.add_argument(
        "-c", "--crop", help="Train using random crop.", action="store_true"
    )
    parser.add_argument(
        "-t",
        "--add-transformed-channels",
        help="Add extra channels to the input mri.",
        action="store_true",
    )
    args = parser.parse_args()

    output_dir = args.output
    epochs = args.epochs
    num_anchors = args.num_anchors
    crop = args.crop
    extra_channels = args.add_transformed_channels
    metadata_film = not args.ignore_metadata

    BATCH_SIZE = 8 if crop else 1
    downsample = not crop

    rng = args.range
    data_range = None if rng == None else [int(idx) for idx in rng.split(":")]
    print(data_range)

    dataset = (
        ISLESDataset(
            range=data_range, add_extra_channels=extra_channels, random_crop=True
        )
    )

    print(len(dataset))

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # print(len(train_dataset))

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True)
    model = LightMedSeg(
        n_classes=2,
        in_channels=5 if extra_channels else 1,
        num_anchors=num_anchors,
        metadata_film=metadata_film,
        downsample=downsample,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    train_model(
        model,
        train_loader,
        val_loader,
        num_epochs=epochs,
        device=device,
        lr=(1e-3, 1e-8),
        # ce_only=True,
        save_path_best=os.path.join(output_dir, "best.pth"),
        save_path_last=os.path.join(output_dir, "last.pth"),
    )

    # image = dataset[0]["image"].to(device)
    # mask = dataset[0]["mask"].to(device)

    # visualize_prediction(model, image, mask)


if __name__ == "__main__":
    main()
