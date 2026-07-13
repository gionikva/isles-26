import torch
from models.models import LightMedSeg
from utils.dataset import ISLESDataset
from utils.training import train_model
from torch.utils.data import DataLoader, random_split


import torch.nn as nn
from model_testing import visualize_prediction

import torch.nn.functional as F
from tqdm import tqdm


def main():
    torch.manual_seed(42)

    BATCH_SIZE = 1

    dataset = ISLESDataset(range=(0, 20))

    train_size = int(0.8 * len(dataset))
    val_size = int(0.2 * len(dataset))

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True)
    model = LightMedSeg(n_classes=2, num_anchors=8)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_model(model, train_loader, val_loader, num_epochs=10, device=device)

    image = dataset[0]["image"].to(device)
    mask = dataset[0]["mask"].to(device)

    visualize_prediction(model, image, mask)


if __name__ == "__main__":
    main()
