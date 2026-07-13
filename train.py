import torch
from models.models import LightMedSeg
from models.loss import LightMedSegLoss
from utils.dataset import ISLESDataset
from torch.utils.data import DataLoader


import torch.nn as nn
from torch.nn import CrossEntropyLoss
from model_testing import visualize_prediction

import torch.nn.functional as F
from tqdm import tqdm




def main():
    dataset = ISLESDataset(range=(0, 20))
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    model = LightMedSeg(n_classes=2, num_anchors=8)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = model.to(device)

    NUM_EPOCHS = 1

    LR = 0.001

    optimizer = AdamW(model.parameters(), lr=LR)

    criterion = CrossEntropyLoss()

    for EPOCH in range(NUM_EPOCHS):
        for batch in tqdm(dataloader):
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)

            predictions = model(images)

            print(masks[0])
            # print(predictions.shape, masks.shape)

            loss = criterion(predictions, masks)

            loss.backward()
            optimizer.step()

    image = dataset[0]["image"].to(device)
    mask = dataset[0]["mask"].to(device)

    visualize_prediction(model, image, mask)


if __name__ == "__main__":
    main()
