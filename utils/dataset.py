import pandas as pd
import numpy as np
from PIL.Image import Image
from torch.utils.data import Dataset
import torch
import nibabel as nib
from glob import glob
from os import scandir, listdir
import os


class ISLESDataset(Dataset):
    def __init__(self, split="train"):
        super().__init__()
        self.metadata = []
        self.features = []
        self.labels = []

        root = "./data"
        dirs = [f for f in scandir(root)]

        for dir in dirs:
            # name = dir.name
            path = dir.path

            self.metadata.append(os.path.join(path, "meta.csv"))
            self.features.append(os.path.join(path, "img.nii.gz"))
            self.labels.append(os.path.join(path, "mask.nii.gz"))

    def __getitem__(self, idx):
        image = torch.tensor(nib.load(self.features[idx]).get_fdata())
        mask = torch.tensor(nib.load(self.labels[idx]).get_fdata())
        meta = pd.read_csv(self.metadata[idx])
        return {"image": image, "mask": mask, "metadata": meta}

    def __len__(self):
        return len(self.features)
