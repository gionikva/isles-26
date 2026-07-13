import pandas as pd
import numpy as np
from PIL.Image import Image
from torch.utils.data import Dataset
import torch
import nibabel as nib
from monai.transforms import ResizeWithPadOrCrop
from glob import glob
from os import scandir, listdir
import os


class ISLESDataset(Dataset):
    # mask_add_bgc: whether to add a background channel to the target mask
    def __init__(self, split="train", range=None, mask_add_bgc=True):
        super().__init__()
        
        self.mask_add_bgc = mask_add_bgc
        
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
    
        if range is not None:
            self.metadata = self.metadata[range[0]:range[1]]
            self.features = self.features[range[0]:range[1]]
            self.labels = self.labels[range[0]:range[1]]
        
        self.standardize_grid = ResizeWithPadOrCrop(spatial_size=(256, 256, 256), mode='constant')

    def __getitem__(self, idx):
        image = torch.tensor(nib.load(self.features[idx]).get_fdata(), dtype=torch.float)
        mask = torch.tensor(nib.load(self.labels[idx]).get_fdata(), dtype=torch.float)
        image = self.standardize_grid(image.unsqueeze(0))
        mask = self.standardize_grid(mask.unsqueeze(0))
        if self.mask_add_bgc:
            bg = 1.0 - mask
            mask = torch.cat([mask, bg], dim = 0)
        meta = pd.read_csv(self.metadata[idx])
        # return {"image": image, "mask": mask, "metadata": meta}
        return {"image": image, "mask": mask}

    def __len__(self):
        return len(self.features)
