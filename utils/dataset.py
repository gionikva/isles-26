import pandas as pd
import numpy as np
from PIL.Image import Image
from torch.utils.data import Dataset
import torch
import nibabel as nib
from monai.transforms import ResizeWithPadOrCrop, RandSpatialCropd, GridSplit
from glob import glob
from os import scandir, listdir
import os

from utils.shared import get_dataset_filepaths


class OctantCropDataset(Dataset):
    # mask_add_bgc: whether to add a background channel to the target mask
    def __init__(self, split="train", range=None, mask_add_bgc=True):
        super().__init__()
        
        self.mask_add_bgc = mask_add_bgc
        
        self.metadata, self.features, self.labels = get_dataset_filepaths(range)
        
        
        self.standardize_grid = ResizeWithPadOrCrop(spatial_size=(256, 256, 256), mode='constant')
        

        
        # For caching
        self.last_dataset_idx = None
        self.patches = None

        # 2. Set the seed for this specific transform
       

        # 3. Apply it to your data dictionary

    def split(self, image, mask):
        img_oct = [
            block
            for z_half in torch.chunk(image, 2, dim=1)
            for y_half in torch.chunk(z_half, 2, dim=2)
            for block in torch.chunk(y_half, 2, dim=3)
        ]   
        
        msk_oct = [
            block
            for z_half in torch.chunk(mask, 2, dim=1)
            for y_half in torch.chunk(z_half, 2, dim=2)
            for block in torch.chunk(y_half, 2, dim=3)
        ]   
        
        return img_oct, msk_oct
    def __getitem__(self, idx):
        dataset_idx = idx // 8
        patch_number = idx % 8
        
        if self.last_dataset_idx == dataset_idx:
            return self.patches[patch_number]
        else:
            self.last_dataset_idx = dataset_idx
            self.patches = []
            full_img = torch.tensor(nib.load(self.features[dataset_idx]).get_fdata(), dtype=torch.float)
            full_mask = torch.tensor(nib.load(self.labels[dataset_idx]).get_fdata(), dtype=torch.float)
            full_img = self.standardize_grid(full_img.unsqueeze(0))
            full_mask = self.standardize_grid(full_mask.unsqueeze(0))
            if self.mask_add_bgc:
                bg = 1.0 - full_mask
                full_mask = torch.cat([full_mask, bg], dim = 0)
                
            img_oct, msk_oct = self.split(full_img, full_mask)
            
            for img_oct, msk_oct in zip(img_oct, msk_oct):
                self.patches.append({"image": img_oct, "mask": msk_oct})
                
            # print(patch_number, len(self.patches))
            
            return self.patches[patch_number]
                

    def __len__(self):
        return 8*len(self.features) 


class ISLESDataset(Dataset):
    # mask_add_bgc: whether to add a background channel to the target mask
    def __init__(self, split="train", range=None, mask_add_bgc=True, random_crop=True, random_seed=42):
        super().__init__()
        
        self.mask_add_bgc = mask_add_bgc
        self.random_crop = random_crop
        
        self.metadata, self.features, self.labels = get_dataset_filepaths(range)

        
        self.standardize_grid = ResizeWithPadOrCrop(spatial_size=(256, 256, 256), mode='constant')
        

        self.cropper = RandSpatialCropd(
            keys=["image", "mask"],
            roi_size=(128, 128, 128),
            random_size=False
        )

        # 2. Set the seed for this specific transform
        self.cropper.set_random_state(seed=random_seed)

        # 3. Apply it to your data dictionary
        
    def parse_metadata(self, filepath):
        # 0: days_post_stroke missing? 0/1
        # 1: chronicity missing? 0/1
        # 2: days_post_stroke: float or nan
        # 3: chronicity: 0/1/2 or nan
        out = torch.empty((4), dtype=torch.float32)
        meta = pd.read_csv(filepath)
        if len(meta) > 0:
            dps = meta['DAYS_POST_STROKE'][0]
            chronicity = meta['CHRONICITY'][0]
            
            # print(type(chronicity))
            
            if np.isnan(dps):
                out[0] = 1.
                out[2] = 0.
            else:
                out[0] = 0.
                out[2] = dps
            
            if np.isnan(chronicity):
                out[1] = 1.
                out[3] = 0.
            else:
                out[1] = 0.
                out[3] = float(chronicity)
            
        else:
            out[0] = 1.
            out[1] = 1.
            out[2] = 0.
            out[3] = 0.
            
        return out
        

    def __getitem__(self, idx):
        image = torch.tensor(nib.load(self.features[idx]).get_fdata(), dtype=torch.float)
        mask = torch.tensor(nib.load(self.labels[idx]).get_fdata(), dtype=torch.float)
        image = self.standardize_grid(image.unsqueeze(0))
        mask = self.standardize_grid(mask.unsqueeze(0))
        if self.mask_add_bgc:
            bg = 1.0 - mask
            mask = torch.cat([bg, mask], dim = 0)
        meta = self.parse_metadata(self.metadata[idx])
        # days_post_stroke = torch.tensor(meta['DAYS_POST_STROKE'][0])
        # print(meta)
        # return {"image": image, "mask": mask, "metadata": meta}
        
        out = {"image": image, "mask": mask, "metadata": meta}
        
        if self.random_crop:
            return self.cropper(out)
        else:
            return out

    def __len__(self):
        return len(self.features)
