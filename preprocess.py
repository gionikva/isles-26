import os
from random import shuffle
from shutil import copy
from os import scandir, listdir, mkdir
import SimpleITK as sitk
import nibabel as nib
import numpy as np
from tqdm import tqdm


def load_raw_fps():
    metadata = []
    mris = []
    masks = []

    root = "./raw_data"
    dirs = [f.path for f in scandir(root) if f.is_dir()]

    for dir in dirs:
        subdirs = [f for f in scandir(dir) if f.is_dir()]
        for subdir in subdirs:
            path = os.path.join(subdir.path, "ses-1/anat")
            name = subdir.name
            metadata.append(os.path.join(path, f"{name}_ses-1_metadata.csv"))

            mris.append(
                os.path.join(path, f"{name}_ses-1_space-orig_desc-brain_T1w.nii.gz")
            )

            masks.append(
                os.path.join(
                    path,
                    f"{name}_ses-1_space-orig_label-lesion_desc-T1lesion_mask.nii.gz",
                )
            )

    return list(zip(metadata, mris, masks))


def n4_bias_field(array):
    arr = np.transpose(array)
    img = sitk.GetImageFromArray(arr)
    img = sitk.Cast(img, sitk.sitkFloat32)
    mask = sitk.OtsuThreshold(img, 0, 1, 200)
    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    shrink_factor = 4
    shrunk = sitk.Shrink(img, [shrink_factor] * img.GetDimension())
    mask_shrunk = sitk.Shrink(mask, [shrink_factor] * img.GetDimension())

    corrector.Execute(shrunk, mask_shrunk)

    log_bias_field = corrector.GetLogBiasFieldAsImage(img)
    corrected_image = img / sitk.Exp(log_bias_field)
    # corrected_image = corrector.Execute(img, mask)

    return np.transpose(sitk.GetArrayFromImage(corrected_image))


def percentile_clip(array):
    lower = np.percentile(array, 1)
    upper = np.percentile(array, 99)
    return np.clip(array, lower, upper)


def z_score(array):
    mean = np.mean(array)
    std = np.std(array)
    return (array - mean) / std


def process_image(path, out_path):
    orig_img = nib.load(path)
    arr = orig_img.get_fdata()
    affine = orig_img.affine
    header = orig_img.header

    arr = n4_bias_field(arr)
    arr = percentile_clip(arr)
    arr = z_score(arr)

    img = nib.Nifti1Image(arr, affine=affine, header=header)
    nib.save(img, out_path)


def main():
    files = load_raw_fps()
    shuffle(files)

    # Ratio of training set images to total images
    train_ratio = 0.9
    n_train = int(len(files) * train_ratio)
    n_test = len(files) - n_train

    print(f"Total data points: {len(files)}")
    for i in tqdm(range(n_train), desc="Train set"):
        os.makedirs(f"./data/train/{i}", exist_ok=True)

        meta, mri, mask = files[i]
        
        copy(meta, f"./data/train/{i}/meta.csv")
        copy(mask, f"./data/train/{i}/mask.nii.gz")
        process_image(mri, f"./data/train/{i}/img.nii.gz")

    for i in tqdm(range(n_test), desc="Test set"):
        os.makedirs(f"./data/test/{i}", exist_ok=True)

        meta, mri, mask = files[i + n_train]

        copy(meta, f"./data/test/{i}/meta.csv")
        copy(mask, f"./data/test/{i}/mask.nii.gz")
        process_image(mri, f"./data/test/{i}/img.nii.gz")


if __name__ == "__main__":
    main()
