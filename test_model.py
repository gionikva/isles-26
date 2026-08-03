import argparse
import torch
import numpy as np
from monai.inferers import sliding_window_inference
from monai.transforms import KeepLargestConnectedComponent
from utils.dataset import ISLESDataset
from models.models import LightMedSeg, LMSBR
from utils.loss import LightMedSegLoss
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm
from utils.eval import (
    compute_dice_f1_instance_difference,
    compute_absolute_volume_difference,
)

# def get_dice_score(prediction, mask):
#     eps = 1e-7

#     mask = mask[:, 1, :, :, :]

#     intersection = (prediction * mask).sum(dim=(1, 2, 3))
#     total = prediction.sum(dim=[1, 2, 3]) + mask.sum(dim=[1, 2, 3])
#     dice_score = (2 * intersection + eps) / (total + eps)

#     return dice_score


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", help="Path to weights.", required=True)
    parser.add_argument(
        "-s",
        "--sw-batch-size",
        help="Batch size for sliding window inference. 1-16. Not relevant with the -c option.",
        type=int,
        default=1,
    )
    parser.add_argument(
        "-b", "--batch-size", help="Batch size for inference.", type=int, default=1
    )
    parser.add_argument(
        "-r",
        "--random-seed",
        help="Random seed value for pytorch and monai.",
        type=int,
    )
    parser.add_argument(
        "-c",
        "--cropped",
        help="Run on 128x128x128 randomly cropped patches. By default use sliding window inference.",
        action="store_true",
    )

    args = parser.parse_args()

    parsed = {}

    parsed["checkpoint_path"] = args.input
    parsed["sw_batch_size"] = args.sw_batch_size
    parsed["batch_size"] = args.batch_size
    parsed["random_seed"] = args.random_seed
    parsed["crop"] = args.cropped

    return parsed


def main():
    args = parse_args()

    random_seed = args["random_seed"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint_path = args["checkpoint_path"]
    crop = args["crop"]
    sw_batch_size = args["sw_batch_size"]
    batch_size = args["batch_size"]

    if random_seed is not None:
        torch.random.manual_seed(random_seed)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_type = checkpoint["model"]

    if model_type == "base":
        model = LightMedSeg.from_checkpoint(checkpoint)
    else:
        model = LMSBR.from_checkpoint(checkpoint)

    model = model.to(device)

    model.eval()

    dataset = ISLESDataset(split="test", random_crop=crop)
    dataloader = DataLoader(dataset, batch_size, True)

    total_dice = 0.0
    total_f1 = 0.0
    total_abs_vol_diff = 0.0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Inferring"):
            image = batch["image"].to(device)
            mask = batch["mask"].to(device)
            metadata = batch["metadata"].to(device)

            if not crop:

                def predictor(x):
                    return model(x, metadata)

                roi_size = (128, 128, 128)

                logits = sliding_window_inference(
                    inputs=image,
                    roi_size=roi_size,
                    sw_batch_size=sw_batch_size,
                    predictor=predictor,
                    overlap=0.5,
                    mode="gaussian",
                )
            else:
                logits = model(image, metadata)

            cleaner = KeepLargestConnectedComponent(applied_labels=[1])

            probs = torch.softmax(logits, dim=1)
            prediction = (probs > 0.5).float()
            prediction = cleaner(prediction)[:, 1, :, :, :]
            mask = mask[:, 1, :, :, :]

            f1, icd, dice = compute_dice_f1_instance_difference(prediction, mask)
            abs_vol_diff = compute_absolute_volume_difference(
                prediction, mask, voxel_size=1.0
            )

            total_dice += dice
            total_f1 += f1
            total_abs_vol_diff += abs_vol_diff

    avg_dice = total_dice / len(dataloader)
    avg_f1 = total_f1 / len(dataloader)
    avg_abs_vol_diff = total_abs_vol_diff / len(dataloader)

    print(f"Average DICE: {avg_dice}")
    print(f"Average Lesion-Wise F1: {avg_f1}")
    print(f"Average Absolute Volume Difference: {avg_abs_vol_diff}")


if __name__ == "__main__":
    main()
