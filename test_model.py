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
        "-f",
        "--full-patches",
        help="Enable if model was trained on full 256x256x256 patches. Otherwise, it will use sliding window inference.",
        action="store_true",
    )
    

    args = parser.parse_args()

    parsed = {}

    parsed["checkpoint_path"] = args.input
    parsed["sw_batch_size"] = args.sw_batch_size
    parsed["batch_size"] = args.batch_size
    parsed["random_seed"] = args.random_seed
    parsed["full_patches"] = args.full_patches

    return parsed


def main():
    args = parse_args()

    random_seed = args["random_seed"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint_path = args["checkpoint_path"]
    full_patches = args["full_patches"]
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

    dataset = ISLESDataset(split="test", random_crop=False)
    dataloader = DataLoader(dataset, batch_size, True)

    total_dice = 0.0
    total_f1 = 0.0
    total_abs_vol_diff = 0.0

    with torch.no_grad():
        test_loop = tqdm(dataloader, desc="Evaluating")
        
        for batch in test_loop:
            image = batch["image"].to(device)
            mask = batch["mask"].to(device)
            metadata = batch["metadata"].to(device)

            if not full_patches:
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
            prediction = cleaner(prediction)[:, 1, :, :, :].detach().cpu().numpy()
            mask = mask[:, 1, :, :, :].detach().cpu().numpy()
            
            batch_dice = 0.0
            batch_f1 = 0.0
            batch_abs_vol_diff = 0.0
            
            
            for i in range(batch_size):
                pred_slice = prediction[i]
                mask_slice = mask[i]
                
                f1, icd, dice = compute_dice_f1_instance_difference(pred_slice, mask_slice)
                abs_vol_diff = compute_absolute_volume_difference(
                    pred_slice, mask_slice, voxel_size=1.0
                )

                batch_dice += dice
                batch_f1 += f1
                batch_abs_vol_diff += abs_vol_diff
                
            

            batch_dice /= batch_size
            batch_f1 /= batch_size
            batch_abs_vol_diff /= batch_size

            total_dice += batch_dice
            total_f1 += batch_f1
            total_abs_vol_diff += batch_abs_vol_diff
            
            test_loop.set_postfix(
                {
                    "DICE": f"{batch_dice:.4f}",
                    "F1": f"{batch_f1:.4f}",
                    "AbsVolDiff": f"{batch_abs_vol_diff:.0f}",
                }
            )

    avg_dice = total_dice / len(dataloader)
    avg_f1 = total_f1 / len(dataloader)
    avg_abs_vol_diff = total_abs_vol_diff / len(dataloader)

    print(f"Average DICE: {avg_dice:.4f}")
    print(f"Average Lesion-Wise F1: {avg_f1:.4f}")
    print(f"Average Absolute Volume Difference: {avg_abs_vol_diff:.0f}")


if __name__ == "__main__":
    main()
