import argparse
import napari
import torch
import numpy as np
from monai.inferers import sliding_window_inference
from utils.dataset import ISLESDataset
from models.models import LightMedSeg, LMSBR
from utils.loss import LightMedSegLoss
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm

# def test_components():
#     anchor_detector = GlobalAnchorDetector(1, 16, 8)
#     ghost_conv = GhostConv3D(1, 8, downscale=False)
#     lspm = LSPM()
#     spatial_film = SpatialAnchorFiLM(8, 8)
#     dataset = ISLESDataset()

#     img = dataset[500]["image"].numpy()
#     mask = dataset[500]["mask"].numpy()

#     x = torch.tensor(img, dtype=torch.float)[None, None, ...]
#     print(x.shape)
#     y = anchor_detector(x)
#     f0 = ghost_conv(x)
#     T, out = lspm(f0)

#     anchors = torch.randn(1, 8, 3)

#     test = spatial_film(anchors, f0)

#     print(f0.shape)


def predict_in_octants(model, image, num_classes=2):
    """
    Splits a 256x256x256 image into 8 octants of 128x128x128,
    runs model inference on each, and reconstructs the full volume.

    Args:
        model: The trained PyTorch model.
        image: Input tensor of shape (B, C, 256, 256, 256).
        num_classes: Number of output channels the model predicts.

    Returns:
        final_mask: The combined argmax segmentation mask of shape (B, 256, 256, 256).
    """
    B, C, D, H, W = image.shape
    device = image.device

    # Pre-allocate an empty tensor to hold the stitched logits
    # Shape: (B, num_classes, 256, 256, 256)
    full_logits = torch.zeros(
        (B, num_classes, D, H, W), device=device, dtype=torch.float16
    )

    # Define the starting indices for our 8 blocks (0 and 128)
    steps = [0, 128]

    model.eval()
    # Loop through the 3 spatial dimensions (Depth, Height, Width)
    for d in steps:
        for h in steps:
            for w in steps:
                # 1. Extract the 128x128x128 patch
                patch = image[:, :, d : d + 128, h : h + 128, w : w + 128]

                # 2. Run the patch through the model
                patch_logits = model(patch)

                # 3. Place the output exactly where it belongs in the full volume
                full_logits[:, :, d : d + 128, h : h + 128, w : w + 128] = patch_logits

                print("finished octant")

    # Convert the raw logits into a final discrete segmentation mask
    # argmax across the channel dimension (dim=1) collapses it to (B, 256, 256, 256)
    final_mask = torch.argmax(full_logits, dim=1)

    return final_mask


def visualize_predictions(data, cropped=True, debug=False):

    size = 128 if cropped else 256

    images = []
    masks = []
    predictions = []

    for image, mask, prediction in data:
        image = image[0].numpy()
        mask = mask[1, :, :, :].numpy()
        prediction = prediction.numpy()
        mask = np.ma.masked_where(mask == 0, mask)
        prediction = np.ma.masked_where(prediction == 0, prediction)
        images.append(image)
        masks.append(mask)
        predictions.append(prediction)

    viewer = napari.Viewer()

    viewer.add_image(np.stack(images), name="Image", colormap="gray")

    viewer.add_labels(np.stack(masks).astype(int), name="Label", opacity=0.5)
    viewer.add_labels(
        np.stack(predictions).astype(int),
        name="Prediction",
        opacity=0.5,
        colormap={1: "blue"},
    )

    napari.run()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", help="Path to weights.", required=True)
    parser.add_argument(
        "--sw-batch-size",
        help="Batch size for sliding window inference. 1-16. Not relevant with the -c option.",
        type=int,
        default=1,
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
    parser.add_argument(
        "-n", "--num-results", help="Number of results to display.", type=int, default=8
    )
    parser.add_argument(
        "-d",
        "--dice-ceiling",
        help="Only display results with a DICE score <= to this value.",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "-s",
        "--size-ceiling",
        help="Only display results with lesions of size (in mm^3) <= this value.",
        type=float
    )

    args = parser.parse_args()

    parsed = {}

    parsed["checkpoint_path"] = args.input
    parsed["sw_batch_size"] = args.sw_batch_size
    parsed["random_seed"] = args.random_seed
    parsed["crop"] = args.cropped
    parsed["num_results"] = args.num_results
    parsed["dice_ceiling"] = args.dice_ceiling
    parsed["size_ceiling"] = args.size_ceiling

    return parsed



def main():

    args = parse_args()

    random_seed = args["random_seed"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint_path = args["checkpoint_path"]
    crop = args["crop"]
    sw_batch_size = args["sw_batch_size"]
    num_results = args["num_results"]
    dice_ceiling = args["dice_ceiling"]
    size_ceiling = args["size_ceiling"]

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
    dataloader = DataLoader(dataset, 1, True)

    results = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Inferring"):
            image = batch["image"].to(device)
            mask = batch["mask"].to(device)
            metadata = batch["metadata"].to(device)

            criterion = LightMedSegLoss()

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

            loss, l_dice, l_ce, l_bdry = criterion(logits, mask)

            dice_score = 1 - l_dice.item()
            lesion_size = mask.sum()

            image = image.detach().cpu().squeeze(0)
            mask = mask.detach().cpu().squeeze(0)
            
            filter1 = dice_score <= dice_ceiling
            filter2 = lesion_size <= size_ceiling if size_ceiling is not None else True
          
            if filter1 and filter2:
                prediction = torch.argmax(logits, dim=1)
                prediction = prediction.detach().cpu().squeeze(0)
                results.append((image, mask, prediction))
                
            if len(results) == num_results:
                break
            
    visualize_predictions(results, cropped=False, debug=False)


if __name__ == "__main__":
    main()
