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

def visualize_predictions(data, cropped=True, debug=False):
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
