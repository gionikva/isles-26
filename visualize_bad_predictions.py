import argparse
import napari
import numpy as np
import torch
from torch.amp import autocast

# from models.models import (
#     GlobalAnchorDetector,
#     GhostConv3D,
#     LightMedSeg,
#     SpatialAnchorFiLM,
# )
from utils.dataset import ISLESDataset
from models.models import LightMedSeg, LMSBR
from utils.loss import LightMedSegLoss
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

    # def show_slices(slices):
    #     fig, axes = plt.subplots(1, len(slices))
    #     for i, (mri, mask, prediction) in enumerate(slices):
    #         axes[i].imshow(mri.T, cmap="gray", origin="lower")
    #         axes[i].imshow(mask.T, cmap='autumn', alpha=0.5, interpolation='none')
    #         axes[i].imshow(prediction.T, cmap='winter', alpha=0.5, interpolation='none')

    images = []
    masks = []
    predictions = []

    for image, mask, prediction in data:
        image = image[0].numpy()
        mask = mask[1, :, :, :].numpy()
        prediction = prediction.numpy()
        mask = np.ma.masked_where(mask == 0, mask)
        prediction = np.ma.masked_where(prediction == 0, prediction)

        # print(image.shape, mask.shape, prediction.shape)

        images.append(image)
        masks.append(mask)
        predictions.append(prediction)

    viewer = napari.Viewer()

    viewer.add_image(np.stack(images), name="Image", colormap="gray")

    viewer.add_labels(np.stack(masks).astype(int), name="Label", opacity=0.5)
    viewer.add_labels(np.stack(predictions).astype(int), name="Prediction", opacity=0.5, colormap={1: "blue"})

    napari.run()

    # mri_0 = image[size//2, :, :]
    # mask_0 = mask[size//2, :, :]
    # prediction_0 = prediction[size//2, :, :]

    # mri_1 = image[:, size//2, :]
    # mask_1 = mask[:, size//2, :]
    # prediction_1 = prediction[:, size//2, :]

    # mri_2 = image[:, :, size//2]
    # mask_2 = mask[:, :, size//2]
    # prediction_2 = prediction[:, :, size//2]

    # show_slices([(mri_0, mask_0, prediction_0), (mri_1, mask_1, prediction_1), (mri_2, mask_2, prediction_2)])
    # plt.tight_layout()
    # plt.show()

    # print(prediction.shape)


def test_model():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", help="Path to weights.", required=True)
    # parser.add_argument(
    #     "-c",
    #     "--crop",
    #     help="Train on cropped octants",
    #     action="store_true"
    # )
    # parser.add_argument(
    #     "-e",
    #     "--epochs",
    #     help="Number of epochs for training/eval",
    #     type=int,
    #     default=40,
    # )
    # parser.add_argument(
    #     "-r",
    #     "--range",
    #     help="Range of datapoints to train on in the format start_idx:end_idx",
    #     type=str,
    #     default=None
    # )
    # parser.add_argument(
    #     "-t",
    #     "--add-transformed-channels",
    #     help="Add extra channels to the input mri",
    #     action="store_true"
    # )
    args = parser.parse_args()

    device = "cuda"
    checkpoint_path = args.input

    # 3. Load the dictionary from the file
    # map_location ensures it loads correctly even if moving from GPU to CPU
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model_type = checkpoint["model_type"]
    hyperparams = checkpoint["hyperparams"]

    add_edges = model_type == "refined"
    # random_crop = hyperparams

    # print(torch.sum(img))
    # print(mask.shape)

    # print (final_image.shape)

    hparams = checkpoint["hyperparams"]

    print(hparams)

    if model_type == "base":

        model = LightMedSeg(
            n_classes=2,
            in_channels=hparams["in_channels"],
            num_anchors=hparams["num_anchors"],
            metadata_film=hparams["metadata_film"],
            downsample=hparams["downsample"],
        )
    else:
        model = LMSBR(
            n_classes=2,
            num_anchors=hparams["num_anchors"],
            metadata_film=hparams["metadata_film"],
        )

    model = model.to(device)

    model.eval()

    # 4. Inject the saved weights into the model
    model.load_state_dict(checkpoint["model_state_dict"])

    dataset = ISLESDataset(random_crop=True, range=(0, 20), add_edges=add_edges)

    dice_loss_threshold = 0.7

    bad_dice = []
    with torch.no_grad():
        for i in tqdm(range(len(dataset)), desc="Inferring"):
            datapoint = dataset[i]
            img = datapoint["image"].unsqueeze(0).to(device)

            # print(img.shape)

            mask = datapoint["mask"].unsqueeze(0).to(device)
            metadata = datapoint["metadata"].unsqueeze(0).to(device)

            criterion = LightMedSegLoss()

            logits = model(img, metadata)

            loss, l_dice, l_ce, l_bdry = criterion(logits, mask)

            img = img.detach().cpu().squeeze(0)
            mask = mask.detach().cpu().squeeze(0)

            if l_dice.item() > dice_loss_threshold:
                prediction = torch.argmax(logits, dim=1)
                prediction = prediction.detach().cpu().squeeze(0)
                bad_dice.append((img, mask, prediction))

    # print(sum(p.numel() for p in model.parameters() if p.requires_grad))
    # print(checkpoint['model_state_dict'])

    # prediction = model(img.unsqueeze(0))
    # print(prediction.shape)

    visualize_predictions(bad_dice, cropped=False, debug=False)


def main():
    # test_components()
    test_model()


if __name__ == "__main__":
    main()
