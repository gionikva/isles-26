from matplotlib import pyplot as plt
import numpy as np
import torch
from models.models import GlobalAnchorDetector, GhostConv3D, LightMedSeg, SpatialAnchorFiLM
from models.lspm import LSPM
from utils.dataset import ISLESDataset
from monai.transforms import ResizeWithPadOrCrop
import torch.nn.functional as F

def test_components():
    anchor_detector = GlobalAnchorDetector(1, 16, 8)
    ghost_conv = GhostConv3D(1, 8, downscale=False)
    lspm = LSPM()
    spatial_film = SpatialAnchorFiLM(8, 8)
    dataset = ISLESDataset()
    
    img = dataset[500]['image'].numpy()
    mask = dataset[500]['mask'].numpy()
    
    x = torch.tensor(img, dtype=torch.float)[None, None, ...]
    print(x.shape)
    y = anchor_detector(x)
    f0 = ghost_conv(x)
    T, out = lspm(f0)
    
    anchors = torch.randn(1, 8, 3)
    
    test = spatial_film(anchors, f0)
    
    
    print(f0.shape)

def visualize_prediction(model, image, mask):
    
    image = image.cpu()
    mask = mask.cpu()
    model = model.cpu()
    prediction = F.softmax(model(image.unsqueeze(0)), dim=1)
    
    def show_slices(slices):
        fig, axes = plt.subplots(1, len(slices))
        for i, (mri, mask, prediction) in enumerate(slices):
            axes[i].imshow(mri.T, cmap="gray", origin="lower")
            axes[i].imshow(mask.T, cmap='autumn', alpha=0.5, interpolation='none')
            axes[i].imshow(prediction.T, cmap='winter', alpha=0.5, interpolation='none')

    print(image.shape)

    image = image.squeeze(0).numpy()
    mask = mask[0, :, :, :].numpy()
    prediction = prediction.squeeze(0)[0, :, :, :].detach().numpy()
    mask = np.ma.masked_where(mask == 0, mask)
    prediction = np.ma.masked_where(prediction < 0.5, prediction)

    mri_0 = image[128, :, :]
    mask_0 = mask[128, :, :]
    prediction_0 = prediction[128, :, :]
    
    mri_1 = image[:, 128, :]
    mask_1 = mask[:, 128, :]
    prediction_1 = prediction[:, 128, :]

    
    mri_2 = image[:, :, 128]
    mask_2 = mask[:, :, 128]
    prediction_2 = prediction[:, :, 128]
    
   
    show_slices([(mri_0, mask_0, prediction_0), (mri_1, mask_1, prediction_1), (mri_2, mask_2, prediction_2)])
    plt.tight_layout()
    plt.show()
    
    print(prediction.shape)

def test_model():
    dataset = ISLESDataset()

    img = dataset[500]['image']
    mask = dataset[500]['mask']
    
    
    
    print(img.shape)
    print(mask.shape)
    
    standardize_grid = ResizeWithPadOrCrop(spatial_size=(256, 256, 256), mode='constant')
    
    # print (final_image.shape)
    model = LightMedSeg(n_classes=2, num_anchors=8)
    
    device = 'cpu'
    checkpoint_path = "lightmedseg_best.pth"

    # 3. Load the dictionary from the file
    # map_location ensures it loads correctly even if moving from GPU to CPU
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # 4. Inject the saved weights into the model
    model.load_state_dict(checkpoint['model_state_dict'])
    
    
    prediction = model(img.unsqueeze(0))
    print(prediction.shape)
    
    visualize_prediction(model, img, mask)
  

def main():
    # test_components()
    test_model()


if __name__ == "__main__":
    main()
