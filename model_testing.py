import torch
from models.models import GlobalAnchorDetector, GhostConv3D, SpatialAnchorFiLM
from models.lspm import LSPM
from utils.dataset import ISLESDataset


def main():
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


if __name__ == "__main__":
    main()
