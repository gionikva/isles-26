from utils.dataset import ISLESDataset
import matplotlib.pyplot as plt

def main():
    dataset = ISLESDataset()
    
    img = dataset[0]['image'].numpy()
    mask = dataset[0]['mask'].numpy()
    
    def show_slices(slices):
        fig, axes = plt.subplots(1, len(slices))
        for i, (mri, mask) in enumerate(slices):
            axes[i].imshow(mri.T, cmap="gray", origin="lower")
            axes[i].imshow(mask.T, cmap='autumn', alpha=0.1, interpolation='none')
            
    print(img.shape)

    mri_0 = img[60, :, :]
    mask_0 = mask[60, :, :]
    
    mri_1 = img[:, 60, :]
    mask_1 = mask[:, 60, :]
    
    mri_2 = img[:, :, 60]
    mask_2 = mask[:, :, 60]
    
   
    show_slices([(mri_0, mask_0), (mri_1, mask_1), (mri_2, mask_2)])
    plt.tight_layout()
    plt.show()
    

if __name__ == "__main__":
    main()