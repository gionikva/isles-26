from utils.dataset import ISLESDataset
from utils.stats import MetadataIterator
import numpy as np
import matplotlib.pyplot as plt

def main():
    dataset = ISLESDataset()
    
    img = dataset[500]['image'].numpy()
    mask = dataset[500]['mask'].numpy()
    meta = dataset[500]['metadata']
    
    print(mask[0] + mask[1])
    # print(meta)
    
    # def show_slices(slices):
    #     fig, axes = plt.subplots(1, len(slices))
    #     for i, (mri, mask) in enumerate(slices):
    #         axes[i].imshow(mri.T, cmap="gray", origin="lower")
    #         axes[i].imshow(mask.T, cmap='autumn', alpha=0.3, interpolation='none')
            
    # print(img.shape)

    # mri_0 = img[60, :, :]
    # mask_0 = mask[60, :, :]
    
    # mri_1 = img[:, 60, :]
    # mask_1 = mask[:, 60, :]
    
    # mri_2 = img[:, :, 60]
    # mask_2 = mask[:, :, 60]
    
   
    # show_slices([(mri_0, mask_0), (mri_1, mask_1), (mri_2, mask_2)])
    # plt.tight_layout()
    # plt.show()

def check_chronicity():
    meta_iter = MetadataIterator()
    total = len(meta_iter)
    has_chronicity = 0
    
    meta_arr = np.empty((total, 2))
    
    for i, meta in enumerate(meta_iter):
        
        # print(meta)


        
        if len(meta) > 0:
            dps = meta['DAYS_POST_STROKE'][0]
            chronicity = meta['CHRONICITY'][0]
            
            meta_arr[i, 0] = dps
            meta_arr[i, 1] = chronicity
        
        
        # print(dps)
        # print(chronicity)
        
        # print(meta['CHRONICITY'][0])
    print(meta_arr)
    not_is_nan = ~np.isnan(meta_arr)
    print(f"Nonzero-dps: {np.count_nonzero(not_is_nan[:, 0])}")
    print(f"Nonzero-chronicity: {np.count_nonzero(not_is_nan[:, 1])}")
    
    # item = dataset[200]

if __name__ == "__main__":
    main()