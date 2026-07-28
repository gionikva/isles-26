# Description
This repo contains the training code for the 2026 ISLES challenge.
## Setup
To install all the dependencies, run `pip install -r requirements.txt`.\
\
For the data, download the [ATLAS 3.0 raw data](https://fcon_1000.projects.nitrc.org/indi/retro/atlas_download.html) and extract it in `isles-26/raw_data`. Be sure to fill out the google form at the bottom of the page to obtain the decryption password. After extracting the data, run `python preprocess.py`. This will save the processed data to `isles-26/data`.
## Training
To to train the model run `python train.py`.
The following is a description of all the parameters:
```
  -h, --help                    Show this help message and exit
  -o, --output OUTPUT           Output directory for the best and last weights.
  -e, --epochs EPOCHS           Number of epochs for training/eval.
  -b, --batch-size BATCH_SIZE   Batch size.
  --lr-range LR_RANGE           Learning rate range in the format max:min. Must follow
                                Python's float format.
  -r, --range RANGE             Range of datapoints to train on in the format 
                                start_idx:end_idx.
  -s, --model-size              Model size: 'small', 'medium' or 'large'.
  -m, --model                   Which model to use: 'base' or 'refined'. The 'refined'
                                model uses boundary refinement to save on VRAM.
  --deep-supervision            Enable deep supervision.
  -d, --ignore-metadata         Disables the metadata FiLM functionality.
  -c, --crop                    Train on 128x128x128 random cropped patches.
  -a, --domain-augment          Enable domain augmentation.
  -p, --resume                  Resume training from the last epoch.
```
The best performance model was trained with the folowing command: \
`python train_model.py -o weights/base_medium_deep -e 100 -b 1 --lr-range 2e-4:1e-9 --deep-supervision -m base -s medium -ca`.