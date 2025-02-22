# In-Context RCA

This repository contains the code for the In-Context RCA project. It includes RCA classifier models, a UNet for segmentation, and utilities to support training and inference using PyTorch.

## Installation

1. Install the required dependencies by running:

    ```bash
    pip install -r requirements.txt
    ```

2. Install SAM 2 following the [SAM 2 installation instructions](https://github.com/facebookresearch/sam2).

3. Download the SAM 2 checkpoints and place the folder inside the `segment-anything-2`.

4. Replace the contents of the `sam2` folder with the ones from `segment-anything-2`. Once the replacement is complete, you may safely delete the `segment-anything-2` folder.

## Repository Structure

- **unet_segmentations**: Contains different quality segmentations generated with a UNet.
- **src**: Contains the supported models along with scripts for preprocessing, generating the datasets and evaluation
- **run_inference.py**: The script used to run the method.

## Usage

To run inference with the trained models, execute:

```bash
python run_inference.py --dataset <dataset_path> --classifier <classifier_name> --output_file <output_file_path>
