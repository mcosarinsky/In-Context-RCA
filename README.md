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
- **scripts**: Contains `run_inference.py` and `custom_inference.py` scripts for reproducing experiments and running the method on user's custom data.

## Datasets
The following datasets were used:

- [SCD](https://www.cardiacatlas.org/sunnybrook-cardiac-data/)  
- [HC18](https://zenodo.org/records/1327317)  
- [PSFHS](https://zenodo.org/records/10969427)  
- [JSRT](http://db.jsrt.or.jp/eng.php)  
- [3D-IRCAdB](https://www.ircad.fr/research/data-sets/liver-segmentation-3d-ircadb-01/)  
- [PH<sup>2</sup>](https://www.fc.up.pt/addi/ph2%20database.html)  
- [NuCLS](https://sites.google.com/view/nucls/single-rater?authuser=0)  
- [ISIC 2018](https://challenge.isic-archive.com/data/#2018)  
- [WBC](https://data.mendeley.com/datasets/w7cvnmn4c5/1)

## Usage

To reproduce experiments on the datasets used in the paper, execute the following command with the `run_inference.py` script:

```bash
python run_inference.py --dataset <dataset_path> --classifier <classifier_name> --output_file <output_file_path>
```

To run inference on your own data, you can use the `custom_inference.py` script:

```bash
python custom_inference.py --ref_dataset <reference_dataset_path> --eval_dataset <eval_data_path> --n_classes <num_of_classes> --classifier <classifier_name> --output_file <output_file_path> 
```

## Citation
If you are using our masks please cite our work:

```
@misc{cosarinsky2025incontextreverseclassificationaccuracy,
      title={In-Context Reverse Classification Accuracy: Efficient Estimation of Segmentation Quality without Ground-Truth}, 
      author={Matias Cosarinsky and Ramiro Billot and Lucas Mansilla and Gabriel Gimenez and Nicolas Gaggión and Guanghui Fu and Enzo Ferrante},
      year={2025},
      eprint={2503.04522},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2503.04522}, 
}
```
