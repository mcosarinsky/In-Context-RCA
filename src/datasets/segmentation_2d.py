import pathlib
import numpy as np
from itertools import chain
from typing import Optional, Callable
from torch.utils.data import Dataset
from ..utils.data_transforms import process_img
from ..utils.io import load_test_folder, load_train_folder


class Seg2D_Dataset(Dataset):
    def __init__(self, split: str, dataset: str, target_size: int = 256, transform: Optional[Callable] = None, grayscale: bool = True):
        if split not in ['Train', 'Test']:
            raise ValueError("Invalid split. Must be one of 'Train', 'Test'")
        datasets = ['hc18', 'camus', 'psfhs', 'scd', 'jsrt', 'ph2', 'isic 2018', '3d-ircadb/liver', 'nucls', 'wbc/cv', 'wbc/jtsc']
        if dataset not in datasets:
            raise ValueError(f"Invalid dataset. Must be one of {datasets}")
        
        dataset = dataset.upper()
        self.grayscale = grayscale
        self.path = pathlib.Path(f'datasets/{dataset}')
        self.dataset = dataset
        self.split = split
        self.target_size = target_size
        self.transform = transform
        self._data = self.load_data()

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        if self.split == 'Test':
            img_file, label_file, seg_file = self._data[idx]
            img = process_img(img_file, self.target_size, grayscale=self.grayscale)
            gt = process_img(label_file, self.target_size, is_seg=True)
            seg = process_img(seg_file, self.target_size, is_seg=True)
            sample = {'image': img, 'GT': gt, 'seg': seg, 'name': img_file.name}
        else:
            img_file, label_file = self._data[idx]
            img = process_img(img_file, self.target_size, grayscale=self.grayscale)
            gt = process_img(label_file, self.target_size, is_seg=True) 
            sample = {'image': img, 'GT': gt, 'name': img_file.name}

        # Normalize ground truth for binary masks
        if len(np.unique(gt)) == 2:
            gt = gt / gt.max()
            sample['GT'] = gt.astype(np.uint8)
        
        if self.transform:
            self.transform(sample)

        if '3D-IRCADB' in self.dataset:
            name = '_'.join(img_file.parts[-2:])
            sample['name'] = name
        return sample

    def load_data(self):
        if self.split == 'Test':
            return load_test_folder(self.path / 'Test', self.dataset)
        else:
            return load_train_folder(self.path / 'Train')
        

class CustomImageDataset(Dataset):
    def __init__(self, folder, type: bool, target_size: int = 256, transform: Optional[Callable] = None):
        """
        Custom dataset class for loading image-mask pairs.
        
        Assumes the following directory structure:
        - folder/
            - images/ (contains image files)
            - masks/ (contains corresponding mask files assuming same name as its image pair)
        
        Parameters:
        - folder (str): Path to the dataset folder containing 'images' and 'masks' subdirectories.
        - target_size (int): Size to which the images and masks will be resized (default is 256).
        - transform (Callable, optional): A function/transform to apply to each sample (default is None).
        """
        if type not in ['reference', 'eval']:
            raise ValueError("Invalid type. Choose either 'reference' or 'eval'.")
        
        self.type = type
        self.target_size = target_size
        self.transform = transform
        self._data = self.load_data(pathlib.Path(folder))

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        img_file, mask_file = self._data[idx]
        img = process_img(img_file, self.target_size)
        mask = process_img(mask_file, self.target_size, is_seg=True)

        if self.type == 'reference':
            sample = {'image': img, 'GT': mask}
        else:
            sample = {'image': img, 'seg': mask}
        
        if self.transform:
            sample = self.transform(sample)
        return sample
    
    def load_data(self, path):
        img_dir = path / 'images'
        mask_dir = path / 'masks'
        
        paths = []
        extensions = ['*.png', '*.bmp', '*.jpg', '*.tiff']
        img_files = sorted(chain.from_iterable(img_dir.rglob(ext) for ext in extensions))
        
        for img_path in img_files:
            relative_path = img_path.relative_to(img_dir)  # Get relative path from img_dir
            name = str(relative_path.with_suffix('')) 
            mask_path = next(mask_dir.glob(f"{name}*"))
            paths.append((img_path, mask_path))

        return paths