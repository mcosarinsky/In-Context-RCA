import pathlib
import numpy as np
from typing import Optional, Callable
from torch.utils.data import Dataset
from ..utils.data_transforms import process_img
from ..utils.io import load_train_folder

class TrainerDataset(Dataset):
    def __init__(self, split: str, dataset: str, target_size: int = 256, transform: Optional[Callable] = None, grayscale: bool = True, transform_img: bool = True):
        if split not in ['Train', 'Test']:
            raise ValueError("Invalid split. Must be one of 'Train', 'Test'")
        datasets = ['hc18', 'camus', 'psfhs', 'scd', 'jsrt', 'ph2', 'isic 2018', '3d-ircadb/liver', 'nucls', 'wbc/cv', 'wbc/jtsc']
        if dataset not in datasets:
            raise ValueError(f"Invalid dataset. Must be one of {datasets}")
        
        dataset = dataset.upper()
        self.grayscale = grayscale
        self.transform_img = transform_img
        self.path = pathlib.Path(f'datasets/{dataset}')
        self.dataset = dataset
        self.split = split
        self.target_size = target_size
        self.transform = transform
        self._data = self.load_data()

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        img_file, label_file = self._data[idx]
        img = process_img(img_file, self.target_size, grayscale=self.grayscale, transform=self.transform_img)
        gt = process_img(label_file, self.target_size, is_seg=True) 
        sample = {'image': img, 'GT': gt, 'name': img_file.name}

        # Normalize ground truth for binary masks
        if len(np.unique(gt)) == 2:
            gt = gt / gt.max()
            sample['GT'] = gt.astype(np.uint8)

        if '3D-IRCADB' in self.dataset:
            name = '_'.join(img_file.parts[-2:])
            sample['name'] = name
        
        if self.transform:
            self.transform(sample)

        return sample

    def load_data(self):
        if self.split == 'Test':
            return load_train_folder(self.path / 'Test')
        else:
            return load_train_folder(self.path / 'Train')