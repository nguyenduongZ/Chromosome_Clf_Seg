import os, sys
import cv2
import numpy as np
import argparse
import torch
import torch.nn.functional as F
import albumentations as A

from torch import Tensor
from typing import *
from PIL import Image
from rich.progress import track
from torch.utils.data import Dataset, DataLoader, random_split

class ChromSeg(Dataset):
    def __init__(self, root = _root, split = 'train'):
        self.root = root
        self._split = split
        self.__mode = "train" if self._split == 'train' else 'test'

        self.resize = A.Compose(
            [
                A.Resize(256, 256),
            ]
        )

        self.aug_transforms = A.Compose(
            [
                A.HorizontalFlip(p=0.2),
            ]
        )

        self.norm = A.Compose(
            [
                A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
            ]
        )

        from glob import glob
        self._images = sorted(glob(self.root + "/images/*"))
        self._masks1 = sorted(glob(self.root + "/masks1/*"))
        self._masks2 = sorted(glob(self.root + "/masks2/*"))

        print("Data Set Setting Up. Done")
    
    def __len__(self):
        return len(self._images)
    
    def __getitem__(self, idx):
        image = np.array(Image.open(self._images[idx]))
        mask1 = np.array(Image.open(self._masks1[idx])) #overlapped
        mask2 = np.array(Image.open(self._masks2[idx])) #nonverlapped

        resized = self.resize(image=image, masks=[mask1, mask2])

        if self.__mode == "train":
            transformed = self.aug_transforms(image=resized['image'], masks= resized['masks'])
            transformed_image = self.norm(image=transformed['image'])['image']
            transformed_mask1 = transformed['masks'][0]
            transformed_mask2 = transformed['masks'][1]
        else:
            transformed_image = self.norm(image=transformed['image'])['image']
            transformed_mask1 = transformed['masks'][0]
            transformed_mask2 = transformed['masks'][1]

        torch_image = torch.from_numpy(transformed_image).permute(2, 0, 1).float()
        torch_mask1 = torch.from_numpy(transformed_mask1).float()
        torch_mask2 = torch.from_numpy(transformed_mask2).float()

        return torch_image, torch_mask1, torch_mask2        
          
    @property
    def mode(self):
        return self.__mode 
    
    @mode.setter
    def mode(self, m):
        if m not in ['train', 'test']:
            raise ValueError(f"mode cannot be {m} and must be ['train', 'test']")
        else:
            self.__mode = m

def get_ds(args):
    train_ds, test_ds = random_split(ChromSeg(), (0.90, 0.10))

    train_dl = DataLoader(train_ds, batch_size=args.bs, shuffle=True, pin_memory=args.pm, num_workers=args.wk)
    test_dl = DataLoader(test_ds, batch_size=args.bs, shuffle=True, pin_memory=args.pm, num_workers=args.wk)

    return args, train_dl, test_dl