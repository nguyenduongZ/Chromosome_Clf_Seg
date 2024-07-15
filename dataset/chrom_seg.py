import os
import ossaudiodev
import torch
import torch.nn.functional as F

from torch import nn, Tensor
from typing import *
from PIL import Image
from torch.utils.data import Dataset
 
root = "/media/mountHDD3/data_storage/z2h/chromosome/dataset/source/original_dataset"

def make_dataset(root):
    imgs=[]
    for filename in os.listdir(root):
        form = filename.split('_')[0]
        if form == 'image':
            tag = filename.split('_')      
            img = os.path.join(root, filename)
            mask1 = os.path.join(root,'binary_label_' + tag[1])
            mask2 = os.path.join(root,'binary_label2_' + tag[1])
            imgs.append((img, mask1, mask2))
    return imgs

class ChromSeg(Dataset):
    def __init__(self, root, transform=None, target_transform=None):
        imgs = make_dataset(root)
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        x_path, y1_path, y2_path = self.imgs[index]

        img_x = Image.open(x_path)
        img_y1 = Image.open(y1_path)
        img_y2 = Image.open(y2_path)
        if self.transform is not None:
            img_x = self.transform(img_x)
        if self.target_transform is not None:
            img_y1 = img_y1.convert('L')
            img_y2 = img_y2.convert('L')
            img_y1 = self.target_transform(img_y1)
            img_y2 = self.target_transform(img_y2)
        return img_x, img_y1, img_y2

    def __len__(self):
        return len(self.imgs)