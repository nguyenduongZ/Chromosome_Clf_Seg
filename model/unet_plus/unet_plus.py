import torch
from torch import nn 
from torchvision.transforms import transforms
import torchvision.models as models
from .unet_plus_core import *

class UnetPlus(nn.Module):
    def __init__(self, args):
        super(UnetPlus, self).__init__()

        self.seg_n_classes = args.seg_n_classes
        self.init_ch = args.init_ch
        n1 = 64
        filters = [n1, n1 * 2, n1 * 4, n1 * 8]

        self.contour = nn.Sequential(
            nn.Conv2d(filters[0]*2, filters[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(filters[0], filters[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(filters[0], self.seg_n_classes, 1)
        )

    def forward(self, x):
        x1 = self.encoder[0](x)
        x2 = self.encoder[1](x1)
        x3 = self.encoder[2](x2)
        x4 = self.encoder[3](x3)
        x5 = self.encoder[4](x4)
        
        x = self.decoder[0](x5, x4)
        x = self.decoder[1](x, x3)
        x = self.decoder[2](x, x2)
        x = self.decoder[3](x, x1)

        return self.decoder[4](x)