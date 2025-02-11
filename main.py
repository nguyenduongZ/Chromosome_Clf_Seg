from dataset import ChromSeg
from trainer import train_model
from model import UNet_plus2
from loss import FocalLoss
from torch import nn, optim
from torch.optim import Adam
from datetime import datetime
from metrics import meanIOU_per_image, Score
from torch.utils.data import DataLoader, random_split
from torchvision.transforms import transforms

import os, sys
import argparse
import numpy as np
import torch
import random
import cv2
import wandb

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default='1')
    parser.add_argument('--model-type', type=str, default='UNet_plus2')
    parser.add_argument('--dataset', type=str, default='dataset')
    parser.add_argument('--idx', type=int, default=0, help='device index used in training')
    parser.add_argument('--wandb', action='store_true', help='toggle to use wandb for online saving')
    parser.add_argument('--log', action='store_true', help='wandb logging')
    args = parser.parse_args()
    assert args.model_type in ['UNet', 'UNet_plus', 'UNet_plus2']
    assert args.dataset in ['dataset']
    
    if args.dataset == 'dataset':
        IMAGE_SIZE = 256
        batch_size = 8
    else:
        print("wrong")

    torch.manual_seed(args.seed)    # reproducible
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    if args.model_type == 'UNet':
        print('Using original UNet')
        model = UNet(3, 1)
        model = model.to(device)
    elif args.model_type == 'UNet_plus':
        print('UNet_plus')
        model = UNet_plus(3, 1)
        model = model.to(device)
    elif args.model_type == 'UNet_plus2':
        print('UNet_plus2')
        model = UNet_plus2(3, 1)
        model = model.to(device)
    else:
        raise Exception('Invalid model type: %s'% args.model_type)

    x_transforms = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    y_transforms = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])

    gamma=1
    alpha=0.75
    criterion = FocalLoss(gamma, alpha, reduction='mean')
    optimizer = optim.Adam(model.parameters())

    root = "/media/mountHDD3/data_storage/z2h/chromosome/dataset/source/original_dataset"

    dataset_train, dataset_val = random_split(ChromSeg(root, transform=x_transforms, target_transform=y_transforms), (0.8, 0.2))
    
    dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=0)
    dataloader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=True, num_workers=0)
    
    model = train_model(args, model, criterion, optimizer, dataloader_train, dataloader_val, num_epochs=100, patience=7)
    
    model = model.cpu()
    dataloader_val = DataLoader(dataset_val, batch_size=1, shuffle=True, num_workers=0)

    iou = 0
    iou2 = 0
    n = 0
    model.eval()
    with torch.no_grad():
        for x, target, c in dataloader_val:
            n = n + 1
            y = model(x)
            y_pred_0 = torch.squeeze(y[0]).numpy()
            y_pred = np.zeros((256,256))
            y_pred[y_pred_0 > 0.5] = 1.0
            y_2_0 = torch.squeeze(y[1]).numpy()
            y_2 = np.zeros((256,256))
            y_2[y_2_0 > 0.5] = 1.0
            y_true = torch.squeeze(target).numpy()
            y_true2 = torch.squeeze(c).numpy()
            output1 = np.reshape(y_pred * 255,(256,256))
            output2 = np.reshape(y_2 * 255,(256,256))

            x_image = torch.squeeze(x).numpy()
            image = np.dstack((x_image[0,...]*255, x_image[1,...]*255, x_image[2,...]*255))

            cv2.imwrite('output1/' + str(n) + ".png", output1)
            cv2.imwrite('output2/' + str(n) + ".png", output2)
            cv2.imwrite('img/' + str(n) + ".png", image)
            try:
                iou += meanIOU_per_image(y_pred, y_true)
                iou2 += meanIOU_per_image(y_2, y_true2)
            except Exception as e:
                print(e)
                print('y_pred: %s'% y_pred)
                print('y_true: %s'% y_true)
                torch.save(model.state_dict(), './models/%s_seed%s_error.pth' % (args.model_type, args.seed))
                sys.exit(0)
    IoU = float(iou/n)
    IoU2 = float(iou2/n)
    
    print('Final_IoU: %s'% IoU)
    print('Final_IoU2: %s'% IoU2)
    # torch.save(model.state_dict(), './models/%s_%s_seed%s_IoU-%0.4f_IoU-%0.4f_gamma-%d_alpha-%0.2f.pth' % (args.model_type, args.dataset, args.seed, IoU, IoU2, gamma, alpha))