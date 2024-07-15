from dataset import ChromSeg
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from datetime import datetime
from model import UNet_plus2
from loss import FocalLoss
from metrics import meanIOU_per_image, Score

import torch
import wandb
import json
import os
import numpy as np
import copy


def train_model(args, model, criterion, optimizer, dataloader_train, dataloader_val, num_epochs=20, patience=30): 

    if torch.cuda.is_available():
        device = torch.device("cuda", index=args.idx)
    else:
        device = torch.device("cpu")

    now = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")

    if args.log:
        run = wandb.init(
        project="Chromosome Segmentation",
        config=args,
        name=now,
        force=True
        )

    run_dir = os.getcwd() + '/runs'
    if not os.path.exists(run_dir):
        os.mkdir(run_dir)

    sv_dir = run_dir + f"/{now}"
    if not os.path.exists(sv_dir):
        os.mkdir(sv_dir)

    min_val_loss = float('inf')
    best_epoch = 0
    best_model_state = None
    last_model = sv_dir + f'/last.pt'
    for epoch in range(num_epochs):
        log_dict = {}
        dt_size = len(dataloader_train.dataset)
        
        # ----------------------TRAIN-----------------------
        model.train()
        epoch_loss = 0
        step = 0
        train_iou1 = 0
        train_iou2 = 0
        
        for x, y, c in dataloader_train:
            step += 1
            inputs = x.to(device)
            labels = y.to(device)
            labels2 = c.to(device)
            
            # zero the parameter gradients
            optimizer.zero_grad()
            
            # forward
            outputs = model(inputs)
            loss = 0.5 * criterion(outputs[0], labels) + 0.5 * criterion(outputs[1], labels2)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

            # Compute IoU for both outputs
            iou1 = meanIOU_per_image(outputs[0], labels)
            iou2 = meanIOU_per_image(outputs[1], labels2)
            train_iou1 += iou1
            train_iou2 += iou2
            
            print("%d/%d,train_loss:%0.3f" % (step, (dt_size - 1) // dataloader_train.batch_size + 1, loss.item()))
        
        train_mean_loss = epoch_loss / step
        train_mean_iou1 = train_iou1 / step
        train_mean_iou2 = train_iou2 / step
        log_dict['train/loss'] = train_mean_loss
        log_dict['train/iou1'] = train_mean_iou1
        log_dict['train/iou2'] = train_mean_iou2

        print("epoch %d training loss:%0.3f, IoU1:%0.3f, IoU2:%0.3f" % (epoch, train_mean_loss, train_mean_iou1, train_mean_iou2))
        
        # ----------------------VALIDATION-----------------------
        with torch.no_grad():
            model.eval()
            epoch_loss = 0
            step = 0
            val_iou1 = 0
            val_iou2 = 0
            
            for x, y, c in dataloader_val:
                step += 1
                inputs = x.to(device)
                labels = y.to(device)
                labels2 = c.to(device)
                
                outputs = model(inputs)
                loss = 0.7 * criterion(outputs[0], labels) + 0.3 * criterion(outputs[1], labels2)
                epoch_loss += loss.item()

                # Compute IoU for both outputs
                iou1 = meanIOU_per_image(outputs[0], labels)
                iou2 = meanIOU_per_image(outputs[1], labels2)
                val_iou1 += iou1
                val_iou2 += iou2

            val_loss = epoch_loss / step
            val_mean_iou1 = val_iou1 / step
            val_mean_iou2 = val_iou2 / step
            log_dict['valid/loss'] = val_loss
            log_dict['valid/iou1'] = val_mean_iou1
            log_dict['valid/iou2'] = val_mean_iou2

            print("epoch %d validation loss:%0.5f, IoU1:%0.3f, IoU2:%0.3f" % (epoch, val_loss, val_mean_iou1, val_mean_iou2))

            if val_loss < min_val_loss:
                best_epoch = epoch
                min_val_loss = val_loss
                best_model_state = copy.deepcopy(model.state_dict())
                torch.save({'args': args, 'model_state_dict': best_model_state}, last_model)
                
            if epoch - best_epoch > patience:
                break

            if args.log:
                run.log(log_dict)
                run.log_model(path=last_model, name=f"{now}-last-model")

    print('Best validation loss %0.5f at epoch %s' % (min_val_loss, best_epoch))
    model.load_state_dict(best_model_state)
    return model
