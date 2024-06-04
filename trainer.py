from dataset import get_ds
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from datetime import datetime

import torch
import wandb
import hashlib
import json
import os

def trainer(args):
    #set up device
    if torch.cuda.is_available():
        device = torch.device("cuda", index=args.idx)
    else:
        device = torch.device("cpu")

    args, train_dl, test_dl = get_ds(args)

    print(f"TRAIN batch: {len(train_dl)}")
    print(f"TEST batch: {len(test_dl)}")

    #run_name
    now = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")

    run_dir = os.getcwd() + "/runs"
    if not os.path.exists(run_dir):
        os.mkdir(run_dir)

    sv_dir = run_dir + f"/{now}"
    if not os.path.exists(sv_dir):
        os.mkdir(run_dir)

    best_model_path = sv_dir + f'/best.pt'
    last_model_path = sv_dir + f'/last.pt'

    # model =
    