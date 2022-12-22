import torch
import os
os.environ['XLA_USE_32BIT_LONG'] = '1'
os.environ['XLA_USE_BF16'] = '1'
# os.environ['XLA_SAVE_TENSORS_FILE'] = 'tensors.log'
# os.environ['XLA_SAVE_TENSORS_FMT'] = 'text'
os.environ['XLA_TRIM_GRAPH_SIZE'] = '1000000'
#os.environ['XRT_DEVICE_MAP'] ="CPU:0;/job:localservice/replica:0/task:0/device:XLA_CPU:0" 
#os.environ['XRT_WORKERS'] = "localservice:0;grpc://localhost:40934"
import torch.nn as nn
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp
import numpy as np
from biopack.copy_env import copy_process
import time
from torchdata.datapipes.iter import IterableWrapper
import optuna
copy_process(5719)

from biopack.trains.xla_muli import XLAMultiTrainer
train = np.load('data/tr_links.npy',allow_pickle=True)[:156]
test = np.load('data/ts_links.npy',allow_pickle=True)[:64]

def objective(trial):
    params = {
            'epochs':trial.suggest_int("epochs", 2, 5),
            'batch_size':4,#trial.suggest_int("bs", 4, 8),
            'lr':trial.suggest_float("lr", 1e-3, 0.1, log=True),
           'b1':trial.suggest_float("b1", 0.4, 0.999),
            'b2':trial.suggest_float("b2", 0.6, 0.9999),
            'weight_decay':trial.suggest_int("weight_decay", 0, 0.05),
            'slide':trial.suggest_float("slide", 1e-4, 1, log=True)
    }
    trr = XLAMultiTrainer('data/res.pth', trial ,train, test, 8)
    rest = trr.train(params)
    return rest


study = optuna.create_study(direction='minimize', pruner=optuna.pruners.MedianPruner())
study.optimize(objective, n_trials=2)





