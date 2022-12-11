import torch
import os
#os.environ['XRT_DEVICE_MAP'] ="CPU:0;/job:localservice/replica:0/task:0/device:XLA_CPU:0" 
#os.environ['XRT_WORKERS'] = "localservice:0;grpc://localhost:40934"
import torch.nn as nn
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp
import numpy as np
from biopack.copy_env import copy_process
import time
from torchdata.datapipes.iter import IterableWrapper
copy_process(73)

train = np.load('data/tr_links.npy',allow_pickle=True)[:1000]
test = np.load('data/ts_links.npy',allow_pickle=True)
from biopack.trains.xla_muli import XLAMultiTrainer
trr = XLAMultiTrainer('data/res.pth',train, test, 1)
print(trr)
trr.train()

