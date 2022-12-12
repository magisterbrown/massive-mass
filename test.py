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
copy_process(72)

train = np.load('data/tr_links.npy',allow_pickle=True)[:32]
test = np.load('data/ts_links.npy',allow_pickle=True)[:16]
from biopack.trains.xla_muli import XLAMultiTrainer
trr = XLAMultiTrainer('data/res.pth',train, test, 1)
print(trr)
trr.train()

