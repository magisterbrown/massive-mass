import torch
import torch.nn as nn
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp
import numpy as np
from biopack.copy_env import copy_process
import time
from torchdata.datapipes.iter import IterableWrapper
copy_process(2946)

train = np.load('data/tr_links.npy',allow_pickle=True)
from biopack.trains.xla_muli import XLAMultiTrainer
trr = XLAMultiTrainer('../res.pth',train, 0)
print(trr)
trr.train()

