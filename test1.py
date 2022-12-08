import torch
import torch.nn as nn
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp
import numpy as np
from biopack.copy_env import copy_process
import time
from torchdata.datapipes.iter import IterableWrapper
copy_process(74)

def mp_fn(index):
      print(index)
      model = nn.Linear(2,3)
      dev = xm.xla_device()
      model.to(dev)
      cpu = torch.device('cpu')
      model.to(cpu)
      xm.rendezvous('init')
      time.sleep(1)
xmp.spawn(mp_fn,  nprocs=8, start_method='fork')


