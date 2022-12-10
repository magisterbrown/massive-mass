import torch
import torch_xla
import torch.nn as nn
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp
import os
res = ""
for i in range(8):
    res+=f

os.environ['XRT_DEVICE_MAP'] ="CPU:0;/job:localservice/replica:0/task:0/device:XLA_CPU:0" 
os.environ['XRT_WORKERS'] = "localservice:0;grpc://localhost:40934"
#os.environ['XRT_TPU_CONFIG']="tpu:0;./xla.yaml"
torch_xla._XLAC._xla_num_devices()
import numpy as np
#from biopack.copy_env import copy_process
import time
#from torchdata.datapipes.iter import IterableWrapper
##copy_process(74)
#torch_xla.set_default_tpu_config('/etc/torch/xla.yaml')
#
#
print(torch_xla._XLAC._xla_get_devices())

def mp_fn(index):
      dev = xm.xla_device()
      model = nn.Linear(2,3)
      model.to(dev)
      cpu = torch.device('cpu')
      model.to(cpu)
      xm.rendezvous('init')
      time.sleep(1)
xmp.spawn(mp_fn,  nprocs=1, start_method='fork')
