import torch
import torch.nn as nn
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp
import numpy as np
from biopack.copy_env import copy_process
import time
from torchdata.datapipes.iter import IterableWrapper
copy_process(59)

train = np.load('data/tr_links.npy',allow_pickle=True)
from biopack.trains.xla_muli import XLAMultiTrainer
trr = XLAMultiTrainer('gs://monet-cool-gan/res.pth',train, 0)
print(trr)
trr.train()

#arsp = np.array_split(train,8)
#def mp_fn(index, splits):
#    xm.rendezvous('init')
#    print(f'{index} sst')
#    trr = splits[xm.get_ordinal()]
#    zpp = IterableWrapper(trr[:,0]).open_files_by_fsspec(mode="rb", anon=True)
#    zpi = iter(zpp)
#    print(zpi)
#    print(next(zpi))
#    time.sleep(1)
#    #print(next(iter(zpp)))
#    # if xm.is_master_ordinal():
#    #   print(next(iter(zpp
#xmp.spawn(mp_fn,  args=(arsp,), nprocs=8, start_method='fork')
