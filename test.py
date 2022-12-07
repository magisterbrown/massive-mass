import torch
import torch.nn as nn
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp
import numpy as np

train = np.load('data/tr_links.npy',allow_pickle=True)
print(train)

arsp = np.array_split(train,8)
def mp_fn(index):
    xm.rendezvous('init')
    print(f'{index} sst')
    #trr = splits[xm.get_ordinal()]
    #zpp = IterableWrapper(trr[:,0]).open_files_by_fsspec(mode="rb", anon=True).webdataset()
    #print(next(iter(zpp)))
    # if xm.is_master_ordinal():
    #   print(next(iter(zpp
xmp.spawn(mp_fn,  nprocs=8, start_method='fork')
