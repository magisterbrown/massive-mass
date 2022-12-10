import torch
import os
import torch.nn as nn
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp
import segmentation_models_pytorch as smp
from biopack.datastuff.train_loader import InputS1Loader, get_zds_from_sus
from torchdata.datapipes.iter import IterableWrapper 
from torch.utils.data import DataLoader
import torch_xla.utils.serialization as xser

import numpy as np
import time
from tqdm import tqdm

class XLAMultiTrainer:
    def __init__(self, save_pth, train_sus, test_sus, procs):
        self.save_pth = save_pth
        self.flags = dict()
        self.flags['seed'] = 420
        self.procs = procs
        self.trains = np.array_split(train_sus, procs)
        self.test_sus = test_sus
        self.inp_proc = InputS1Loader()

    def train(self, hyper_params=dict()):
        model = smp.Unet(
                    encoder_name="resnet34",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
                    encoder_weights=None,     # use `imagenet` pre-trained weights for encoder initialization
                    in_channels=4,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
                    classes=1,                      # model output channels (number of classes in your dataset)
                                )
        self.model=xmp.MpModelWrapper(model)
        xmp.spawn(self.mp_fn, args=(self.trains, hyper_params),  nprocs=self.procs, start_method='fork')

    def mp_fn(self, index, splits, hyper_params):
        torch.manual_seed(self.flags['seed'])
        device = xm.xla_device()
        inside_model = self.model.to(device)
        self.train_loop(inside_model, splits[xm.get_ordinal()], **hyper_params)

        if xm.is_master_ordinal():
            print('Train Done')
            device = torch.device('cpu')
            inside_model = inside_model.to(device)
            torch.save(inside_model.state_dict(), self.save_pth)

    def train_loop(self,inside_model, 
            split, 
            epochs=2,
            batch_size=3):
        inside_model.train()
        device = xm.xla_device()
        train_ds = self.get_train_dataset(split)
        train_dl = DataLoader(train_ds, batch_size=3, drop_last=True)
        loss_module = nn.MSELoss(reduction='mean')
        optimizer = torch.optim.Adam(inside_model.parameters(), lr=0.02)
        sz=split.shape[0]
        for ep in range(epochs):
            break
            ct=0
            if xm.is_master_ordinal():
                prog = tqdm(total=sz)
            for i, batch in enumerate(train_dl):
                inputs, lables = batch
                inputs = inputs.to(dtype=torch.float32,device=device)
                lables = lables.to(dtype=torch.float32,device=device)

                optimizer.zero_grad()
                out = inside_model(inputs)
                loss = loss_module(out, lables)
                loss.backward()
                xm.optimizer_step(optimizer)
                xm.mark_step()
                if(xm.is_master_ordinal()):
                    prog.update(inputs.shape[0])
                if(ct>=3):
                    break
                ct+=1 
            if(xm.is_master_ordinal()):
                prog.close()
                print(f'Epoch {ep} finished')
                self.validation_loop(inside_model)
        pass
    def validation_loop(self, model):
        with torch.no_grad():
            loss_module = nn.MSELoss(reduction='mean')
            device = xm.xla_device()
            model.eval()
            test_ds = self.get_train_dataset(self.test_sus)
            test_dl = DataLoader(test_ds, batch_size=1, drop_last=False)
            
            for i, batch in enumerate(test_dl):
                inputs, lables = batch
                print(i)
                inputs = inputs.to(dtype=torch.float32,device=device)
                lables = lables.to(dtype=torch.float32,device=device)

                out = model(inputs)
                loss = loss_module(out, lables)
                print(loss)



            pass

    def get_train_dataset(self, subs):
        return get_zds_from_sus(subs, self.inp_proc)





