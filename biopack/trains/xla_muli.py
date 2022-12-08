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

class XLAMultiTrainer:
    def __init__(self, save_pth, train_sus, test_sus):
        self.save_pth = save_pth
        self.flags = dict()
        self.flags['seed'] = 420
        self.trains = np.array_split(train_sus, 8)
        self.test_sus = test_sus
        self.inp_proc = InputS1Loader()

    def train(self):
        print('bef spawn')
        xmp.spawn(self.mp_fn, args=(self.trains,),  nprocs=1, start_method='fork')
        #device = torch.device("cpu")
        #self.model = self.model.to(device)
        #torch.save(self.model.state_dict(), self.save_pth)

    def mp_fn(self, index, splits):
        torch.manual_seed(self.flags['seed'])
        #model = smp.Unet(
        #            encoder_name="resnet34",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
        #            encoder_weights=None,     # use `imagenet` pre-trained weights for encoder initialization
        #            in_channels=4,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
        #            classes=1,                      # model output channels (number of classes in your dataset)
        #                        )
        model = nn.Linear(3,4)

        if xm.is_master_ordinal():
            print(index)
            print(os.cpu_count())
        dp = next(model.parameters())
        old = dp.device
        print(old)
        model = model.to(old)
        device = xm.xla_device()
        #if xm.is_master_ordinal():
        #    print(std['encoder.conv1.weight'][:3])
        print(device)
        model = model.to(device)
        print(model.state_dict())
        #model = model.to(old)
        #std = inside_model.state_dict()
        #if xm.is_master_ordinal():
        #    cpu = torch.device("cpu")
        #    inside_model.to(cpu)
        #    print(std)
        #self.train_loop(inside_model, splits[xm.get_ordinal()])
        ##xm.rendezvous('save')
        ##xm.save(inside_model.state_dict(), self.save_pth)
        #xm.rendezvous('init')
        #print('Train Done')
        #std = inside_model.state_dict()
        #if xm.is_master_ordinal():
        #    print(std['encoder.conv1.weight'][:3])
        #    
        #xm.rendezvous('end')
        #time.sleep(1)
        #xm.save(inside_model.state_dict(), self.save_pth, global_master=True)

        #if xm.is_master_ordinal():
        #    from pudb.remote import set_trace
        #    set_trace(term_size=(80, 24)) 
        ###    pass
        #    device = torch.device("cpu")
        #    inside_model = inside_model.to(device)
        #    print('COnverted')
        #    pass
            #torch.save(smd.state_dict(), self.save_pth)

    def train_loop(self,inside_model, split, epochs=1):
        inside_model.train()
        device = xm.xla_device()
        train_ds = self.get_train_dataset(split)
        train_dl = DataLoader(train_ds, batch_size=3)
        loss_module = nn.MSELoss(reduction='mean')
        optimizer = torch.optim.Adam(inside_model.parameters(), lr=0.02)
        for ep in range(epochs):
            ct=0
            for i, batch in enumerate(train_dl):
                inputs, lables = batch
                inputs = inputs.to(dtype=torch.float32,device=device)
                lables = lables.to(dtype=torch.float32,device=device)

                optimizer.zero_grad()
                out = inside_model(inputs)
                print(lables.shape)
                loss = loss_module(out, lables)
                loss.backward()
                xm.optimizer_step(optimizer)
                print('Ns')
                #if(xm.is_master_ordinal()):
                #    print(loss)
                print(f'CT {ct}')
                if(ct>=4):
                    break
                ct+=1 
        pass


    def get_train_dataset(self, subs):
        return get_zds_from_sus(subs, self.inp_proc)





