import torch
import os
import torch.nn as nn
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp
import segmentation_models_pytorch as smp
from biopack.datastuff.train_loader import InputS1Loader, get_zds_from_sus
from torchdata.datapipes.iter import IterableWrapper 
from torch.utils.data import DataLoader
from multiprocessing import Manager
import torch_xla.utils.serialization as xser
import torch.optim.lr_scheduler as lr_scheduler
from multiprocessing import Value
import optuna

import numpy as np
import time
from tqdm import tqdm
import struct

class BestScore:
    def __init__(self):
        self.min_step = 0
        self.min_score = float('inf')
        self.alpha = 0.1

    def update(self, step, score):
        if(score< self.min_score+self.alpha):
            self.min_score = score
            self.min_step = step
        elif(step-self.min_step>3):
            return True

        return False

class XLAMultiTrainer:
    def __init__(self, save_pth, trial_id, storage, name, train_sus, test_sus, procs):
        self.save_pth = save_pth
        self.flags = dict()
        self.flags['seed'] = 420
        self.procs = procs
        self.trains = np.array_split(train_sus, procs)
        self.test_sus = np.array_split(test_sus, procs)
        self.inp_proc = InputS1Loader()

        self.trial_id = trial_id
        self.storage = storage
        self.name = name

    def init_model(self):
        model = smp.Unet(
                    encoder_name="resnet34",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
                    encoder_weights=None,     # use `imagenet` pre-trained weights for encoder initialization
                    in_channels=4,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
                    classes=1)                      # model output channels (number of classes in your dataset)
        return model

    def train(self, hyper_params=dict()):
        torch.manual_seed(self.flags['seed'])
        model = self.init_model()        
        self.res = Value('f', 100.0)                        
        self.res_step = Value('i', 0)                        
        self.prune = Value('b', False)                        
        self.model=xmp.MpModelWrapper(model)
        xmp.spawn(self.mp_fn, args=(hyper_params, ),  nprocs=self.procs, start_method='fork')
        #if self.prune.value:
        #    raise optuna.TrialPruned()
        return self.res.value, self.res_step.value

    def mp_fn(self, index, hyper_params):
        torch.manual_seed(self.flags['seed'])
        device = xm.xla_device()
        inside_model = self.model.to(device)
        self.loaded_study = optuna.load_study(study_name=self.name, storage=self.storage)
        self.bst = BestScore()
        try:
            pass
            self.train_loop(inside_model,  **hyper_params)
        except optuna.exceptions.TrialPruned:
            pass

        if xm.is_master_ordinal():
            print('Train Done')
            device = torch.device('cpu')
            inside_model = inside_model.to(device)
            torch.save(inside_model.state_dict(), self.save_pth)
        xm.rendezvous('alldone')
        time.sleep(1)


    def train_loop(self,inside_model,  
            epochs=2,
            batch_size=4,
            lr=0.02,
            b1=0.9,
            b2=0.999,
            weight_decay=0.01,
            slide=0.1):
        inside_model.train()
        device = xm.xla_device()
        buff = 32
        train_ds = self.get_train_dataset(self.trains).prefetch(buff).shuffle(buffer_size=buff)
        train_dl = DataLoader(train_ds, batch_size=batch_size,shuffle=False, drop_last=True)
        loss_module = nn.MSELoss(reduction='mean')
        optimizer = torch.optim.AdamW(inside_model.parameters(), lr=lr,betas=(b1,b2),weight_decay=weight_decay)
        sz=self.trains[0].shape[0]
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(sz/batch_size)*epochs, eta_min=lr*slide)
        for ep in range(epochs):
            lossed = -1
            inside_model.train()
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
                scheduler.step()
                if(xm.is_master_ordinal()):
                    prog.update(inputs.shape[0])
                    lossed = loss.item() if lossed==-1 else lossed*0.9+loss.item()*0.1
                    prog.set_description(f'Running loss {loss}')
                    prog.refresh()
                xm.mark_step()
            if(xm.is_master_ordinal()):
                prog.close()
                print(f'Epoch {ep} finished')
            self.validation_loop(inside_model, ep, batch_size)
        pass
    def validation_loop(self, model, step, batch_size):
        loss_module = nn.MSELoss(reduction='none')
        device = xm.xla_device()
        model.eval()
        sz = self.test_sus[0].shape[0]
        test_ds = self.get_train_dataset(self.test_sus).prefetch(8)
        test_dl = DataLoader(test_ds, batch_size=4, shuffle=False, drop_last=False)
        ct = 0 
        summs = 0
        for i, batch in enumerate(test_dl):
            inputs, lables = batch
            inputs = inputs.to(dtype=torch.float32,device=device)
            lables = lables.to(dtype=torch.float32,device=device)

            with torch.no_grad():
                out = model(inputs)
                loss = self.rmse(loss_module(out, lables))
                summs+=loss.item()
                ct+=1

        fins = xm.rendezvous('finval', payload=struct.pack('f',summs/ct))
        if xm.is_master_ordinal():
            if fins:
                fins = [struct.unpack('f', x) for x in fins]
                print(fins)
                res = np.mean(fins)
                print(f'Final avg rmse {res}')
            else:
                res = summs/ct
                print(f'Final rmse {res}')

            self.prune.value = self.bst.update(step, res)
            if res > 80:
                self.prune.value=True
            if res != res:
                self.prune.value=True
            self.res.value=self.bst.min_score
            self.res_step.value=self.bst.min_step
            #ntr = self.loaded_study._storage._backend.get_trial(self.trial_id)
            #ntr.report(res, step)
            #self.prune.value = ntr.should_prune()

        xm.rendezvous('prune')
        time.sleep(1)
        if self.prune.value:
            raise optuna.TrialPruned()

    @staticmethod
    def rmse(batch):
        mn = torch.mean(batch, [1,2,3])
        sq = torch.sqrt(mn)
        return torch.mean(sq)

    def get_train_dataset(self, subs):
        orbit = xm.get_ordinal()
        return get_zds_from_sus(subs[orbit], self.inp_proc)





