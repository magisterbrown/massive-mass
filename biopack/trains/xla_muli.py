import torch
import torch.nn as nn
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp
import segmentation_models_pytorch as smp
from biopack.datastuff.train_loader import InputS1Loader, get_zds_from_sus
from torchdata.datapipes.iter import IterableWrapper 
from torch.utils.data import DataLoader
import numpy as np

class XLAMultiTrainer:
    def __init__(self, save_pth, train_sus, test_sus):
        self.save_pth = save_pth
        self.flags = dict()
        self.flags['seed'] = 420
        self.trains = np.array_split(train_sus, 8)
        self.test_sus = test_sus
        self.inp_proc = InputS1Loader()

    def train(self):
        model = smp.Unet(
                    encoder_name="resnet34",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
                    encoder_weights=None,     # use `imagenet` pre-trained weights for encoder initialization
                    in_channels=4,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
                    classes=1,                      # model output channels (number of classes in your dataset)
                                )
        self.model=xmp.MpModelWrapper(model)
        xmp.spawn(self.mp_fn,  nprocs=8, start_method='fork')
        pass

    def mp_fn(self, index):
        torch.manual_seed(self.flags['seed'])
        print(index)
        xm.rendezvous('init')
        device = xm.xla_device()
        self.model = self.model.to(device)
        self.train_loop()
        if xm.is_master_ordinal():
            device = torch.device("cpu")
            self.model = self.model.to(device)
            torch.save(self.model.state_dict(), self.save_pth)

    def train_loop(self, epochs=1):
        self.model.train()
        train_ds = self.get_train_dataset()
        train_dl = DataLoader(train_ds, batch_size=4)
        loss_module = nn.MSELoss(reduction='mean')
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.02)
        print('before llop')
        for ep in range(epochs):
            print(f'Rpoch {ep}')
            for i, batch in enumerate(train_dl):
                inputs, lables = batch
                print(inputs.shape)
                break
            break
        pass

    def get_train_dataset(self):
        ordinal = xm.get_ordinal()
        subs = self.trains[ordinal]

        return get_zds_from_sus(subs, self.inp_proc)





