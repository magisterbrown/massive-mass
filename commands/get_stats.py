from .base_command import BaseCommand
import pandas as pd
import numpy as np
from biopack.datastuff.s3loading import get_ds_s1_list

class GetStats(BaseCommand):
    def __init__(self, inputs):
        self.add_arg('p','path','input path to files', path=True)
        super().__init__(inputs)
        
        data = self.args.path
        self.features = pd.read_csv(f'{data}features_sample.csv')
        self.masks = pd.read_csv(f'{data}train_agbm_metadata.csv')
        self.bs = 5

    def submit(self):
        s1_august = self.features[np.logical_and(self.features['satellite']=='S1', self.features['month']=='August')]
        s3links = s1_august['s3path_eu'].values
        ds = get_ds_s1_list(s3links).batch(self.bs)
        alldfs = list()
        el =0
        for bc in ds:
            names, arl= list(zip(*bc))
            packed = np.stack(arl)
            bp = batch_process(packed)
            bp['names'] = names
            alldfs.append(pd.DataFrame(bp))
            el+=1
            if el==3:
                break

        print(pd.concat(alldfs,ignore_index=True))


def batch_process(batch: np.array):
    axis = (1,2)
    mask = batch>-9998
    cols = { 'mean': np.mean(batch, axis=axis, where=mask),
            'misses': np.sum(np.logical_not(mask), axis=axis),
            'mean_of_squares': np.mean(np.square(batch), axis=axis, where=mask),
            'max': batch.max(axis=axis, where=mask, initial=-np.inf),
            'min': batch.min(axis=axis, where=mask, initial=np.inf),
            }
    res = dict()
    for k,v in cols.items():
        for key,col in enumerate(v.T):
            res[f'{k}_{key}'] = col

    return res
 
