from .base_command import BaseCommand
import pandas as pd
import numpy as np
import numpy.ma as ma
from biopack.datastuff.s3loading import get_ds_s1_list, get_ds_s1_lab_list
from tqdm import tqdm

    
class GetStatsBase(BaseCommand):
    def __init__(self, inputs):
        self.add_arg('p','path','input path to files', path=True)
        self.add_arg('o','out','output filename')
        super().__init__(inputs)
        
    def submit(self):
        ds = self.get_links_ds(self.s3links)
        alldfs = list()
        for bc in tqdm(ds):
            names, arl= list(zip(*bc))
            packed = np.stack(arl)
            bp = batch_process(packed)
            bp['names'] = names
            alldfs.append(pd.DataFrame(bp))
            break

        res = pd.concat(alldfs,ignore_index=True)
        res.to_csv(f'{self.data}{self.args.out}')

    def get_links_ds(self, links: list):
        return get_ds_s1_list(links).batch(self.bs)

class GetStats(GetStatsBase):
    
    def __init__(self, inputs):
        super().__init__(inputs)

        self.data = self.args.path
        features = pd.read_csv(f'{self.data}features_metadata.csv')
        self.bs = 5
        s1_august = features[np.logical_and(features['satellite']=='S1', features['month']=='August')]
        self.s3links = s1_august['s3path_eu'].values

class GetOutStats(GetStatsBase):

    def __init__(self, inputs):
        super().__init__(inputs)

        self.data = self.args.path
        features = pd.read_csv(f'{self.data}train_agbm_metadata.csv')
        self.bs = 5
        self.s3links = features['s3path_eu'].values

    def get_links_ds(self, links: list):
        return get_ds_s1_lab_list(links).batch(self.bs)


def batch_process(batch: np.array):
    axis = (1,2)
    mask = batch<-9998
    batch = ma.masked_array(batch, mask=mask)
    cols = { 'mean': np.mean(batch, axis=axis),
            'misses': np.sum(mask, axis=axis),
            'mean_of_squares': np.mean(np.square(batch), axis=axis),
            'max': batch.max(axis=axis),
            'min': batch.min(axis=axis),
            }
    res = dict()
    for k,v in cols.items():
        for key,col in enumerate(v.T):
            res[f'{k}_{key}'] = col

    return res
 
