from .base_command import BaseCommand
import pandas as pd
import numpy as np
import numpy.ma as ma
from biopack.datastuff.s3loading import get_ds_s1_list, get_ds_s1_lab_list
from tqdm import tqdm
from multiprocessing import Pool, Process, Manager
import multiprocessing
import itertools


    
class GetStatsBase(BaseCommand):
    def __init__(self, inputs):
        self.add_arg('p','path','input path to files', path=True)
        self.add_arg('o','out','output filename')
        self.add_arg('c','coc','concurrent processes')
        super().__init__(inputs)
        self.cores = int(self.args.coc)
        
    def submit(self):
        links_p = np.array_split(self.s3links, self.cores)
        manager = Manager()

        queue = manager.Queue()
        prt = Process(target=self.printer, args=(queue,len(self.s3links)) )
        prt.start()

        links_p = [ (lk, queue) for lk in links_p]
        
        with Pool(self.cores) as p:
            proc_res = p.starmap(self.paral, links_p)
        
        queue.put('stop')
        prt.join()

        alldfs = list(itertools.chain.from_iterable(proc_res)) 
        self.save(alldfs)


    def save(self, produced: list):
        res = pd.concat(alldfs,ignore_index=True)
        res.to_csv(f'{self.data}{self.args.out}')

    def paral(self, links:list, queue):
        ds = self.get_links_ds(links)
        alldfs = list()
        for key, bc in enumerate(ds):
            names, arl= list(zip(*bc))
            packed = np.stack(arl)
            bp = self.batch_process(names, packed)
            alldfs.append(bp)
            queue.put(len(names))

        return alldfs
    
    @staticmethod
    def batch_process(names, batch):
        return dict()

    def printer(self, queue, total):
        res = 0
        pbar = tqdm(total=total)
        while(True):
            if(not queue.empty()):
                val = queue.get()
                if val=='stop':
                    break
                pbar.update(val)

        pbar.close()



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

    @staticmethod
    def batch_process(names, batch: np.array):
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
        
        res['names'] = names

        return pd.DataFrame(res)
 

class GetOutStats(GetStatsBase):

    def __init__(self, inputs):
        super().__init__(inputs)

        self.data = self.args.path
        features =pd.read_csv(f'{self.data}train_agbm_metadata.csv')
        self.bs = 32
        self.s3links = features['s3path_eu'].values

    def get_links_ds(self, links: list):
        return get_ds_s1_lab_list(links).batch(self.bs)

    @staticmethod
    def batch_process(names, batch: np.array):
        axis = (1,2)
        cols = { 'mean': np.mean(batch, axis=axis),
                'mean_of_squares': np.mean(np.square(batch), axis=axis),
                'max': batch.max(axis=axis),
                'min': batch.min(axis=axis),
                }
        cols['names'] = names

        return pd.DataFrame(cols)

class GetDistribution(GetStatsBase):
    def __init__(self, inputs):
        super().__init__(inputs)
        self.data = self.args.path
        features =pd.read_csv(f'{self.data}train_agbm_metadata.csv')
        self.bs = 64
        self.s3links = features['s3path_eu'].values

    def get_links_ds(self, links: list):
        return get_ds_s1_lab_list(links).batch(self.bs)

    @staticmethod
    def batch_process(names, batch: np.array):
        rounded = np.around(batch*4,0)/4
        counts = np.unique(rounded, return_counts=True)
        return pd.DataFrame({'counts':counts[1], 'vals':counts[0]})

    def save(self, produced: list):
        allcounts = pd.concat(produced, ignore_index=True)
        res = allcounts.groupby('vals').sum()
        res.to_csv(f'{self.data}{self.args.out}')
