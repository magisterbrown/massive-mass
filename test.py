import pandas as pd
import numpy as np
import io

from biopack.datastuff.s3loading import get_ds_s1_lab_list, WiredAssStream
from torchdata.datapipes.iter import IterableWrapper
from torch.utils.data.datapipes.iter import Zipper
from biopack.datastuff.train_loader import get_simple_train_dl


feature_metadata = pd.read_csv('data/features_metadata.csv')
train_agbm_metadata = pd.read_csv('data/train_agbm_metadata.csv')

s1 = (feature_metadata['satellite']=='S1').to_numpy()
august = (feature_metadata['month']=='August').to_numpy()
train = (feature_metadata['split']=='train').to_numpy()

s1_august = feature_metadata[np.all(np.stack([s1,august,train]),axis=0)]

l1,l2 = get_simple_train_dl(s1_august, train_agbm_metadata, 'eu')
#import pdb; pdb.set_trace()
#next(iter(l1))
zep = Zipper(l1,l2)
itt = iter(zep)
import pdb; pdb.set_trace()
nxx = next(itt)

