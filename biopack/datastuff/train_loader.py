from torchdata.datapipes.iter import IterableWrapper, Zipper
from .s3loading import get_ds_s1_lab_list, WiredAssStream, NormalZipper
import numpy as np

class InputS1Loader(WiredAssStream):

    def process_layers(self, img: np.array):
        return img

def get_simple_train_dl(features_df, amgb_df, place):
    allindf = features_df.merge(amgb_df,on='chip_id')
    allindf = allindf[[f's3path_{place}_x',f's3path_{place}_y']].values

    inputs = allindf[:,0]
    lables = allindf[:,1]


    inp_proc = InputS1Loader()
    input_stream = IterableWrapper(inputs).open_files_by_fsspec(mode="rb", anon=True).map(inp_proc)
    lab_stream = get_ds_s1_lab_list(lables)

    zipped = NormalZipper(input_stream, lab_stream)

    return zipped

