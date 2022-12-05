from torchdata.datapipes.iter import IterableWrapper, Zipper
from .s3loading import get_ds_s1_lab_list, WiredAssStream, NormalZipper
import numpy as np

class InputS1Loader(WiredAssStream):
    def __init__(self, mean=[-11.02, -17.36, -11.11, -17.31], std=[3.41, 4.42, 3.24, 4.57]):
        self.mean = np.expand_dims(np.array(mean),[-1,-2])
        self.std =  np.expand_dims(np.array(std),[-1,-2])
        self.axis = (1,2)


    def process_layers(self, img: np.array):
        img = np.moveaxis(img, -1, 0)

        for key, channel in enumerate(img):
            neg_misses = np.all(channel<-9998)
            zer_misses = np.all(channel==0)
            misses = np.logical_or(neg_misses, zer_misses)
            if misses:
                img[key] = np.random.normal(size=channel.shape)*self.std+self.mean

        img=(img-self.mean)/self.std

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

