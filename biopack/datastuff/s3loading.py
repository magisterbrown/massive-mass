from torchdata.datapipes.iter import IterableWrapper, IterDataPipe
import numpy as np
import cv2

class WiredAssStream:

    def __call__(self, inp):
        name, stream_wrapper = inp
        byted = bytearray(stream_wrapper.__getstate__().read())
        byted = self.process_bytes(byted)
        npbuff= np.frombuffer(byted, np.float32)
        res = cv2.imdecode(npbuff, cv2.IMREAD_UNCHANGED)
        res = self.process_layers(res)

        return name, res

    def process_bytes(self, bt: bytearray):
        return bt

    def process_layers(self ,img: np.array):
        return img

class WiredAssStreamOut(WiredAssStream):
    def process_bytes(self, bt: bytearray):
        bt.append(0)
        bt.append(0)
        return bt

class NormalZipper(IterDataPipe):
    def __init__(self, source_dp: IterDataPipe, lab_dp: IterDataPipe, add_names=False) -> None:
        super().__init__()
        if not add_names:
            cleaner = lambda x:x[1]
            source_dp = source_dp.map(cleaner)
            lab_dp = lab_dp.map(cleaner)
            
        self.dp = source_dp
        self.lab_dp = lab_dp
        
    def __iter__(self):
        for d in zip(self.dp, self.lab_dp):
            yield d

def get_ds_s1_list(links: list):
    was = WiredAssStream()
    dp = IterableWrapper(links).open_files_by_fsspec(mode="rb", anon=True).map(was)
    return dp

def get_ds_s1_lab_list(links: list):
    was = WiredAssStreamOut()
    dp = IterableWrapper(links).open_files_by_fsspec(mode="rb", anon=True).map(was)
    return dp
