from torchdata.datapipes.iter import IterableWrapper
import numpy as np
import cv2

class WiredAssStream:

    def __call__(self, inp):
        name, stream_wrapper = inp
        byted = bytearray(stream_wrapper.__getstate__().read())
        byted = self.process_bytes(byted)
        npbuff= np.frombuffer(byted, np.float32)
        res = cv2.imdecode(npbuff, cv2.IMREAD_UNCHANGED)

        return name, res

    def process_bytes(self, bt: bytearray):
        return bt

def get_ds_s1_list(links: list):
    was = WiredAssStream()
    dp = IterableWrapper(links).open_files_by_fsspec(mode="rb", anon=True).map(was)
    return dp
