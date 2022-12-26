from segmentation_models_pytorch.base import SegmentationModel,SegmentationHead
from segmentation_models_pytorch.decoders.unet.decoder import UnetDecoder
from torchvision.models import efficientnet_v2_s
from torchvision.ops import Conv2dNormActivation
import torch.nn as nn
import torch

class SkipEffnetV2(nn.Module):
    def __init__(self):
        super().__init__()
        effnet = efficientnet_v2_s()
        effnet.features[0] = Conv2dNormActivation(4, 24, kernel_size=3, stride=2, norm_layer=nn.BatchNorm2d, activation_layer=nn.SiLU)
        self.str1 = effnet.features[:2]
        self.str2 = effnet.features[2]
        self.str3 = effnet.features[3]
        self.str4 = effnet.features[4:6]
        self.str5 = effnet.features[6:]

        self.output_stride = 32

    def forward(self, x):
        res = list()
        res.append(x)
        x = self.str1(x)
        res.append(x)
        x = self.str2(x)
        res.append(x)
        x = self.str3(x)
        res.append(x)
        x = self.str4(x)
        res.append(x)
        x = self.str5(x)
        res.append(x)

        return res

class EffUnet(SegmentationModel):

    def __init__(self):
        super().__init__()
        self.encoder = SkipEffnetV2()
        decoder_channels = (256, 128, 64, 32, 16)
        self.decoder = UnetDecoder(
            encoder_channels=(4,24,48,64,160,1280),
            decoder_channels=decoder_channels,
            n_blocks=5,
            use_batchnorm=True,
            center=False,
            attention_type=None,
        )

        self.segmentation_head = SegmentationHead(
            in_channels=decoder_channels[-1],
            out_channels=1,
            activation=nn.ReLU,
            kernel_size=3,
        )

        self.classification_head = None

        self.name = "CoolNet"
        self.initialize()
