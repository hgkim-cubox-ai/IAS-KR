import torch
import torch.nn as nn

from backbone import ResNet

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from types_ import *


class IASModel(nn.Module):
    def __init__(self, cfg: Dict[str, Any]) -> None:
        super(IASModel, self).__init__()
        
        self.cfg = cfg
        self.backbone = ResNet(cfg['backbone'])

    def forward(self, x):
        return x


if __name__ == '__main__':
    net = IASModel({'backbone': 'resnet50'})
    print(net.backbone)