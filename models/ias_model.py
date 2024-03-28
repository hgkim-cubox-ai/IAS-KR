import torch
import torch.nn as nn

from .backbone import CNN_ResNet

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from types_ import *


class IASModel(nn.Module):
    def __init__(self, cfg: Dict[str, Any]) -> None:
        super(IASModel, self).__init__()
        
        self.cfg = cfg
        self.backbone = CNN_ResNet(cfg['backbone'])
        self.fc = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 3),
            nn.Softmax()
        )

    def forward(self, x):
        B, P, C, H, W = x.size()
        x = x.view(B*P, C, H, W)
        
        x = self.backbone(x)
        x = x.view(B*P, -1)
        x = self.fc(x)
        x = x.view(B, P, -1)
        
        x = torch.mean(x, dim=1)
        
        return x


if __name__ == '__main__':
    net = IASModel({'backbone': 'resnet50'})
    p = torch.randn([7, 9, 3, 128, 128])
    out = net(p)
