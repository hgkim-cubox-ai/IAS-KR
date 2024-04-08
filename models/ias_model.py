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
            nn.Softmax(dim=1)
        )
    
    def _forward_image(self, img):
        B = img.size(0)
        img = self.backbone(img)
        img = img.view(B, -1)
        img = self.fc(img)
        return img
    
    def _forward_patch(self, patches):
        B, P, C, H, W = patches.size()
        patches = patches.view(B*P, C, H, W)
        if torch.isnan(patches).sum().item() > 0:
            print('')
        patches = self.backbone(patches)
        patches = patches.view(B*P, -1)
        patches = self.fc(patches)
        patches = patches.view(B, P, -1)
        patches = torch.mean(patches, dim=1)
        return patches

    def forward(self, x):
        if x.dim() == 4:
            return self._forward_image(x)
        elif x.dim() == 5:
            return self._forward_patch(x)
        else:
            raise ValueError

if __name__ == '__main__':
    net = IASModel({'backbone': 'resnet50'})
    p = torch.randn([7, 9, 3, 128, 128])
    out = net(p)
