import torch
import torch.nn as nn

from .backbone import ResNet

from types_ import *


class IASModel(nn.Module):
    def __init__(self, cfg: Dict[str, Any]) -> None:
        super(IASModel, self).__init__()
        
        self.cfg = cfg
        self.backbone = ResNet(cfg['backbone'])
    
    def crop_patches(self, img):
        C, H, W = img.size()
        patches = torch.zeros([self.cfg['n_patches'], C, H, W])
        
        xs = torch.randint(0, W-self.cfg['patch_size'])
        ys = torch.randint(0, H-self.cfg['patch_size'])
        
        for i in range(len(xs)):
            patches[i] = img[:, ys:ys+self.cfg['patch_size'], xs:xs+self.cfg['patch_size']]
        
        return patches
    
    def forward(self, x):
        return x


if __name__ == '__main__':
    net = IASModel({'backbone': 'resnet50'})
    print