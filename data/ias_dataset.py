import os
import json
from glob import glob

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
from torchvision import transforms

import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from utils import is_image_file




class IASDataset(Dataset):
    def __init__(self, cfg, dataset_name, train: bool = True) -> None:
        
        self.cfg = cfg
        
        # Image path
        img_paths = os.path.join(cfg['data_path'], dataset_name, '*.*')
        img_paths = glob(img_paths)
        self.img_paths = [i for i in img_paths if is_image_file(i)]
        # Json
        self.json_paths = [os.path.splitext(i)[0]+'.json' for i in self.img_paths]
    
    def __len__(self):
        return len(self.img_paths)
    
    def __getitem__(self, idx: int):
        return None


if __name__ == '__main__':
    dataset = IASDataset(
        {'data_path': 'C:/Users/heegyoon/Desktop/data/IAS/kr/processed'},
        'cubox_4k_2211'
    )
    print(len(dataset))