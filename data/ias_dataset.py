import os
import json
from glob import glob

import torch
from torch.utils.data import Dataset
from torchvision.io import read_image
from torchvision import transforms

import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from utils import is_image_file


LABEL_DICT = {
    'Real': ['real', 'Real', '1.리얼신분증'],
    'Paper': ['paper', 'Paper', '2.복사신분증', 'synthetic'],
    'Display': ['smartphone', 'SmartPhone', 'tablet', 'Tablet', '4.리플레이',
                'monitor', 'Monitor', '5.캡처', 'Notebook', 'macbook',
                'Display']
}


class IASDataset(Dataset):
    def __init__(self, cfg, dataset_name, train: bool = True) -> None:
        
        self.cfg = cfg
        self.is_train = train
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize(mean=0.5, std=0.5)
        ])
        
        # Image path
        img_paths = os.path.join(cfg['data_path'], dataset_name, '*.*')
        img_paths = glob(img_paths)
        self.img_paths = [i for i in img_paths if is_image_file(i)]
        # Label
        self.labels = []
        json_paths = [os.path.splitext(i)[0]+'.json' for i in self.img_paths]
        for json_path in json_paths:
            with open(json_path, 'r') as f:
                annots = json.load(f)
            if annots['spoof_type'] in LABEL_DICT['Real']:
                self.labels.append(0)
            elif annots['spoof_type'] in LABEL_DICT['Paper']:
                self.labels.append(1)
            elif annots['spoof_type'] in LABEL_DICT['Display']:
                self.labels.append(2)
            else:
                raise ValueError(f'Invalid spoof type. {json_path}')
    
    def __len__(self):
        return len(self.img_paths)
    
    def __getitem__(self, idx: int):
        img = read_image(self.img_paths[idx])
        img = self.transform(img)
        label = self.labels[idx]
        
        return img, label


if __name__ == '__main__':
    dataset = IASDataset(
        {'data_path': 'C:/Users/heegyoon/Desktop/data/IAS/kr/processed'},
        'cubox_4k_2211'
    )
    from torch.utils.data import Dataset, DataLoader
    loader = DataLoader(dataset, 7, False)
    print(len(dataset))