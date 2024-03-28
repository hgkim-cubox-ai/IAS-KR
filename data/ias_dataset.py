import os
import json
from glob import glob

import torch
from torch.utils.data import Dataset
from torchvision.io import read_image
from torchvision import transforms
from torchvision.transforms.functional import crop

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
        self.random_crop = transforms.Compose([
            transforms.RandomCrop(cfg['patch_size'])
        ])
        self.resize = transforms.Compose([
            transforms.Resize(
                (cfg['resize']['height'], cfg['resize']['width']),
                transforms.InterpolationMode.BICUBIC)
        ])
        
        # Image path
        img_paths = os.path.join(cfg['data_path'], dataset_name, '*.*')
        img_paths = glob(img_paths)
        self.img_paths = [i for i in img_paths if is_image_file(i)]
        # Label
        self.labels = []
        self.id_bboxes = []
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
            
            if 'idcard_bbox' in annots:
                bbox = annots['idcard_bbox']    # [top, bottom, left, right]
                bbox = [bbox[0], bbox[2], bbox[1]-bbox[0], bbox[3]-bbox[2]]
            else:
                bbox = None
            self.id_bboxes.append(bbox)
    
    def __len__(self):
        return len(self.img_paths)
    
    def imshow(self, img):
        import matplotlib.pyplot as plt
        from torchvision.transforms.functional import to_pil_image
        p = to_pil_image(img)
        plt.imshow(p)
        plt.show()
    
    def __getitem__(self, idx: int):
        img = read_image(self.img_paths[idx])
        label = self.labels[idx]
        
        # Crop idcard
        bbox = self.id_bboxes[idx]  # [top, left, height, width]
        if bbox is not None:
            img = crop(img, bbox[0], bbox[1], bbox[2], bbox[3])
        
        # Crop patch
        patches = torch.stack(
            [self.random_crop(img) for _ in range(self.cfg['n_patches'])]
        )
        
        # Resize image
        img = self.resize(img)
        
        # Normalize
        mean = torch.mean(img.float(), dim=(1,2), keepdim=True)
        std = torch.std(img.float(), dim=(1,2), keepdim=True)
        img = (img - mean) / std
        mean = torch.mean(patches.float(), dim=(2,3), keepdim=True)
        std = torch.std(patches.float(), dim=(2,3), keepdim=True)
        patches = (patches - mean) / std
        
        return img, patches, label


if __name__ == '__main__':
    dataset = IASDataset(
        {
            'data_path': 'C:/Users/heegyoon/Desktop/data/IAS/kr/processed',
            'patch_size': 128,
            'n_patches': 9,
            'resize': {'height': 224, 'width': 224}
        },
        'shinhan'
    )
    from torch.utils.data import DataLoader
    loader = DataLoader(dataset, 7, False)
    
    for i, (patches, label) in enumerate(loader):
        print(torch.sum(patches, dim=(3,4)))