import os
import cv2
import json
from glob import glob

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms.functional import crop

from torchvision.io import read_image

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
                # bbox = [bbox[0], bbox[2], bbox[1]-bbox[0], bbox[3]-bbox[2]]
            else:
                bbox = None
            self.id_bboxes.append(bbox)
    
    def __len__(self):
        return len(self.img_paths)
    
    def imshow(self, img, img2=None, resize=False, fname=None):
        import matplotlib.pyplot as plt
        from torchvision.transforms.functional import to_pil_image
        _resize = transforms.Compose([
            transforms.Resize(
                (448,448),
                transforms.InterpolationMode.BICUBIC)
        ])
        if resize:
            img = _resize(img)
            img2 = _resize(img2)
        p = to_pil_image(img)
        p2 = to_pil_image(img2)
        plt.subplot(211)
        plt.title(fname)
        plt.imshow(p, cmap='gray')
        plt.subplot(212)
        plt.imshow(p2, cmap='gray')
        plt.show()
        
        # import numpy as np
        
        # img = resize(img)
        # n = np.transpose(img.numpy(), [2,1,0])
        # n = n[:,:,::-1]
        # cv2.imshow('n', n)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
    
    def np_to_tensor(self, img):
        img = torch.from_numpy(img)
        img = torch.permute(img, [2,0,1])   # HWC -> CHW
        return img
    
    def read_image(self, path):
        # return: RGB, CxHxW Tensor
        img = cv2.imread(path)  # torchvision이나 pil로 읽으면 돌아감
        img = img[:,:,::-1].copy()  # RGB -> RGB
        return img
    
    def __getitem__(self, idx: int):
        label = self.labels[idx]
        img = self.read_image(self.img_paths[idx])
        
        # Crop idcard
        bbox = self.id_bboxes[idx]  # [top, left, height, width]
        if bbox is not None:
            img = img[bbox[0]:bbox[1], bbox[2]:bbox[3], :]
        
        img = self.np_to_tensor(img)
        patches = torch.stack(
            [self.random_crop(img) for _ in range(self.cfg['n_patches'])]
        )
        
        # self.imshow(img, patches[0], resize=True,
        #             fname=os.path.basename(self.img_paths[idx]))
        
        # Resize image
        img = self.resize(img)
        
        # Normalize
        mean = torch.mean(img.float(), dim=(1,2), keepdim=True)
        std = torch.std(img.float(), dim=(1,2), keepdim=True)
        img = (img - mean) / std
        mean = torch.mean(patches.float(), dim=(2,3), keepdim=True)
        std = torch.std(patches.float(), dim=(2,3), keepdim=True)
        patches = (patches - mean) / std
        
        
        
        if torch.isnan(patches).sum().item() > 0:
            print(self.img_paths[idx])
        
        return {'img': img, 'patches': patches, 'label': label}


if __name__ == '__main__':
    dataset = IASDataset(
        {
            'data_path': 'C:/Users/heegyoon/Desktop/data/IAS/kr/processed',
            'patch_size': 256,
            'n_patches': 9,
            'resize': {'height': 224, 'width': 224}
        },
        'IAS_cubox_train_230102_renew'
    )
    from torch.utils.data import DataLoader
    from tqdm import tqdm
    loader = DataLoader(dataset, 1, False)
    
    print(len(loader))
    for i, (patches) in tqdm(enumerate(loader)):
        pass