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
            with open(json_path, 'r', encoding='cp949') as f:
                annots = json.load(f)
            if annots['spoof_type'] in LABEL_DICT['Real']:
                self.labels.append(1)
            elif annots['spoof_type'] in LABEL_DICT['Paper']:
                self.labels.append(0)
            elif annots['spoof_type'] in LABEL_DICT['Display']:
                self.labels.append(0)
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
    
    def crop_patch(self, img):
        N = self.cfg['n_patches']
        H = self.cfg['patch_size']
        W = self.cfg['patch_size']
        patches = torch.zeros([N, 3, H, W])
        for n in range(N):
            patch = self.random_crop(img)
            std = torch.std(patch.float(), dim=(1,2), keepdim=True)
        return patches        
    
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
        std = torch.std(patches.float(), dim=(2,3), keepdim=True) + 1e-12
        patches = (patches - mean) / std
        
        data_dict = {'label': torch.tensor(label).float()}
        if self.cfg['input'] == 'image':
            data_dict['input'] = img
        elif self.cfg['input'] == 'patch':
            data_dict['input'] = patches
        else:
            raise ValueError

        # if torch.isnan(patches).sum().item() > 0:
        #     print(f'dataset {self.img_paths[idx]}')
        
        return data_dict


if __name__ == '__main__':
    dataset = IASDataset(
        {
            'data_path': 'C:/Users/heegyoon/Desktop/data/IAS/kr/processed',
            'patch_size': 256,
            'n_patches': 9,
            'resize': {'height': 224, 'width': 224},
            'input': 'image'
        },
        'shinhan'
    )
    from torch.utils.data import DataLoader
    from tqdm import tqdm
    loader = DataLoader(dataset, 1, False)
    
    real = 0
    fake = 0
    
    # print(len(loader))
    for i, (data_dict) in tqdm(enumerate(loader)):
        if data_dict['label'].item() == 1:
            real += 1
        else:
            fake += 1
    
    print(len(loader), real, fake)


"""
Dataset                         total   real    fake
--------------------------------------------------------
[train]
cubox_4k_2211                   1847    694     1153
IAS_cubox_train_230102_renew    67573   5355    62218
IAS_cubox_train_230117_extra    3552    355     3197
real_driver                     478     478     0
real_id                         375     375     0
real_passport                   901     901     0
                                74726   8158    66568
--------------------------------------------------------
[test]
shinhan                         1374    127     1247
--------------------------------------------------------
"""