import os
import cv2
import numpy as np
import json
from glob import glob

import torch
from torch.utils.data import Dataset
from torchvision import transforms
# from torchvision.transforms.functional import crop
# from torchvision.io import read_image

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
            transforms.ToTensor(),
            transforms.RandomCrop(cfg['patch_size']),
            transforms.Normalize(mean=0.5, std=0.5)
        ])
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(
                (cfg['size']['height'], cfg['size']['width']),
                transforms.InterpolationMode.BICUBIC
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=0.5, std=0.5)
        ])

        # Prepare data
        img_paths = os.path.join(cfg['data_path'], dataset_name, '*.*')
        img_paths = glob(img_paths)
        self.img_paths = [i for i in img_paths if is_image_file(i)]
        self.json_paths = [os.path.splitext(i)[0]+'.json' for i in self.img_paths]
        assert len(self.img_paths) == len(self.json_paths)
    
    def __len__(self):
        return len(self.img_paths)
    
    def imshow(self, img, fname=None):
        if isinstance(img, np.ndarray):
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            cv2.imshow('img',img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        elif isinstance(img, torch.Tensor):
            import matplotlib.pyplot as plt
            from torchvision.transforms.functional import to_pil_image
            plt.imshow(to_pil_image(img))
            plt.show()
        # import numpy as np
        # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        # hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV_FULL)
        # h, s, v = cv2.split(hsv)
        # hsv_out = np.vstack([h, s, v])
        # cv2.imshow(f'{fname}', cv2.resize(img, dsize=None, fx=0.5, fy=0.5))
        # cv2.imshow('hsv', cv2.resize(hsv_out, dsize=None, fx=0.25, fy=0.25))
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

    def read_data(self, idx):
        # Image
        img = cv2.imread(self.img_paths[idx])  # torchvision이나 pil로 읽으면 돌아감
        img = img[:,:,::-1].copy()  # BGR -> RGB
        if self.cfg['color'] == 'hsv':
            img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV_FULL)
        # Label
        with open(self.json_paths[idx], 'r', encoding='cp949') as f:
            annots = json.load(f)
        return img, annots
    
    def align_idcard(self, img, keypoints, cls, dsize_factor = None):
        keypoints = np.array(keypoints)
        if cls[0] == 0:
            idcard_ratio = np.array((86, 54))
        elif cls[0] == 1:
            idcard_ratio = np.array((125, 88))
        else:
            raise ValueError(f'Wrong cls: {cls}')

        if dsize_factor is None:
            dsize_factor = round(np.sqrt(cv2.contourArea(np.expand_dims(keypoints, 1))) / idcard_ratio[0])

        dsize = idcard_ratio * dsize_factor  # idcard size unit: mm
        dst = np.array(((0, 0), (0, dsize[1]), dsize, (dsize[0], 0)), np.float32)

        M = cv2.getPerspectiveTransform(keypoints.astype(np.float32), dst)
        img = cv2.warpPerspective(img, M, dsize)
        return img

    def preprocess_input(self, img, annots):
        # Label real/fake
        if annots['spoof_type'] in LABEL_DICT['Real']:
            label = 1
        elif annots['spoof_type'] in LABEL_DICT['Paper']:
            label = 0
        elif annots['spoof_type'] in LABEL_DICT['Display']:
            label = 0
        else:
            raise ValueError(f'Invalid spoof type.')
        
        # Aligend or cropped image
        if self.cfg['type'] == 'aligned':
            if 'idcard_kpts' in annots:
                img = self.align_idcard(img, annots['idcard_kpts'], annots['cls'])
        else:   # Crop from raw image
            if 'idcard_bbox' in annots:
                bbox = annots['idcard_bbox']    # [top, bottom, left, right]
                img = img[bbox[0]:bbox[1], bbox[2]:bbox[3], :]
        # img = torch.permute(torch.from_numpy(img), [2,0,1])   # HWC -> CHW
        
        # Return dict
        data_dict = {'label': torch.tensor(label).float()}
        if self.cfg['input'] == 'image':
            img = self.transform(img)
            data_dict['input'] = img.detach()
        elif self.cfg['input'] == 'patch':
            patches = torch.stack(
                [self.random_crop(img) for _ in range(self.cfg['n_patches'])]
            )
            data_dict['input'] = patches.detach()
        else:
            raise ValueError
        return data_dict
    
    def __getitem__(self, idx):
        img, annots = self.read_data(idx)
        return self.preprocess_input(img, annots)


if __name__ == '__main__':
    dataset = IASDataset(
        {
            'data_path': 'C:/Users/heegyoon/Desktop/data/IAS/kr/processed',
            'patch_size': 256,
            'n_patches': 9,
            'size': {'height': 144, 'width': 224},
            'input': 'image',
            'color': 'rgb',
            'type': 'aligned'
        },
        'cubox_4k_2211'
    )
    from torch.utils.data import DataLoader
    from tqdm import tqdm
    loader = DataLoader(dataset, 1, True)
    
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