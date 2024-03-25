import torch
from PIL import Image
from torchvision import datasets, transforms

from types_ import *


class MNIST(datasets.MNIST):
    def __init__(self, cfg: Dict[str, Any], train: bool = True) -> None:
        super().__init__(
            root=cfg['data_path'], train=train,
            transform=transforms.ToTensor(),
            download=True
        )
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        img, target = self.data[idx], int(self.targets[idx])
        
        img = Image.fromarray(img.numpy(), mode='L')
        
        if self.transform is not None:
            img = self.transform(img)
        
        return {'img': img, 'target': target}


if __name__ == '__main__':
    from torch.utils.data import DataLoader
    dataset = MNIST(
        {'data_path':'C:/Users/heegyoon/Desktop/data'},
        train=True,
        transform=transforms.ToTensor(),
    )
    
    loader = DataLoader(dataset, 7, True)
    
    for i, data_dict in enumerate(loader):
        print(data_dict['img'].size())
        print(data_dict['target'].size())
        
        import os
        os.environ['KMP_DUPLICATE_LIB_OK']='True'
        import matplotlib.pyplot as plt
        plt.imshow(data_dict['img'][0][0], cmap='gray')
        plt.show()
        
        print('')