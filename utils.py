import os
import shutil
import random
import numpy as np
from collections import OrderedDict

import torch
import torch.distributed as dist

from types_ import *


def set_seed(seed: int = 777) -> None:
    """
    Set the seed for reproducible result
    """
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def prepare_path(path: str) -> None:
    """
    Make directory if not exist.
    Else, raise error.

    Args:
        path (str): path to directory
    """
    try:
        os.makedirs(path, exist_ok=True)
    except:
        raise ValueError(f'Existing path!: {path}')
"""
    Basic settings for training.
    Set seed, prepare save path, etc.

    Args:
        cfg (Dict): config as dictionary
    """

def setup(cfg: Dict[str, Any]) -> int:
    """
    Basic settings for training. (default to multi-gpu)

    Args:
        cfg (Dict[str, Any]): config as dictionary.

    Returns:
        int: rank
    """
    dist.init_process_group(cfg['distributed']['backend'])
    rank = dist.get_rank()
    
    seed = cfg['seed'] * dist.get_world_size() + rank
    set_seed(seed)
    
    if rank == 0:
        prepare_path(cfg['save_path'])
    
    return rank


def send_data_dict_to_device(
    data: Dict[str, Any], rank: int
) -> Dict[str, Any]:
    """
    Send data from cpu to gpu.

    Args:
        data (Dict[str, Any]): data dictionary from data loader
        rank (int): cpu or cuda or rank

    Returns:
        Dict[str, Any]: data dictionary on rank
    """
    for k, v in data.items():
        if isinstance(v, torch.Tensor):
            data[k] = v.detach().to(rank)
        elif isinstance(v, list):
            data_list = []
            for i in range(len(v)):
                data_list.append(v[i].detach().to(rank))
            data[k] = data_list
    
    return data


def save_checkpoint(is_best, state, save_path):    
    save_path = os.path.join(save_path, 'saved_models')
    filename = '%d.pth' % (state['epoch'])
    best_filename = 'best_' + filename
    file_path = os.path.join(save_path, filename)
    best_file_path = os.path.join(save_path, best_filename)
    torch.save(state, file_path)
    
    # Remove previous best model
    if is_best:
        saved_models = os.listdir(save_path)
        for saved_model in saved_models:
            if saved_model.startswith('best'):
                os.remove(os.path.join(save_path, saved_model))
        shutil.copyfile(file_path, best_file_path)


def is_image_file(filename: str) -> bool:
    """
    Check the input is image file

    Args:
        filename (str): path to file

    Returns:
        bool: True or False
    """
    
    IMG_EXTENSIONS = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG',
                      '.ppm', '.PPM', '.bmp', '.BMP']
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def calculate_accuracy(pred, gt):
    pred = torch.max(pred.data, dim=1)[1]
    acc = (pred == gt).sum() / len(pred)
    
    idx_r = (gt == 0).nonzero().view(-1)
    idx_p = (gt == 1).nonzero().view(-1)
    idx_d = (gt == 2).nonzero().view(-1)
    acc_r = (pred[idx_r] == gt[idx_r]).float().sum()
    acc_p = (pred[idx_p] == gt[idx_p]).float().sum()
    acc_d = (pred[idx_d] == gt[idx_d]).float().sum()
    
    acc.mul_(100)
    acc_r.div_(len(idx_r)+1e-8).mul_(100)
    acc_p.div_(len(idx_p)+1e-8).mul_(100)
    acc_d.div_(len(idx_d)+1e-8).mul_(100)
   
    return acc, [acc_r, acc_p, acc_d]


def terminate():
    pass


class AverageMeter(object):
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


# class AverageMeter(object):
#     def __init__(self):
#         self.items = OrderedDict()
    
#     def add(self, key: str, value: Any):
#         if key not in self.items:
#             self.items[key] = []
#         self.items[key].append(value)
    
#     def means(self):
#         return OrderedDict([
#             (k, np.mean(v)) for k, v in self.items.items() if len(v) > 0
#         ])
    
#     def reset(self):
#         for k in self.items:
#             self.items[k] = []


if __name__ == '__main__':
    torch.manual_seed(777)
    for _ in range(5):
        print(torch.randn(7, device='cuda:0'))