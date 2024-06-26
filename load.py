import os

import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, ConcatDataset
from torch.utils.data.distributed import DistributedSampler

from types_ import *
from data import IASDataset
from models import MODEL_DICT
from losses import LOSS_FN_DICT


def load_dataloader_dict(cfg: Dict[str, Any]) -> Dict[str, DataLoader]:
    """
    Return data dictionary.
    data_split: train/val/test from config

    Args:
        cfg (Dict): config as dictionary

    Returns:
        Dict: {'train': train_loader, 'test': test_loader, 'val': val_loader}
    """
    dataloader_dict = {}
    all_datasets = cfg['datasets']
    for data_split, dataset_names in all_datasets.items():
        # If test (or val) set is not used.
        if dataset_names is None:
            continue
        
        dataset = IASDataset(cfg, dataset_names, data_split=='train')
        sampler = DistributedSampler(dataset) if data_split=='train' else None
        dataloader_dict[data_split] = DataLoader(
            dataset=dataset, batch_size=cfg['batch_size'],
            num_workers=cfg['num_workers'], pin_memory=True, sampler=sampler
        )
    
    return dataloader_dict


def load_model_and_optimizer(
    cfg: Dict[str, Any], rank: int
) -> Tuple[torch.nn.Module, torch.optim.Optimizer]:
    """
    Args:
        rank (int): if ddp, rank = int. else cpu or cuda:0

    Returns:
        Tuple[torch.nn.Module, torch.optim.Optimizer]: model, optimizer.
    """
    device_id = rank % torch.cuda.device_count()
    
    # Model
    model = MODEL_DICT[cfg['model']](cfg)
    model.to(device_id)
    model = DDP(model, device_ids=[device_id])
    print(f'Model on rank {rank}')
    
    # Load weight
    if cfg['pretrained_model'] is not None:
        saved = torch.load(os.path.join(os.getcwd(), cfg['pretrained_model']))
        model.load_state_dict(saved['state_dict'])
    
    # Optimizer
    optimizer = None
    params = filter(lambda p: p.requires_grad, model.parameters())
    if cfg['optimizer'] == 'sgd':
        optimizer = torch.optim.SGD(
            params=params, lr=cfg['base_lr'], weight_decay=cfg['weight_decay']
        )
    elif cfg['optimizer'] == 'adam':
        optimizer = torch.optim.Adam(
            params=params, lr=cfg['base_lr'], weight_decay=cfg['weight_decay']
        )
    
    return model, optimizer


def load_loss_fn_dict(cfg: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """
    loss_fn_dict = {
        'name': {
            'fn': MSE()
            'weight': 1.0
        }
    }
    Returns:
        Dict[str, Dict[str, Any]]: loss function dict
    """
    loss_fn_dict = {}
    for loss_fn, weight in cfg['loss_functions'].items():
        loss_fn_dict[loss_fn] = {
            'fn': LOSS_FN_DICT[loss_fn](),
            'weight': weight
        }
    
    return loss_fn_dict