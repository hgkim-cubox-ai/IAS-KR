from tqdm import tqdm
import torch
from utils import (save_checkpoint, send_data_dict_to_device,
                   calculate_accuracy,
                   AverageMeter)


def _train(cfg, rank, loader, model, optimizer, loss_fn_dict, epoch):
    model.train()
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    real_meter = AverageMeter()
    fake_meter = AverageMeter()
    loss_fn = loss_fn_dict['bce']['fn']
    
    with tqdm(loader, desc=f'[Train] Epoch {epoch+1}', ncols=150, unit='batch') as t:
        for i, input_dict in enumerate(t):
            batch_size = input_dict['input'].size(0)
            input_dict = send_data_dict_to_device(input_dict, rank)
            
            pred = model(input_dict['input'])
            label = input_dict['label']
            
            loss = loss_fn(pred, label.view(-1,1))
                                    
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            acc, real_acc, fake_acc = calculate_accuracy(pred.data, label.data)
            
            loss_meter.update(loss.item(), batch_size)
            acc_meter.update(acc.item(), batch_size)
            real_meter.update(real_acc.item(), batch_size)
            fake_meter.update(fake_acc.item(), batch_size)
            
            t.set_postfix(
                loss=loss_meter.avg,
                acc=acc_meter.avg,
                real=real_meter.avg,
                fake=fake_meter.avg,
            )
    
    return loss_meter.avg, acc_meter.avg


def _validate(cfg, rank, loader, model, loss_fn_dict, epoch, data_split):
    model.eval()
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    real_meter = AverageMeter()
    fake_meter = AverageMeter()
    loss_fn = loss_fn_dict['bce']['fn']
    
    with tqdm(loader, desc=f'[{data_split}] Epoch {epoch+1}', ncols=150, unit='batch') as t:
        for i, input_dict in enumerate(t):
            batch_size = input_dict['input'].size(0)
            input_dict = send_data_dict_to_device(input_dict, rank)
            
            with torch.no_grad():
                pred = model(input_dict['input'])
            label = input_dict['label']
            
            loss = loss_fn(pred, label.view(-1,1))
            
            if torch.isnan(loss) or torch.isinf(loss):
                print('invalid input detected at iteration ', i)
                        
            acc, real_acc, fake_acc = calculate_accuracy(pred.data, label.data)
            
            loss_meter.update(loss.item(), batch_size)
            acc_meter.update(acc.item(), batch_size)
            real_meter.update(real_acc.item(), batch_size)
            fake_meter.update(fake_acc.item(), batch_size)
            
            t.set_postfix(
                loss=loss_meter.avg,
                acc=acc_meter.avg,
                real=real_meter.avg,
                fake=fake_meter.avg,
            )
    
    return loss_meter.avg, acc_meter.avg
            

def train(cfg, rank, dataloader_dict, model, optimizer, loss_fn_dict):
    results_dict = {}   # to be saved
    for data_split in dataloader_dict:
        results_dict[data_split] = {}
    
    max_acc = 0
    
    for epoch in range(cfg['num_epochs']):
        # Train
        results_dict['train'][epoch] = _train(cfg, rank, dataloader_dict['train'],
                                              model, optimizer, loss_fn_dict, epoch)
        
        # Val, test
        for data_split in [d for d in dataloader_dict if d != 'train']:
            results_dict[data_split][epoch] = _validate(cfg, rank, dataloader_dict[data_split],
                                                        model, loss_fn_dict, epoch, data_split)

        cur_acc = results_dict['test'][epoch][1]
        is_best = cur_acc > max_acc
        max_acc = max(cur_acc, max_acc)
        if rank == 0:
            save_checkpoint(
                is_best,
                {'epoch': epoch+1, 'state_dict': model.state_dict()},
                cfg['save_path']
            )