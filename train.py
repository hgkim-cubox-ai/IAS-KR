from tqdm import tqdm

from utils import (save_checkpoint, send_data_dict_to_device,
                   calculate_accuracy,
                   AverageMeter)


def _train(cfg, rank, loader, model, optimizer, loss_fn_dict, epoch):
    model.train()
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    r_meter = AverageMeter()
    p_meter = AverageMeter()
    d_meter = AverageMeter()
    loss_fn = loss_fn_dict['crossentropy']['fn']
    
    with tqdm(loader, desc=f'[Train] Epoch {epoch+1}', ncols=150, unit='batch') as t:
        for i, input_dict in enumerate(t):
            import torch
            if torch.isnan(input_dict['patches']).sum().item() > 0:
                print(i)
            batch_size = input_dict['img'].size(0)
            input_dict = send_data_dict_to_device(input_dict, rank)
            
            if torch.isnan(input_dict['patches']).sum().item() > 0:
                print(i)
            
            pred = model(input_dict['patches'])
            label = input_dict['label']
            
            loss = loss_fn(pred, label)
            
            if torch.isnan(loss) or torch.isinf(loss):
                print('invalid input detected at iteration ', i)
                        
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            acc, accs = calculate_accuracy(pred.data, label.data)
            
            loss_meter.update(loss.item(), batch_size)
            acc_meter.update(acc.item(), batch_size)
            r_meter.update(accs[0].item(), batch_size)
            p_meter.update(accs[1].item(), batch_size)
            d_meter.update(accs[2].item(), batch_size)
            
            t.set_postfix(
                loss=loss_meter.avg,
                acc=acc_meter.avg,
                real=r_meter.avg,
                paper=p_meter.avg,
                display=d_meter.avg
            )
    
    return loss_meter.avg, acc_meter.avg


def _validate(cfg, rank, loader, model, loss_fn_dict, epoch, data_split):
    model.eval()
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    
    with tqdm(loader, desc=f'[{data_split}] Epoch {epoch+1}', ncols=150, unit='batch') as t:
        for i, input_dict in enumerate(t):
            break
            input_dict = send_data_dict_to_device(input_dict, rank)
            output_dict = model(input_dict, loss_fn_dict)
            
            loss_meter.update(output_dict['loss'].item(), cfg['batch_size'])
            acc_meter.update(output_dict['acc'].item(), cfg['batch_size'])
    
    return {'loss': loss_meter.avg, 'acc': acc_meter.avg}
            

def train(cfg, rank, dataloader_dict, model, optimizer, loss_fn_dict):
    results_dict = {}   # to be saved
    for data_split in dataloader_dict:
        results_dict[data_split] = {}
    
    max_acc = 0
    
    for epoch in range(cfg['num_epochs']):
        # Train
        results_dict['train'][epoch] = _train(cfg, rank, dataloader_dict['train'],
                                              model, optimizer, loss_fn_dict, epoch)
        print(f'{rank}/train/{epoch}')
        # Val, test
        # for data_split in [d for d in dataloader_dict if d != 'train']:
        #     results_dict[data_split][epoch] = _validate(cfg, rank, dataloader_dict[data_split],
        #                                                 model, loss_fn_dict, epoch, data_split)

        # val_acc = results_dict['test'][epoch]['acc']
        # is_best = val_acc > max_acc
        # max_acc = max(val_acc, max_acc)
        # print('')
        # test_loss = loss_meter.avg
        # is_best = test_loss < min_loss
        # min_loss = min(min_loss, test_loss)
        # save_checkpoint(
        #     is_best,
        #     {'epoch': epoch+1, 'state_dict': model.state_dict()},
        #     dataloader_dict.save_path
        # )


