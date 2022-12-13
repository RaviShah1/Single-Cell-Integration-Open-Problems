import gc
import numpy as np
from tqdm import tqdm
import torch

from utils import criterion, correlation_score
from config import CFG

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class AverageMeter(object):
    """Computes and stores the average and current value"""
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


def train_one_epoch(model, optimizer, scheduler, dataloader, device, epoch):
    """ Trains one epoch and returns loss """
    model.train()
    
    losses = AverageMeter()
    corr = AverageMeter()
    
    bar = tqdm(enumerate(dataloader), total=len(dataloader))
    for step, data in bar:
        X, y = data["X"], data["y"]
        
        batch_size = X.size(0)

        outputs = model(X)

        n = outputs.size(0)
        loss = criterion(outputs, y)
        losses.update(loss.item(), n)
        loss.backward()
        
        outputs = outputs.detach().cpu().numpy()
        y = y.detach().cpu().numpy()
        corr_score = correlation_score(y, outputs)
        corr.update(corr_score, n)
        
        optimizer.step()
        optimizer.zero_grad()

        if scheduler is not None:
            scheduler.step()
        
        bar.set_postfix(Epoch=epoch, Train_Loss=losses.avg, Corr=corr.avg,
                        LR=optimizer.param_groups[0]['lr'])
    gc.collect()
    
    return losses.avg


@torch.no_grad()
def valid_one_epoch(model, optimizer, dataloader, device, epoch):
    """ Evaluates one epoch and returns loss """
    model.eval()
    
    losses = AverageMeter()
    corr = AverageMeter()
    
    bar = tqdm(enumerate(dataloader), total=len(dataloader))
    for step, data in bar:        
        X, y = data["X"], data["y"]
        
        batch_size = X.size(0)

        outputs = model(X)
        
        n = outputs.size(0)
        loss = criterion(outputs, y)
        losses.update(loss.item(), n)
        
        outputs = outputs.detach().cpu().numpy()
        y = y.detach().cpu().numpy()
        corr_score = correlation_score(y, outputs)
        corr.update(corr_score, n)
        
        bar.set_postfix(Epoch=epoch, Valid_Loss=losses.avg, Corr=corr.avg,
                        LR=optimizer.param_groups[0]['lr'])   
    
    gc.collect()
    
    return losses.avg


def train_one_fold(model, 
                   optimizer, 
                   scheduler, 
                   train_loader, 
                   valid_loader, 
                   fold):
    """ Trains and saves a full fold of a pytorch model """
    best_epoch_loss = np.inf
    model.to(device)

    for epoch in range(CFG.epochs):
        gc.collect()
        train_epoch_loss = train_one_epoch(model, 
                                           optimizer, 
                                           scheduler, 
                                           dataloader=train_loader, 
                                           device=device, 
                                           epoch=epoch)

        val_epoch_loss = valid_one_epoch(model,
                                         optimizer, 
                                         valid_loader, 
                                         device=device, epoch=epoch)
        
        if val_epoch_loss <= best_epoch_loss:
            print(f"Validation Loss Improved ({best_epoch_loss} ---> {val_epoch_loss})")
            best_epoch_loss = val_epoch_loss
            torch.save(model.state_dict(), f"model_f{fold}.bin")
            
    print("Best Loss: {:.4f}".format(best_epoch_loss))