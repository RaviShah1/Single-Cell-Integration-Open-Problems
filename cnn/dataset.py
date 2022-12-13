import torch
from torch.utils.data import DataLoader, Dataset
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class CtieseqDataset(Dataset):
    """
    Train, Validation or Test dataset for CITEseq samples
    Prepares data for simple vector to vector NN
    """
    def __init__(self, X, y=None):
        self.train = False 
        if y is not None:
            self.train = True
        self.X = X
        self.y = y
            
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        X = self.X[idx]
        
        if self.train:
            y = self.y[idx]
            return {
                "X" : torch.tensor([X]).to(device),
                "y" : torch.tensor(y).to(device)
            }
        else:
            return {
                "X" : torch.tensor([X]).to(device)
            }