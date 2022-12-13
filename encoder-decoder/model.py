import torch
import torch.nn as nn
import torch.nn.functional as F

class FCBlock(nn.Module):
    """
    A Pytorch Block for a fully connected Layer
    Includes Linear, Activation Function, and Dropout
    """
    def __init__(self, input_dim, hidden_dim, dropout):
        super().__init__()
        self.fc = nn.Linear(input_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        x = self.fc(x)
        x = F.selu(x)
        x = self.dropout(x)
        return x

class Encoder(nn.Module):
    """
    Encoder module to generate embeddings of a RNA vector
    """
    def __init__(self):
        super().__init__()
        self.l0 = FCBlock(240, 120, 0.05)
        self.l1 = FCBlock(120, 60, 0.05)
        self.l2 = FCBlock(60, 30, 0.05)
        
    def forward(self, x):
        x = self.l0(x)
        x = self.l1(x)
        x = self.l2(x)
        return x
    
class Decoder(nn.Module):
    """
    Decoder module to extract Protein sequences from RNA embeddings
    """
    def __init__(self):
        super().__init__()
        self.l0 = FCBlock(30, 70, 0.05)
        self.l1 = FCBlock(70, 100, 0.05)
        self.l2 = FCBlock(100, 140, 0.05)
        
    def forward(self, x):
        x = self.l0(x)
        x = self.l1(x)
        x = self.l2(x)
        return x
    
class CtieseqModel(nn.Module):
    """
    Wrapper for the Encoder and Decoder modules
    Converts RNA sequence to Protein sequence
    """
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        
    def forward(self, x):
        embeddings = self.encoder(x)
        outputs = self.decoder(embeddings)
        return outputs
