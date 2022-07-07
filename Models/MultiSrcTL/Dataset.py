import torch
from torch.utils.data import Dataset
from torch.utils.data import Dataset, DataLoader
import numpy as np

class StockDataset(Dataset):
    def __init__(self, X, Y, W):
        self.X = X
        self.Y = Y
        self.W = W
        

    def __getitem__(self, idx):
        x = self.X[idx]
        y = self.Y[idx]
        w = self.W[idx]
        return x, y, w
    

    def __len__(self):
        return len(self.Y)
    
def collate_fn(batch):
    x, y, w = zip(*batch)
    x = np.array(x)
    y = np.array(y)
    w = np.array(w)

    x = torch.tensor(x).to(dtype=torch.float32)
    y = torch.tensor(y).to(dtype=torch.float32)
    w = torch.tensor(w).to(dtype=torch.float32)
    return x, y, w

def get_set_and_loader(X, Y, W, batch_size = 64, shuffle = True):
    dataset = StockDataset(X=X, Y=Y, W=W)

    if batch_size == 0:
        batch_size = len(dataset)
        
    loader = DataLoader(dataset = dataset, 
                        batch_size = batch_size, 
                        shuffle = shuffle, 
                        collate_fn=collate_fn)

    return dataset, loader