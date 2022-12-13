import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader, Dataset

from model import CtieseqModel
from dataset import CtieseqDataset


def predict(fold):
    preds = list()
    ds = CtieseqDataset(cite_test_x)
    dl = DataLoader(ds, batch_size=32, shuffle=False)
    model = CtieseqModel()
    model.load_state_dict(torch.load(f"model_f{fold}.bin"))
    model.eval()

    bar = tqdm(enumerate(dl), total=len(dl))
    for step, data in bar:        
        X = data["X"]

        batch_size = X.size(0)

        outputs = model(X)
        preds.append(outputs.detach().cpu().numpy())
    test_pred = np.concatenate(preds)
    return test_pred

cite_test_x = np.load("../input/citeseq-pca-240-preprocessing/cite_test_x.npy")
print(cite_test_x.shape)

test_preds = np.array([predict(0), predict(1), predict(2), predict(3), predict(4)])
test_preds = np.mean(test_preds, axis=0)
print(test_preds.shape)