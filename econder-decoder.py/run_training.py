import os
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, Dataset

from dataset import CtieseqDataset
from model import CtieseqModel
from config import CFG
from utils import get_optimizer, get_scheduler
from train import train_one_fold


DATA_DIR = "/kaggle/input/open-problems-multimodal/"
FP_CELL_METADATA = os.path.join(DATA_DIR,"metadata.csv")

FP_CITE_TRAIN_INPUTS = os.path.join(DATA_DIR,"train_cite_inputs.h5")
FP_CITE_TRAIN_TARGETS = os.path.join(DATA_DIR,"train_cite_targets.h5")
FP_CITE_TEST_INPUTS = os.path.join(DATA_DIR,"test_cite_inputs.h5")

FP_MULTIOME_TRAIN_INPUTS = os.path.join(DATA_DIR,"train_multi_inputs.h5")
FP_MULTIOME_TRAIN_TARGETS = os.path.join(DATA_DIR,"train_multi_targets.h5")
FP_MULTIOME_TEST_INPUTS = os.path.join(DATA_DIR,"test_multi_inputs.h5")

FP_SUBMISSION = os.path.join(DATA_DIR,"sample_submission.csv")
FP_EVALUATION_IDS = os.path.join(DATA_DIR,"evaluation_ids.csv")

if __name__ == "__main__":

    # read data
    cite_train_x = np.load("../input/citeseq-pca-240-preprocessing/cite_train_x.npy")
    print(cite_train_x.shape)

    cite_test_x = np.load("../input/citeseq-pca-240-preprocessing/cite_test_x.npy")
    print(cite_train_x.shape)

    cite_train_y = pd.read_hdf(FP_CITE_TRAIN_TARGETS).values
    print(cite_train_y.shape)

    # run training
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    score_list = []
    for fold, (idx_tr, idx_va) in enumerate(kf.split(cite_train_x)):
        print(f"\nfold = {fold}")
        X_tr = cite_train_x[idx_tr] 
        y_tr = cite_train_y[idx_tr]
        
        X_va = cite_train_x[idx_va]
        y_va = cite_train_y[idx_va]
        
        ds_tr = CtieseqDataset(X_tr, y_tr)
        ds_va = CtieseqDataset(X_tr, y_tr)
        dl_tr = DataLoader(ds_tr, batch_size=CFG.tr_batch_size, shuffle=True)
        dl_va = DataLoader(ds_va, batch_size=CFG.va_batch_size, shuffle=False)
        
        model = CtieseqModel()
        optimizer = get_optimizer(model, CFG.lr, CFG.weight_decay, CFG.betas)
        scheduler = get_scheduler(optimizer) 

        train_one_fold(model, optimizer, scheduler, dl_tr, dl_va, fold)