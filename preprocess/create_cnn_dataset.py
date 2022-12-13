import os
import pandas as pd
import scipy
import gc

from col_names import constant_cols, important_cols
from dimensionality_reduction import pca_both


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

    # metadata
    metadata_df = pd.read_csv(FP_CELL_METADATA, index_col='cell_id')
    metadata_df = metadata_df[metadata_df.technology=="citeseq"]

    # Read train and convert to sparse matrix
    X = pd.read_hdf(FP_CITE_TRAIN_INPUTS).drop(columns=constant_cols)
    cell_index = X.index
    meta = metadata_df.reindex(cell_index)
    X0 = X[important_cols].values
    print(f"Original X shape: {str(X.shape):14} {X.size*4/1024/1024/1024:2.3f} GByte")
    gc.collect()
    X = scipy.sparse.csr_matrix(X.values)
    gc.collect()

    # Read test and convert to sparse matrix
    Xt = pd.read_hdf(FP_CITE_TEST_INPUTS).drop(columns=constant_cols)
    cell_index_test = Xt.index
    meta_test = metadata_df.reindex(cell_index_test)
    X0t = Xt[important_cols].values
    print(f"Original Xt shape: {str(Xt.shape):14} {Xt.size*4/1024/1024/1024:2.3f} GByte")
    gc.collect()
    Xt = scipy.sparse.csr_matrix(Xt.values)

    # create dataset from CNN
    all_X = []
    all_Xt = []
    for n in range(64):
        one_X, one_Xt = pca_both(X, Xt)
        all_X.append(one_X)
        all_Xt.append(one_Xt)