import scipy
import random
from sklearn.decomposition import TruncatedSVD


def pca_both(X, Xt):
    both = scipy.sparse.vstack([X, Xt])
    assert both.shape[0] == 119651
    
    col_idx = random.sample(range(0, 20856), 4500)
    both = both[:,col_idx]
    
    print(f"Shape of both before SVD: {both.shape}")
    svd = TruncatedSVD(n_components=64, random_state=1) # 512
    both = svd.fit_transform(both)
    print(f"Shape of both after SVD:  {both.shape}")

    # Hstack the svd output with the important features
    X = both[:70988]
    Xt = both[70988:]
    del both
    #X = np.hstack([X, X0])
    #Xt = np.hstack([Xt, X0t])
    #print(f"Reduced X shape:  {str(X.shape):14} {X.size*4/1024/1024/1024:2.3f} GByte")
    #print(f"Reduced Xt shape: {str(Xt.shape):14} {Xt.size*4/1024/1024/1024:2.3f} GByte")
    
    return X, Xt