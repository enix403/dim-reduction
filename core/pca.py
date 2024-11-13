import torch as tr

def reduce_pca(X, dim_low):
    # Center the sample points
    X_centered = X - X.mean(dim=0, keepdim=True) # # shape = (N, H)
    
    # Calculate the covariance matrix
    X_cov = (X.T @ X) / (X.shape[0] - 1) # shape = (H, H)
    
    # Find eigenvalues and eigenvectors
    L, V = tr.linalg.eigh(X_cov)
    # L will have shape (H,)
    # V will have shape (H, H)
    
    # Sort the eigenvectors in decreasing values of their
    # respective eigenvectors, and get first dim_low vectors
    # These will be our principle components
    idx = tr.argsort(L, dim=0, descending=True)
    idx = idx[:dim_low]
    L = L[idx]
    V = V[:, idx]
    
    # Now the columns of V are our principle components
    
    # print(L.shape) # shape = (L,)
    # print(V.shape) # shape = (H, L)
    
    # Project the sample points on the principle components
    
    X_projected = X @ V
    
    # print(X_projected.shape) # (N, L)

    return X_projected