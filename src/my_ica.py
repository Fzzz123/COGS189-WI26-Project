import numpy as np


def center(X: np.ndarray):#To avoid weird terms like 𝜇𝜇𝑇 showing up.
    mu = X.mean(axis=1, keepdims=True)
    Xc = X - mu
    return Xc, mu

def cov_matrix(Xc: np.ndarray):#get cov
    n_ch, T = Xc.shape
    return (Xc @ Xc.T) / T

def whiten_pca(X: np.ndarray, n_components: int = None):#Although I really like that percentage-based setting, it’s too hard. I’ll deal with it when I get a chance later.
    Xc, mu = center(X)
    C = cov_matrix(Xc)
    
    d, E = np.linalg.eigh(C)   #It’s basically like extracting features from this new matrix and then rearranging them. This step is actually very similar to PCA.
    order = np.argsort(d)[::-1]#The main goal is dimensionality reduction. d denotes the eigenvalues, and E denotes the eigenvectors.
    d, E = d[order], E[:, order] 
    
    n_ch = X.shape[0]

    d_co = d[:n_components]
    d_stand = np.sqrt(d_co)
    E_co = E[:, :n_components]

    K = (np.diag(1.0 / d_stand) @ E_co.T)       
    
    Xw = K @ Xc                              #Because we diagonalized d_co, the matrices that follow are all diagonal.
    return Xw, K, mu                         #Then this step essentially incorporates the proportion of the eigenvalues corresponding to Xc, and produces Xw.
    
def _sym_decorrelation(W: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    #Because different w may gradually become very similar, and at the same time we want the length to stay stable.
    s, u = np.linalg.eigh(W @ W.T)
    s = np.clip(s, eps, None)
    inv_sqrt = u @ np.diag(1.0 / np.sqrt(s)) @ u.T
    return inv_sqrt @ W

def icaa(
    X: np.ndarray,                   
    n_components: int = None,
    max_iter: int = 500,
    random_state: int = 0,
    tol: float = 1e-4
):

    Xw, K, mu = whiten_pca(X, n_components=n_components)

    k = Xw.shape[0]
    T = Xw.shape[1]
    rng = np.random.default_rng(random_state)
    W = rng.standard_normal((k, k))
    W = _sym_decorrelation(W)

    def g(Y):
        return np.tanh(Y)

    def gprime(Y):
        T = np.tanh(Y)
        return 1.0 - T*T
        
    for _ in range(max_iter):
        W_old = W.copy()

        Y = W @ Xw       
        G = g(Y)                         
        Gp = gprime(Y)                   

        W = (G @ Xw.T) / T - np.diag(Gp.mean(axis=1)) @ W
        W = _sym_decorrelation(W)

        lim = np.max(np.abs(np.abs(np.diag(W @ W_old.T)) - 1.0))
        if lim < tol:
            break

    S = W @ Xw                       

    W_full = W @ K    

    A = np.linalg.pinv(W_full) 

    return S, W_full, A, mu

