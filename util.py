import numpy as np



#Define One-Hot encoding function
def OneHot(X, n, negative_class=0.):
    """Function for encoding ordinal labels into OneHot representation."""
    X = np.asarray(X).flatten()
    if n is None:
        n = np.max(X) + 1
    Xoh = np.ones((len(X), n)) * negative_class
    for i in range(len(X)):
        m = X[i]
        Xoh[i,m] = 1
    return Xoh