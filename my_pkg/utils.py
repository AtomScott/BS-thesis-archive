import numpy as np

def format_input(X, y):
    X = [X[np.where(y==t)] for t in np.unique(y)]
    return X, np.unique(y)


def gram_schmidt(vectors):
    '''Return orthogonal vectors in same dimension
    '''
    basis = []
    for v in vectors.transpose():
        w = v - sum( np.dot(v,b)*b  for b in basis )
        basis.append(w/np.linalg.norm(w))
    return np.array(basis)

def normalize_columns(X):
    '''Normalizes the columns of a matrix
    '''
    return np.asarray([x/np.linalg.norm(x) for x in X])

def canonical_angles(S1, S2):
    '''Compute canonical angles between subspace S1 and S2
    
    Warning
    -------
    S1 and S2 must be two orthonormal matrixes fo size D by m.
    Orthonormal (orthogonal) matrices are matrices in which 
    the columns vectors form an orthonormal set (each column vector 
    has length one and is orthogonal to all the other colum vectors). 
    '''
    U, S, V = np.linalg.svd(S1@S2.transpose())
    return S
    

def grassman_kernel(S1, S2):
    '''Compute similarity 
    '''
    angles = np.power(canonical_angles(S1, S2),2)
    sim = angles.mean()
    return sim

def plot_stats(classifier, X, y):
    classifier.classes_ = np.unique(y)
    y_pred = classifier.predict(X)
    disp = metrics.plot_confusion_matrix(classifier, X, y)
    disp.figure_.suptitle(f"Confusion Matrix score: {classifier.score( X, y)}")
    print("Classification report for classifier %s:\n%s\n"
          % (classifier, metrics.classification_report(y, y_pred)))
    plt.show()