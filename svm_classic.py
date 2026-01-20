
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline


def compute_kernel_matrix(X, kernel_fn, dtype=np.float32):
    """
    Calcola K (N,N) sfruttando simmetria: K[i,j] = K[j,i].
    kernel_fn(a,b) deve restituire uno scalare.
    """
    X = np.asarray(X)
    n = X.shape[0]
    K = np.empty((n, n), dtype=dtype)

    for i in range(n):
        K[i, i] = kernel_fn(X[i], X[i])
        for j in range(i + 1, n):
            kij = kernel_fn(X[i], X[j])
            K[i, j] = kij
            K[j, i] = kij
    return K


def train_svm_precomputed_kernel(K_train, y_train, C=1.0, cache_size_mb=1024):
    """
    Training SVM con kernel precomputato: usa solver classico (SMO) interno a LIBSVM.
    """
    K_train = np.asarray(K_train, dtype=np.float32)
    y_train = np.asarray(y_train)

    model = SVC(
        kernel="precomputed",
        C=C,
        cache_size=cache_size_mb,
    )
    model.fit(K_train, y_train)
    return model


def predict_svm_precomputed_kernel(model, K_test):
    """
    Predizione con kernel precomputato tra test e train: K_test shape (N_test, N_train).
    """
    K_test = np.asarray(K_test, dtype=np.float32)
    return model.predict(K_test)


def train_svm_features(X_train, y_train, C=1.0, gamma="scale", cache_size_mb=1024):
    """
    Training SVM classico su feature (senza precomputare K): scaler + RBF.
    """
    X_train = np.asarray(X_train, dtype=np.float32)
    y_train = np.asarray(y_train)

    model = make_pipeline(
        StandardScaler(),
        SVC(kernel="rbf", C=C, gamma=gamma, cache_size=cache_size_mb),
    )
    model.fit(X_train, y_train)
    return model
