from tqdm.notebook import tqdm
from dataset import get_dataset
from qkernel import get_quantum_kernel, FeatureMapKind
import numpy as np
import time
from sklearn.decomposition import PCA
from qsvm import QASVM
from sklearn.metrics import f1_score
from itertools import product

"""
Dataset

Objective: find the best seed for sampling the sub-dataset
"""


def search_seed():
    try:
        with open('primes.txt') as f:
            primes = [int(x) for x in ' '.join(f.readlines()).split()]
    except FileNotFoundError:
        print('WARNING: File primes.txt not found, list of primes used: https://t5k.org/lists/small/10000.txt')
        print('INFO: seeds will fallback to the list [1, 10_000]')
        primes = [i + 1 for i in range(10_000)]

    curr_diff, selected_prime = 9999, None
    for p in tqdm(primes):
        new_diff = get_dataset(seed=p, get_deviation=True)
        if new_diff < curr_diff:
            curr_diff, selected_prime = new_diff, p

    return selected_prime, curr_diff


search_seed()

"""
Quantum Kernel

Objective: fine the bests parameters for the quantum kernel
"""

train_features, _, train_labels, _ = get_dataset()
train_labels = train_labels * 2 - 1


def evaluate_kernel(entanglement, num_qubits, reps, f_map):
    if num_qubits != train_features.shape[1]:
        pca = PCA(n_components=num_qubits)
        fix_train = pca.fit_transform(train_features)
    else:
        fix_train = train_features

    start = time.perf_counter()
    qkernel = get_quantum_kernel(f_map_name=f_map, num_qubits=num_qubits, reps=reps, entanglement=entanglement)
    K = qkernel.evaluate(x_vec=fix_train)
    y = (train_labels * 2) - 1
    alignment = (y.T @ K @ y) / (np.linalg.norm(K ** 2, 'fro') * y.shape[0])
    duration = time.perf_counter() - start

    return alignment, duration


for reps_, num_qubits_, f_map_ in product([1, 2, 3], [4, 8, 16, 30], FeatureMapKind):
    al, dur = evaluate_kernel(num_qubits=num_qubits_, reps=reps_, f_map=f_map_, entanglement='linear')
    print(f'LOG: num_qubits {num_qubits_} - reps {reps_} - f_map {f_map_.name} - Alignment {al} - Duration {dur}')

"""
Quantum Support Vector Machine

Objective: find the best value for the regularization parameter C
"""

train_features, test_features, train_labels, test_labels = get_dataset()
for big_c in range(1, 13):
    svm = QASVM(big_c=2 ** big_c - 1)
    svm.fit(train_features, train_labels)
    predictions = svm.predict(test_features)
    print(f'C: {2 ** big_c - 1} - F1: {round(f1_score(test_labels, predictions), 3)}')
