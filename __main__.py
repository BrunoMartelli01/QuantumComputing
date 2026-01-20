from sklearn.decomposition import PCA
from sklearn.metrics import classification_report

from dataset import get_dataset
from qkernel import FeatureMapKind, get_quantum_kernel
from qsvm import QASVM
from svm_classic import train_svm_features ,train_svm_precomputed_kernel, predict_svm_precomputed_kernel

def main_QUBO():
    train_features, test_features, train_labels, test_labels = get_dataset()
    kernel_params = [
        {'num_qubits': 30, 'f_map_name': FeatureMapKind.SU2HR, 'reps': 1},
        {'num_qubits': 8, 'f_map_name': FeatureMapKind.ZMAP, 'reps': 2},
        {'num_qubits': 16, 'f_map_name': FeatureMapKind.SU2RR, 'reps': 1},
    ]
    svm_params = [3, 15, 63, 255, 2047]

    for k in kernel_params:
        if k['num_qubits'] != train_features.shape[1]:
            pca = PCA(n_components=k['num_qubits'])
            fix_train = pca.fit_transform(train_features)
            fix_test = pca.transform(test_features)
        else:
            fix_train = train_features
            fix_test = test_features
        qkernel = get_quantum_kernel(**k)
        for big_c in svm_params:
            svm = QASVM(big_c=big_c, kernel_func=lambda x, y: qkernel.evaluate(x, y))
            svm.fit(fix_train, train_labels)
            predictions = svm.predict(fix_test)
            print('num_qubits:', k['num_qubits'], '- f_map:', k['f_map_name'].name, '- reps:', k['reps'], '- C:', big_c)
            print(classification_report(test_labels, predictions))

def main_Classic():
    train_features, test_features, train_labels, test_labels = get_dataset()
    kernel_params = [
        {'num_qubits': 30, 'f_map_name': FeatureMapKind.SU2HR, 'reps': 1},
        {'num_qubits': 8, 'f_map_name': FeatureMapKind.ZMAP, 'reps': 2},
        {'num_qubits': 16, 'f_map_name': FeatureMapKind.SU2RR, 'reps': 1},
    ]
    svm_params = [3, 15, 63, 255, 2047]

    for k in kernel_params:
        if k['num_qubits'] != train_features.shape[1]:
            pca = PCA(n_components=k['num_qubits'])
            fix_train = pca.fit_transform(train_features)
            fix_test = pca.transform(test_features)
        else:
            fix_train = train_features
            fix_test = test_features
        qkernel = get_quantum_kernel(**k)
        K_train = qkernel.evaluate(fix_train, fix_train)
        K_test = qkernel.evaluate(fix_test, fix_train)

        for big_c in svm_params:
            svm = train_svm_precomputed_kernel(K_train, train_labels, C=big_c)
            predictions = predict_svm_precomputed_kernel(svm, K_test)

            print('num_qubits:', k['num_qubits'], '- f_map:', k['f_map_name'].name, '- reps:', k['reps'], '- C:', big_c)
            print(classification_report(test_labels, predictions))
if __name__ == '__main__':
    main_Classic()
