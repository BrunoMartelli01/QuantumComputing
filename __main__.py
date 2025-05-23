from sklearn.decomposition import PCA
from sklearn.metrics import classification_report

from dataset import get_dataset
from qkernel import FeatureMapKind, get_quantum_kernel
from qsvm import QASVM


def main():
    train_features, test_features, train_labels, test_labels = get_dataset()
    kernel_params = [
        {'num_qubits': 30, 'f_map_name': FeatureMapKind.SU2HR, 'reps': 1},
        {'num_qubits': 8, 'f_map_name': FeatureMapKind.ZMAP, 'reps': 3},
        {'num_qubits': 8, 'f_map_name': FeatureMapKind.ZMAP, 'reps': 2},
        {'num_qubits': 16, 'f_map_name': FeatureMapKind.SU2RR, 'reps': 1},
        {'num_qubits': 8, 'f_map_name': FeatureMapKind.SU2RR, 'reps': 1},
    ]
    svm_params = [7, 63, 255]

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


if __name__ == '__main__':
    main()
