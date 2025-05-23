from qiskit.circuit.library import zz_feature_map, z_feature_map
from qiskit_aer.primitives import SamplerV2
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from qiskit_machine_learning.state_fidelities import ComputeUncompute
from enum import Enum
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector


class FeatureMapKind(Enum):
    ZMAP = 0
    ZZMAP = 1
    SU2RR = 2
    SU2HR = 3


def get_quantum_kernel(f_map_name, num_qubits, reps, entanglement='linear', shots=5):
    assert entanglement in ['linear', 'circular', 'full'], 'Invalid entanglement'
    assert f_map_name in FeatureMapKind, 'Invalid feature map name'

    if f_map_name == FeatureMapKind.ZMAP:
        feature_map = z_feature_map(feature_dimension=num_qubits, reps=reps, entanglement=entanglement)
    elif f_map_name == FeatureMapKind.ZZMAP:
        feature_map = zz_feature_map(feature_dimension=num_qubits, reps=reps, entanglement=entanglement)
    elif f_map_name == FeatureMapKind.SU2RR:
        feature_map = su2_feature_map_rr(feature_dimension=num_qubits, reps=reps, entanglement=entanglement)
    else:  # f_map_name == FeatureMapKind.SU2HR:
        feature_map = su2_feature_map_hr(feature_dimension=num_qubits, reps=reps, entanglement=entanglement)
    sampler = SamplerV2(default_shots=shots, options={'backend_options': {'method': 'matrix_product_state'}})
    fidelity = ComputeUncompute(sampler=sampler)
    return FidelityQuantumKernel(fidelity=fidelity, feature_map=feature_map)


def su2_feature_map_hr(feature_dimension, reps, entanglement):
    x = ParameterVector('x', feature_dimension)
    qc = QuantumCircuit(feature_dimension)
    for _ in range(reps):
        for i in range(feature_dimension):
            qc.h(i)
            qc.rz(x[i], i)
            qc.ry(x[i], i)
        qc = entangle(qc, entanglement)
        for i in range(feature_dimension):
            qc.rz(x[i], i)
    return qc


def su2_feature_map_rr(feature_dimension, reps, entanglement):
    x = ParameterVector('x', feature_dimension)
    qc = QuantumCircuit(feature_dimension)
    for _ in range(reps):
        for i in range(feature_dimension):
            qc.ry(x[i], i)
            qc.rz(x[i], i)
        qc = entangle(qc, entanglement)
        for i in range(feature_dimension):
            qc.ry(x[i], i)
            qc.rz(x[i], i)
    return qc


def entangle(qc, entanglement):
    if entanglement == "linear":
        for i in range(qc.num_qubits - 1):
            qc.cx(i, i + 1)
    elif entanglement == "circular":
        qc.cx(qc.num_qubits - 1, 0)
        for i in range(qc.num_qubits - 1):
            qc.cx(i, i + 1)
    elif entanglement == "full":
        for i in range(qc.num_qubits):
            for j in range(i + 1, qc.num_qubits):
                qc.cx(i, j)
    return qc
