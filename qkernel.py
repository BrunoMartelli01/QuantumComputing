from qiskit.circuit.library import zz_feature_map, z_feature_map
from qiskit_aer.primitives import SamplerV2
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from qiskit_machine_learning.state_fidelities import ComputeUncompute


def get_quantum_kernel(shots=5, f_map_name='z_map', num_qubits=2, reps=2, entanglement='linear'):
    f_map = zz_feature_map if f_map_name == 'zz_map' else z_feature_map
    sampler = SamplerV2(default_shots=shots, options={'backend_options': {'method': 'matrix_product_state'}})
    fidelity = ComputeUncompute(sampler=sampler)
    feature_map = f_map(feature_dimension=num_qubits, reps=reps, entanglement=entanglement)
    return FidelityQuantumKernel(fidelity=fidelity, feature_map=feature_map)
