# End2End Quantum learning

## Problem Statement

The goal is to implement a **fully quantum pipeline** for training a **Support Vector Machine (SVM)** model using **quantum kernel methods**.

## Problem Motivation

1. In the current literature, the term **qSVM (quantum SVM)** often refers to **incomplete quantum implementations**:
   - In the **gate-based** paradigm, only the kernel computation is quantum, while the SVM training is performed **classically**.
   - In **quantum annealing**, optimization is performed on quantum hardware, but the **kernel (if present)** is computed **classically**.

2. **Integrating different quantum computing paradigms** offers a path toward **distributed quantum computation**:
   - Technologies are seen not as competitors but as **complementary**.
   - The goal is to exploit the **strengths of each paradigm**, similar to modern **classical HPC**, where CPUs, GPUs, and FPGAs collaborate depending on the task.

## Degrees of Freedom and Design Choices

### 1. Choice of Kernel

- **Classical reference kernel**: RBF (Radial Basis Function), used for baseline comparisons.
- **Quantum kernel**: ZZFeatureMap, ZFeatureMap, custom kernels.
- While kernel quality depends heavily on the dataset, for a Proof-of-Concept it is sufficient to use **standard**, **reasonably good** kernels.

### 2. Choice of SVM Model

- For the **quantum annealing** part, the **dual form** of the SVM problem is used, with the hyperparameter `C` controlling the soft margin (classification error tolerance).
- For **classical benchmarking**, the `SVC` class from `scikit-learn` is used, which solves the dual problem via **gradient-based optimization**.

### 3. Simulation vs. Real Hardware

- **Quantum gate execution**: While execution on real hardware (e.g., IBM QPU) is possible, **simulations using MPS (Matrix Product State)** or **Tensor Networks (TN)** are preferred due to time constraints.
- **Quantum annealing**: Although real hardware (e.g., D-Wave or Fujitsu) is available, **simulated annealing** will be used due to lack of direct access.

> Simulation removes noise-related issues from quantum processing units (QPUs) and simplifies development by eliminating inter-system communication overhead.

### HPC/QHPC Considerations

In an ideal HPC or QHPC setting, network latency is minimized.
In a local or experimental environment, hybrid execution on real hardware would involve sequences like:

- **local machine -> IBM QPU -> local machine** x N (where N is the number of kernel entries)
- then: **local machine -> D-Wave QPU -> local machine**

Such a setup is not easily scalable in non-distributed environments.

## Available scripts

- `__main.py__`: Entry point to execute final tests with optimal parameters
- `dataset.py`: Generates dataset using `load_breast_cancer` from scikit-learn
- `parameter_study.py`: Explores optimal parameters for quantum SVM
- `qkernel_log_to_csv.py`: Converts kernel study logs to a CSV format for easier analysis
- `qkernel.py`: Implements quantum kernel construction
- `qsvm.py`: Quantum SVM implementation using quantum annealing

## Other files

- `End2EndQuantumLearning.pdf`: Final project report (output of LaTeX build in the report folder)
- `primes.txt`: List of prime numbers used to generate dataset subsets
- `requirements.txt`: Python dependencies for running the project
