from typing import Callable

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
import dimod
from dwave.samplers import SimulatedAnnealingSampler
from dimod.generators.integer import binary_encoding
import pyomo.environ as pyo
import os


class QASVM(BaseEstimator, ClassifierMixin):
    def __init__(self, big_c: int, kernel_func: Callable[[np.array, np.array], np.array], num_reads: int = 10):
        self.big_c: int = big_c
        self.sv_labels: np.array = None
        self.sv_alphas: np.array = None
        self.sv_examples: np.array = None
        self.b: float | None = None
        self.kernel: Callable[[np.array, np.array], np.array] = kernel_func
        self.num_reads: int = num_reads

    def fit(self, examples: np.array, labels: np.array):
        svm_fix_labels = labels * 2 - 1  # SVM labels must be -1/1 not 0/1
        examples_kernel = self.kernel(examples, examples)
        cqm = self.create_model(examples_kernel, svm_fix_labels)
        sampled_sol = self.sample_from_cqm(cqm)
        self.extract_support_vectors(sampled_sol, cqm, examples, svm_fix_labels, examples_kernel)

    def create_model(self, kernel: np.array, labels: np.array):
        n_samples, _ = kernel.shape
        model = pyo.ConcreteModel()
        model.alpha = pyo.Var(range(n_samples), domain=pyo.NonNegativeIntegers, bounds=(0, self.big_c))
        model.obj = pyo.Objective(rule=lambda curr_model: (
                0.5 * ((labels * curr_model.alpha) @ kernel @ (labels * curr_model.alpha).T)
                - sum(curr_model.alpha[i] for i in range(n_samples))
        ), sense=pyo.minimize)
        model.constraint = pyo.Constraint(rule=lambda curr_model: sum(curr_model.alpha * labels) == 0)
        model.write('qsvm.lp')
        cqm = dimod.lp.load('qsvm.lp')
        os.remove('qsvm.lp')

        return cqm

    def sample_from_cqm(self, cqm: dimod.ConstrainedQuadraticModel):
        bqm, _default_back_function = dimod.cqm_to_bqm(cqm)
        solver = SimulatedAnnealingSampler()
        bqm_sampleset = solver.sample(bqm, num_reads=self.num_reads)
        bqm_sol = bqm_sampleset.first.sample

        # Convert back to CQM solution
        # _default_back_function from D-Wave doesn't work
        integers = {v: binary_encoding(v, int(cqm.upper_bound(v))) for v in cqm.variables}
        cqm_sample = {}
        for v, bqm in integers.items():
            cqm_sample[v] = 0
            for u in bqm.variables:
                new_s = np.uint64(bqm_sol[u])  # Added conversion
                cqm_sample[v] += new_s * u[1]
        return cqm_sample

    def extract_support_vectors(self, sampled_sol, cqm, examples, labels, kernel):
        indexes = {var: i for i, var in enumerate(cqm.variables)}
        sv_idxs = [indexes[k] for k, v in sampled_sol.items() if 0 < v < self.big_c]
        sv_alphas = np.array([sampled_sol[k] for k, v in sampled_sol.items() if 0 < v < self.big_c])
        assert len(sv_idxs) > 0, 'No support vectors found'
        self.sv_examples = examples[sv_idxs]
        self.sv_labels = labels[sv_idxs]
        self.sv_alphas = sv_alphas
        sub_kernel = kernel[np.ix_(sv_idxs, sv_idxs)]
        alpha_y = self.sv_alphas * self.sv_labels
        self.b = np.mean(self.sv_labels - np.dot(sub_kernel, alpha_y))

    def predict(self, examples: np.array) -> np.array:
        assert self.sv_labels is not None, 'No sv_labels, you need to fit before training'
        assert self.sv_examples is not None, 'No sv_examples, you need to fit before training'
        assert self.sv_alphas is not None, 'No sv_alphas, you need to fit before training'
        assert self.b is not None, 'No b, you need to fit before training'

        pred = np.sign(np.dot(self.sv_alphas * self.sv_labels, self.kernel(self.sv_examples, examples)) + self.b)
        return np.int8((pred + 1) / 2)  # return labels in 0/1 form
