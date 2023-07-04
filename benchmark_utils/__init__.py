# `benchmark_utils` is a module in which you can define code to reuse in
# the benchmark objective, datasets, and solvers. The folder should have the
# name `benchmark_utils`, and code defined inside will be importable using
# the usual import syntax
from .build_deepinv_models import build_DPIR_model, build_wavelet_model, build_waveletdict_model
from .build_deepinv_datasets import build_set3c_dataset, build_fastMRI_dataset