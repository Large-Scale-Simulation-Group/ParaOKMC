# Weak scaling (§4.3, Fig. 10)

The weak-scaling series was run on one cluster node with two AMD EPYC 7452 (64 cores);
`weak_scaling.csv` is the weak-scaling half of the master
table `../4.1_benchmark_workloads/bscc_a_measured_20260717.csv` (same columns). The raw
`.out` logs of all jobs are in `../4.1_benchmark_workloads/logs_v4_v5_v7.tar.gz`, and the
workloads, protocol and parser are described in `../4.1_benchmark_workloads/README.md`.
Figure 10 uses the rows with `lib_arm` = `new` (diffusion) and `new_on`/`new` (pair) at
n_p = 1, 2, 4, 8, 16, 32, 48, 64.
