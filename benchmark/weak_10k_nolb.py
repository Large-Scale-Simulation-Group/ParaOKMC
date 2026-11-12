"""
Load balancing comparison - every LB off, to compare against the run with LB on
mpirun -np 4  python weak_10k_nolb.py
mpirun -np 8  python weak_10k_nolb.py
mpirun -np 16 python weak_10k_nolb.py
"""
from mpi4py import MPI
from bench_common import run_bench

nprocs = MPI.COMM_WORLD.size
per_rank = 10000
n_H = per_rank * nprocs

g = max(50, int((n_H / 10.0) ** (1/3)) + 1)
run_bench(f'Weak 10k/rank NO-LB (total {n_H:,})',
          n_H=n_H, grid_size=(g, g, g),
          enable_lb=False, output_file='benchmark_nolb.json')
