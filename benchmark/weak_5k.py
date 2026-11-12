"""
Weak scaling - 5,000 H per rank
mpirun -np 1  python weak_5k.py    →   5,000 H
mpirun -np 2  python weak_5k.py    →  10,000 H
mpirun -np 4  python weak_5k.py    →  20,000 H
mpirun -np 8  python weak_5k.py    →  40,000 H
mpirun -np 16 python weak_5k.py    →  80,000 H
mpirun -np 32 python weak_5k.py    → 160,000 H
"""
from mpi4py import MPI
from bench_common import run_bench

nprocs = MPI.COMM_WORLD.size
per_rank = 5000
n_H = per_rank * nprocs

# The grid grows with the particle count, keeping the density constant
g = max(50, int((n_H / 10.0) ** (1/3)) + 1)
run_bench(f'Weak Scaling — 5k/rank (total {n_H:,})',
          n_H=n_H, grid_size=(g, g, g))
