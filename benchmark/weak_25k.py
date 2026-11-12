"""
Weak scaling - 25,000 H per rank
mpirun -np 1  python weak_25k.py   →  25,000 H
mpirun -np 2  python weak_25k.py   →  50,000 H
mpirun -np 4  python weak_25k.py   → 100,000 H
mpirun -np 8  python weak_25k.py   → 200,000 H
mpirun -np 16 python weak_25k.py   → 400,000 H
mpirun -np 32 python weak_25k.py   → 800,000 H
"""
from mpi4py import MPI
from bench_common import run_bench

nprocs = MPI.COMM_WORLD.size
per_rank = 25000
n_H = per_rank * nprocs

g = max(50, int((n_H / 10.0) ** (1/3)) + 1)
run_bench(f'Weak Scaling — 25k/rank (total {n_H:,})',
          n_H=n_H, grid_size=(g, g, g))
