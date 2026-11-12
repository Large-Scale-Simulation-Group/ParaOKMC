"""
Strong scaling - large (200,000 H)
mpirun -np 1  python strong_200k.py
mpirun -np 2  python strong_200k.py
mpirun -np 4  python strong_200k.py
mpirun -np 8  python strong_200k.py
mpirun -np 16 python strong_200k.py
mpirun -np 32 python strong_200k.py
"""
from bench_common import run_bench
run_bench('Strong Scaling — 200k H', n_H=200000, grid_size=(130, 130, 130))
