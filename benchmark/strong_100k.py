"""
Strong scaling - medium (100,000 H)
mpirun -np 1  python strong_100k.py
mpirun -np 2  python strong_100k.py
mpirun -np 4  python strong_100k.py
mpirun -np 8  python strong_100k.py
mpirun -np 16 python strong_100k.py
mpirun -np 32 python strong_100k.py
"""
from bench_common import run_bench
run_bench('Strong Scaling — 100k H', n_H=100000, grid_size=(100, 100, 100))
