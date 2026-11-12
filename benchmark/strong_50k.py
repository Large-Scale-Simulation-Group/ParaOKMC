"""
Strong scaling - small (50,000 H)
mpirun -np 1  python strong_50k.py
mpirun -np 2  python strong_50k.py
mpirun -np 4  python strong_50k.py
mpirun -np 8  python strong_50k.py
mpirun -np 16 python strong_50k.py
mpirun -np 32 python strong_50k.py
"""
from bench_common import run_bench
run_bench('Strong Scaling — 50k H', n_H=50000, grid_size=(80, 80, 80))
