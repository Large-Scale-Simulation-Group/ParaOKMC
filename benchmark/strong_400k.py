"""
Strong scaling - xlarge (400,000 H)
mpirun -np 1  python strong_400k.py
mpirun -np 2  python strong_400k.py
mpirun -np 4  python strong_400k.py
mpirun -np 8  python strong_400k.py
mpirun -np 16 python strong_400k.py
mpirun -np 32 python strong_400k.py
"""
from bench_common import run_bench
run_bench('Strong Scaling — 400k H', n_H=400000, grid_size=(160, 160, 160))
