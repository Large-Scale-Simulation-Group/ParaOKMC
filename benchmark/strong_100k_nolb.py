"""
Load balancing comparison - every LB off, to compare against the run with LB on
mpirun -np 4  python strong_100k_nolb.py
mpirun -np 8  python strong_100k_nolb.py
mpirun -np 16 python strong_100k_nolb.py
"""
from bench_common import run_bench
run_bench('Strong 100k NO-LB', n_H=100000, grid_size=(100, 100, 100),
          enable_lb=False, output_file='benchmark_nolb.json')
