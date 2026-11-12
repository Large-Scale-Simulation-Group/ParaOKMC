"""快速 A/B 对比：cell list vs cKDTree 路径的 sim_t 演化"""
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mpi4py import MPI
from paraokmc.v16_engine import create_okmc_engine_v16
from paraokmc.event_compiler import merge_event_config

# 复用 e1 的配置
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..', 'example', 'e1'))
from run_cascade_annealing import build_events, create_config

comm = MPI.COMM_WORLD
rank = comm.rank

def run_with_cell_list(use_cl, label, max_steps=5000):
    base_config = create_config(temperature=300.0, n_sia=100, n_v=100)
    base_config['max_steps'] = max_steps
    base_config['random_seed'] = 42
    base_config['verbose'] = False

    config = merge_event_config(base_config, build_events())
    engine = create_okmc_engine_v16(config)

    # 切换 cell list 开关
    if hasattr(engine, '_event_builder'):
        eb = engine._event_builder
        if hasattr(eb, '_vectorized_builder'):
            eb._vectorized_builder.use_cell_list = use_cl

    stats = engine.run()

    if rank == 0:
        print(f"\n[{label}] steps={stats['total_steps']}  "
              f"sim_t={stats['total_time']:.6e}s  "
              f"objs={stats.get('final_objects', '?')}")

    # 收集存活数
    objs = engine.rma_object_pool.get_local_objects(active_only=True)
    sia_local = sum(1 for o in objs if 'SIA' in str(o['type']).strip())
    v_local = sum(1 for o in objs if str(o['type']).strip().startswith('V'))
    sia_total = comm.reduce(sia_local, op=MPI.SUM, root=0)
    v_total = comm.reduce(v_local, op=MPI.SUM, root=0)

    if rank == 0:
        print(f"[{label}] SIA clusters={sia_total}  V clusters={v_total}")

    engine.destroy()
    return stats

if rank == 0:
    print("=" * 60)
    print("A/B: Cell List vs cKDTree sim_t comparison")
    print("=" * 60)

stats_cl = run_with_cell_list(True, "CellList")
stats_kd = run_with_cell_list(False, "cKDTree ")
