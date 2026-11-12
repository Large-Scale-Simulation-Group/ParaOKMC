"""
运行时负载均衡器（Runtime Load Balancer）专项测试

测试策略：
    - 初始对象全部集中在 box 的一个角落（局部热点）
    - 关闭增量更新、预测负载均衡、对象负载均衡
    - 只开启运行时负载均衡（runtime_load_balancer）
    → 观察速率计算期间事件是否能动态分配给空闲 rank

运行：
    mpirun -np 2 python run_runtime_lb_test.py
    mpirun -np 4 python run_runtime_lb_test.py
"""

import numpy as np
import argparse
import time
from mpi4py import MPI
from numba import njit
from paraokmc.v16_engine import create_okmc_engine_v16
from paraokmc.event_class import SingleObjectEvent, PairObjectEvent
from paraokmc.event_compiler import merge_event_config
from paraokmc.jit_rate_engine import RateDeps

# ==================== 物理常数 ====================

k_B   = 8.617e-5   # eV/K
a_0   = 3.17       # Å  (W bcc)
R_c   = 5.0        # Å  捕获半径

# ==================== 事件类（复用 e1 的物理） ====================

@njit
def _sia_migration_kernel(n, temperature):
    n_s = max(n, 1.0)
    return 6e12 * (n_s ** (-0.5)) * np.exp(-0.013 / (8.617e-5 * temperature))

class SIAMigration(SingleObjectEvent):
    object_types = ['SIA']
    params = []
    rate_deps = RateDeps(obj1=['n'], physics=['temperature'])
    rate_kernel = _sia_migration_kernel

    @classmethod
    def execute(cls, event, objs, config):
        o = objs[0]
        step = 3.17 * 0.5
        dx, dy, dz = np.random.normal(0, step, 3)
        return [{'id': int(o['id']),
                 'x': float(o['x']) + dx,
                 'y': float(o['y']) + dy,
                 'z': float(o['z']) + dz}]

@njit
def _v_migration_kernel(n, temperature):
    n_s = max(n, 1.0)
    return n_s * (1000.0 ** (-(n_s - 1))) * 6e12 * np.exp(-1.66 / (8.617e-5 * temperature))

class VMigration(SingleObjectEvent):
    object_types = ['V']
    params = []
    rate_deps = RateDeps(obj1=['n'], physics=['temperature'])
    rate_kernel = _v_migration_kernel

    @classmethod
    def execute(cls, event, objs, config):
        o = objs[0]
        step = 3.17 * 0.5
        dx, dy, dz = np.random.normal(0, step, 3)
        return [{'id': int(o['id']),
                 'x': float(o['x']) + dx,
                 'y': float(o['y']) + dy,
                 'z': float(o['z']) + dz}]

@njit
def _sia_v_recomb_kernel(n_sia, n_v, distance, temperature):
    n1 = max(n_sia, 1.0)
    n2 = max(n_v, 1.0)
    d  = max(distance, 0.1)
    a2_6 = 3.17**2 / 6.0
    D_sia = a2_6 * 6e12 * (n1 ** (-0.5)) * np.exp(-0.013 / (8.617e-5 * temperature))
    D_v   = a2_6 * n2 * (1000.0 ** (-(n2-1))) * 6e12 * np.exp(-1.66 / (8.617e-5 * temperature))
    return 4.0 * 3.141592653589793 * (D_sia + D_v) * 5.0 / d

class SIAVRecombination(PairObjectEvent):
    type_pairs = [('SIA', 'V')]
    max_distance = R_c * 3
    params = ['distance']
    rate_deps = RateDeps(obj1=['n'], obj2=['n'], params=['distance'],
                         physics=['temperature'], kind='pair')
    rate_kernel = _sia_v_recomb_kernel

    @classmethod
    def execute(cls, event, objs, config):
        sia, v = objs[0], objs[1]
        n_sia = max(int(sia['float_props'][1]), 1)
        n_v   = max(int(v['float_props'][1]), 1)
        diff  = n_sia - n_v
        result = [
            {'id': int(sia['id']), 'deleted': True},
            {'id': int(v['id']),   'deleted': True},
        ]
        if diff > 0:
            result.append({'id': -1, 'type': 'SIA', 'new': True,
                           'x': float(sia['x']), 'y': float(sia['y']),
                           'z': float(sia['z']), 'float_props': [0.0, diff]})
        elif diff < 0:
            result.append({'id': -1, 'type': 'V', 'new': True,
                           'x': float(v['x']), 'y': float(v['y']),
                           'z': float(v['z']), 'float_props': [0.0, -diff]})
        return result


# ==================== 热点初始条件 ====================

def create_hotspot_initial_objects(n_sia, n_v, grid_size, lc, hotspot_fraction=0.1):
    box = [grid_size[i] * lc for i in range(3)]
    hx, hy, hz = box[0] * hotspot_fraction, box[1] * hotspot_fraction, box[2] * hotspot_fraction

    def initial_objects_func(rank, size, config):
        saved_state = np.random.get_state()
        np.random.seed(config.get('_user_seed', 42))
        objects = []
        for i in range(n_sia):
            objects.append({'id': i, 'type': 'SIA',
                'x': np.random.uniform(0, hx), 'y': np.random.uniform(0, hy),
                'z': np.random.uniform(0, hz), 'float_props': [0.0, 1.0]})
        for i in range(n_v):
            objects.append({'id': n_sia + i, 'type': 'V',
                'x': np.random.uniform(0, hx), 'y': np.random.uniform(0, hy),
                'z': np.random.uniform(0, hz), 'float_props': [0.0, 1.0]})
        np.random.set_state(saved_state)
        return objects if rank == 0 else []

    return initial_objects_func


# ==================== 配置 ====================

def create_config(temperature=300.0, n_sia=80, n_v=80):
    lc = a_0  # 3.17 Å
    lattice_grid = (50, 50, 50)
    h_factor = 1
    cap = n_sia + n_v + 200

    base_config = {
        'physics': {'temperature': temperature, 'lattice_constant': lc},
        'objects': {
            'SIA': {'properties': {'n': 1}},
            'V':   {'properties': {'n': 1}},
        },
        'domain_decomposition': {
            'lattice_constant': lc,
            'lattice_grid': lattice_grid,
            'h_factor': h_factor,
            'periodic': [True, True, True],
            'max_interaction_distance': R_c * 3,
        },
        'initial_objects_func': create_hotspot_initial_objects(
            n_sia, n_v, lattice_grid, lc, hotspot_fraction=0.1),
        'lattice_constant': lc,
        'grid_size':        lattice_grid,
        'temperature':      temperature,
        'max_steps':        100,
        'max_time':         1.0,
        'verbose':          False,
        'random_seed':      42,
        'progress_interval': 2000,
        'rma_object_pool': {'initial_capacity': cap, 'auto_expand': True},
        'rma_event_pool': {
            'capacity_zero': 10, 'capacity_single': cap,
            'capacity_pair': cap * 20, 'auto_expand': True,
        },
        # ===== 核心：只开 runtime_load_balancer，其余全关 =====
        'incremental_update':      {'enabled': False},
        'predictive_rebalance':    {'enabled': False},
        'adaptive_weight_tracker': {'enabled': False},
        'runtime_load_balancer':   {'enabled': True, 'check_interval': 16},
        'object_load_balancer':    {'enabled': False},
    }
    events_config = {
        'SIA_migration':   SIAMigration,
        'V_migration':     VMigration,
        'SIA_V_recomb':    SIAVRecombination,
    }
    return merge_event_config(base_config, events_config)


# ==================== 主程序 ====================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Runtime Load Balancer 专项测试')
    parser.add_argument('--temperature', type=float, default=300.0)
    parser.add_argument('--n_sia', type=int, default=80)
    parser.add_argument('--n_v',   type=int, default=80)
    parser.add_argument('--max_steps', type=int, default=100)
    args = parser.parse_args()

    comm = MPI.COMM_WORLD
    rank = comm.rank
    size = comm.size

    config = create_config(args.temperature, args.n_sia, args.n_v)
    config['max_steps'] = args.max_steps

    if rank == 0:
        print("=" * 60)
        print("Runtime Load Balancer 专项测试")
        print(f"  ranks={size}  T={args.temperature}K")
        print(f"  n_sia={args.n_sia}  n_v={args.n_v}")
        print(f"  热点分布: 仅 rank 0 持有所有对象")
        print(f"  runtime_load_balancer: ON  (check_interval=16)")
        print(f"  其余负载均衡:          全部 OFF")
        print("=" * 60)

    engine = create_okmc_engine_v16(config)

    t0 = time.perf_counter()
    stats = engine.run()
    wall = time.perf_counter() - t0

    n_local = engine.rma_object_pool.get_local_count()
    cs = engine.rma_event_pool.count_single
    cp = engine.rma_event_pool.count_pair

    all_objs   = comm.gather(n_local, root=0)
    all_single = comm.gather(cs, root=0)
    all_pair   = comm.gather(cp, root=0)

    # 收集 Runtime LB 统计信息
    if hasattr(engine, 'runtime_load_balancer') and engine.runtime_load_balancer:
        lb_stats = engine.runtime_load_balancer.stats
        all_sent = comm.gather(lb_stats.get('total_tasks_sent', 0), root=0)
        all_received = comm.gather(lb_stats.get('total_tasks_received', 0), root=0)
        all_steals = comm.gather(lb_stats.get('total_steals', 0), root=0)
    else:
        all_sent = all_received = all_steals = None

    if rank == 0:
        print(f"\n{'=' * 60}")
        print(f"[结果] steps={engine.current_step}  sim_time={engine.current_time:.4e}s  wall={wall:.1f}s")
        print(f"[结果] 对象分布:   {all_objs}  (总计 {sum(all_objs)})")
        print(f"[结果] single事件: {all_single}")
        print(f"[结果] pair事件:   {all_pair}")

        # Runtime LB 统计
        if all_sent is not None:
            print(f"\n[Runtime LB] 事件迁移统计:")
            print(f"  发送: {all_sent}")
            print(f"  接收: {all_received}")
            print(f"  偷取次数: {all_steals}")

        # 详细性能分解
        n = engine.current_step
        if n > 0:
            print(f"\n[性能] 平均每步耗时:")
            print(f"  build:  {engine.stats['build_time']/n*1e3:.1f}ms")
            print(f"  rate:   {engine.stats['rate_time']/n*1e3:.1f}ms")
            print(f"  select: {engine.stats['select_time']/n*1e3:.1f}ms")
            print(f"  exec:   {engine.stats['execute_time']/n*1e3:.1f}ms")
        print("=" * 60)
