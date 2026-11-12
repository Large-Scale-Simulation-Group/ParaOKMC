"""
v1.6.9 JIT 事件构建器测试

测试目标：
1. 验证 JIT 事件构建器的正确性
2. 验证守恒定律
"""

import numpy as np
from mpi4py import MPI
import time
from numba import njit
from paraokmc.jit_rate_engine import RateDeps

# 导入事件类系统
from paraokmc.event_class import (
    SingleObjectEvent, PairObjectEvent,
    get_physics_param,
)

# 导入编译器
from paraokmc.event_compiler import merge_event_config

# 导入配置工具和引擎
from paraokmc.config_system import create_initial_objects_func
from paraokmc.v16_engine import create_okmc_engine_v16


# ============================================================================
# JIT kernels (must be defined before classes that reference them)
# ============================================================================

@njit
def arrhenius_kernel(E_m, nu, temperature, k_B):
    return nu * np.exp(-E_m / (k_B * temperature))


@njit
def capture_kernel(distance, k_prefactor):
    return k_prefactor / (distance * distance + 0.1)


# ============================================================================
# 事件类定义
# ============================================================================

class MigrationEvent(SingleObjectEvent):
    """迁移事件"""
    object_types = ['H', 'V', 'HV']
    params = ['E_m', 'nu']
    rate_deps = RateDeps(obj1=['E_m', 'nu'], physics=['temperature', 'k_B'])
    rate_kernel = arrhenius_kernel

    @classmethod
    def rate(cls, objs, params, config):
        T = get_physics_param(config, 'temperature', default=500.0)
        k_B = 8.617e-5
        E_m = params.get('E_m', 0.5)
        nu = params.get('nu', 1e13)
        return float(nu * np.exp(-E_m / (k_B * T)))

    @classmethod
    def execute(cls, event, objs, config):
        obj = objs[0]
        step_size = config.get('migration_step', 1.0)
        box_size = config.get('domain_decomposition', {}).get('box_size', [100, 100, 100])

        dx = np.random.uniform(-step_size, step_size)
        dy = np.random.uniform(-step_size, step_size)
        dz = np.random.uniform(-step_size, step_size)

        new_x = (float(obj['x']) + dx) % box_size[0]
        new_y = (float(obj['y']) + dy) % box_size[1]
        new_z = (float(obj['z']) + dz) % box_size[2]

        return [{'id': int(obj['id']), 'x': new_x, 'y': new_y, 'z': new_z}]


class CaptureEvent(PairObjectEvent):
    """捕获事件（简化版本）"""
    type_pairs = [('H', 'V')]
    max_distance = 5.0
    params = ['distance']
    product_type = 'HV'
    rate_deps = RateDeps(physics=['k_prefactor'], kind='pair')
    rate_kernel = capture_kernel

    @classmethod
    def rate(cls, objs, params, config):
        distance = params.get('distance', 1.0)
        k = get_physics_param(config, 'k_prefactor', default=1e12)
        return float(k / (distance ** 2 + 0.1))

    @classmethod
    def execute(cls, event, objs, config):
        h_obj = None
        v_obj = None

        for obj in objs:
            obj_type = str(obj['type']).strip()
            if obj_type == 'H':
                h_obj = obj
            elif obj_type == 'V':
                v_obj = obj

        if h_obj is None or v_obj is None:
            h_obj, v_obj = objs[0], objs[1]

        id_gen = config['id_generator']

        new_x = (float(h_obj['x']) + float(v_obj['x'])) / 2
        new_y = (float(h_obj['y']) + float(v_obj['y'])) / 2
        new_z = (float(h_obj['z']) + float(v_obj['z'])) / 2

        return [
            {'id': int(h_obj['id']), 'deleted': True},
            {'id': int(v_obj['id']), 'deleted': True},
            {
                'id': id_gen.generate_id('HV'),
                'type': 'HV',
                'x': new_x,
                'y': new_y,
                'z': new_z,
                'float_props': [1.0],
                'new': True,
            }
        ]


def run_simulation(n_objects: int = 50, max_steps: int = 20):
    """运行模拟并返回结果（使用 JIT 事件构建器）"""
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    # 物理参数（通过 ic_func 注入 UnifiedPhysics）
    def physics_ic(physics, config):
        physics.set_param('temperature', 500.0)
        physics.set_param('k_B', 8.617e-5)
        physics.set_param('k_prefactor', 1e12)

    physics_config = {
        'ic_func': physics_ic,
    }

    objects = {
        'H': {'properties': {'radius': 0.5, 'E_m': 0.3, 'nu': 1e13}},
        'V': {'properties': {'radius': 0.8, 'E_m': 1.0, 'nu': 1e13}},
        'HV': {'properties': {'radius': 1.0, 'E_m': 1.5, 'nu': 1e13}},
    }

    initial_conditions = {
        'H': {'count': n_objects, 'distribution': 'random'},
        'V': {'count': n_objects, 'distribution': 'random'},
    }

    grid_size = (30, 30, 30)
    lc = 1.0

    base_config = {
        'physics': physics_config,
        'objects': objects,
        'domain_decomposition': {
            'lattice_constant': lc,
            'lattice_grid': list(grid_size),
            'h_factor': 1,
            'box_size': [grid_size[0] * lc, grid_size[1] * lc, grid_size[2] * lc],
            'periodic': [True, True, True],
            'max_interaction_distance': 5.0,
        },
        'initial_objects_func': create_initial_objects_func(
            initial_conditions, objects, grid_size, lc),
        'max_steps': max_steps,
        'verbose': False,
        'migration_step': 1.0,
        'rma_object_pool': {'initial_capacity': 500, 'auto_expand': True},
        'rma_event_pool': {
            'capacity_zero': 10, 'capacity_single': 2000,
            'capacity_pair': 2000, 'auto_expand': True,
        },
        'incremental_update': {'enabled': False},
        'predictive_rebalance': {'enabled': False},
        'runtime_load_balancer': {'enabled': False},
    }

    events_config = {
        'migration': MigrationEvent,
        'capture': CaptureEvent,
    }

    config = merge_event_config(base_config, events_config)

    # 运行
    engine = create_okmc_engine_v16(config)

    t0 = time.perf_counter()
    stats = engine.run()
    elapsed = time.perf_counter() - t0

    # 收集结果
    local_objs = engine.rma_object_pool.get_local_objects()
    all_objs_list = comm.gather(local_objs, root=0)

    result = None
    if rank == 0:
        all_objs = np.concatenate([arr for arr in all_objs_list if len(arr) > 0])

        type_counts = {}
        for obj in all_objs:
            t = str(obj['type']).strip()
            type_counts[t] = type_counts.get(t, 0) + 1

        result = {
            'elapsed': elapsed,
            'total_objects': len(all_objs),
            'type_counts': type_counts,
            'stats': stats,
        }

    engine.destroy()
    return result


def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:
        print("=" * 70)
        print("v1.6.9 JIT Event Builder Test (cKDTree + query_pairs)")
        print("=" * 70)
        print(f"MPI Configuration: {size} ranks")
        print()

    # 测试配置
    n_objects = 200
    max_steps = 50

    # 运行测试（JIT 事件构建器）
    if rank == 0:
        print("-" * 70)
        print("[Test] JIT event builder + cKDTree + query_pairs")
        print("-" * 70)

    result = run_simulation(n_objects=n_objects, max_steps=max_steps)

    if rank == 0 and result:
        print(f"  Elapsed: {result['elapsed']*1000:.2f} ms")
        print(f"  Total objects: {result['total_objects']}")
        print(f"  Type counts: {result['type_counts']}")

        h_count = result['type_counts'].get('H', 0)
        v_count = result['type_counts'].get('V', 0)
        hv_count = result['type_counts'].get('HV', 0)

        h_conserved = (h_count + hv_count) == n_objects
        v_conserved = (v_count + hv_count) == n_objects

        print(f"  Conservation H: {h_count} + {hv_count} = {h_count + hv_count} (expected {n_objects}) {'✓' if h_conserved else '✗'}")
        print(f"  Conservation V: {v_count} + {hv_count} = {v_count + hv_count} (expected {n_objects}) {'✓' if v_conserved else '✗'}")

    comm.Barrier()

    # 总结
    if rank == 0:
        print()
        print("=" * 70)
        print("Summary")
        print("=" * 70)

        if result:
            print(f"  Total time: {result['elapsed']*1000:.2f} ms")
            print(f"  Objects: {result['total_objects']}")
            print(f"  Method: cKDTree + query_pairs (O(N log N))")

        print("=" * 70)


if __name__ == "__main__":
    main()
