#!/usr/bin/env python3
"""
Bug 1 测试：RMA 窗口初始化扩容同步

测试场景：
    initial_objects_func 创建的对象数量超过 initial_capacity（默认100）
    触发 _expand_boundary_array()，但窗口仍指向旧缓冲区
    如果不调用 sync_window_rebuild()，rebuild_ghost_cache() 会 Segfault

预期结果：
    修复前：MPI 访问越界 / Segfault / 程序崩溃
    修复后：正常运行，所有测试通过

运行方式：
    mpirun -np 2 python test_bug1_rma_window_expand.py
    mpirun -np 4 python test_bug1_rma_window_expand.py
"""

import numpy as np
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


def many_initial_objects(my_rank, my_size, config):
    """
    初始化大量对象，超过 initial_capacity

    策略：每个 rank 创建 500 个对象
    如果 initial_capacity=100，必然触发多次扩容
    """
    objects = []
    box = config['domain_decomposition']['box_size']

    # 每个 rank 创建 500 个对象（远超 initial_capacity=100）
    n_objects_per_rank = 500

    for i in range(n_objects_per_rank):
        # 对象均匀分布在该 rank 的子域内
        x = (my_rank + np.random.random()) * box[0] / my_size
        y = np.random.random() * box[1]
        z = np.random.random() * box[2]

        objects.append({
            'id': my_rank * 10000 + i,  # 确保 ID 唯一
            'type': 'Particle',
            'x': x, 'y': y, 'z': z,
            'energy': 1.0,
        })

    return objects


def my_calculate_rate(event_type, objs, params, config):
    """简单速率函数"""
    return 1.0


def my_execute_event(event, objs, config):
    """简单执行函数：小位移"""
    obj = objs[0]
    return [{
        'id': int(obj['id']),
        'x': float(obj['x']) + 0.01,
        'y': float(obj['y']),
        'z': float(obj['z']),
    }]


def my_migration_builder(local_objects, config, my_rank):
    """迁移事件构建器"""
    from paraokmc.rma_event_pool import create_event

    event_dtype = config.get('_event_dtype', None)
    events = []

    for i, obj in enumerate(local_objects):
        event = create_event(
            event_id=my_rank * 10000000 + i,
            event_type='migration',
            obj_ids=[int(obj['id'])],
            params=[0.5, 1e12],
            owner_rank=my_rank,
            dtype=event_dtype
        )
        events.append(event)

    return events


# 配置：故意设置小的 initial_capacity 来触发扩容
config = {
    'verbose': False,  # 关闭详细输出，只看结果
    'max_steps': 3,    # 只运行几步验证不崩溃
    'max_time': 100.0,
    'random_seed': 42,

    'domain_decomposition': {
        'nx': size,
        'ny': 1,
        'nz': 1,
        'lattice_constant': 10.0,
        'lattice_grid': [10, 10, 10],
        'h_factor': 1,
        'box_size': [100.0, 100.0, 100.0],
        'periodic': [True, True, True],
        'max_interaction_distance': 5.0,
    },

    'rma_object_pool': {
        'initial_capacity': 100,  # 故意设小，触发扩容
        'auto_expand': True,
        'enable_boundary_separation': True,
        'enable_large_objects': False,
    },

    'rma_event_pool': {
        'capacity_zero': 10,
        'capacity_single': 1000,
        'capacity_pair': 100,
    },

    'incremental_update': {'enabled': False},
    'predictive_rebalance': {'enabled': False},
    'runtime_load_balancer': {'enabled': False},
    'adaptive_weight': {'enabled': False},

    # 用户函数
    'initial_objects_func': many_initial_objects,
    'calculate_rate_func': my_calculate_rate,
    'execute_event_func': my_execute_event,

    # 事件构建器
    'single_obj_builders': {
        'migration': my_migration_builder,
    },

    # 对象类型定义
    'objects': {
        'Particle': {
            'float_props': ['energy'],
        },
    },

    # 事件类型定义
    'events': {
        'migration': {
            'category': 'single',
            'object_types': ['Particle'],
            'params': ['E_m', 'nu'],
        },
    },
}


def main():
    from paraokmc.v16_engine import OKMCEngineV16

    if rank == 0:
        print("=" * 60)
        print("Bug 1 测试：RMA 窗口初始化扩容同步")
        print("=" * 60)
        print(f"MPI ranks: {size}")
        print(f"每 rank 对象数: 500")
        print(f"initial_capacity: 100 (会触发多次扩容)")
        print()

    # 这里如果 Bug 1 未修复，程序会在 engine.run() 中崩溃
    # 因为 rebuild_ghost_cache() 会使用未重建的旧窗口
    try:
        engine = OKMCEngineV16(config)
        stats = engine.run()

        # 收集所有 rank 的对象数
        local_count = engine.rma_object_pool.get_local_count()
        all_counts = comm.gather(local_count, root=0)

        if rank == 0:
            print()
            print("=" * 60)
            print("测试结果")
            print("=" * 60)
            print(f"总步数: {stats['total_steps']}")
            print(f"执行事件数: {stats['total_events']}")
            print(f"各 rank 对象数: {all_counts}")
            print(f"总对象数: {sum(all_counts)}")
            print()
            print("✓ 程序正常完成，没有崩溃")
            print("✓ Bug 1 修复验证: PASS")
            print("=" * 60)

    except Exception as e:
        if rank == 0:
            print()
            print("=" * 60)
            print("测试失败！")
            print("=" * 60)
            print(f"异常: {type(e).__name__}: {e}")
            print()
            print("✗ Bug 1 修复验证: FAIL")
            print("=" * 60)
        raise

    comm.Barrier()


if __name__ == '__main__':
    main()
