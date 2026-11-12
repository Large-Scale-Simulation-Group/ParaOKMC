#!/usr/bin/env python3
"""
v1.6.10 测试：事件执行中修改物理模块

测试目标：
1. 验证 execute_func 中可以调用 physics.set_param_in_event()
2. 验证 execute_func 中可以调用 physics.add_field_delta_in_event()
3. 验证所有 rank 在广播后统一应用相同的物理变更

运行方式：
    mpirun -np 2 python test_physics_in_event.py
    mpirun -np 4 python test_physics_in_event.py
"""

import numpy as np
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# ============================================================================
# 测试配置
# ============================================================================

def my_initial_objects(my_rank, my_size, config):
    """初始化对象：每个rank生成一个测试对象"""
    objects = []
    box = config['domain_decomposition']['box_size']

    # 每个rank生成一个对象（位于rank对应的子域）
    x = (my_rank + 0.5) * box[0] / my_size
    y = box[1] / 2
    z = box[2] / 2

    objects.append({
        'id': my_rank,
        'type': 'TestParticle',
        'x': x, 'y': y, 'z': z,
        'energy': 1.0,  # 用于触发事件
    })

    return objects


def my_calculate_rate(event_type, objs, params, config):
    """计算速率：简单返回固定速率"""
    return 1.0


def my_execute_event(event, objs, config):
    """
    执行事件：测试在事件中修改物理模块

    这是测试的核心！
    我们在这里调用 physics.set_param_in_event() 和 physics.add_field_delta_in_event()
    这些变更会被收集并随 changed_objects 一起广播给所有 rank
    """
    physics = config.get('physics')
    obj = objs[0]
    obj_id = int(obj['id'])
    x, y, z = float(obj['x']), float(obj['y']), float(obj['z'])

    # 测试1：修改全局参数
    # 每执行一次事件，counter 增加 1
    if physics is not None:
        try:
            current = physics.get_param('counter')
        except KeyError:
            current = 0.0
        physics.set_param_in_event('counter', current + 1.0)

        # 测试2：添加场增量
        # 在对象位置产生一个局部温度spike
        # 使用点模式（radius=0），只影响最近邻网格点
        physics.add_field_delta_in_event(
            'temperature',
            position=(x, y, z),
            delta=10.0,  # 增加10K
            radius=0.0   # 点模式
        )

    # 简单位移对象（测试正常的 changed_objects 机制仍然工作）
    new_x = x + 0.1
    if new_x > config['domain_decomposition']['box_size'][0]:
        new_x = 0.1

    return [{
        'id': obj_id,
        'x': new_x,
        'y': y,
        'z': z,
    }]


def my_migration_builder(local_objects, config, my_rank):
    """
    迁移事件构建器

    参数：
        local_objects: 本地对象数组（NumPy结构化数组）
        config: 配置字典
        my_rank: 当前rank

    返回：
        事件列表，每个事件是NumPy结构化数组（使用create_event创建）
    """
    from paraokmc.rma_event_pool import create_event

    # 获取事件权重
    event_weights = config.get('event_weights', {})
    weight = event_weights.get('migration', 1.0)

    # 从config获取event_dtype
    event_dtype = config.get('_event_dtype', None)

    events = []
    for i, obj in enumerate(local_objects):
        obj_type = str(obj['type']).strip()
        if obj_type == 'TestParticle':
            event = create_event(
                event_id=my_rank * 10000000 + i,
                event_type='migration',
                obj_ids=[int(obj['id'])],
                params=[0.5, 1e12],  # [E_m, nu]
                owner_rank=my_rank,
                weight=weight,
                dtype=event_dtype
            )
            events.append(event)
    return events


def my_physics_ic(physics, config):
    """物理模块初始化函数：定义参数和场"""
    # 定义参数
    physics.define_param('counter', 0.0)
    physics.define_param('temperature_base', 300.0)

    # 定义温度场（10x10x10网格）
    physics.define_field('temperature', (10, 10, 10), initial_value=300.0)


# ============================================================================
# 配置
# ============================================================================

config = {
    'verbose': True,
    'max_steps': 5,  # 只运行5步，足够测试
    'max_time': 100.0,
    'random_seed': 42,

    'domain_decomposition': {
        'nx': size,
        'ny': 1,
        'nz': 1,
        'lattice_constant': 10.0,
        'lattice_grid': [1, 1, 1],
        'h_factor': 1,
        'box_size': [10.0, 10.0, 10.0],
        'periodic': [True, True, True],
        'max_interaction_distance': 2.0,
    },

    'rma_object_pool': {
        'initial_capacity': 100,
        'auto_expand': True,
        'enable_boundary_separation': False,
        'enable_large_objects': False,
    },

    'rma_event_pool': {
        'capacity_zero': 10,
        'capacity_single': 100,
        'capacity_pair': 100,
    },

    'incremental_update': {'enabled': False},
    'predictive_rebalance': {'enabled': False},
    'runtime_load_balancer': {'enabled': False},
    'adaptive_weight': {'enabled': False},

    # 用户函数
    'initial_objects_func': my_initial_objects,
    'calculate_rate_func': my_calculate_rate,
    'execute_event_func': my_execute_event,

    # 事件构建器
    'single_obj_builders': {
        'migration': my_migration_builder,
    },

    # 物理模块配置（引擎会创建UnifiedPhysics实例）
    'physics': {
        'ic_func': my_physics_ic,  # 初始化函数
    },

    # 对象类型定义（用于TypeCompiler）
    'objects': {
        'TestParticle': {
            'float_props': ['energy'],
        },
    },

    # 事件类型定义
    'events': {
        'migration': {
            'category': 'single',
            'object_types': ['TestParticle'],
            'params': ['E_m', 'nu'],
        },
    },
}


# ============================================================================
# 主测试
# ============================================================================

def main():
    from paraokmc.v16_engine import OKMCEngineV16

    if rank == 0:
        print("=" * 60)
        print("v1.6.10 测试：事件执行中修改物理模块")
        print("=" * 60)
        print()

    # 创建并运行引擎
    engine = OKMCEngineV16(config)

    # 引擎创建后，physics已经被注入到config中
    physics = config.get('physics')

    # 记录初始状态（在run()之前，physics.initialize()已在引擎构造时调用）
    # 但我们需要在run()后检查，因为ic_func在run()的Step 0时调用
    # 所以这里先记录"预期"的初始值
    expected_initial_counter = 0.0
    expected_initial_max_temp = 300.0

    if rank == 0:
        print(f"预期初始 counter = {expected_initial_counter}")
        print(f"预期初始温度场最大值 = {expected_initial_max_temp}")
        print()

    # 运行引擎
    stats = engine.run()

    # 获取physics对象（从config中）
    physics = engine.config.get('physics')
    if physics is None:
        if rank == 0:
            print("ERROR: physics object not found in config!")
        return

    # 验证结果
    final_counter = physics.get_param('counter')
    final_temp_field = physics.get_field('temperature')
    final_max_temp = np.max(final_temp_field)
    temp_changed = final_max_temp > expected_initial_max_temp

    if rank == 0:
        print()
        print("=" * 60)
        print("测试结果")
        print("=" * 60)
        print(f"完成步数: {stats['total_steps']}")
        print(f"执行事件数: {stats['total_events']}")
        print()
        print(f"最终 counter = {final_counter}")
        print(f"预期 counter = {stats['total_events']}.0")
        print(f"counter 测试: {'PASS' if final_counter == stats['total_events'] else 'FAIL'}")
        print()
        print(f"最终温度场最大值 = {final_max_temp}")
        print(f"温度场是否变化: {temp_changed}")
        print(f"温度场测试: {'PASS' if temp_changed else 'FAIL'}")
        print()

    # 验证所有rank的physics状态一致
    all_counters = comm.gather(final_counter, root=0)
    all_max_temps = comm.gather(final_max_temp, root=0)

    if rank == 0:
        counters_consistent = len(set(all_counters)) == 1
        temps_consistent = len(set(all_max_temps)) == 1

        print(f"各rank counter值: {all_counters}")
        print(f"各rank counter一致性: {'PASS' if counters_consistent else 'FAIL'}")
        print()
        print(f"各rank最大温度: {all_max_temps}")
        print(f"各rank温度场一致性: {'PASS' if temps_consistent else 'FAIL'}")
        print()

        # 综合判断
        all_pass = (
            final_counter == stats['total_events'] and
            temp_changed and
            counters_consistent and
            temps_consistent
        )
        print("=" * 60)
        print(f"综合测试结果: {'ALL PASS' if all_pass else 'SOME FAILED'}")
        print("=" * 60)

    comm.Barrier()


if __name__ == '__main__':
    main()
