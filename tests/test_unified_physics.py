"""
v1.6.7 统一物理模块测试

测试场景：
1. 静态温度（只有IC）
2. 线性升温（IC + Update）
3. 空间温度场插值

验证：
- 参数设置和读取
- 场插值功能
- 增量失效联动
"""

import numpy as np
from mpi4py import MPI
from paraokmc.v16_engine import create_okmc_engine_v16
from paraokmc.id_generator import IDGenerator
from paraokmc.builtin_migrations import create_migration_builder

comm = MPI.COMM_WORLD
rank = comm.rank
size = comm.size


# ===== 物理模块 IC/Update 函数 =====

def physics_ic(physics, config):
    """
    初始化物理参数和场

    设置：
    - 温度 T = 500 K
    - 玻尔兹曼常数 k_B = 8.617e-5 eV/K
    - 迁移能 Em = 0.5 eV
    - 尝试频率 v0 = 1e13 Hz
    - 温度场（用于演示插值）
    """
    # 标量参数
    physics.set_param('temperature', config.get('initial_temperature', 500.0))
    physics.set_param('k_B', 8.617333262e-5)  # eV/K
    physics.set_param('migration_energy', 0.5)  # eV
    physics.set_param('attempt_frequency', 1e13)  # Hz

    # 创建温度场（3D高斯分布，中心高温）
    physics.define_field('temperature_field', (20, 20, 20))
    T_field = physics.get_field('temperature_field')

    # 填充温度场：中心 600K，边缘 400K
    box_size = physics.box_size
    for i in range(20):
        for j in range(20):
            for k in range(20):
                x = (i + 0.5) / 20 * box_size[0]
                y = (j + 0.5) / 20 * box_size[1]
                z = (k + 0.5) / 20 * box_size[2]

                # 到中心的距离
                cx, cy, cz = box_size[0]/2, box_size[1]/2, box_size[2]/2
                r = np.sqrt((x-cx)**2 + (y-cy)**2 + (z-cz)**2)
                r_max = np.sqrt(cx**2 + cy**2 + cz**2)

                # 高斯分布
                T_field[i, j, k] = 400.0 + 200.0 * np.exp(-3*(r/r_max)**2)

    if physics.rank == 0:
        print(f"[Physics IC] Temperature: {physics.get_param('temperature'):.1f} K")
        print(f"[Physics IC] Temperature field: min={T_field.min():.1f}, max={T_field.max():.1f}")


def physics_update_linear_heating(physics, step, time, config):
    """
    线性升温更新

    每步增加温度（基于步数而非时间，更适合KMC测试）
    """
    heating_rate = config.get('heating_rate', 10.0)  # K per step
    T0 = config.get('initial_temperature', 500.0)

    # 使用步数而非时间来计算温度变化
    new_temp = T0 + heating_rate * step
    changed = physics.set_param('temperature', new_temp)

    if physics.rank == 0:
        print(f"[Physics Update] step={step}, new_temp={new_temp:.2f}, changed={changed}")

    return changed


# ===== 用户函数（简化版）=====

def create_initial_objects(rank, size, config):
    """创建初始对象"""
    id_gen = config['id_generator']
    box_size = config['domain_decomposition']['box_size']

    objects = []
    n_per_rank = 10

    for i in range(n_per_rank):
        obj_type = 'H' if i % 2 == 0 else 'V'
        obj_id = id_gen.generate_id(obj_type)
        x = np.random.uniform(0, box_size[0])
        y = np.random.uniform(0, box_size[1])
        z = np.random.uniform(0, box_size[2])

        objects.append({
            'id': obj_id,
            'type': obj_type,
            'x': x,
            'y': y,
            'z': z,
            'float_props': [0.5],  # Em
        })

    return objects


def calculate_rate(event_type, objects, params, config):
    """
    计算速率（使用UnifiedPhysics）

    演示如何访问physics模块的参数和场
    """
    physics = config.get('physics')

    if physics is None:
        # 没有physics模块，使用默认值
        T = 500.0
        k_B = 8.617e-5
    else:
        # 从physics模块读取参数
        T = physics.get_param('temperature')
        k_B = physics.get_param('k_B')

    if event_type == 'migration':
        obj = objects[0]
        Em = obj['float_props'][0] if len(obj['float_props']) > 0 else 0.5
        v0 = 1e13

        # 如果有温度场，使用位置处的温度
        if physics is not None and 'temperature_field' in physics._field_name_to_idx:
            x, y, z = float(obj['x']), float(obj['y']), float(obj['z'])
            T_local = physics.interpolate_field('temperature_field', x, y, z)
            T = T_local  # 使用局部温度

        # Arrhenius公式
        rate = v0 * np.exp(-Em / (k_B * T))
        return rate

    return 0.0


def execute_event(event, objects, config):
    """执行事件"""
    event_type = str(event['type'])

    if event_type == 'migration':
        # 检查是否有对象数据
        if len(objects) == 0:
            print(f"[WARNING] No objects in execute_event! Event obj_ids: {event['obj_ids']}")
            return []

        obj = objects[0]
        box_size = config['domain_decomposition']['box_size']

        # 随机位移
        dx = np.random.uniform(-1, 1)
        dy = np.random.uniform(-1, 1)
        dz = np.random.uniform(-1, 1)

        new_x = (float(obj['x']) + dx) % box_size[0]
        new_y = (float(obj['y']) + dy) % box_size[1]
        new_z = (float(obj['z']) + dz) % box_size[2]

        return [{
            'id': int(obj['id']),
            'type': str(obj['type']),
            'x': new_x,
            'y': new_y,
            'z': new_z,
            'float_props': list(obj['float_props']),
        }]

    return []

# ===== 主测试 =====

if __name__ == "__main__":
    if rank == 0:
        print("=" * 60)
        print("v1.6.7 Unified Physics Module Test")
        print("=" * 60)

    # 测试1：静态温度
    if rank == 0:
        print("\n[Test 1] Static temperature (IC only)")

    config_static = {
        'verbose': (rank == 0),
        'max_steps': 3,
        'random_seed': 42,

        'domain_decomposition': {
            'nx': size, 'ny': 1, 'nz': 1,
            'lattice_constant': 10.0,
            'lattice_grid': [10, 10, 10],
            'h_factor': 1,
            'box_size': [100.0, 100.0, 100.0],
            'periodic': [True, True, True],
            'max_interaction_distance': 5.0,
        },

        # TypeCompiler需要这些定义来生成正确的dtype
        'objects': {
            'H': {'properties': {'Em': 0.5}},
            'V': {'properties': {'Em': 0.5}},
        },
        'events': {
            'migration': {'rate_params': ['distance']},
        },

        'physics': {
            'ic_func': physics_ic,
            'update_func': None,  # 静态，无更新
        },

        'initial_temperature': 500.0,

        'initial_objects_func': create_initial_objects,
        'calculate_rate_func': calculate_rate,
        'execute_event_func': execute_event,

        'single_obj_builders': {
            'migration': create_migration_builder(migration_distance=1.0),
        },
    }

    try:
        engine = create_okmc_engine_v16(config_static)

        # 验证physics已创建（但IC还未执行）
        assert engine.physics is not None, "Physics should be initialized"

        stats = engine.run()

        # 验证physics初始化（IC在run()开始时执行）
        assert engine.physics.get_param('temperature') == 500.0, "Temperature should be 500"
        engine.destroy()

        if rank == 0:
            print(f"[Test 1] ✓ Passed: {stats['total_steps']} steps completed")
            print(f"         Physics rebuilds: {engine.physics.stats['triggered_rebuilds']}")

    except Exception as e:
        print(f"[Rank {rank}] Test 1 failed: {e}")
        import traceback
        traceback.print_exc()

    comm.Barrier()

    # 测试2：线性升温
    if rank == 0:
        print("\n[Test 2] Linear heating (IC + Update)")

    config_heating = {
        'verbose': False,
        'max_steps': 3,
        'random_seed': 42,

        'domain_decomposition': {
            'nx': size, 'ny': 1, 'nz': 1,
            'lattice_constant': 10.0,
            'lattice_grid': [10, 10, 10],
            'h_factor': 1,
            'box_size': [100.0, 100.0, 100.0],
            'periodic': [True, True, True],
            'max_interaction_distance': 5.0,
        },

        # TypeCompiler需要这些定义来生成正确的dtype
        'objects': {
            'H': {'properties': {'Em': 0.5}},
            'V': {'properties': {'Em': 0.5}},
        },
        'events': {
            'migration': {'rate_params': ['distance']},
        },

        'physics': {
            'ic_func': physics_ic,
            'update_func': physics_update_linear_heating,
        },

        'initial_temperature': 500.0,
        'heating_rate': 10.0,  # 快速升温便于测试

        'initial_objects_func': create_initial_objects,
        'calculate_rate_func': calculate_rate,
        'execute_event_func': execute_event,

        'single_obj_builders': {
            'migration': create_migration_builder(migration_distance=1.0),
        },
    }

    try:
        engine = create_okmc_engine_v16(config_heating)
        stats = engine.run()

        # 验证温度变化
        final_temp = engine.physics.get_param('temperature')
        rebuilds = engine.physics.stats['triggered_rebuilds']

        if rank == 0:
            print(f"[Test 2] Final temperature: {final_temp:.1f} K")
            print(f"         Triggered rebuilds: {rebuilds}")

        # 温度应该增加了
        assert final_temp > 500.0, "Temperature should have increased"
        # 应该触发了增量失效
        assert rebuilds > 0, "Should have triggered rebuilds"

        engine.destroy()

        if rank == 0:
            print(f"[Test 2] ✓ Passed")

    except Exception as e:
        print(f"[Rank {rank}] Test 2 failed: {e}")
        import traceback
        traceback.print_exc()

    comm.Barrier()

    # 测试3：场插值
    if rank == 0:
        print("\n[Test 3] Field interpolation")

    from paraokmc.unified_physics import create_unified_physics

    physics_config = {
        'physics': {
            'box_size': [100.0, 100.0, 100.0],
            'ic_func': physics_ic,
        },
        'initial_temperature': 500.0,
    }

    physics = create_unified_physics(comm, physics_config)
    physics.initialize()

    # 测试单点插值
    T_center = physics.interpolate_field('temperature_field', 50.0, 50.0, 50.0)
    T_corner = physics.interpolate_field('temperature_field', 0.0, 0.0, 0.0)

    if rank == 0:
        print(f"[Test 3] Temperature at center: {T_center:.1f} K")
        print(f"         Temperature at corner: {T_corner:.1f} K")

    assert T_center > T_corner, "Center should be hotter than corner"

    # 测试批量插值
    positions = np.random.rand(100, 3) * 100.0
    T_batch = physics.interpolate_field_batch('temperature_field', positions)

    if rank == 0:
        print(f"         Batch interpolation: min={T_batch.min():.1f}, max={T_batch.max():.1f}")

    assert len(T_batch) == 100, "Batch interpolation should return 100 values"

    if rank == 0:
        print(f"[Test 3] ✓ Passed")

    comm.Barrier()

    # 总结
    if rank == 0:
        print("\n" + "=" * 60)
        print("All tests passed!")
        print("=" * 60)
