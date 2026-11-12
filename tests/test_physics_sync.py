"""
v1.6.10 Physics Sync Test

测试目标：
1. 验证事件执行期间的 physics 参数修改能正确广播到所有 rank
2. 验证事件执行期间的 physics 场增量能正确广播和应用
3. 验证所有 rank 的 physics 状态保持同步

测试方法：
- 定义一个特殊事件，在执行时修改 physics 参数和场
- 运行多步模拟，验证所有 rank 的 physics 状态一致
"""

import numpy as np
from mpi4py import MPI
import sys
from numba import njit
from paraokmc.jit_rate_engine import RateDeps

# 导入事件类系统
from paraokmc.event_class import SingleObjectEvent, get_physics_param
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


# ============================================================================
# 事件类定义
# ============================================================================

class MigrationEvent(SingleObjectEvent):
    """普通迁移事件"""
    object_types = ['A']
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
        step_size = 1.0
        box_size = config.get('domain_decomposition', {}).get('box_size', [100, 100, 100])

        dx = np.random.uniform(-step_size, step_size)
        dy = np.random.uniform(-step_size, step_size)
        dz = np.random.uniform(-step_size, step_size)

        new_x = (float(obj['x']) + dx) % box_size[0]
        new_y = (float(obj['y']) + dy) % box_size[1]
        new_z = (float(obj['z']) + dz) % box_size[2]

        return [{'id': int(obj['id']), 'x': new_x, 'y': new_y, 'z': new_z}]


class PhysicsModifyEvent(SingleObjectEvent):
    """
    特殊事件：在执行时修改 physics 参数和场

    用于测试 v1.6.10 的 physics 同步机制
    """
    object_types = ['B']
    params = ['rate_constant']

    @classmethod
    def rate(cls, objs, params, config):
        """使用固定速率 - 高速率确保被选中"""
        rate_constant = params.get('rate_constant', 1.0)
        return float(rate_constant * 1e10)

    @classmethod
    def execute(cls, event, objs, config):
        """执行时修改 physics 参数和场"""
        obj = objs[0]
        physics = config.get('physics')

        if physics is not None:
            # 1. 修改参数：每次执行增加 temperature 10K
            current_T = physics.get_param('temperature')
            physics.set_param_in_event('temperature', current_T + 10.0)

            # 2. 修改场：在对象位置添加 +1.0 的增量
            x, y, z = float(obj['x']), float(obj['y']), float(obj['z'])
            if 'test_field' in physics._field_name_to_idx:
                physics.add_field_delta_in_event('test_field', (x, y, z), 1.0, radius=0.0)

        # 返回空变更（对象不变）
        return []


def run_test(n_steps: int = 10):
    """运行测试"""
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:
        print("=" * 70)
        print("v1.6.10 Physics Sync Test")
        print("=" * 70)
        print(f"MPI Configuration: {size} ranks")
        print()

    # 配置
    # 注意：physics 配置会传给 UnifiedPhysics 的构造函数
    # 需要在 param_names 中声明参数，并通过 ic_func 设置初始值
    def physics_ic_func(physics, config):
        """初始化物理参数"""
        physics.set_param('temperature', 500.0)
        physics.set_param('k_B', 8.617e-5)

    physics_config = {
        'param_names': ['temperature', 'k_B'],
        'field_shapes': [(10, 10, 10)],
        'field_names': ['test_field'],
        'ic_func': physics_ic_func,  # 添加初始化函数
    }

    objects = {
        'A': {'properties': {'radius': 0.5}},
        'B': {'properties': {'rate_const': 1.0}},  # 特殊对象，用于触发 physics 修改
    }

    events = {
        'A_migration': {
            'objects': ['A'],
            'rate_params': ['E_m', 'nu'],
            'E_m': 0.3,
            'nu': 1e13
        },
        'B_modify_physics': {
            'objects': ['B'],
            'rate_params': ['rate_constant'],
            'rate_constant': 100.0  # 增大速率常数，确保 B 事件更容易被选中
        },
    }

    initial_conditions = {
        'A': {'count': 10, 'distribution': 'random'},
        'B': {'count': 2, 'distribution': 'random'},
    }

    grid_size = (20, 20, 20)
    lc = 1.0

    # 编译配置
    base_config = {
        'objects': objects,
        'events': {
            'A_migration': {'objects': ['A'], 'rate_params': ['E_m', 'nu'],
                            'E_m': 0.3, 'nu': 1e13},
            'B_modify_physics': {'objects': ['B'], 'rate_params': ['rate_constant'],
                                 'rate_constant': 100.0},
        },
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
        'physics': physics_config,
        'max_steps': n_steps,
        'verbose': True,
        'rma_object_pool': {'initial_capacity': 100, 'auto_expand': True},
        'rma_event_pool': {
            'capacity_zero': 10, 'capacity_single': 500,
            'capacity_pair': 100, 'auto_expand': True,
        },
        'incremental_update': {'enabled': False},
        'predictive_rebalance': {'enabled': False},
        'runtime_load_balancer': {'enabled': False},
    }

    # 合并事件类
    events_config = {
        'A_migration': MigrationEvent,
        'B_modify_physics': PhysicsModifyEvent,
    }
    config = merge_event_config(base_config, events_config)

    # 创建引擎
    engine = create_okmc_engine_v16(config)

    # 记录初始 physics 状态（在 physics.initialize() 之后）
    # 注意：engine.run() 内部会调用 physics.initialize()，所以这里读取的是未初始化的值
    # 我们需要在 run() 之后验证最终状态
    initial_T_expected = 500.0  # 期望的初始温度

    if rank == 0:
        print(f"Expected initial temperature: {initial_T_expected}")
        print()
        print("-" * 70)
        print("Running simulation...")
        print("-" * 70)

    # 运行模拟
    stats = engine.run()

    # 收集最终 physics 状态
    final_T = engine.physics.get_param('temperature') if engine.physics else None
    final_field = engine.physics.get_field('test_field') if engine.physics else None
    field_sum = np.sum(final_field) if final_field is not None else 0.0

    # 验证所有 rank 的 physics 状态一致
    all_temps = comm.gather(final_T, root=0)
    all_field_sums = comm.gather(field_sum, root=0)

    comm.Barrier()

    # 输出结果
    if rank == 0:
        print()
        print("-" * 70)
        print("Results")
        print("-" * 70)

        print(f"\nTemperature across ranks: {all_temps}")
        print(f"Field sum across ranks: {all_field_sums}")

        # 验证一致性
        temp_consistent = len(set(all_temps)) == 1
        field_consistent = len(set([round(s, 6) for s in all_field_sums])) == 1

        print()
        print("-" * 70)
        print("Verification")
        print("-" * 70)

        print(f"Temperature consistent across all ranks: {'✓' if temp_consistent else '✗'}")
        print(f"Field sum consistent across all ranks: {'✓' if field_consistent else '✗'}")

        if temp_consistent:
            if final_T is not None:
                delta_T = final_T - initial_T_expected
                print(f"Temperature change: {initial_T_expected} → {final_T} (Δ = {delta_T})")
            else:
                print(f"Warning: final_T is None (physics module not initialized)")
            # 每次 PhysicsModifyEvent 执行增加 10K
            # 但不是每步都执行 PhysicsModifyEvent

        if field_consistent:
            print(f"Field total sum: {field_sum:.2f}")

        # 总体测试结果
        all_passed = temp_consistent and field_consistent
        print()
        print("=" * 70)
        print(f"Overall test result: {'PASSED ✓' if all_passed else 'FAILED ✗'}")
        print("=" * 70)

    engine.destroy()

    return rank == 0 and temp_consistent and field_consistent


def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    try:
        success = run_test(n_steps=20)
        if rank == 0:
            sys.exit(0 if success else 1)
    except Exception as e:
        print(f"[Rank {rank}] Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
