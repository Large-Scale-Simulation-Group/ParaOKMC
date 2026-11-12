"""
v1.6.8 历史存储功能测试

测试场景：
1. 启用历史存储
2. 保存和获取上一步参数/场
3. 计算参数/场的变化率
"""

import numpy as np
from mpi4py import MPI
from paraokmc.unified_physics import UnifiedPhysics, MODE_TWO_WAY


def test_history_disabled():
    """测试历史存储禁用时的行为"""
    comm = MPI.COMM_WORLD
    rank = comm.rank

    config = {
        'box_size': [50.0, 50.0, 50.0],
        'field_coupling': {
            'enable_history': False,  # 禁用历史
        },
    }

    physics = UnifiedPhysics(comm, config)
    physics.set_param('temperature', 500.0)
    physics.define_field('temperature_field', (10, 10, 10))

    # 禁用时，save_history不应该做任何事
    physics.save_history(0.0)
    physics.save_history(1.0)

    # 所有获取历史的方法应该返回None
    assert physics.get_prev_param('temperature') is None
    assert physics.get_param_rate('temperature') is None
    assert physics.get_prev_field('temperature_field') is None
    assert physics.get_field_rate('temperature_field') is None
    assert physics.get_time_delta() == 0.0

    if rank == 0:
        print("[Test 1] History disabled: PASSED")

    return True


def test_param_history():
    """测试参数历史存储"""
    comm = MPI.COMM_WORLD
    rank = comm.rank

    config = {
        'box_size': [50.0, 50.0, 50.0],
        'field_coupling': {
            'enable_history': True,  # 启用历史
        },
    }

    physics = UnifiedPhysics(comm, config)
    physics.set_param('temperature', 500.0)

    # 第一次保存历史（基线）
    physics.save_history(0.0)
    assert physics._history_initialized

    # 修改参数
    physics.set_param('temperature', 600.0)

    # 使用update_time更新当前时间
    physics.update_time(1.0)

    # 验证历史数据
    assert physics.get_prev_param('temperature') == 500.0
    assert physics.get_param('temperature') == 600.0
    assert physics.get_time_delta() == 1.0

    # 验证变化率
    rate = physics.get_param_rate('temperature')
    assert rate is not None
    assert abs(rate - 100.0) < 1e-10  # (600-500) / 1.0 = 100

    if rank == 0:
        print("[Test 2] Parameter history: PASSED")

    return True


def test_field_history():
    """测试场历史存储"""
    comm = MPI.COMM_WORLD
    rank = comm.rank

    config = {
        'box_size': [50.0, 50.0, 50.0],
        'field_coupling': {
            'enable_history': True,
        },
    }

    physics = UnifiedPhysics(comm, config)
    physics.define_field('temperature', (5, 5, 5), initial_value=300.0)

    # 保存历史（基线）
    physics.save_history(0.0)

    # 修改场
    new_field = np.full((5, 5, 5), 400.0)
    physics.set_field('temperature', new_field)

    # 使用update_time更新时间
    physics.update_time(2.0)  # dt = 2.0

    # 验证历史数据
    prev_field = physics.get_prev_field('temperature')
    assert prev_field is not None
    assert prev_field[0, 0, 0] == 300.0

    current_field = physics.get_field('temperature')
    assert current_field[0, 0, 0] == 400.0

    # 验证变化率
    rate_field = physics.get_field_rate('temperature')
    assert rate_field is not None
    # (400 - 300) / 2.0 = 50
    assert abs(rate_field[0, 0, 0] - 50.0) < 1e-10

    if rank == 0:
        print("[Test 3] Field history: PASSED")

    return True


def test_prev_field_interpolator():
    """测试上一步场的插值函数"""
    comm = MPI.COMM_WORLD
    rank = comm.rank

    config = {
        'box_size': [100.0, 100.0, 100.0],
        'field_coupling': {
            'enable_history': True,
        },
    }

    physics = UnifiedPhysics(comm, config)
    physics.define_field('stress', (10, 10, 10), initial_value=0.0)

    # 设置初始场为线性分布
    field = physics.get_field('stress')
    x = np.linspace(0, 100, 10)
    for i in range(10):
        field[i, :, :] = x[i]  # 沿x方向线性
    physics.set_field('stress', field)

    # 保存历史
    physics.save_history(0.0)

    # 修改场
    new_field = np.full((10, 10, 10), 50.0)  # 均匀场
    physics.set_field('stress', new_field)
    physics.save_history(1.0)

    # 获取上一步的插值函数
    prev_interp = physics.get_prev_field_interpolator('stress')
    assert prev_interp is not None

    # 测试插值（上一步是线性分布）
    val_at_50 = prev_interp(50.0, 50.0, 50.0)
    assert abs(val_at_50 - 50.0) < 1.0  # 应该接近50

    # 当前场是均匀的
    current_interp = physics.get_field_interpolator('stress')
    val_current = current_interp(25.0, 25.0, 25.0)
    assert abs(val_current - 50.0) < 1e-10

    if rank == 0:
        print("[Test 4] Previous field interpolator: PASSED")

    return True


def test_time_delta_zero():
    """测试dt=0时的行为"""
    comm = MPI.COMM_WORLD
    rank = comm.rank

    config = {
        'box_size': [50.0, 50.0, 50.0],
        'field_coupling': {
            'enable_history': True,
        },
    }

    physics = UnifiedPhysics(comm, config)
    physics.set_param('temperature', 500.0)
    physics.define_field('temperature_field', (5, 5, 5))

    # 两次相同时间保存
    physics.save_history(1.0)
    physics.set_param('temperature', 600.0)
    physics.save_history(1.0)  # 相同时间

    # dt = 0，变化率应该返回None
    assert physics.get_time_delta() == 0.0
    assert physics.get_param_rate('temperature') is None
    assert physics.get_field_rate('temperature_field') is None

    if rank == 0:
        print("[Test 5] Zero time delta: PASSED")

    return True


# ===== 主测试 =====

if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    rank = comm.rank

    if rank == 0:
        print("=" * 60)
        print("v1.6.8 History Storage Tests")
        print("=" * 60)

    tests = [
        test_history_disabled,
        test_param_history,
        test_field_history,
        test_prev_field_interpolator,
        test_time_delta_zero,
    ]

    all_passed = True
    for test in tests:
        try:
            test()
        except AssertionError as e:
            if rank == 0:
                print(f"FAILED: {test.__name__}")
                print(f"  Error: {e}")
            all_passed = False
        except Exception as e:
            if rank == 0:
                print(f"ERROR: {test.__name__}")
                print(f"  {type(e).__name__}: {e}")
            all_passed = False

    if rank == 0:
        print("=" * 60)
        if all_passed:
            print("All history tests PASSED!")
        else:
            print("Some tests FAILED!")
        print("=" * 60)
