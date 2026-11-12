"""
v1.6.7 自适应网格场测试

测试非均匀直角网格(Non-uniform Rectilinear Grid)功能：
1. define_adaptive_field() 基本功能
2. 与 define_field() 的兼容性
3. 非均匀网格插值正确性
4. MPI同步正确性
5. 内存优化效果验证
"""

import numpy as np
from mpi4py import MPI
from paraokmc.unified_physics import UnifiedPhysics, create_unified_physics


def test_adaptive_field_basic():
    """测试1：基本功能"""
    comm = MPI.COMM_WORLD
    rank = comm.rank

    if rank == 0:
        print("\n[Test 1] Adaptive field basic functionality")
        print("-" * 50)

    config = {
        'physics': {
            'box_size': [100.0, 100.0, 100.0],
        }
    }

    physics = create_unified_physics(comm, config)

    # 定义非均匀X轴：在x=50处加密
    x_coarse_1 = np.arange(0, 40, 10.0)      # [0, 10, 20, 30]
    x_fine = np.arange(40, 60, 2.0)           # [40, 42, 44, ..., 58] - 10 points
    x_coarse_2 = np.arange(60, 101, 10.0)     # [60, 70, 80, 90, 100]

    x_coords = np.unique(np.concatenate([x_coarse_1, x_fine, x_coarse_2]))

    # Y和Z轴保持均匀
    y_coords = np.linspace(0, 100, 20)
    z_coords = np.linspace(0, 100, 20)

    # 定义自适应场
    idx = physics.define_adaptive_field('stress', x_coords, y_coords, z_coords)

    if rank == 0:
        print(f"  X axis points: {len(x_coords)} (non-uniform)")
        print(f"  Y axis points: {len(y_coords)} (uniform)")
        print(f"  Z axis points: {len(z_coords)} (uniform)")
        print(f"  Total grid points: {len(x_coords) * len(y_coords) * len(z_coords)}")
        print(f"  Equivalent uniform grid: 100 x 20 x 20 = 40000")

    # 验证形状
    field = physics.get_field('stress')
    assert field.shape == (len(x_coords), len(y_coords), len(z_coords))

    # 验证坐标轴
    x_ret, y_ret, z_ret = physics.get_field_axes('stress')
    assert np.allclose(x_ret, x_coords)
    assert np.allclose(y_ret, y_coords)
    assert np.allclose(z_ret, z_coords)

    if rank == 0:
        print("  ✓ Basic adaptive field test passed")

    return True


def test_uniform_field_compatibility():
    """测试2：均匀场兼容性"""
    comm = MPI.COMM_WORLD
    rank = comm.rank

    if rank == 0:
        print("\n[Test 2] Uniform field compatibility")
        print("-" * 50)

    config = {
        'physics': {
            'box_size': [100.0, 100.0, 100.0],
        }
    }

    physics = create_unified_physics(comm, config)

    # 使用旧接口定义均匀场
    physics.define_field('temperature', (20, 20, 20))

    # 验证自动生成的坐标轴
    x, y, z = physics.get_field_axes('temperature')

    assert len(x) == 20
    assert len(y) == 20
    assert len(z) == 20

    # 验证均匀间距
    dx = np.diff(x)
    assert np.allclose(dx, dx[0])

    if rank == 0:
        print(f"  Generated X axis: [{x[0]:.1f}, {x[1]:.1f}, ..., {x[-1]:.1f}]")
        print(f"  dx = {dx[0]:.2f} (uniform)")
        print("  ✓ Uniform field compatibility test passed")

    return True


def test_adaptive_interpolation():
    """测试3：非均匀网格插值正确性"""
    comm = MPI.COMM_WORLD
    rank = comm.rank

    if rank == 0:
        print("\n[Test 3] Adaptive grid interpolation")
        print("-" * 50)

    config = {
        'physics': {
            'box_size': [100.0, 100.0, 100.0],
        }
    }

    physics = create_unified_physics(comm, config)

    # 定义非均匀X轴
    x_coords = np.array([0, 10, 40, 45, 50, 55, 60, 90, 100.0])
    y_coords = np.linspace(0, 100, 10)
    z_coords = np.linspace(0, 100, 10)

    physics.define_adaptive_field('test_field', x_coords, y_coords, z_coords)

    # 填充一个简单的解析函数：f(x,y,z) = x
    field = physics.get_field('test_field')
    X, Y, Z = np.meshgrid(x_coords, y_coords, z_coords, indexing='ij')
    field[:] = X  # f(x,y,z) = x

    # 测试插值
    test_points = [
        (25.0, 50.0, 50.0),   # 在粗网格区域内
        (47.5, 50.0, 50.0),   # 在细网格区域内
        (52.5, 50.0, 50.0),   # 在细网格区域内
    ]

    for px, py, pz in test_points:
        interpolated = physics.interpolate_field('test_field', px, py, pz)
        expected = px  # f(x,y,z) = x

        if rank == 0:
            print(f"  Point ({px:.1f}, {py:.1f}, {pz:.1f}): "
                  f"interpolated={interpolated:.4f}, expected={expected:.4f}")

        # 线性插值应该是精确的（对于f(x)=x）
        assert np.isclose(interpolated, expected, rtol=1e-10)

    # 批量插值测试
    positions = np.array([[25, 50, 50], [47.5, 50, 50], [52.5, 50, 50]], dtype=np.float64)
    batch_results = physics.interpolate_field_batch('test_field', positions)
    expected_batch = positions[:, 0]  # f(x,y,z) = x

    assert np.allclose(batch_results, expected_batch, rtol=1e-10)

    if rank == 0:
        print("  ✓ Adaptive interpolation test passed")

    return True


def test_mpi_sync():
    """测试4：MPI同步正确性"""
    comm = MPI.COMM_WORLD
    rank = comm.rank
    size = comm.size

    if rank == 0:
        print("\n[Test 4] MPI synchronization")
        print("-" * 50)

    config = {
        'physics': {
            'box_size': [100.0, 100.0, 100.0],
        }
    }

    physics = create_unified_physics(comm, config)

    # 所有rank定义相同形状的场，但只有rank 0填充数据
    x_coords = np.array([0, 25, 50, 75, 100.0])
    y_coords = np.linspace(0, 100, 5)
    z_coords = np.linspace(0, 100, 5)

    physics.define_adaptive_field('sync_test', x_coords, y_coords, z_coords)

    if rank == 0:
        # 使用set_field正确触发_fields_changed标志
        new_data = np.full((5, 5, 5), 42.0)
        physics.set_field('sync_test', new_data)

    # 同步到所有rank
    # 注意：_sync_all现在区分params和fields，只有_fields_changed=True时才同步场
    # rank 0 的 set_field 已经设置了 _fields_changed=True
    physics._sync_all()

    # 验证所有rank的数据一致
    field = physics.get_field('sync_test')
    x, y, z = physics.get_field_axes('sync_test')

    # 收集所有rank的数据进行比较
    all_field_sums = comm.allgather(field.sum())
    all_x_sums = comm.allgather(x.sum())

    # 所有rank应该有相同的值（来自rank 0的42.0）
    assert all(s == all_field_sums[0] for s in all_field_sums), f"Field sums differ: {all_field_sums}"
    assert all(s == all_x_sums[0] for s in all_x_sums), f"X coord sums differ: {all_x_sums}"

    if rank == 0:
        print(f"  MPI ranks: {size}")
        print(f"  Field sum (all ranks): {all_field_sums}")
        print(f"  X coords sum (all ranks): {all_x_sums}")
        print("  ✓ MPI synchronization test passed")

    return True


def test_memory_efficiency():
    """测试5：内存优化效果验证"""
    comm = MPI.COMM_WORLD
    rank = comm.rank

    if rank == 0:
        print("\n[Test 5] Memory efficiency comparison")
        print("-" * 50)

    config = {
        'physics': {
            'box_size': [100.0, 100.0, 100.0],
        }
    }

    physics = create_unified_physics(comm, config)

    # 场景：裂纹尖端应力场
    # 需求：x=50处高分辨率(0.5nm)，其他区域低分辨率(10nm)

    # 方案1：自适应网格
    x_coarse_1 = np.arange(0, 45, 10.0)        # 5 points
    x_fine = np.arange(45, 55, 0.5)            # 20 points
    x_coarse_2 = np.arange(55, 101, 10.0)      # 5 points
    x_adaptive = np.unique(np.concatenate([x_coarse_1, x_fine, x_coarse_2]))

    y_adaptive = np.linspace(0, 100, 20)
    z_adaptive = np.linspace(0, 100, 20)

    physics.define_adaptive_field('stress_adaptive', x_adaptive, y_adaptive, z_adaptive)
    adaptive_field = physics.get_field('stress_adaptive')
    adaptive_memory = adaptive_field.nbytes

    # 方案2：等效均匀网格（要覆盖细网格区域的分辨率）
    # 全局使用0.5nm分辨率
    uniform_nx = 200  # 100nm / 0.5nm
    uniform_shape = (uniform_nx, 20, 20)
    uniform_memory = uniform_shape[0] * uniform_shape[1] * uniform_shape[2] * 8  # float64

    if rank == 0:
        print(f"  Adaptive grid:")
        print(f"    Shape: {adaptive_field.shape}")
        print(f"    Points: {adaptive_field.size}")
        print(f"    Memory: {adaptive_memory / 1024:.1f} KB")
        print(f"\n  Equivalent uniform grid (0.5nm resolution):")
        print(f"    Shape: {uniform_shape}")
        print(f"    Points: {uniform_shape[0] * uniform_shape[1] * uniform_shape[2]}")
        print(f"    Memory: {uniform_memory / 1024:.1f} KB")
        print(f"\n  Memory reduction: {uniform_memory / adaptive_memory:.1f}x")

    # 验证自适应网格确实更小
    assert adaptive_memory < uniform_memory

    if rank == 0:
        print("  ✓ Memory efficiency test passed")

    return True


def test_validation_errors():
    """测试6：输入验证"""
    comm = MPI.COMM_WORLD
    rank = comm.rank

    if rank == 0:
        print("\n[Test 6] Input validation")
        print("-" * 50)

    config = {
        'physics': {
            'box_size': [100.0, 100.0, 100.0],
        }
    }

    physics = create_unified_physics(comm, config)

    # 测试非递增坐标
    try:
        x_bad = np.array([0, 50, 25, 100.0])  # 非递增
        y_ok = np.linspace(0, 100, 10)
        z_ok = np.linspace(0, 100, 10)
        physics.define_adaptive_field('bad_field', x_bad, y_ok, z_ok)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        if rank == 0:
            print(f"  ✓ Non-increasing coords rejected: {str(e)[:50]}...")

    # 测试太少的点
    try:
        x_short = np.array([50.0])  # 只有1个点
        physics.define_adaptive_field('short_field', x_short, y_ok, z_ok)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        if rank == 0:
            print(f"  ✓ Too few points rejected: {str(e)[:50]}...")

    if rank == 0:
        print("  ✓ Input validation test passed")

    return True


def main():
    """运行所有测试"""
    comm = MPI.COMM_WORLD
    rank = comm.rank

    if rank == 0:
        print("=" * 60)
        print("v1.6.7 Adaptive Physics Field Tests")
        print("=" * 60)

    tests = [
        test_adaptive_field_basic,
        test_uniform_field_compatibility,
        test_adaptive_interpolation,
        test_mpi_sync,
        test_memory_efficiency,
        test_validation_errors,
    ]

    all_passed = True
    for test in tests:
        try:
            test()
        except Exception as e:
            if rank == 0:
                print(f"  ✗ FAILED: {e}")
            all_passed = False

    if rank == 0:
        print("\n" + "=" * 60)
        if all_passed:
            print("All tests passed!")
        else:
            print("Some tests failed!")
        print("=" * 60)

    return all_passed


if __name__ == "__main__":
    main()
