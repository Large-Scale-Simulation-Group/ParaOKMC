"""
v1.6.8 Dedalus外场耦合器测试

测试内容：
1. 高精度场存储和历史功能
2. 场压缩器（均匀/自适应）
3. 对象栅格化
4. 热传导求解器
5. 与UnifiedPhysics集成

运行方式：
    mpirun -np 2 python test_dedalus_coupler.py
"""

import numpy as np
from mpi4py import MPI
import sys
import os

# 设置环境变量减少Dedalus线程警告
os.environ.setdefault('OMP_NUM_THREADS', '1')

from paraokmc.dedalus_coupler import (
    HighPrecisionFieldStore,
    FieldCompressor,
    DedalusCoupler,
    DedalusHeatSolver,
    DedalusCouplerAdapter,
    DEDALUS_AVAILABLE,
    DEDALUS_VERSION,
)
from paraokmc.unified_physics import UnifiedPhysics, MODE_INWARD

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


def test_high_precision_store():
    """测试高精度场存储"""
    print(f"\n[Test 1] High Precision Field Store")

    store = HighPrecisionFieldStore(comm, enable_history=True)

    # 注册场
    shape = (10, 10, 10)
    x = np.linspace(0, 1, 10)
    y = np.linspace(0, 1, 10)
    z = np.linspace(0, 1, 10)

    store.register_field('temperature', shape, shape,
                         (slice(None),)*3, (x, y, z), 'grid')

    # 设置初始值
    T_init = np.full(shape, 300.0)
    store.current_fields['temperature'][:] = T_init

    # 保存历史
    store.save_history(0.0)

    # 更新场
    T_new = np.full(shape, 350.0)
    store.update_field('temperature', T_new)
    store.update_time(0.1)

    # 检查历史
    T_prev = store.get_prev('temperature')
    T_curr = store.get_current('temperature')

    assert T_prev is not None
    assert np.allclose(T_prev, 300.0)
    assert np.allclose(T_curr, 350.0)

    # 检查变化率
    rate = store.get_field_rate('temperature')
    expected_rate = (350 - 300) / 0.1
    assert rate is not None
    assert np.allclose(rate, expected_rate)

    print(f"  - History storage: PASSED")
    print(f"  - Field rate: {rate[0,0,0]:.1f} K/s (expected {expected_rate:.1f})")
    return True


def test_field_compressor_uniform():
    """测试均匀压缩"""
    print(f"\n[Test 2] Field Compressor (Uniform)")

    compressor = FieldCompressor(comm, target_size=(5, 5, 5))

    # 创建测试场
    nx, ny, nz = 20, 20, 20
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)
    z = np.linspace(0, 1, nz)

    # 创建有梯度的场
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    field = 300 + 50 * np.sin(2*np.pi*X) * np.cos(2*np.pi*Y)

    # 只有rank 0执行压缩
    compressed, x_new, y_new, z_new = compressor.compress(
        field, x, y, z, method='uniform'
    )

    if rank == 0:
        assert compressed.shape == (5, 5, 5)
        assert len(x_new) == 5
        print(f"  - Original: {field.shape}")
        print(f"  - Compressed: {compressed.shape}")
        print(f"  - Ratio: {compressed.size / field.size:.2%}")

    return True


def test_field_compressor_adaptive():
    """测试自适应压缩"""
    print(f"\n[Test 3] Field Compressor (Adaptive)")

    compressor = FieldCompressor(
        comm,
        gradient_threshold=0.1,
        min_points=5,
        max_compression_ratio=0.5
    )

    # 创建测试场：大部分平坦，中间有尖峰
    nx, ny, nz = 30, 30, 30
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)
    z = np.linspace(0, 1, nz)

    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

    # 中心有尖峰的场
    r = np.sqrt((X - 0.5)**2 + (Y - 0.5)**2 + (Z - 0.5)**2)
    field = 300 + 100 * np.exp(-r**2 / 0.01)  # 尖峰在中心

    compressed, x_new, y_new, z_new = compressor.compress(
        field, x, y, z, method='adaptive'
    )

    if rank == 0:
        print(f"  - Original: {field.shape}")
        print(f"  - Compressed: {compressed.shape}")
        print(f"  - Ratio: {compressed.size / field.size:.2%}")
        # 自适应应该在中心保留更多点

    return True


def test_object_rasterization():
    """测试对象栅格化"""
    print(f"\n[Test 4] Object Rasterization")

    box_size = (10.0, 10.0, 10.0)
    config = {
        'resolution': (20, 20, 20),
    }

    coupler = DedalusCoupler(comm, box_size, config)

    # 创建测试对象
    n_obj = 5
    obj_dtype = np.dtype([
        ('id', np.int64),
        ('type', 'U32'),
        ('x', np.float64),
        ('y', np.float64),
        ('z', np.float64),
        ('energy', np.float64),
    ])

    objects = np.zeros(n_obj, dtype=obj_dtype)
    objects['id'] = np.arange(n_obj)
    objects['type'] = 'vacancy'
    objects['x'] = [2.0, 5.0, 5.0, 5.0, 8.0]  # 一些在中心附近
    objects['y'] = [5.0, 2.0, 5.0, 8.0, 5.0]
    objects['z'] = [5.0, 5.0, 5.0, 5.0, 5.0]
    objects['energy'] = [1.0, 1.0, 2.0, 1.0, 1.0]  # 中心能量更高

    # 测试不同核函数
    delta_field = coupler.rasterize_objects(objects, 'energy', kernel='delta')
    gauss_field = coupler.rasterize_objects(objects, 'energy', kernel='gaussian', sigma=1.0)
    uniform_field = coupler.rasterize_objects(objects, 'energy', kernel='uniform', sigma=1.0)

    print(f"  - Delta kernel sum: {delta_field.sum():.2f} (expected ~6.0)")
    print(f"  - Gaussian kernel sum: {gauss_field.sum():.2f}")
    print(f"  - Uniform kernel sum: {uniform_field.sum():.2f}")

    # Delta应该精确等于总能量
    assert np.isclose(delta_field.sum(), 6.0, atol=0.1)

    return True


def test_heat_solver():
    """测试热传导求解器"""
    print(f"\n[Test 5] Heat Solver")

    box_size = (10.0, 10.0, 10.0)
    config = {
        'resolution': (16, 16, 16),
        'compression': {
            'method': 'uniform',
            'target_size': (8, 8, 8),
        }
    }

    coupler = DedalusCoupler(comm, box_size, config)

    # 注册温度场
    coupler.register_field('temperature', 300.0)

    # 创建求解器
    heat_solver = DedalusHeatSolver(
        kappa=1.0,
        dt=0.01,
        bc_type='periodic',
        T_boundary=300.0
    )
    coupler.register_solver('heat', heat_solver, ['temperature'])

    # 添加热源（模拟cascade）
    obj_dtype = np.dtype([
        ('id', np.int64),
        ('type', 'U32'),
        ('x', np.float64),
        ('y', np.float64),
        ('z', np.float64),
        ('energy', np.float64),
    ])

    objects = np.zeros(1, dtype=obj_dtype)
    objects['id'] = 0
    objects['type'] = 'cascade'
    objects['x'] = 5.0
    objects['y'] = 5.0
    objects['z'] = 5.0
    objects['energy'] = 100.0  # 高能量热源

    # 执行求解
    coupler.save_history(0.0)
    results = coupler.solve_and_compress('heat', 0, 0.01, objects)

    if rank == 0 and 'temperature' in results:
        compressed_T, x_c, y_c, z_c = results['temperature']
        print(f"  - Compressed shape: {compressed_T.shape}")
        print(f"  - T range: [{compressed_T.min():.1f}, {compressed_T.max():.1f}] K")

        # 热源位置应该温度升高
        assert compressed_T.max() > 300.0

    return True


def test_unified_physics_integration():
    """测试与UnifiedPhysics集成"""
    print(f"\n[Test 6] UnifiedPhysics Integration")

    box_size = (10.0, 10.0, 10.0)
    coupler_config = {
        'resolution': (16, 16, 16),
        'compression': {
            'method': 'uniform',
            'target_size': (8, 8, 8),
        }
    }

    # 创建DedalusCoupler
    coupler = DedalusCoupler(comm, box_size, coupler_config)
    coupler.register_field('temperature', 300.0)

    heat_solver = DedalusHeatSolver(kappa=1.0, dt=0.01, bc_type='periodic')
    coupler.register_solver('heat', heat_solver, ['temperature'])

    # 创建适配器
    adapter = DedalusCouplerAdapter(coupler, 'heat')

    # 创建UnifiedPhysics（注意：只需comm和config两个参数）
    physics_config = {
        'box_size': box_size,
        'coupling': {
            'mode': MODE_INWARD,
            'external_solver': adapter,
        }
    }

    physics = UnifiedPhysics(comm, physics_config)

    # 定义温度场（使用正确的API：name, shape, initial_value）
    physics.define_field('temperature', (8, 8, 8), 300.0)

    # 注册求解器
    physics.register_solver('heat', adapter, fields=['temperature'])

    # 通过适配器求解（这会调用DedalusCoupler）
    adapter.initialize(physics, {})

    print(f"  - Adapter created: {adapter is not None}")
    print(f"  - Supported fields: {adapter.get_supported_fields()}")

    return True


def run_all_tests():
    """运行所有测试"""
    if rank == 0:
        print("="*60)
        print("v1.6.8 Dedalus Coupler Tests")
        print("="*60)
        print(f"MPI Size: {size}")
        print(f"Dedalus Available: {DEDALUS_AVAILABLE}")
        if DEDALUS_AVAILABLE:
            print(f"Dedalus Version: {DEDALUS_VERSION}")

    comm.Barrier()

    tests = [
        ("High Precision Store", test_high_precision_store),
        ("Uniform Compression", test_field_compressor_uniform),
        ("Adaptive Compression", test_field_compressor_adaptive),
        ("Object Rasterization", test_object_rasterization),
        ("Heat Solver", test_heat_solver),
        ("UnifiedPhysics Integration", test_unified_physics_integration),
    ]

    passed = 0
    failed = 0

    for name, test_func in tests:
        try:
            result = test_func()
            if result:
                passed += 1
                if rank == 0:
                    print(f"[{name}] PASSED")
            else:
                failed += 1
                if rank == 0:
                    print(f"[{name}] FAILED")
        except Exception as e:
            failed += 1
            if rank == 0:
                print(f"[{name}] FAILED with exception: {e}")
                import traceback
                traceback.print_exc()

        comm.Barrier()

    if rank == 0:
        print("\n" + "="*60)
        print(f"Results: {passed}/{passed+failed} tests passed")
        if failed == 0:
            print("All tests PASSED!")
        else:
            print(f"{failed} tests FAILED")
        print("="*60)

    return failed == 0


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
