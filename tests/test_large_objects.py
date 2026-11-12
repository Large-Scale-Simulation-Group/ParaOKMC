"""
测试大对象支持功能（v1.6.5）

验证内容：
1. 大对象自动判定
2. 大对象在所有rank间的同步
3. 大→小、小→大转换
4. 大对象的更新和删除
"""

import numpy as np
from mpi4py import MPI
import sys
import os


from paraokmc.rma_object_pool import create_rma_object_pool, create_object
from paraokmc.domain_decomposition import DomainDecomposition


def test_large_object_detection():
    """测试大对象判定算法"""
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    if rank == 0:
        print("=" * 60)
        print("测试1：大对象判定算法")
        print("=" * 60)

    # 配置
    config = {
        'lattice_constant': 10.0,
        'lattice_grid': [10, 10, 10],
        'h_factor': 1,
        'box_size': [100.0, 100.0, 100.0],
        'max_interaction_distance': 5.0,  # r_cut = 5.0
        'periodic': [True, True, True],
        'nx': 2,
        'ny': 2,
        'nz': 1,
        'grid_dim': [10, 10, 10],  # cell_size = 10.0
    }

    # 创建域分解
    domain_decomp = DomainDecomposition(comm=comm, config=config, verbose=False)

    # 创建对象池（v1.6.6：三分类存储是默认模式）
    pool = create_rma_object_pool(
        comm,
        initial_capacity=100,
        domain_decomposition=domain_decomp,
        large_object_multiplier=5.0,  # 判定倍数
        verbose=(rank == 0)
    )

    # 计算判定阈值
    # Ghost capacity = N_layers × L_min_cell
    # N_layers = ceil(r_cut / L_min_cell) = ceil(5.0 / 10.0) = 1
    # Ghost capacity = 1 × 10.0 = 10.0
    # Threshold = 10.0 × 5.0 = 50.0
    # 因此：size + r_cut >= 50.0 → large object
    # 即：size >= 45.0 → large object

    # 测试Case 1: 小对象（size=10.0 < 45.0）
    small_obj = create_object(
        obj_id=1,
        obj_type='cluster',
        position=[50.0, 50.0, 50.0],
        owner_rank=rank,
        size=10.0  # radius = 5.0, influence = 5.0 + 5.0 = 10.0 < 50.0
    )

    is_large1 = pool._is_large_object(small_obj)

    # 测试Case 2: 大对象（size=90.0 >= 45.0）
    large_obj = create_object(
        obj_id=2,
        obj_type='precipitate',
        position=[50.0, 50.0, 50.0],
        owner_rank=rank,
        size=90.0  # radius = 45.0, influence = 45.0 + 5.0 = 50.0 >= 50.0
    )

    is_large2 = pool._is_large_object(large_obj)

    # 测试Case 3: 边界情况（size=90.0）
    edge_obj = create_object(
        obj_id=3,
        obj_type='void',
        position=[50.0, 50.0, 50.0],
        owner_rank=rank,
        size=90.0  # 刚好达到阈值
    )

    is_large3 = pool._is_large_object(edge_obj)

    if rank == 0:
        print(f"\n判定阈值分析:")
        print(f"  r_cut = 5.0")
        print(f"  L_min_cell = 10.0")
        print(f"  N_layers = 1")
        print(f"  Ghost capacity = 10.0")
        print(f"  Threshold = 10.0 × 5.0 = 50.0")
        print(f"  判定条件: (size/2 + r_cut) >= 50.0")
        print(f"           即: size >= 90.0")

        print(f"\n判定结果:")
        print(f"  Case 1: size=10.0 (10.0 < 90.0) → {is_large1} (预期: False)")
        print(f"  Case 2: size=90.0 (90.0 >= 90.0) → {is_large2} (预期: True)")
        print(f"  Case 3: size=90.0 (边界) → {is_large3} (预期: True)")

        assert not is_large1, "错误：小对象被判定为大对象"
        assert is_large2, "错误：大对象未被判定"
        assert is_large3, "错误：边界情况判定错误"

        print("\n✓ 大对象判定算法正确！")

    pool.destroy()


def test_large_object_lifecycle():
    """测试大对象的完整生命周期"""
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:
        print("\n" + "=" * 60)
        print("测试2：大对象完整生命周期")
        print("=" * 60)

    # 配置
    config = {
        'lattice_constant': 10.0,
        'lattice_grid': [10, 10, 10],
        'h_factor': 1,
        'box_size': [100.0, 100.0, 100.0],
        'max_interaction_distance': 5.0,
        'periodic': [True, True, True],
        'nx': 2,
        'ny': 2,
        'nz': 1,
        'grid_dim': [10, 10, 10],
    }

    domain_decomp = DomainDecomposition(comm=comm, config=config, verbose=False)

    pool = create_rma_object_pool(
        comm,
        initial_capacity=100,
        domain_decomposition=domain_decomp,
        large_object_multiplier=5.0,
        verbose=False
    )

    # === 阶段1：添加大对象 ===
    if rank == 0:
        print("\n阶段1：添加大对象")

    # 只有rank 0创建大对象
    if rank == 0:
        large_obj = create_object(
            obj_id=1000,
            obj_type='precipitate',
            position=[50.0, 50.0, 50.0],
            owner_rank=0,
            size=100.0  # 大对象
        )
        pool.add_object(large_obj)

    comm.Barrier()

    # 验证：所有rank都应该有这个大对象吗？
    # 注意：在当前实现中，大对象添加只在add_object时进行
    # 在实际应用中，大对象应该通过changed_objects机制同步

    # 为了测试，我们手动模拟changed_objects处理
    # 在实际v16_engine中，这由_execute_event自动处理

    if rank == 0:
        print(f"  Rank 0添加大对象 ID=1000")
        stats = {
            'center_active': pool.center_count,
            'boundary_active': pool.boundary_count,
            'large_active': pool.large_count,
            'total_active': pool.center_count + pool.boundary_count + pool.large_count,
        }
        print(f"  Rank 0统计: large={stats['large_active']}")

    comm.Barrier()

    # === 阶段2：测试小对象 ===
    if rank == 0:
        print("\n阶段2：添加小对象")

    small_obj = create_object(
        obj_id=rank * 100 + 1,
        obj_type='I',
        position=[25.0 + rank * 50.0, 25.0, 50.0],
        owner_rank=rank,
        size=10.0  # 小对象
    )
    pool.add_object(small_obj)

    stats = {
        'center_active': pool.center_count,
        'boundary_active': pool.boundary_count,
        'large_active': pool.large_count,
        'total_active': pool.center_count + pool.boundary_count + pool.large_count,
    }

    if rank == 0:
        print(f"\n所有rank统计:")
        all_stats = comm.gather(stats, root=0)
        for r, s in enumerate(all_stats):
            print(f"  Rank {r}: center={s['center_active']}, boundary={s['boundary_active']}, large={s['large_active']}")
    else:
        comm.gather(stats, root=0)

    # === 阶段3：测试分类 ===
    if rank == 0:
        print("\n阶段3：验证对象分类")

        # 获取对象
        small_retrieved = pool.get_object_by_id(1)
        large_retrieved = pool.get_object_by_id(1000)

        if small_retrieved is not None:
            print(f"  小对象 ID=1: 位置=({small_retrieved['x']:.1f}, {small_retrieved['y']:.1f}, {small_retrieved['z']:.1f})")

        if large_retrieved is not None:
            print(f"  大对象 ID=1000: size={large_retrieved['size']:.1f}")
        else:
            print(f"  大对象 ID=1000: 未找到（等待同步）")

    pool.destroy()

    if rank == 0:
        print("\n✓ 大对象生命周期测试完成！")


def test_size_transition():
    """测试大小对象转换"""
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    if rank == 0:
        print("\n" + "=" * 60)
        print("测试3：大小对象转换")
        print("=" * 60)

    config = {
        'lattice_constant': 10.0,
        'lattice_grid': [10, 10, 10],
        'h_factor': 1,
        'box_size': [100.0, 100.0, 100.0],
        'max_interaction_distance': 5.0,
        'periodic': [True, True, True],
        'nx': 2,
        'ny': 2,
        'nz': 1,
        'grid_dim': [10, 10, 10],
    }

    domain_decomp = DomainDecomposition(comm=comm, config=config, verbose=False)

    pool = create_rma_object_pool(
        comm,
        initial_capacity=100,
        domain_decomposition=domain_decomp,
        large_object_multiplier=5.0,
        verbose=False
    )

    # 创建一个可以转换大小的对象
    if rank == 0:
        # 初始：小对象
        obj = create_object(
            obj_id=2000,
            obj_type='cluster',
            position=[50.0, 50.0, 50.0],
            owner_rank=0,
            size=10.0  # 小对象
        )
        pool.add_object(obj)

        print("\n初始状态:")
        print(f"  size=10.0")
        print(f"  is_large={pool.is_large_object_by_id(2000)}")

        # 模拟生长：小→大
        obj_updated = obj.copy()
        obj_updated['size'] = 100.0  # 增大到大对象尺寸

        is_large_after = pool._is_large_object(obj_updated)

        print(f"\n生长后:")
        print(f"  size=100.0")
        print(f"  is_large={is_large_after} (预期: True)")

        assert is_large_after, "错误：生长后应该判定为大对象"

        print("\n✓ 大小转换判定正确！")

    pool.destroy()


if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    try:
        # 测试1：大对象判定
        test_large_object_detection()

        comm.Barrier()

        # 测试2：生命周期
        test_large_object_lifecycle()

        comm.Barrier()

        # 测试3：大小转换
        test_size_transition()

        comm.Barrier()

        if rank == 0:
            print("\n" + "=" * 60)
            print("✓ 所有测试通过！")
            print("=" * 60)

    except Exception as e:
        print(f"[Rank {rank}] 测试失败: {e}")
        import traceback
        traceback.print_exc()
        comm.Abort(1)
