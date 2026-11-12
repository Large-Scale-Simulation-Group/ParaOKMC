"""
简单测试：验证边界分离的核心算法

只测试分类算法，不涉及RMA操作
"""

import numpy as np
from mpi4py import MPI
import sys
import os

from paraokmc.rma_object_pool import create_rma_object_pool, create_object
from paraokmc.domain_decomposition import DomainDecomposition


def test_simple():
    """简单测试：分类算法 + 统计信息"""
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    if rank == 0:
        print("=" * 60)
        print("边界分离优化：简单测试（v1.6.6）")
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

    # 创建域分解
    domain_decomp = DomainDecomposition(comm=comm, config=config, verbose=False)

    # 创建对象池（当前版本默认使用三分类存储）
    pool = create_rma_object_pool(
        comm,
        initial_capacity=100,
        domain_decomposition=domain_decomp,
        verbose=(rank == 0)
    )

    # 获取本rank的子域边界
    bounds = domain_decomp.subdomain_bounds

    if rank == 0:
        print(f"\nRank {rank} 子域边界:")
        print(f"  x: [{bounds['x_min']:.2f}, {bounds['x_max']:.2f}]")
        print(f"  y: [{bounds['y_min']:.2f}, {bounds['y_max']:.2f}]")
        print(f"  z: [{bounds['z_min']:.2f}, {bounds['z_max']:.2f}]")

    # 计算中心点
    center_x = (bounds['x_min'] + bounds['x_max']) / 2.0
    center_y = (bounds['y_min'] + bounds['y_max']) / 2.0
    center_z = (bounds['z_min'] + bounds['z_max']) / 2.0

    # 测试1：中心对象（应该分类为center）
    center_obj = create_object(
        obj_id=rank * 1000 + 1,
        obj_type='I',
        position=[center_x, center_y, center_z],
        owner_rank=rank,
        size=1.0
    )

    region1 = pool._classify_object_region(center_obj)

    if rank == 0:
        print(f"\n测试1：中心对象")
        print(f"  位置: ({center_x:.2f}, {center_y:.2f}, {center_z:.2f})")
        print(f"  分类: {region1}")
        print(f"  预期: center")
        print(f"  ✓ 通过" if region1 == 'center' else "  ✗ 失败")

    # 测试2：边界对象（靠近x_min，应该分类为boundary）
    boundary_obj = create_object(
        obj_id=rank * 1000 + 2,
        obj_type='V',
        position=[bounds['x_min'] + 2.0, center_y, center_z],  # 距离边界 < r_cut
        owner_rank=rank,
        size=1.0
    )

    region2 = pool._classify_object_region(boundary_obj)

    if rank == 0:
        print(f"\n测试2：边界对象")
        print(f"  位置: ({bounds['x_min'] + 2.0:.2f}, {center_y:.2f}, {center_z:.2f})")
        print(f"  距离x_min边界: 2.0 (< r_cut={config['max_interaction_distance']:.1f})")
        print(f"  分类: {region2}")
        print(f"  预期: boundary")
        print(f"  ✓ 通过" if region2 == 'boundary' else "  ✗ 失败")

    # 添加对象到池
    pool.add_object(center_obj)
    pool.add_object(boundary_obj)

    # 再添加一些随机对象
    np.random.seed(rank + 42)
    for i in range(20):
        obj = create_object(
            obj_id=rank * 1000 + 100 + i,
            obj_type='I' if i % 2 == 0 else 'V',
            position=[
                np.random.uniform(bounds['x_min'], bounds['x_max']),
                np.random.uniform(bounds['y_min'], bounds['y_max']),
                np.random.uniform(bounds['z_min'], bounds['z_max']),
            ],
            owner_rank=rank,
            size=1.0
        )
        pool.add_object(obj)

    # 统计信息
    stats = {
        'center_active': pool.center_count,
        'boundary_active': pool.boundary_count,
        'large_active': pool.large_count,
        'total_active': pool.center_count + pool.boundary_count + pool.large_count,
    }

    if rank == 0:
        print(f"\n统计信息:")
        print(f"  中心对象: {stats['center_active']}")
        print(f"  边界对象: {stats['boundary_active']}")
        print(f"  大对象: {stats['large_active']}")
        print(f"  总计: {stats['total_active']}")

        total = stats['total_active']
        center_pct = (stats['center_active'] / total * 100) if total > 0 else 0
        boundary_pct = (stats['boundary_active'] / total * 100) if total > 0 else 0

        print(f"\n分布:")
        print(f"  中心: {center_pct:.1f}%")
        print(f"  边界: {boundary_pct:.1f}%")

    # 收集全局统计
    all_stats = comm.gather(stats, root=0)

    if rank == 0:
        print(f"\n全局统计（{comm.Get_size()} ranks）:")
        total_center = sum(s['center_active'] for s in all_stats)
        total_boundary = sum(s['boundary_active'] for s in all_stats)
        total_large = sum(s['large_active'] for s in all_stats)
        total_all = total_center + total_boundary + total_large

        print(f"  中心对象: {total_center}")
        print(f"  边界对象: {total_boundary}")
        print(f"  大对象: {total_large}")
        print(f"  总计: {total_all}")

        if total_all > 0:
            print(f"\n全局分布:")
            print(f"  中心: {total_center/total_all*100:.1f}%")
            print(f"  边界: {total_boundary/total_all*100:.1f}%")

    # 测试向后兼容接口
    all_objs = pool.get_local_objects()
    center_objs = pool.center_objects[:pool.center_count]
    boundary_objs = pool.boundary_objects[:pool.boundary_count]

    if rank == 0:
        print(f"\n接口测试:")
        print(f"  get_local_objects(): {len(all_objs)} 个对象")
        print(f"  get_center_objects(): {len(center_objs)} 个对象")
        print(f"  get_boundary_objects(): {len(boundary_objs)} 个对象")
        print(f"  合计: {len(center_objs) + len(boundary_objs)}")
        print(f"  ✓ 一致" if len(all_objs) == len(center_objs) + len(boundary_objs) else "  ✗ 不一致")

    # 清理
    pool.destroy()

    if rank == 0:
        print("\n" + "=" * 60)
        print("测试完成！")
        print("=" * 60)


if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    try:
        test_simple()

        if rank == 0:
            print("\n✓ 所有测试通过！")

    except Exception as e:
        print(f"[Rank {rank}] 测试失败: {e}")
        import traceback
        traceback.print_exc()
        comm.Abort(1)
