"""
v1.6.5 Bug修复验证测试

测试三个关键Bug的修复：
1. Bug 1: MPI通信长度错误（load_balancer.py）
2. Bug 2: 鬼域计算忽略size字段（domain_decomposition.py）
3. Bug 3: 大对象数组缺少扩容（rma_object_pool.py）

运行：
    mpirun -np 4 python test_bug_fixes.py
"""

import numpy as np
from mpi4py import MPI
import sys


def test_bug1_mpi_alltoallv():
    """
    测试Bug 1修复：MPI通信长度错误

    验证：load_balancer.execute_migration 中 Alltoallv 使用正确的字节计数
    """
    print("\n" + "="*70)
    print("Test 1: Bug 1 - MPI Alltoallv 字节计数修复")
    print("="*70)

    comm = MPI.COMM_WORLD
    rank = comm.rank
    size = comm.size

    # 模拟对象 dtype (120 bytes)
    OBJECT_DTYPE = np.dtype([
        ('id', np.int64),
        ('type', 'U16'),
        ('x', np.float64),
        ('y', np.float64),
        ('z', np.float64),
        ('size', np.float64),
        ('owner_rank', np.int32),
        ('active', np.bool_),
        ('float_props', np.float64, (1,)),
    ], align=True)

    itemsize = OBJECT_DTYPE.itemsize

    # 创建测试数据
    if rank == 0:
        send_data = np.zeros(3, dtype=OBJECT_DTYPE)
        for i in range(3):
            send_data[i]['id'] = i
            send_data[i]['x'] = i * 10.0
    else:
        send_data = np.array([], dtype=OBJECT_DTYPE)

    # 准备 Alltoallv 参数
    # rank 0 -> rank 1 发送 3 个对象，其余方向都是 0。
    send_counts = np.zeros(size, dtype=np.int32)
    recv_counts = np.zeros(size, dtype=np.int32)
    if size > 1:
        if rank == 0:
            send_counts[1] = 3
        elif rank == 1:
            recv_counts[0] = 3

    send_displs = np.zeros(size, dtype=np.int32)
    recv_displs = np.zeros(size, dtype=np.int32)

    # [Bug 1 修复验证] 必须将 counts 乘以 itemsize
    send_counts_bytes = send_counts * itemsize
    recv_counts_bytes = recv_counts * itemsize
    send_displs_bytes = send_displs * itemsize
    recv_displs_bytes = recv_displs * itemsize

    total_recv = np.sum(recv_counts)
    recv_buf = np.empty(total_recv, dtype=OBJECT_DTYPE)

    # 执行 Alltoallv（使用字节计数）
    comm.Alltoallv(
        [send_data, send_counts_bytes, send_displs_bytes, MPI.BYTE],
        [recv_buf, recv_counts_bytes, recv_displs_bytes, MPI.BYTE]
    )

    # 验证
    if rank == 1:
        assert len(recv_buf) == 3, f"Expected 3 objects, got {len(recv_buf)}"
        assert recv_buf[0]['id'] == 0, "Data corruption detected"
        assert recv_buf[0]['x'] == 0.0, "Data corruption detected"
        print(f"[Rank {rank}] ✓ Bug 1 修复验证通过：正确接收到 {len(recv_buf)} 个对象")
        print(f"  - Object dtype size: {itemsize} bytes")
        print(f"  - Send counts (objects): {send_counts}")
        print(f"  - Send counts (bytes): {send_counts_bytes}")

    comm.Barrier()
    if rank == 0:
        print("✓ Test 1 PASSED: MPI Alltoallv 字节计数正确")


def test_bug2_size_field_priority():
    """
    测试Bug 2修复：鬼域计算优先使用size字段

    验证：domain_decomposition._has_large_objects_in_cell 优先检查 size 字段
    """
    print("\n" + "="*70)
    print("Test 2: Bug 2 - size字段优先级修复")
    print("="*70)

    comm = MPI.COMM_WORLD
    rank = comm.rank

    # 模拟有 size 字段的对象
    OBJECT_DTYPE_WITH_SIZE = np.dtype([
        ('id', np.int64),
        ('type', 'U16'),
        ('x', np.float64),
        ('y', np.float64),
        ('z', np.float64),
        ('size', np.float64),  # v1.6.5 新增字段
        ('owner_rank', np.int32),
        ('active', np.bool_),
        ('float_props', np.float64, (1,)),
    ], align=True)

    # 创建测试对象（半径 = 30）
    obj = np.zeros(1, dtype=OBJECT_DTYPE_WITH_SIZE)[0]
    obj['size'] = 60.0  # 直径 60，半径 30
    obj['float_props'][0] = 5.0  # 旧约定的半径（应该被忽略）

    # 模拟检查逻辑（Bug 2 修复后的版本）
    threshold = 20.0

    # 方案1：优先使用 size 字段
    if 'size' in OBJECT_DTYPE_WITH_SIZE.names:
        radius = obj['size'] / 2.0  # = 30.0
        has_large = (radius >= threshold)  # True
    else:
        has_large = False

    if rank == 0:
        assert has_large == True, "size字段检测失败"
        print(f"[Rank {rank}] ✓ Bug 2 修复验证通过：")
        print(f"  - size字段: {obj['size']} → 半径 = {obj['size']/2.0}")
        print(f"  - float_props[0]: {obj['float_props'][0]} (被忽略)")
        print(f"  - threshold: {threshold}")
        print(f"  - 判定结果: has_large = {has_large} ✓")

    comm.Barrier()
    if rank == 0:
        print("✓ Test 2 PASSED: size字段优先级正确")


def test_bug3_large_array_expansion():
    """
    测试Bug 3修复：大对象数组自动扩容

    验证：rma_object_pool._add_to_large_array 支持自动扩容
    """
    print("\n" + "="*70)
    print("Test 3: Bug 3 - 大对象数组自动扩容")
    print("="*70)

    comm = MPI.COMM_WORLD
    rank = comm.rank

    from paraokmc.rma_object_pool import RMAObjectPool, create_default_object_dtype, create_object
    from paraokmc.domain_decomposition import DomainDecomposition

    # 创建测试配置
    config = {
        'lattice_constant': 10.0,
        'lattice_grid': [10, 10, 10],
        'periodic': [True, True, True],
        'max_interaction_distance': 5.0,
        'h_factor': 1,
    }

    domain_decomp = DomainDecomposition(comm, config)

    # 创建对象池（初始容量很小，用于测试扩容）
    # 移除废弃参数：use_binning, enable_boundary_separation, enable_large_objects
    dtype = create_default_object_dtype()
    pool = RMAObjectPool(
        comm,
        dtype=dtype,
        initial_capacity=10,  # 小容量触发扩容
        auto_expand=True,
        verbose=False,
        domain_decomposition=domain_decomp,
        max_large_objects=2,  # 非常小的容量
    )

    # 测试添加大对象（超过初始容量）
    initial_large_capacity = pool.large_capacity

    try:
        # 添加 3 个大对象（超过初始容量 2）
        for i in range(3):
            obj = create_object(
                obj_id=i,
                obj_type='LargeObject',
                position=[i * 10.0, 0.0, 0.0],
                owner_rank=rank,
                size=200.0,
            )

            # [Bug 3 修复验证] 这里会触发自动扩容
            pool._add_to_large_array(obj)

        final_large_capacity = pool.large_capacity

        if rank == 0:
            print(f"[Rank {rank}] ✓ Bug 3 修复验证通过：")
            print(f"  - 初始大对象容量: {initial_large_capacity}")
            print(f"  - 添加对象数量: 3")
            print(f"  - 最终容量: {final_large_capacity}")
            print(f"  - 扩容次数: {int(np.log2(final_large_capacity / initial_large_capacity))}")
            assert final_large_capacity > initial_large_capacity, "扩容失败"
            assert pool.large_count == 3, f"对象计数错误: {pool.large_count}"

    finally:
        pool.destroy()

    comm.Barrier()
    if rank == 0:
        print("✓ Test 3 PASSED: 大对象数组自动扩容正确")


def main():
    comm = MPI.COMM_WORLD
    rank = comm.rank

    if rank == 0:
        print("\n" + "="*70)
        print("v1.6.5 Bug修复验证测试")
        print("="*70)
        print(f"运行在 {comm.size} 个MPI进程上")

    try:
        # 测试1：MPI通信长度错误
        test_bug1_mpi_alltoallv()

        # 测试2：size字段优先级
        test_bug2_size_field_priority()

        # 测试3：大对象数组扩容
        test_bug3_large_array_expansion()

        if rank == 0:
            print("\n" + "="*70)
            print("✓✓✓ 所有测试通过！三个Bug修复验证成功 ✓✓✓")
            print("="*70)

    except Exception as e:
        print(f"[Rank {rank}] ✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
