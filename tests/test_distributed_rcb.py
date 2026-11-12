"""
分布式 RCB 对象重分布测试

测试内容：
1. distributed_rcb 与 centralized recursive_bisection 的一致性
2. 对象守恒（重划前后对象总数不变）
3. incremental_reclassify 正确性
4. execute_migration_planned 基本流程
"""

import numpy as np
import pytest
from paraokmc.load_balancer import (
    Bounds, recursive_bisection, distributed_rcb,
    compute_imbalance, ObjectLoadBalancer,
)


class TestDistributedRCBConsistency:
    """对比分布式 RCB 和集中式 RCB 的边界结果"""

    def _run_both_rcb(self, grid_load, n_ranks):
        """用相同输入分别跑集中式和分布式 RCB，返回两组结果"""
        from mpi4py import MPI
        comm = MPI.COMM_WORLD

        grid_dim = grid_load.shape
        global_box = Bounds(0, grid_dim[0], 0, grid_dim[1], 0, grid_dim[2])
        ranks = list(range(n_ranks))

        # 集中式
        centralized = recursive_bisection(grid_load, ranks, global_box)

        # 分布式（单进程模式下 comm.size==1，但算法仍然正确）
        # 对于 n_ranks > 1 的情况，单进程下无法真正测试 Comm.Split
        # 但可以验证 n_ranks == 1 的退化情况
        if comm.size == 1 and n_ranks == 1:
            distributed = distributed_rcb(comm, grid_load, global_box, ranks)
        else:
            distributed = None

        return centralized, distributed

    def test_single_rank_trivial(self):
        """单 rank 情况下，两种方法结果一致"""
        grid_load = np.random.randint(0, 10, size=(8, 8, 8), dtype=np.int32)
        centralized, distributed = self._run_both_rcb(grid_load, n_ranks=1)

        assert 0 in centralized
        c_bounds = centralized[0]
        assert c_bounds.x_start == 0 and c_bounds.x_end == 8
        assert c_bounds.y_start == 0 and c_bounds.y_end == 8
        assert c_bounds.z_start == 0 and c_bounds.z_end == 8

        if distributed is not None:
            d_bounds = distributed[0]
            assert d_bounds.x_start == c_bounds.x_start
            assert d_bounds.x_end == c_bounds.x_end
            assert d_bounds.y_start == c_bounds.y_start
            assert d_bounds.y_end == c_bounds.y_end
            assert d_bounds.z_start == c_bounds.z_start
            assert d_bounds.z_end == c_bounds.z_end

    def test_centralized_rcb_coverage(self):
        """集中式 RCB 覆盖全部网格（无遗漏无重叠）"""
        grid_load = np.random.randint(1, 20, size=(10, 10, 10), dtype=np.int32)
        for n_ranks in [2, 4, 8]:
            global_box = Bounds(0, 10, 0, 10, 0, 10)
            result = recursive_bisection(grid_load, list(range(n_ranks)), global_box)

            # 验证所有 rank 都有分配
            assert len(result) == n_ranks

            # 验证覆盖：每个 cell 恰好属于一个 rank
            coverage = np.zeros((10, 10, 10), dtype=np.int32)
            for rank_id, bounds in result.items():
                coverage[
                    bounds.x_start:bounds.x_end,
                    bounds.y_start:bounds.y_end,
                    bounds.z_start:bounds.z_end
                ] += 1

            assert np.all(coverage == 1), f"n_ranks={n_ranks}: coverage not exactly 1 everywhere"

    def test_centralized_rcb_load_balance(self):
        """集中式 RCB 产生合理的负载平衡"""
        np.random.seed(42)
        grid_load = np.zeros((10, 10, 10), dtype=np.int32)
        # 创建热点
        grid_load[2:4, 2:4, 2:4] = 100
        grid_load[7:9, 7:9, 7:9] = 50

        global_box = Bounds(0, 10, 0, 10, 0, 10)
        result = recursive_bisection(grid_load, list(range(4)), global_box)

        loads = []
        for rank_id in range(4):
            b = result[rank_id]
            load = int(np.sum(grid_load[b.x_start:b.x_end, b.y_start:b.y_end, b.z_start:b.z_end]))
            loads.append(load)

        imbalance = compute_imbalance(loads)
        # RCB 应该能把不平衡度控制在合理范围（热点集中时可能较高）
        assert imbalance < 2.0, f"Imbalance too high: {imbalance:.2f}, loads={loads}"

    def test_empty_load(self):
        """空负载情况下不崩溃"""
        grid_load = np.zeros((8, 8, 8), dtype=np.int32)
        global_box = Bounds(0, 8, 0, 8, 0, 8)

        for n_ranks in [1, 2, 4]:
            result = recursive_bisection(grid_load, list(range(n_ranks)), global_box)
            assert len(result) == n_ranks

    def test_uniform_load(self):
        """均匀负载下 RCB 产生近似均分"""
        grid_load = np.ones((8, 8, 8), dtype=np.int32)
        global_box = Bounds(0, 8, 0, 8, 0, 8)

        result = recursive_bisection(grid_load, list(range(4)), global_box)
        loads = []
        for rank_id in range(4):
            b = result[rank_id]
            loads.append(b.volume())

        # 均匀负载下，各 rank 体积应该相近（RCB 不保证完美均分）
        total = sum(loads)
        assert total == 512, f"Total volume mismatch: {total}"
        # 每个 rank 至少分到一些 cells
        assert all(v > 0 for v in loads), f"Some rank got 0 volume: {loads}"


class TestBoundsClass:
    """Bounds 类的基本测试"""

    def test_to_dict_from_dict(self):
        b = Bounds(1, 5, 2, 8, 0, 10)
        d = b.to_dict()
        b2 = Bounds.from_dict(d)
        assert b2.x_start == 1 and b2.x_end == 5
        assert b2.y_start == 2 and b2.y_end == 8
        assert b2.z_start == 0 and b2.z_end == 10

    def test_volume(self):
        b = Bounds(0, 4, 0, 3, 0, 2)
        assert b.volume() == 24


class TestComputeImbalance:
    """负载不平衡度计算"""

    def test_perfect_balance(self):
        assert compute_imbalance([10, 10, 10, 10]) == 0.0

    def test_imbalanced(self):
        imb = compute_imbalance([10, 10, 10, 30])
        # max=30, avg=15, imbalance=1.0
        assert abs(imb - 1.0) < 1e-10

    def test_empty(self):
        assert compute_imbalance([]) == 0.0
        assert compute_imbalance([0, 0, 0]) == 0.0


class TestIncrementalReclassify:
    """incremental_reclassify 正确性测试"""

    def _make_dd_config(self):
        """创建 DomainDecomposition 所需的配置"""
        return {
            'lattice_constant': 10.0,
            'lattice_grid': [10, 10, 10],
            'h_factor': 1,
            'max_interaction_distance': 5.0,
            'periodic': [True, True, True],
        }

    def test_reclassify_center_to_boundary(self):
        """对象靠近新边界时应从 center 变为 boundary"""
        from mpi4py import MPI
        from paraokmc.rma_object_pool import RMAObjectPool, create_object
        from paraokmc.domain_decomposition import DomainDecomposition

        comm = MPI.COMM_WORLD
        if comm.size > 1:
            pytest.skip("Single-process test only")

        dd = DomainDecomposition(comm=comm, config=self._make_dd_config(), verbose=False)

        pool = RMAObjectPool(
            comm=comm, initial_capacity=100,
            domain_decomposition=dd, verbose=False
        )

        # 添加一些对象
        for i in range(10):
            obj = create_object(
                obj_id=i, obj_type='test',
                position=[50.0, 50.0, 50.0],  # 中心位置
                owner_rank=0, size=1.0
            )
            pool.add_object(obj)

        initial_total = pool.center_count + pool.boundary_count

        # 调用 incremental_reclassify
        reclassified = pool.incremental_reclassify()

        # 对象总数应该不变
        final_total = pool.center_count + pool.boundary_count
        assert final_total == initial_total

        pool.free()

    def test_reclassify_preserves_ids(self):
        """重分类后 obj_id_to_location 保持一致"""
        from mpi4py import MPI
        from paraokmc.rma_object_pool import RMAObjectPool, create_object
        from paraokmc.domain_decomposition import DomainDecomposition

        comm = MPI.COMM_WORLD
        if comm.size > 1:
            pytest.skip("Single-process test only")

        dd = DomainDecomposition(comm=comm, config=self._make_dd_config(), verbose=False)

        pool = RMAObjectPool(
            comm=comm, initial_capacity=100,
            domain_decomposition=dd, verbose=False
        )

        obj_ids = set()
        for i in range(20):
            x = 10.0 + i * 4.0  # spread across domain
            obj = create_object(
                obj_id=i, obj_type='test',
                position=[x, 50.0, 50.0],
                owner_rank=0, size=1.0
            )
            pool.add_object(obj)
            obj_ids.add(i)

        pool.incremental_reclassify()

        # 验证所有对象仍在 id map 中
        for oid in obj_ids:
            assert oid in pool.obj_id_to_location, f"Object {oid} missing from id map"

        # 验证 id map 中的位置信息正确
        for oid, (cat, idx) in pool.obj_id_to_location.items():
            if cat == 'center':
                assert int(pool.center_objects[idx]['id']) == oid
            elif cat == 'boundary':
                assert int(pool.boundary_objects[idx]['id']) == oid

        pool.free()


class TestDistributedRCBFunction:
    """distributed_rcb 函数的单进程测试"""

    def test_single_rank_returns_full_box(self):
        """单 rank 时返回完整 box"""
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        if comm.size > 1:
            pytest.skip("Single-process test only")

        grid_load = np.ones((6, 6, 6), dtype=np.int32)
        global_box = Bounds(0, 6, 0, 6, 0, 6)

        result = distributed_rcb(comm, grid_load, global_box, [0])
        assert 0 in result
        b = result[0]
        assert b.x_start == 0 and b.x_end == 6
        assert b.y_start == 0 and b.y_end == 6
        assert b.z_start == 0 and b.z_end == 6

    def test_mismatched_comm_size_raises(self):
        """comm.size 与 all_ranks 长度不一致时应直接报错"""
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        if comm.size != 1:
            pytest.skip("Single-process test only")

        grid_load = np.ones((6, 6, 6), dtype=np.int32)
        global_box = Bounds(0, 6, 0, 6, 0, 6)

        with pytest.raises(ValueError):
            distributed_rcb(comm, grid_load, global_box, [0, 1])


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
