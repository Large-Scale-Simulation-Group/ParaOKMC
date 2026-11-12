"""
Cell List vs cKDTree 正确性 + 性能对比测试

测试：
1. 正确性：cell list 路径 vs cKDTree 路径产生完全一致的 pair events
2. 性能：两条路径的 build_pair_events 耗时对比
3. JIT 内核单元测试：build_neighbor_map, build_cell_bins, jit_find_pairs_cell_list

用法：
    python tests/test_cell_list.py                    # 单进程
    mpirun -np 4 python tests/test_cell_list.py       # 4 ranks
"""

import numpy as np
import time
import sys

from mpi4py import MPI

from paraokmc.jit_cell_list import (
    build_neighbor_map, build_cell_bins, jit_find_pairs_cell_list,
)
from paraokmc.jit_event_builder import (
    build_match_tables, NUMBA_AVAILABLE,
)
from paraokmc.vectorized_event_builder import VectorizedPairEventBuilder
from paraokmc.rma_event_pool import create_default_event_dtype

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

passed = 0
failed = 0


def report(name, ok, detail=""):
    global passed, failed
    if ok:
        passed += 1
        if rank == 0:
            print(f"  [PASS] {name} {detail}")
    else:
        failed += 1
        if rank == 0:
            print(f"  [FAIL] {name} {detail}")


# ============================================================================
# Test 1: build_neighbor_map 单元测试
# ============================================================================

def test_neighbor_map():
    if rank == 0:
        print("\n[Test 1] build_neighbor_map")

    # 3x3x3 网格，全周期，cell_size=10, max_dist=10
    # 每个 cell 应该有 27 个邻居（含自身）
    nmap, ncounts = build_neighbor_map(3, 3, 3, 10.0, 10.0, 10.0, 10.0,
                                        True, True, True)
    report("3x3x3 PBC: all cells have 27 neighbors",
           np.all(ncounts == 27),
           f"(min={ncounts.min()}, max={ncounts.max()})")

    # 3x3x3 非周期，角落 cell 应该有 8 个邻居
    nmap2, ncounts2 = build_neighbor_map(3, 3, 3, 10.0, 10.0, 10.0, 10.0,
                                          False, False, False)
    # cell (0,0,0) = id 0
    report("3x3x3 no-PBC: corner cell has 8 neighbors",
           ncounts2[0] == 8,
           f"(got {ncounts2[0]})")
    # cell (1,1,1) = id 13, center should have 27
    center_id = 1 + 1*3 + 1*3*3
    report("3x3x3 no-PBC: center cell has 27 neighbors",
           ncounts2[center_id] == 27,
           f"(got {ncounts2[center_id]})")

    # 6x6x6 网格，cell_size=25, max_dist=25 → rx=ry=rz=1 → 27 neighbors
    nmap3, ncounts3 = build_neighbor_map(6, 6, 6, 25.0, 25.0, 25.0, 25.0,
                                          True, True, True)
    report("6x6x6 PBC: all cells have 27 neighbors",
           np.all(ncounts3 == 27),
           f"(min={ncounts3.min()}, max={ncounts3.max()})")


# ============================================================================
# Test 2: build_cell_bins 单元测试
# ============================================================================

def test_cell_bins():
    if rank == 0:
        print("\n[Test 2] build_cell_bins")

    N = 100
    np.random.seed(42)
    box = 150.0
    x = np.random.rand(N) * box
    y = np.random.rand(N) * box
    z = np.random.rand(N) * box

    Gx, Gy, Gz = 6, 6, 6
    cs = box / 6.0

    heads, nexts = build_cell_bins(x, y, z, N, cs, cs, cs,
                                    Gx, Gy, Gz, box, box, box,
                                    True, True, True)

    # 验证：遍历所有 cell，统计总对象数应等于 N
    total = 0
    for c in range(Gx * Gy * Gz):
        i = heads[c]
        while i >= 0:
            total += 1
            i = nexts[i]
    report("total objects in bins == N", total == N,
           f"(got {total}, expected {N})")

    # 验证：每个对象确实在正确的 cell 中
    correct = 0
    for c in range(Gx * Gy * Gz):
        iz_c = c // (Gx * Gy)
        iy_c = (c % (Gx * Gy)) // Gx
        ix_c = c % Gx
        i = heads[c]
        while i >= 0:
            ex_ix = int(np.floor((x[i] % box) / cs))
            ex_iy = int(np.floor((y[i] % box) / cs))
            ex_iz = int(np.floor((z[i] % box) / cs))
            ex_ix = min(ex_ix, Gx - 1)
            ex_iy = min(ex_iy, Gy - 1)
            ex_iz = min(ex_iz, Gz - 1)
            if ix_c == ex_ix and iy_c == ex_iy and iz_c == ex_iz:
                correct += 1
            i = nexts[i]
    report("all objects in correct cell", correct == N,
           f"(correct={correct}/{N})")


# ============================================================================
# Test 3: jit_find_pairs_cell_list vs cKDTree+filter 正确性
# ============================================================================

def test_cell_list_vs_kdtree():
    if rank == 0:
        print("\n[Test 3] jit_find_pairs_cell_list vs cKDTree+filter")

    from scipy.spatial import cKDTree
    from paraokmc.jit_event_builder import jit_compute_distances_pbc, jit_filter_pairs_lut

    N = 160
    np.random.seed(77 + rank)
    box = 150.0
    max_dist = 25.0

    x = np.random.rand(N).astype(np.float64) * box
    y = np.random.rand(N).astype(np.float64) * box
    z = np.random.rand(N).astype(np.float64) * box
    obj_ids = np.arange(rank * N, (rank + 1) * N, dtype=np.int64)
    # 2 types: H=0, V=1
    type_ids = np.random.randint(0, 2, N, dtype=np.int32)
    owner_ranks = np.full(N, rank, dtype=np.int32)

    # match table: H-V → event 0
    vf = {'capture': {'type_pairs': [('H', 'V')], 'max_distance': max_dist}}
    t2i = {'H': 0, 'V': 1}
    et2i = {'capture': 0}
    match_table, max_dist_table = build_match_tables(vf, t2i, et2i)

    # --- Cell list path ---
    Gx = max(1, int(np.floor(box / max_dist)))
    Gy, Gz = Gx, Gx
    cs = box / Gx

    nmap, ncounts = build_neighbor_map(Gx, Gy, Gz, max_dist, cs, cs, cs,
                                        True, True, True)
    heads, nexts = build_cell_bins(x, y, z, N, cs, cs, cs,
                                    Gx, Gy, Gz, box, box, box,
                                    True, True, True)

    max_pairs = N * (N - 1) // 2
    out_idx1 = np.empty(max_pairs, dtype=np.int64)
    out_idx2 = np.empty(max_pairs, dtype=np.int64)
    out_min_ids = np.empty(max_pairs, dtype=np.int64)
    out_max_ids = np.empty(max_pairs, dtype=np.int64)
    out_owners = np.empty(max_pairs, dtype=np.int32)
    out_distances = np.empty(max_pairs, dtype=np.float64)
    out_event_types = np.empty(max_pairs, dtype=np.int32)

    n_found = jit_find_pairs_cell_list(
        x, y, z, obj_ids, type_ids, owner_ranks, N,
        heads, nexts, nmap, ncounts,
        match_table, max_dist_table,
        box, box, box, True, True, True,
        np.int32(rank),
        out_idx1, out_idx2, out_min_ids, out_max_ids,
        out_owners, out_distances, out_event_types, max_pairs
    )

    # 收集 cell list 结果：(min_id, max_id) 集合
    cl_pairs = set()
    for k in range(n_found):
        cl_pairs.add((int(out_min_ids[k]), int(out_max_ids[k])))

    # --- cKDTree path（周期感知：boxsize 环面拓扑 + 最小镜像距离）---
    positions = np.column_stack([x, y, z]) % box
    tree = cKDTree(positions, boxsize=box)
    pairs_set = tree.query_pairs(r=max_dist, output_type='set')

    if len(pairs_set) > 0:
        pairs_arr = np.array(list(pairs_set), dtype=np.int64)
        i1, i2 = pairs_arr[:, 0], pairs_arr[:, 1]

        ids1 = obj_ids[i1]
        ids2 = obj_ids[i2]
        owners1 = owner_ranks[i1]
        owners2 = owner_ranks[i2]
        tids1 = type_ids[i1]
        tids2 = type_ids[i2]

        dists = np.empty(len(pairs_arr), dtype=np.float64)
        min_ids_ref = np.empty(len(pairs_arr), dtype=np.int64)
        max_ids_ref = np.empty(len(pairs_arr), dtype=np.int64)
        jit_compute_distances_pbc(x[i1], y[i1], z[i1], x[i2], y[i2], z[i2],
                                  ids1, ids2, box, box, box, True, True, True,
                                  dists, min_ids_ref, max_ids_ref)

        out_et_ref = np.empty(len(pairs_arr), dtype=np.int32)
        jit_filter_pairs_lut(tids1, tids2, dists,
                             owners1, owners2, ids1, ids2,
                             match_table, max_dist_table,
                             np.int32(rank), out_et_ref)

        kd_pairs = set()
        for k in range(len(pairs_arr)):
            if out_et_ref[k] >= 0:
                kd_pairs.add((int(min_ids_ref[k]), int(max_ids_ref[k])))
    else:
        kd_pairs = set()

    report("pair count matches",
           len(cl_pairs) == len(kd_pairs),
           f"(cell_list={len(cl_pairs)}, kdtree={len(kd_pairs)})")

    # 集合完全一致
    missing = kd_pairs - cl_pairs
    extra = cl_pairs - kd_pairs
    report("pair sets identical",
           len(missing) == 0 and len(extra) == 0,
           f"(missing={len(missing)}, extra={len(extra)})")

    if len(missing) > 0 and rank == 0:
        print(f"    First 5 missing: {list(missing)[:5]}")
    if len(extra) > 0 and rank == 0:
        print(f"    First 5 extra: {list(extra)[:5]}")


# ============================================================================
# Test 4: 完整 build_pair_events 两条路径对比
# ============================================================================

def test_full_builder_consistency():
    if rank == 0:
        print("\n[Test 4] Full build_pair_events: cell_list vs cKDTree")

    from paraokmc.grid_system import GridSystem

    obj_dtype = np.dtype([
        ('id', np.int64), ('type', 'U16'), ('type_id', np.int32),
        ('x', np.float64), ('y', np.float64), ('z', np.float64),
        ('owner_rank', np.int32), ('active', np.bool_),
    ])
    event_dtype = create_default_event_dtype()

    N = 160
    np.random.seed(42 + rank)
    box = 150.0
    max_dist = 25.0

    objs = np.zeros(N, dtype=obj_dtype)
    objs['id'] = np.arange(rank * N, (rank + 1) * N)
    types = np.where(np.random.rand(N) < 0.5, 'H', 'V')
    objs['type'] = types
    objs['type_id'] = np.where(types == 'H', 0, 1)
    objs['x'] = np.random.rand(N) * box
    objs['y'] = np.random.rand(N) * box
    objs['z'] = np.random.rand(N) * box
    objs['owner_rank'] = rank
    objs['active'] = True

    class MockGridSystem:
        def __init__(self):
            self.periodic = [True, True, True]

    class MockDomain:
        def __init__(self):
            self.comm = comm
            self.rank = rank
            self.size = size
            self.max_interaction_distance = max_dist
            self.box_size = np.array([box, box, box])
            self.grid_system = MockGridSystem()

    class MockPool:
        def __init__(self, objs):
            self._objs = objs
            self.large_count = 0
        def get_local_objects(self):
            return self._objs
        def get_ghost_objects(self):
            return np.array([], dtype=obj_dtype)

    pool = MockPool(objs)
    config = {
        'pair_type_filters': {'capture': None},
        'pair_param_extractors': {},
        'event_weights': {'capture': 1.0},
        'vectorized_pair_filters': {
            'capture': {
                'type_pairs': [('H', 'V')],
                'max_distance': max_dist,
            }
        },
        '_type_to_int': {'H': 0, 'V': 1},
    }

    # Cell list path
    builder_cl = VectorizedPairEventBuilder(
        MockDomain(), config, obj_dtype, event_dtype,
        use_cell_list=True)
    events_cl = builder_cl.build_pair_events(pool, current_step=1)

    # cKDTree path
    builder_kd = VectorizedPairEventBuilder(
        MockDomain(), config, obj_dtype, event_dtype,
        use_cell_list=False)
    events_kd = builder_kd.build_pair_events(pool, current_step=1)

    report("event count matches",
           len(events_cl) == len(events_kd),
           f"(cell_list={len(events_cl)}, kdtree={len(events_kd)})")

    if len(events_cl) > 0 and len(events_kd) > 0:
        # 比较 (min_id, max_id) 集合
        cl_set = set(zip(events_cl['obj_ids'][:, 0].tolist(),
                         events_cl['obj_ids'][:, 1].tolist()))
        kd_set = set(zip(events_kd['obj_ids'][:, 0].tolist(),
                         events_kd['obj_ids'][:, 1].tolist()))
        report("pair ID sets identical",
               cl_set == kd_set,
               f"(diff={len(cl_set ^ kd_set)})")


# ============================================================================
# Test 5: 性能基准
# ============================================================================

def test_performance():
    if rank == 0:
        print("\n[Test 5] Performance benchmark")

    from paraokmc.grid_system import GridSystem

    obj_dtype = np.dtype([
        ('id', np.int64), ('type', 'U16'), ('type_id', np.int32),
        ('x', np.float64), ('y', np.float64), ('z', np.float64),
        ('owner_rank', np.int32), ('active', np.bool_),
    ])
    event_dtype = create_default_event_dtype()

    N = 160
    box = 150.0
    max_dist = 25.0

    class MockGridSystem:
        def __init__(self):
            self.periodic = [True, True, True]

    class MockDomain:
        def __init__(self):
            self.comm = comm
            self.rank = rank
            self.size = size
            self.max_interaction_distance = max_dist
            self.box_size = np.array([box, box, box])
            self.grid_system = MockGridSystem()

    config = {
        'pair_type_filters': {'capture': None},
        'pair_param_extractors': {},
        'event_weights': {'capture': 1.0},
        'vectorized_pair_filters': {
            'capture': {
                'type_pairs': [('H', 'V')],
                'max_distance': max_dist,
            }
        },
        '_type_to_int': {'H': 0, 'V': 1},
    }

    def make_objs(seed):
        np.random.seed(seed)
        objs = np.zeros(N, dtype=obj_dtype)
        objs['id'] = np.arange(rank * N, (rank + 1) * N)
        types = np.where(np.random.rand(N) < 0.5, 'H', 'V')
        objs['type'] = types
        objs['type_id'] = np.where(types == 'H', 0, 1)
        objs['x'] = np.random.rand(N) * box
        objs['y'] = np.random.rand(N) * box
        objs['z'] = np.random.rand(N) * box
        objs['owner_rank'] = rank
        objs['active'] = True
        return objs

    class MockPool:
        def __init__(self, objs):
            self._objs = objs
            self.large_count = 0
        def get_local_objects(self):
            return self._objs
        def get_ghost_objects(self):
            return np.array([], dtype=obj_dtype)

    # Warmup (JIT compilation)
    objs_warmup = make_objs(999)
    pool_warmup = MockPool(objs_warmup)
    builder_cl = VectorizedPairEventBuilder(
        MockDomain(), config, obj_dtype, event_dtype, use_cell_list=True)
    builder_kd = VectorizedPairEventBuilder(
        MockDomain(), config, obj_dtype, event_dtype, use_cell_list=False)
    builder_cl.build_pair_events(pool_warmup, current_step=0)
    builder_kd.build_pair_events(pool_warmup, current_step=0)

    # Benchmark
    iters = 50
    seeds = [42 + i for i in range(iters)]

    # Cell list
    t0 = time.perf_counter()
    for i in range(iters):
        objs = make_objs(seeds[i])
        pool = MockPool(objs)
        builder_cl.build_pair_events(pool, current_step=i)
    t_cl = (time.perf_counter() - t0) / iters

    # cKDTree
    t0 = time.perf_counter()
    for i in range(iters):
        objs = make_objs(seeds[i])
        pool = MockPool(objs)
        builder_kd.build_pair_events(pool, current_step=i)
    t_kd = (time.perf_counter() - t0) / iters

    if rank == 0:
        speedup = t_kd / t_cl if t_cl > 0 else float('inf')
        print(f"  N={N} objects, max_dist={max_dist}, box={box}")
        print(f"  Cell list:  {t_cl*1e3:.3f} ms/step")
        print(f"  cKDTree:    {t_kd*1e3:.3f} ms/step")
        print(f"  Speedup:    {speedup:.1f}x")

    report("benchmark completed", True)


# ============================================================================
# Main
# ============================================================================

if __name__ == '__main__':
    if rank == 0:
        print("=" * 60)
        print("Cell List vs cKDTree Tests")
        print(f"Numba available: {NUMBA_AVAILABLE}")
        print(f"MPI: {size} ranks")
        print("=" * 60)

    if not NUMBA_AVAILABLE:
        if rank == 0:
            print("[SKIP] Numba not available")
        sys.exit(0)

    test_neighbor_map()
    test_cell_bins()
    test_cell_list_vs_kdtree()
    test_full_builder_consistency()
    test_performance()

    comm.Barrier()

    if rank == 0:
        print("\n" + "=" * 60)
        print(f"Results: {passed} passed, {failed} failed")
        print("=" * 60)

    if failed > 0:
        sys.exit(1)
