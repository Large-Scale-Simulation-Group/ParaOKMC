"""
JIT Event Builder 测试

测试目标：
1. jit_compute_distances() 正确性
2. jit_filter_pairs() 正确性（对比 NumPy 参考实现）
3. 完整 build_pair_events() JIT vs 非 JIT 结果一致性
4. 性能基准测试

用法：
    python test_jit_event_builder.py               # 单进程
    mpirun -np 2 python test_jit_event_builder.py  # MPI
"""

import numpy as np
import time
import sys

from mpi4py import MPI

from paraokmc.jit_event_builder import (
    jit_compute_distances, jit_filter_pairs, jit_filter_pairs_lut,
    jit_embed_objects_flat,
    build_type_id_array, build_allowed_pairs_table, build_type_to_int,
    build_match_tables,
    NUMBA_AVAILABLE,
)

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
# Test 1: jit_compute_distances 正确性
# ============================================================================

def test_compute_distances():
    if rank == 0:
        print("\n[Test 1] jit_compute_distances correctness")

    N = 5000
    np.random.seed(42 + rank)
    x1, y1, z1 = np.random.rand(N), np.random.rand(N), np.random.rand(N)
    x2, y2, z2 = np.random.rand(N), np.random.rand(N), np.random.rand(N)
    ids1 = np.random.randint(0, 100000, N, dtype=np.int64)
    ids2 = np.random.randint(0, 100000, N, dtype=np.int64)

    out_d = np.empty(N, dtype=np.float64)
    out_min = np.empty(N, dtype=np.int64)
    out_max = np.empty(N, dtype=np.int64)

    jit_compute_distances(x1, y1, z1, x2, y2, z2, ids1, ids2,
                          out_d, out_min, out_max)

    ref_d = np.sqrt((x1 - x2)**2 + (y1 - y2)**2 + (z1 - z2)**2)
    ref_min = np.minimum(ids1, ids2)
    ref_max = np.maximum(ids1, ids2)

    report("distances match", np.allclose(out_d, ref_d, atol=1e-12))
    report("min_ids match", np.array_equal(out_min, ref_min))
    report("max_ids match", np.array_equal(out_max, ref_max))

    # Edge case: identical IDs
    ids_same = np.arange(N, dtype=np.int64)
    jit_compute_distances(x1, y1, z1, x2, y2, z2, ids_same, ids_same,
                          out_d, out_min, out_max)
    report("identical IDs", np.array_equal(out_min, out_max))


# ============================================================================
# Test 2: jit_filter_pairs 正确性
# ============================================================================

def test_filter_pairs():
    if rank == 0:
        print("\n[Test 2] jit_filter_pairs correctness")

    N = 8000
    np.random.seed(123 + rank)

    # 3 types: H=0, V=1, SIA=2
    type_ids1 = np.random.randint(0, 3, N, dtype=np.int32)
    type_ids2 = np.random.randint(0, 3, N, dtype=np.int32)
    distances = np.random.rand(N) * 10.0
    owner_ids1 = np.full(N, rank, dtype=np.int32)
    owner_ids2 = np.full(N, rank, dtype=np.int32)
    obj_ids1 = np.arange(N, dtype=np.int64)
    obj_ids2 = np.arange(N, 2 * N, dtype=np.int64)

    # 2 event types:
    # capture (0): H-V, max_dist=5.0
    # recomb (1): SIA-V, max_dist=3.0
    allowed = np.array([[0, 1], [1, 2]], dtype=np.int32)
    max_ds = np.array([5.0, 3.0], dtype=np.float64)
    etype_for = np.array([0, 1], dtype=np.int32)

    out_et = np.empty(N, dtype=np.int32)
    jit_filter_pairs(type_ids1, type_ids2, distances,
                     owner_ids1, owner_ids2, obj_ids1, obj_ids2,
                     allowed, max_ds, etype_for,
                     np.int32(rank), out_et)

    # NumPy reference for capture (H-V, dist <= 5.0)
    hv_mask = (((type_ids1 == 0) & (type_ids2 == 1)) |
               ((type_ids1 == 1) & (type_ids2 == 0))) & (distances <= 5.0)
    # NumPy reference for recomb (SIA-V, dist <= 3.0)
    sv_mask = (((type_ids1 == 1) & (type_ids2 == 2)) |
               ((type_ids1 == 2) & (type_ids2 == 1))) & (distances <= 3.0)

    jit_capture = (out_et == 0)
    jit_recomb = (out_et == 1)

    # Note: if a pair matches both (impossible here since type pairs are disjoint),
    # JIT picks the first match. For disjoint pairs, counts should match exactly.
    report("capture count",
           jit_capture.sum() == hv_mask.sum(),
           f"(JIT={jit_capture.sum()}, ref={hv_mask.sum()})")
    report("recomb count",
           jit_recomb.sum() == sv_mask.sum(),
           f"(JIT={jit_recomb.sum()}, ref={sv_mask.sum()})")

    # Ownership test: set some owners to a different rank
    owner_ids1_mixed = np.copy(owner_ids1)
    owner_ids1_mixed[:N // 2] = (rank + 1) % max(size, 2)
    jit_filter_pairs(type_ids1, type_ids2, distances,
                     owner_ids1_mixed, owner_ids2, obj_ids1, obj_ids2,
                     allowed, max_ds, etype_for,
                     np.int32(rank), out_et)

    # obj_ids1 < obj_ids2 always, so min_id owner = owner_ids1
    # First half has wrong owner → should be filtered out
    first_half_matched = (out_et[:N // 2] >= 0).sum()
    report("ownership filter", first_half_matched == 0,
           f"(first half matched: {first_half_matched}, expected 0)")


# ============================================================================
# Test 3: build_pair_events JIT vs non-JIT 一致性
# ============================================================================

def test_build_pair_events_consistency():
    if rank == 0:
        print("\n[Test 3] build_pair_events JIT builder vs NumPy reference")

    from scipy.spatial import cKDTree
    from paraokmc.vectorized_event_builder import VectorizedPairEventBuilder
    from paraokmc.rma_event_pool import create_default_event_dtype

    # 构造最小 mock 对象
    class MockDomain:
        def __init__(self):
            self.comm = comm
            self.rank = rank
            self.size = size
            self.max_interaction_distance = 5.0

    obj_dtype = np.dtype([
        ('id', np.int64), ('type', 'U16'),
        ('x', np.float64), ('y', np.float64), ('z', np.float64),
        ('owner_rank', np.int32), ('active', np.bool_),
    ])
    event_dtype = create_default_event_dtype()

    # 生成对象
    N = 300
    np.random.seed(77 + rank)
    objs = np.zeros(N, dtype=obj_dtype)
    objs['id'] = np.arange(rank * N, (rank + 1) * N)
    objs['type'] = np.where(np.random.rand(N) < 0.5, 'H', 'V')
    objs['x'] = np.random.rand(N) * 30
    objs['y'] = np.random.rand(N) * 30
    objs['z'] = np.random.rand(N) * 30
    objs['owner_rank'] = rank
    objs['active'] = True

    # Mock RMA pool
    class MockPool:
        def __init__(self, objs):
            self._objs = objs
            self.large_count = 0
        def get_local_objects(self):
            return self._objs
        def get_ghost_objects(self):
            return np.array([], dtype=obj_dtype)

    pool = MockPool(objs)

    config_base = {
        'pair_type_filters': {'capture': lambda o1, o2, d, c, r: True},
        'pair_param_extractors': {},
        'event_weights': {'capture': 1.0},
        '_event_name_to_int': {'capture': 0},
        'vectorized_pair_filters': {
            'capture': {
                'type_pairs': [('H', 'V')],
                'max_distance': 5.0,
            }
        },
    }

    # Run JIT builder
    builder = VectorizedPairEventBuilder(
        MockDomain(), config_base, obj_dtype, event_dtype)
    events_jit = builder.build_pair_events(pool, current_step=1)

    # NumPy reference: manually compute expected H-V pairs within distance
    positions = np.column_stack([objs['x'], objs['y'], objs['z']])
    tree = cKDTree(positions)
    pairs_set = tree.query_pairs(r=5.0, output_type='set')
    if len(pairs_set) > 0:
        pairs_arr = np.array(list(pairs_set), dtype=np.int64)
        i1, i2 = pairs_arr[:, 0], pairs_arr[:, 1]
        t1 = objs['type'][i1]
        t2 = objs['type'][i2]
        id1 = objs['id'][i1]
        id2 = objs['id'][i2]
        # H-V or V-H type match
        hv_mask = ((t1 == 'H') & (t2 == 'V')) | ((t1 == 'V') & (t2 == 'H'))
        # Ownership: min_id owner == rank
        min_id_owner = np.where(id1 < id2,
                                objs['owner_rank'][i1],
                                objs['owner_rank'][i2])
        own_mask = (min_id_owner == rank)
        n_ref = int((hv_mask & own_mask).sum())
    else:
        n_ref = 0

    n_jit = len(events_jit)
    report("event count matches NumPy ref", n_jit == n_ref,
           f"(JIT={n_jit}, ref={n_ref})")

    if n_jit > 0:
        # Verify all events are capture type
        all_capture = np.all(events_jit['event_type_id'] == 0)
        report("all events are capture type", all_capture)

        # Verify obj_ids are properly ordered (min, max)
        ids_ordered = np.all(events_jit['obj_ids'][:, 0] < events_jit['obj_ids'][:, 1])
        report("obj_ids ordered (min < max)", ids_ordered)


# ============================================================================
# Test 4: 性能基准
# ============================================================================

def test_performance():
    if rank == 0:
        print("\n[Test 4] Performance benchmark")

    N = 50000
    np.random.seed(42)
    x1, y1, z1 = np.random.rand(N), np.random.rand(N), np.random.rand(N)
    x2, y2, z2 = np.random.rand(N), np.random.rand(N), np.random.rand(N)
    ids1 = np.arange(N, dtype=np.int64)
    ids2 = np.arange(N, 2 * N, dtype=np.int64)

    out_d = np.empty(N, dtype=np.float64)
    out_min = np.empty(N, dtype=np.int64)
    out_max = np.empty(N, dtype=np.int64)

    # Warmup
    jit_compute_distances(x1, y1, z1, x2, y2, z2, ids1, ids2,
                          out_d, out_min, out_max)

    # JIT timing
    iters = 50
    t0 = time.perf_counter()
    for _ in range(iters):
        jit_compute_distances(x1, y1, z1, x2, y2, z2, ids1, ids2,
                              out_d, out_min, out_max)
    t_jit = (time.perf_counter() - t0) / iters

    # NumPy timing
    t0 = time.perf_counter()
    for _ in range(iters):
        _ = np.sqrt((x1 - x2)**2 + (y1 - y2)**2 + (z1 - z2)**2)
        _ = np.minimum(ids1, ids2)
        _ = np.maximum(ids1, ids2)
    t_np = (time.perf_counter() - t0) / iters

    if rank == 0:
        speedup = t_np / t_jit if t_jit > 0 else float('inf')
        print(f"  compute_distances (N={N}):")
        print(f"    JIT:   {t_jit*1e3:.3f} ms")
        print(f"    NumPy: {t_np*1e3:.3f} ms")
        print(f"    Speedup: {speedup:.1f}x")

    # Filter benchmark (LUT version)
    type_ids1 = np.random.randint(0, 3, N, dtype=np.int32)
    type_ids2 = np.random.randint(0, 3, N, dtype=np.int32)
    distances = np.random.rand(N) * 10.0
    owner_ids1 = np.zeros(N, dtype=np.int32)
    owner_ids2 = np.zeros(N, dtype=np.int32)

    # Build LUT: H=0, SIA=1, V=2
    vf = {'capture': {'type_pairs': [('H', 'V')], 'max_distance': 5.0},
          'recomb': {'type_pairs': [('SIA', 'V')], 'max_distance': 3.0}}
    t2i = {'H': 0, 'SIA': 1, 'V': 2}
    et2i = {'capture': 0, 'recomb': 1}
    mt, mdt = build_match_tables(vf, t2i, et2i)
    out_et = np.empty(N, dtype=np.int32)

    # Warmup
    jit_filter_pairs_lut(type_ids1, type_ids2, distances,
                         owner_ids1, owner_ids2, ids1, ids2,
                         mt, mdt, np.int32(0), out_et)

    t0 = time.perf_counter()
    for _ in range(iters):
        jit_filter_pairs_lut(type_ids1, type_ids2, distances,
                             owner_ids1, owner_ids2, ids1, ids2,
                             mt, mdt, np.int32(0), out_et)
    t_jit_f = (time.perf_counter() - t0) / iters

    # NumPy reference (2 event types + ownership, separate passes)
    t0 = time.perf_counter()
    for _ in range(iters):
        hv = (((type_ids1 == 0) & (type_ids2 == 2)) |
              ((type_ids1 == 2) & (type_ids2 == 0))) & (distances <= 5.0)
        sv = (((type_ids1 == 1) & (type_ids2 == 2)) |
              ((type_ids1 == 2) & (type_ids2 == 1))) & (distances <= 3.0)
    t_np_f = (time.perf_counter() - t0) / iters

    if rank == 0:
        speedup_f = t_np_f / t_jit_f if t_jit_f > 0 else float('inf')
        print(f"  filter_pairs_lut (N={N}, 2 event types):")
        print(f"    JIT LUT: {t_jit_f*1e3:.3f} ms")
        print(f"    NumPy:   {t_np_f*1e3:.3f} ms")
        print(f"    Speedup: {speedup_f:.1f}x")

    report("benchmark completed", True)


# ============================================================================
# Main
# ============================================================================

if __name__ == '__main__':
    if rank == 0:
        print("=" * 60)
        print("JIT Event Builder Tests")
        print(f"Numba available: {NUMBA_AVAILABLE}")
        print(f"MPI: {size} ranks")
        print("=" * 60)

    if not NUMBA_AVAILABLE:
        if rank == 0:
            print("[SKIP] Numba not available")
        sys.exit(0)

    test_compute_distances()
    test_filter_pairs()
    test_build_pair_events_consistency()
    test_performance()

    comm.Barrier()

    if rank == 0:
        print("\n" + "=" * 60)
        print(f"Results: {passed} passed, {failed} failed")
        print("=" * 60)

    if failed > 0:
        sys.exit(1)
