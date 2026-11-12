# Copyright (c) 2026 Beihang University
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
JIT event builder - Numba + OpenMP prange accelerated event construction

Three JIT kernels:
1. jit_compute_distances()  - fused distance computation + ID normalization
2. jit_filter_pairs()       - fused type, distance and ownership filtering
3. jit_embed_objects()      - parallelized object embedding

Helpers:
- build_allowed_pairs_table()  - build the (K, 2) int32 lookup table
- build_type_id_array()        - type string -> int32 array
- build_max_distance_table()   - build the per-event-type maximum distance table

Performance: avoids Python loops and NumPy temporaries; prange gives multithreading
"""

import numpy as np
from typing import Dict, List, Tuple, Optional

try:
    from numba import njit, prange, types
    from numba.typed import Dict as NumbaDict
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False

    def njit(*args, **kwargs):
        if args and callable(args[0]):
            return args[0]
        def wrapper(fn):
            return fn
        return wrapper

    def prange(*args):
        return range(*args)


# ============================================================================
# JIT kernel 1: distance computation + ID normalization
# ============================================================================

@njit(parallel=True)
def jit_compute_distances(x1, y1, z1, x2, y2, z2, ids1, ids2,
                          out_distances, out_min_ids, out_max_ids):
    """Fused distance computation + ID normalization kernel

    A single prange loop performs:
    - the Euclidean distance (avoiding three temporary difference arrays plus sqrt)
    - min/max ID normalization (avoiding np.minimum/np.maximum temporaries)
    """
    n = len(x1)
    for i in prange(n):
        dx = x1[i] - x2[i]
        dy = y1[i] - y2[i]
        dz = z1[i] - z2[i]
        out_distances[i] = np.sqrt(dx * dx + dy * dy + dz * dz)
        a, b = ids1[i], ids2[i]
        if a < b:
            out_min_ids[i] = a
            out_max_ids[i] = b
        else:
            out_min_ids[i] = b
            out_max_ids[i] = a


@njit(parallel=True)
def jit_compute_distances_pbc(x1, y1, z1, x2, y2, z2, ids1, ids2,
                              box_x, box_y, box_z,
                              periodic_x, periodic_y, periodic_z,
                              out_distances, out_min_ids, out_max_ids):
    """Fused distance (minimum image along periodic axes) + ID normalization kernel"""
    n = len(x1)
    for i in prange(n):
        dx = x1[i] - x2[i]
        dy = y1[i] - y2[i]
        dz = z1[i] - z2[i]
        if periodic_x:
            dx = dx - box_x * np.rint(dx / box_x)
        if periodic_y:
            dy = dy - box_y * np.rint(dy / box_y)
        if periodic_z:
            dz = dz - box_z * np.rint(dz / box_z)
        out_distances[i] = np.sqrt(dx * dx + dy * dy + dz * dz)
        a, b = ids1[i], ids2[i]
        if a < b:
            out_min_ids[i] = a
            out_max_ids[i] = b
        else:
            out_min_ids[i] = b
            out_max_ids[i] = a


# ============================================================================
# JIT kernel 2: unified filter kernel
# ============================================================================

@njit(parallel=True)
def jit_filter_pairs(type_ids1, type_ids2, distances,
                     owner_ids1, owner_ids2, obj_ids1, obj_ids2,
                     allowed_pairs, max_dists, event_type_for_pair,
                     my_rank, out_event_types):
    """Unified filter kernel - type, distance and ownership in one pass (inner-loop version)

    Used when K (the number of allowed_pairs) is large.
    """
    n = len(type_ids1)
    k = len(allowed_pairs)
    for i in prange(n):
        out_event_types[i] = -1
        t1 = type_ids1[i]
        t2 = type_ids2[i]
        d = distances[i]

        # Ownership check: the owner of min_id must be my_rank
        if obj_ids1[i] <= obj_ids2[i]:
            owner = owner_ids1[i]
        else:
            owner = owner_ids2[i]
        if owner != my_rank:
            continue

        # Walk all allowed type pairs
        for j in range(k):
            pa = allowed_pairs[j, 0]
            pb = allowed_pairs[j, 1]
            md = max_dists[j]
            if d > md:
                continue
            if (t1 == pa and t2 == pb) or (t1 == pb and t2 == pa):
                out_event_types[i] = event_type_for_pair[j]
                break


@njit(parallel=True)
def jit_filter_pairs_lut(type_ids1, type_ids2, distances,
                         owner_ids1, owner_ids2, obj_ids1, obj_ids2,
                         match_table, max_dist_table,
                         my_rank, out_event_types):
    """Unified filter kernel - 2D lookup table version (O(1) type matching)

    Args:
        match_table: (n_types, n_types) int32, match_table[t1][t2] = event_type_id, -1 = no match
        max_dist_table: (n_types, n_types) float64, maximum distance per type pair
    """
    n = len(type_ids1)
    for i in prange(n):
        out_event_types[i] = -1
        t1 = type_ids1[i]
        t2 = type_ids2[i]

        # [fix] Self-pair check: an object must not pair with itself
        if obj_ids1[i] == obj_ids2[i]:
            continue

        # Ownership check
        if obj_ids1[i] <= obj_ids2[i]:
            owner = owner_ids1[i]
        else:
            owner = owner_ids2[i]
        if owner != my_rank:
            continue

        # O(1) lookup table match
        etype = match_table[t1, t2]
        if etype >= 0 and distances[i] <= max_dist_table[t1, t2]:
            out_event_types[i] = etype


# ============================================================================
# JIT kernel 3: object embedding kernel
# ============================================================================

@njit(parallel=True)
def jit_embed_objects_flat(idx1, idx2, n_events, n_fields,
                           src_data, out_obj1, out_obj2):
    """Object embedding kernel - copies structured array fields column by column in parallel

    src_data: (N_total, n_fields) float64, all numeric object fields flattened
    out_obj1: (n_events, n_fields) float64
    out_obj2: (n_events, n_fields) float64
    """
    for i in prange(n_events):
        i1 = idx1[i]
        i2 = idx2[i]
        for f in range(n_fields):
            out_obj1[i, f] = src_data[i1, f]
            out_obj2[i, f] = src_data[i2, f]


# ============================================================================
# Helpers: building the lookup tables
# ============================================================================

def build_type_to_int(type_strings):
    """Build the type_to_int mapping from a set of type strings

    Args:
        type_strings: an iterable of type strings

    Returns:
        dict: {'H': 0, 'V': 1, ...}
    """
    return {t: i for i, t in enumerate(sorted(set(type_strings)))}


def build_type_id_array(type_str_array, type_to_int):
    """Array of type strings -> int32 array

    Args:
        type_str_array: (N,) array of Unicode strings
        type_to_int: dict, type name -> integer ID

    Returns:
        (N,) int32 array
    """
    n = len(type_str_array)
    out = np.full(n, -1, dtype=np.int32)
    for tname, tid in type_to_int.items():
        mask = (type_str_array == tname)
        out[mask] = tid
    return out


@njit(cache=True)
def _binary_search_int64(sorted_values: np.ndarray, target: np.int64) -> bool:
    """Membership query in a sorted int64 array."""
    left = 0
    right = len(sorted_values) - 1
    while left <= right:
        mid = (left + right) // 2
        value = sorted_values[mid]
        if value == target:
            return True
        if value < target:
            left = mid + 1
        else:
            right = mid - 1
    return False


@njit(parallel=True, cache=True)
def jit_scan_reusable_single_events(prev_obj_ids: np.ndarray,
                                    prev_status: np.ndarray,
                                    valid_obj_ids_sorted: np.ndarray,
                                    changed_obj_ids_sorted: np.ndarray,
                                    status_done: np.int32,
                                    status_sent: np.int32,
                                    out_actions: np.ndarray):
    """Scan the old single events in parallel: drop, reuse as DONE, or reuse as TODO.

    out_actions:
    - 0: drop
    - 1: reuse, keep DONE
    - 2: reuse, but reset to TODO
    """
    n_prev = len(prev_obj_ids)
    for i in prange(n_prev):
        out_actions[i] = 0
        if prev_status[i] == status_sent:
            continue

        obj_id = prev_obj_ids[i]
        if not _binary_search_int64(valid_obj_ids_sorted, obj_id):
            continue
        if _binary_search_int64(changed_obj_ids_sorted, obj_id):
            continue

        if prev_status[i] == status_done:
            out_actions[i] = 1
        else:
            out_actions[i] = 2


def build_allowed_pairs_table(vectorized_filters, type_to_int, event_type_to_int):
    """Build the JIT lookup tables from vectorized_filters

    Args:
        vectorized_filters: {event_name: {'type_pairs': [...], 'max_distance': float}}
        type_to_int: {'H': 0, 'V': 1, ...}
        event_type_to_int: {'capture': 0, 'recombination': 1, ...}

    Returns:
        allowed_pairs: (K, 2) int32
        max_dists: (K,) float64
        event_type_for_pair: (K,) int32
    """
    rows = []
    dists = []
    etypes = []

    for event_name, fconfig in vectorized_filters.items():
        etype_id = event_type_to_int.get(event_name, -1)
        if etype_id < 0:
            continue
        max_d = fconfig.get('max_distance', 1e30)
        for t1, t2 in fconfig.get('type_pairs', []):
            tid1 = type_to_int.get(t1, -1)
            tid2 = type_to_int.get(t2, -1)
            if tid1 < 0 or tid2 < 0:
                continue
            # Normalize: store only (min, max) to avoid duplicates
            a, b = min(tid1, tid2), max(tid1, tid2)
            key = (a, b, etype_id)
            if key not in {(r[0], r[1], e) for r, e in zip(rows, etypes)}:
                rows.append((a, b))
                dists.append(max_d)
                etypes.append(etype_id)

    if not rows:
        return (np.empty((0, 2), dtype=np.int32),
                np.empty(0, dtype=np.float64),
                np.empty(0, dtype=np.int32))

    return (np.array(rows, dtype=np.int32),
            np.array(dists, dtype=np.float64),
            np.array(etypes, dtype=np.int32))


def build_match_tables(vectorized_filters, type_to_int, event_type_to_int):
    """Build the 2D lookup table used by jit_filter_pairs_lut

    Returns:
        match_table: (n_types, n_types) int32, -1 = no match
        max_dist_table: (n_types, n_types) float64
    """
    n_types = max(type_to_int.values()) + 1 if type_to_int else 1
    match_table = np.full((n_types, n_types), -1, dtype=np.int32)
    max_dist_table = np.zeros((n_types, n_types), dtype=np.float64)

    for event_name, fconfig in vectorized_filters.items():
        etype_id = event_type_to_int.get(event_name, -1)
        if etype_id < 0:
            continue
        max_d = fconfig.get('max_distance', 1e30)
        for t1, t2 in fconfig.get('type_pairs', []):
            tid1 = type_to_int.get(t1, -1)
            tid2 = type_to_int.get(t2, -1)
            if tid1 < 0 or tid2 < 0:
                continue
            # Fill symmetrically
            if match_table[tid1, tid2] < 0:
                match_table[tid1, tid2] = etype_id
                max_dist_table[tid1, tid2] = max_d
            if match_table[tid2, tid1] < 0:
                match_table[tid2, tid1] = etype_id
                max_dist_table[tid2, tid1] = max_d

    return match_table, max_dist_table


# ============================================================================
# JIT kernel 4: single-object event type classification
# ============================================================================

@njit(parallel=True)
def jit_filter_single_objects_lut(type_ids, owner_ranks, my_rank,
                                   match_table_2d, out_event_types_2d):
    """Single-object event type classification kernel - 2D lookup table version

    One object type may map to several event types (H may have both migration
    and dissociation). A 2D lookup table gives O(1) type matching.

    Args:
        type_ids: (N,) int32, type_id of each object
        owner_ranks: (N,) int32, owner_rank of each object
        my_rank: int32, this rank
        match_table_2d: (n_types, max_events_per_type) int32
            match_table_2d[type_id, k] = event_type_id, -1 = end
        out_event_types_2d: (N, max_events_per_type) int32
            output: the list of event types for each object
    """
    n = len(type_ids)
    max_k = match_table_2d.shape[1]
    for i in prange(n):
        for k in range(max_k):
            out_event_types_2d[i, k] = -1
        if owner_ranks[i] != my_rank:
            continue
        tid = type_ids[i]
        if tid >= 0 and tid < match_table_2d.shape[0]:
            for k in range(max_k):
                out_event_types_2d[i, k] = match_table_2d[tid, k]


@njit(parallel=True)
def jit_count_single_events_lut(type_ids, owner_ranks, eligible_mask, my_rank,
                                match_table_2d, out_counts):
    """Single-object events, pass one: count how many events each object will produce."""
    n = len(type_ids)
    max_k = match_table_2d.shape[1]
    for i in prange(n):
        count = 0
        if not eligible_mask[i] or owner_ranks[i] != my_rank:
            out_counts[i] = 0
            continue

        tid = type_ids[i]
        if tid >= 0 and tid < match_table_2d.shape[0]:
            for k in range(max_k):
                if match_table_2d[tid, k] < 0:
                    break
                count += 1
        out_counts[i] = count


@njit(parallel=True)
def jit_fill_single_events_lut(type_ids, owner_ranks, eligible_mask, my_rank,
                               match_table_2d, offsets,
                               out_event_types, out_object_indices):
    """Single-object events, pass two: fill the compact output using prefix-sum offsets."""
    n = len(type_ids)
    max_k = match_table_2d.shape[1]
    for i in prange(n):
        if not eligible_mask[i] or owner_ranks[i] != my_rank:
            continue

        tid = type_ids[i]
        if tid < 0 or tid >= match_table_2d.shape[0]:
            continue

        base = offsets[i]
        count = 0
        for k in range(max_k):
            etype = match_table_2d[tid, k]
            if etype < 0:
                break
            out_event_types[base + count] = etype
            out_object_indices[base + count] = i
            count += 1


@njit(parallel=True)
def jit_count_valid_event_rows(event_types, out_counts):
    """Count the valid entries among the candidate pairs (event_type >= 0)."""
    n = len(event_types)
    for i in prange(n):
        out_counts[i] = 1 if event_types[i] >= 0 else 0


@njit(parallel=True)
def jit_compact_pair_rows(event_types, idx1, idx2, min_ids, max_ids, owners, distances,
                          offsets,
                          out_event_types, out_idx1, out_idx2,
                          out_min_ids, out_max_ids, out_owners, out_distances):
    """Compact the pair candidate arrays using the prefix-sum result."""
    n = len(event_types)
    for i in prange(n):
        if event_types[i] < 0:
            continue
        dst = offsets[i]
        out_event_types[dst] = event_types[i]
        out_idx1[dst] = idx1[i]
        out_idx2[dst] = idx2[i]
        out_min_ids[dst] = min_ids[i]
        out_max_ids[dst] = max_ids[i]
        out_owners[dst] = owners[i]
        out_distances[dst] = distances[i]


# ============================================================================
# Helpers: building the single-object event lookup table
# ============================================================================

# ============================================================================
# JIT kernel 5: per-cell zero-object event filling
# ============================================================================

@njit(parallel=True)
def jit_build_per_cell_zero_events(
    cell_ids,           # (N_cells,) int64 — cell ID
    cell_x, cell_y, cell_z,  # (N_cells,) float64 - cell centre coordinates
    field_values_2d,    # (N_cells, n_field_deps) float64 - field values
    out_obj_ids_0,      # (N_cells,) int64 - output obj_ids[:, 0]
    out_params,         # (N_cells, max_params) float64 - output params
    n_field_deps        # int - number of field_deps
):
    """Fill obj_ids and params of per-cell zero-object events in parallel

    params layout:
    - [0 : n_field_deps]  - field values (e.g. stress_xx)
    - [n_field_deps]      — cell_x
    - [n_field_deps + 1]  — cell_y
    - [n_field_deps + 2]  — cell_z
    """
    n = len(cell_ids)
    for i in prange(n):
        out_obj_ids_0[i] = cell_ids[i]
        for f in range(n_field_deps):
            out_params[i, f] = field_values_2d[i, f]
        out_params[i, n_field_deps] = cell_x[i]
        out_params[i, n_field_deps + 1] = cell_y[i]
        out_params[i, n_field_deps + 2] = cell_z[i]


def build_single_match_table(event_registry, type_to_int):
    single_events = event_registry.get_single_events()
    if not single_events or not type_to_int:
        n_types = max(type_to_int.values()) + 1 if type_to_int else 1
        return (np.full((n_types, 1), -1, dtype=np.int32),
                {}, {})

    # Event name -> int mapping
    event_name_to_int = {name: idx for idx, name in enumerate(single_events)}
    event_int_to_name = {idx: name for name, idx in event_name_to_int.items()}

    # Count the events belonging to each type_id
    n_types = max(type_to_int.values()) + 1
    type_events = {}  # type_id → [event_type_id, ...]

    for event_name in single_events:
        event_class = event_registry.get(event_name)
        if event_class is None:
            continue
        etype_id = event_name_to_int[event_name]
        for type_name in event_class.object_types:
            tid = type_to_int.get(type_name, -1)
            if tid < 0:
                continue
            if tid not in type_events:
                type_events[tid] = []
            type_events[tid].append(etype_id)

    # Determine max_events_per_type
    max_k = max((len(v) for v in type_events.values()), default=1)
    max_k = max(max_k, 1)

    # Build the 2D table
    match_table_2d = np.full((n_types, max_k), -1, dtype=np.int32)
    for tid, etype_ids in type_events.items():
        for k, eid in enumerate(etype_ids):
            match_table_2d[tid, k] = eid

    return match_table_2d, event_int_to_name, event_name_to_int


# ============================================================================
# Self-test
# ============================================================================

if __name__ == '__main__':
    import time

    print("=" * 60)
    print("JIT Event Builder - Self Test")
    print("=" * 60)
    print(f"Numba available: {NUMBA_AVAILABLE}")

    N = 10000
    np.random.seed(42)

    # ---- Test 1: jit_compute_distances ----
    print("\n[Test 1] jit_compute_distances")
    x1 = np.random.rand(N)
    y1 = np.random.rand(N)
    z1 = np.random.rand(N)
    x2 = np.random.rand(N)
    y2 = np.random.rand(N)
    z2 = np.random.rand(N)
    ids1 = np.random.randint(0, 100000, N, dtype=np.int64)
    ids2 = np.random.randint(0, 100000, N, dtype=np.int64)

    out_d = np.empty(N, dtype=np.float64)
    out_min = np.empty(N, dtype=np.int64)
    out_max = np.empty(N, dtype=np.int64)

    # Warmup
    jit_compute_distances(x1, y1, z1, x2, y2, z2, ids1, ids2,
                          out_d, out_min, out_max)

    # Verify
    ref_d = np.sqrt((x1 - x2)**2 + (y1 - y2)**2 + (z1 - z2)**2)
    ref_min = np.minimum(ids1, ids2)
    ref_max = np.maximum(ids1, ids2)
    assert np.allclose(out_d, ref_d), "Distance mismatch"
    assert np.array_equal(out_min, ref_min), "min_id mismatch"
    assert np.array_equal(out_max, ref_max), "max_id mismatch"

    t0 = time.perf_counter()
    for _ in range(100):
        jit_compute_distances(x1, y1, z1, x2, y2, z2, ids1, ids2,
                              out_d, out_min, out_max)
    t_jit = (time.perf_counter() - t0) / 100

    t0 = time.perf_counter()
    for _ in range(100):
        _ = np.sqrt((x1 - x2)**2 + (y1 - y2)**2 + (z1 - z2)**2)
        _ = np.minimum(ids1, ids2)
        _ = np.maximum(ids1, ids2)
    t_np = (time.perf_counter() - t0) / 100

    print(f"  JIT: {t_jit*1e6:.1f} us, NumPy: {t_np*1e6:.1f} us, "
          f"speedup: {t_np/t_jit:.1f}x")
    print("  PASSED")

    # ---- Test 2: jit_filter_pairs ----
    print("\n[Test 2] jit_filter_pairs")
    # 2 types: H=0, V=1; 1 event type: capture=0
    type_ids1 = np.random.randint(0, 2, N, dtype=np.int32)
    type_ids2 = np.random.randint(0, 2, N, dtype=np.int32)
    distances = np.random.rand(N) * 10.0
    owner_ids1 = np.zeros(N, dtype=np.int32)
    owner_ids2 = np.zeros(N, dtype=np.int32)
    obj_ids1 = np.arange(N, dtype=np.int64)
    obj_ids2 = np.arange(N, N * 2, dtype=np.int64)

    allowed = np.array([[0, 1]], dtype=np.int32)  # H-V
    max_ds = np.array([5.0], dtype=np.float64)
    etype_for = np.array([0], dtype=np.int32)
    out_et = np.empty(N, dtype=np.int32)

    jit_filter_pairs(type_ids1, type_ids2, distances,
                     owner_ids1, owner_ids2, obj_ids1, obj_ids2,
                     allowed, max_ds, etype_for, 0, out_et)

    # Reference: NumPy
    ref_mask = (((type_ids1 == 0) & (type_ids2 == 1)) |
                ((type_ids1 == 1) & (type_ids2 == 0))) & (distances <= 5.0)
    jit_mask = (out_et >= 0)
    assert np.array_equal(jit_mask, ref_mask), \
        f"Filter mismatch: JIT={jit_mask.sum()}, ref={ref_mask.sum()}"
    print(f"  Matched {jit_mask.sum()} pairs out of {N}")
    print("  PASSED")

    # ---- Test 3: jit_embed_objects_flat ----
    print("\n[Test 3] jit_embed_objects_flat")
    n_total = 500
    n_fields = 8
    n_events = 200
    src = np.random.rand(n_total, n_fields)
    i1 = np.random.randint(0, n_total, n_events, dtype=np.int64)
    i2 = np.random.randint(0, n_total, n_events, dtype=np.int64)
    o1 = np.empty((n_events, n_fields), dtype=np.float64)
    o2 = np.empty((n_events, n_fields), dtype=np.float64)

    jit_embed_objects_flat(i1, i2, n_events, n_fields, src, o1, o2)

    assert np.array_equal(o1, src[i1]), "obj1 embed mismatch"
    assert np.array_equal(o2, src[i2]), "obj2 embed mismatch"
    print("  PASSED")

    # ---- Test 4: build helpers ----
    print("\n[Test 4] Helper functions")
    t2i = build_type_to_int(['H', 'V', 'SIA', 'H'])
    assert len(t2i) == 3
    assert set(t2i.keys()) == {'H', 'V', 'SIA'}

    types_str = np.array(['H', 'V', 'H', 'SIA', 'V'])
    tid_arr = build_type_id_array(types_str, t2i)
    assert tid_arr[0] == t2i['H']
    assert tid_arr[1] == t2i['V']
    assert tid_arr[3] == t2i['SIA']

    vf = {
        'capture': {'type_pairs': [('H', 'V')], 'max_distance': 5.0},
        'recomb': {'type_pairs': [('SIA', 'V')], 'max_distance': 3.0},
    }
    et2i = {'capture': 0, 'recomb': 1}
    ap, md, ef = build_allowed_pairs_table(vf, t2i, et2i)
    assert len(ap) == 2
    assert len(md) == 2
    print(f"  allowed_pairs: {ap.tolist()}")
    print(f"  max_dists: {md.tolist()}")
    print("  PASSED")

    # ---- Test 5: jit_filter_single_objects_lut ----
    print("\n[Test 5] jit_filter_single_objects_lut")
    # 3 types: H=0, V=1, HV=2
    # 2 event types: migration=0 (H,V), dissociation=1 (HV)
    N5 = 1000
    type_ids5 = np.random.randint(0, 3, N5, dtype=np.int32)
    owner_ranks5 = np.zeros(N5, dtype=np.int32)  # all owned by rank 0

    match_2d = np.full((3, 2), -1, dtype=np.int32)
    match_2d[0, 0] = 0  # H → migration
    match_2d[1, 0] = 0  # V → migration
    match_2d[2, 0] = 1  # HV → dissociation

    out_2d = np.empty((N5, 2), dtype=np.int32)
    jit_filter_single_objects_lut(type_ids5, owner_ranks5, np.int32(0),
                                   match_2d, out_2d)

    # Verify
    for i in range(N5):
        tid = type_ids5[i]
        if tid == 0 or tid == 1:
            assert out_2d[i, 0] == 0, f"Expected migration(0) for type {tid}, got {out_2d[i, 0]}"
        elif tid == 2:
            assert out_2d[i, 0] == 1, f"Expected dissociation(1) for type HV, got {out_2d[i, 0]}"
        assert out_2d[i, 1] == -1, f"Expected -1 for slot 1, got {out_2d[i, 1]}"

    # Test ownership filtering
    owner_ranks5_mixed = np.random.randint(0, 2, N5, dtype=np.int32)
    jit_filter_single_objects_lut(type_ids5, owner_ranks5_mixed, np.int32(0),
                                   match_2d, out_2d)
    for i in range(N5):
        if owner_ranks5_mixed[i] != 0:
            assert out_2d[i, 0] == -1, f"Non-owned object should have -1"

    print(f"  Classified {N5} objects, ownership filtering verified")
    print("  PASSED")

    # ---- Test 6: build_single_match_table ----
    print("\n[Test 6] build_single_match_table")

    # Mock EventRegistry
    class MockEventClass:
        def __init__(self, obj_types):
            self.object_types = obj_types
    class MockRegistry:
        def __init__(self):
            self._events = {
                'migration': MockEventClass(['H', 'V']),
                'dissociation': MockEventClass(['HV']),
            }
            self._single = ['migration', 'dissociation']
        def get_single_events(self):
            return self._single
        def get(self, name):
            return self._events.get(name)

    mock_reg = MockRegistry()
    mock_t2i = {'H': 0, 'V': 1, 'HV': 2}
    mt2d, ei2n, en2i = build_single_match_table(mock_reg, mock_t2i)

    assert mt2d.shape[0] == 3  # 3 types
    assert mt2d[0, 0] == en2i['migration']  # H → migration
    assert mt2d[1, 0] == en2i['migration']  # V → migration
    assert mt2d[2, 0] == en2i['dissociation']  # HV → dissociation
    print(f"  match_table_2d shape: {mt2d.shape}")
    print(f"  event_name_to_int: {en2i}")
    print("  PASSED")

    # ---- Test 7: jit_build_per_cell_zero_events ----
    print("\n[Test 7] jit_build_per_cell_zero_events")
    N7 = 500
    cell_ids7 = np.arange(N7, dtype=np.int64)
    cx = np.random.rand(N7)
    cy = np.random.rand(N7)
    cz = np.random.rand(N7)
    n_fields = 2
    max_params = 8
    fv = np.random.rand(N7, n_fields)
    out_oid = np.empty(N7, dtype=np.int64)
    out_p = np.zeros((N7, max_params), dtype=np.float64)

    # Warmup
    jit_build_per_cell_zero_events(cell_ids7, cx, cy, cz, fv,
                                    out_oid, out_p, n_fields)

    # Verify
    assert np.array_equal(out_oid, cell_ids7), "cell_id mismatch"
    assert np.allclose(out_p[:, 0], fv[:, 0]), "field[0] mismatch"
    assert np.allclose(out_p[:, 1], fv[:, 1]), "field[1] mismatch"
    assert np.allclose(out_p[:, n_fields], cx), "cell_x mismatch"
    assert np.allclose(out_p[:, n_fields + 1], cy), "cell_y mismatch"
    assert np.allclose(out_p[:, n_fields + 2], cz), "cell_z mismatch"
    print(f"  Filled {N7} per-cell events with {n_fields} field deps")
    print("  PASSED")

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)


@njit(parallel=True, cache=True)
def jit_mark_invalid_pair_events(
    x1, y1, z1, x2, y2, z2,
    ids1, ids2,
    changed_ids_sorted,
    poison_mask,
    gx, gy, gz,
    cell_sx, cell_sy, cell_sz,
    box_x, box_y, box_z,
    periodic_x, periodic_y, periodic_z,
    out_keep,
):
    """[pair-inc] Decide when a pair event is structurally invalid (uniform verdict, no communication)

    Invalid <=> (an embedded object ID is in the packet) OR (an embedded position is in a poisoned cell)

    - ID predicate: kills every pair of the changed object itself, including the
      'escapee' event on the old-position side even when old coordinates are absent
    - Zone predicate: kills the collateral events inside the rebuild scope, so that
      kill <=> rebuild scope holds identically. Every input (packet IDs, zone mask,
      grid parameters) is identical on all ranks, so owner and host always agree.

    Args:
        x1..z2: (N,) coordinates of embedded object slots 0 and 1
        ids1, ids2: (N,) int64 IDs of the two objects of the event
        changed_ids_sorted: (M,) int64 object IDs from the state packet (sorted)
        poison_mask: (total_cells,) bool poisoned-zone mask (pair grid)
        gx/gy/gz, cell_s*, box_*, periodic_*: pair grid parameters
          (the cell reduction matches build_cell_bins exactly)
        out_keep: (N,) int8 output, 1 = structurally retained
    """
    n = x1.shape[0]
    for i in prange(n):
        keep = 1

        # ID predicate (binary search)
        if _binary_search_int64(changed_ids_sorted, ids1[i]) or \
           _binary_search_int64(changed_ids_sorted, ids2[i]):
            keep = 0
        else:
            # Zone predicate: embedded position -> cell -> mask
            for slot in range(2):
                if slot == 0:
                    px = x1[i]
                    py = y1[i]
                    pz = z1[i]
                else:
                    px = x2[i]
                    py = y2[i]
                    pz = z2[i]

                if periodic_x:
                    px = px % box_x
                if periodic_y:
                    py = py % box_y
                if periodic_z:
                    pz = pz % box_z

                ix = int(np.floor(px / cell_sx))
                iy = int(np.floor(py / cell_sy))
                iz = int(np.floor(pz / cell_sz))
                if ix >= gx:
                    ix = gx - 1
                if ix < 0:
                    ix = 0
                if iy >= gy:
                    iy = gy - 1
                if iy < 0:
                    iy = 0
                if iz >= gz:
                    iz = gz - 1
                if iz < 0:
                    iz = 0

                if poison_mask[ix + iy * gx + iz * gx * gy]:
                    keep = 0
                    break

        out_keep[i] = keep


@njit(parallel=True)
def jit_mark_stale_single_events(event_obj_ids, valid_ids_sorted, stale_ids_sorted,
                                 out_keep):
    """For the in-place update: mark which single events in the pool stay valid.

    An event is kept (keep=1) if and only if its host object:
      - still exists (is in valid_ids_sorted), and
      - did not change this step (is not in stale_ids_sorted)
    Events of changed or vanished objects are dropped and rebuilt by the caller.

    All inputs are sorted int64 arrays, searched by bisection, O(N log M).
    """
    n = event_obj_ids.shape[0]
    nv = valid_ids_sorted.shape[0]
    ns = stale_ids_sorted.shape[0]

    for i in prange(n):
        oid = event_obj_ids[i]
        keep = 0
        if oid >= 0 and nv > 0:
            # Binary search: does the object still exist?
            lo = 0
            hi = nv - 1
            found = False
            while lo <= hi:
                mid = (lo + hi) >> 1
                v = valid_ids_sorted[mid]
                if v == oid:
                    found = True
                    break
                elif v < oid:
                    lo = mid + 1
                else:
                    hi = mid - 1
            if found:
                keep = 1
                # Binary search: did the object change this step?
                if ns > 0:
                    lo = 0
                    hi = ns - 1
                    while lo <= hi:
                        mid = (lo + hi) >> 1
                        v = stale_ids_sorted[mid]
                        if v == oid:
                            keep = 0
                            break
                        elif v < oid:
                            lo = mid + 1
                        else:
                            hi = mid - 1
        out_keep[i] = keep


@njit(parallel=True)
def jit_mark_stale_single_events_any_owner(event_obj_ids, stale_ids_sorted, out_keep):
    """In-place update with an event-moving LB on (foreign events may be hosted here).

    Ownership is not a validity test in that case: an event of an object owned by
    another rank is just as valid as a local one. Keep every event whose object did
    not change this step; the events of changed or deleted objects were already removed
    pool-wide (remove_events_for_objects, from the broadcast change set) and the owner
    rebuilds them. Sorted int64 inputs, bisection, O(N log M).
    """
    n = event_obj_ids.shape[0]
    ns = stale_ids_sorted.shape[0]

    for i in prange(n):
        oid = event_obj_ids[i]
        keep = 0
        if oid >= 0:
            keep = 1
            if ns > 0:
                lo = 0
                hi = ns - 1
                while lo <= hi:
                    mid = (lo + hi) >> 1
                    v = stale_ids_sorted[mid]
                    if v == oid:
                        keep = 0
                        break
                    elif v < oid:
                        lo = mid + 1
                    else:
                        hi = mid - 1
        out_keep[i] = keep
