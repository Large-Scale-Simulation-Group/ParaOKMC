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
JIT cell list - a fused Numba kernel replacing the cKDTree + filter pipeline

Three @njit functions:
1. build_neighbor_map()  - precompute the cell neighbour table (cached and reused)
2. build_cell_bins()     - linked-list cell bins (O(N) insertion)
3. jit_find_pairs_cell_list() - fused pair finding + filtering kernel

Performance target: pair finding + filtering from ~20 ms down to <1 ms (160 objects)
"""

import numpy as np

try:
    from numba import njit
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False

    def njit(*args, **kwargs):
        if args and callable(args[0]):
            return args[0]
        def wrapper(fn):
            return fn
        return wrapper


# ============================================================================
# 1. build_neighbor_map - precompute the cell neighbour table
# ============================================================================

@njit
def build_neighbor_map(Gx, Gy, Gz, max_dist, cell_size_x, cell_size_y,
                       cell_size_z, periodic_x, periodic_y, periodic_z):
    """Precompute the list of neighbour cell IDs for every cell (including itself)

    Args:
        Gx, Gy, Gz: grid dimensions
        max_dist: maximum interaction distance
        cell_size_x/y/z: physical size of a single cell
        periodic_x/y/z: periodic boundary conditions

    Returns:
        neighbor_offsets: (total_cells, max_neighbors) int32
        neighbor_counts: (total_cells,) int32
    """
    total_cells = Gx * Gy * Gz

    # Search radius (in cells)
    rx = int(np.ceil(max_dist / cell_size_x))
    ry = int(np.ceil(max_dist / cell_size_y))
    rz = int(np.ceil(max_dist / cell_size_z))

    # Upper bound on the neighbour count (including itself)
    max_neighbors = (2 * rx + 1) * (2 * ry + 1) * (2 * rz + 1)

    neighbor_offsets = np.empty((total_cells, max_neighbors), dtype=np.int32)
    neighbor_counts = np.zeros(total_cells, dtype=np.int32)

    for iz in range(Gz):
        for iy in range(Gy):
            for ix in range(Gx):
                c = ix + iy * Gx + iz * Gx * Gy
                cnt = 0
                for dz in range(-rz, rz + 1):
                    for dy in range(-ry, ry + 1):
                        for dx in range(-rx, rx + 1):
                            tx = ix + dx
                            ty = iy + dy
                            tz = iz + dz

                            # Periodic wrap / validity check
                            if periodic_x:
                                tx = tx % Gx
                            else:
                                if tx < 0 or tx >= Gx:
                                    continue
                            if periodic_y:
                                ty = ty % Gy
                            else:
                                if ty < 0 or ty >= Gy:
                                    continue
                            if periodic_z:
                                tz = tz % Gz
                            else:
                                if tz < 0 or tz >= Gz:
                                    continue

                            nc = tx + ty * Gx + tz * Gx * Gy
                            # Deduplicate (PBC may map different offsets to one cell)
                            dup = False
                            for k in range(cnt):
                                if neighbor_offsets[c, k] == nc:
                                    dup = True
                                    break
                            if not dup:
                                neighbor_offsets[c, cnt] = nc
                                cnt += 1
                neighbor_counts[c] = cnt

    return neighbor_offsets, neighbor_counts


# ============================================================================
# 2. build_cell_bins - linked-list cell bins
# ============================================================================

@njit
def build_cell_bins(x, y, z, n, cell_size_x, cell_size_y, cell_size_z,
                    Gx, Gy, Gz, box_size_x, box_size_y, box_size_z,
                    periodic_x, periodic_y, periodic_z):
    """Linked-list cell bins: O(N) sequential insertion

    Args:
        x, y, z: (N,) object coordinates
        n: number of objects
        cell_size_x/y/z: physical cell size
        Gx, Gy, Gz: grid dimensions
        box_size_x/y/z: box dimensions
        periodic_x/y/z: periodic boundaries

    Returns:
        cell_heads: (total_cells,) int32, list head of each cell (-1 = empty)
        cell_next: (N,) int32, linked-list next pointer (-1 = end)
    """
    total_cells = Gx * Gy * Gz
    cell_heads = np.full(total_cells, -1, dtype=np.int32)
    cell_next = np.full(n, -1, dtype=np.int32)

    for i in range(n):
        # Handle periodic boundaries
        px = x[i]
        py = y[i]
        pz = z[i]
        if periodic_x:
            px = px % box_size_x
        if periodic_y:
            py = py % box_size_y
        if periodic_z:
            pz = pz % box_size_z

        # Compute the cell index
        ix = int(np.floor(px / cell_size_x))
        iy = int(np.floor(py / cell_size_y))
        iz = int(np.floor(pz / cell_size_z))

        # Clamp at the boundary
        if ix >= Gx:
            ix = Gx - 1
        if ix < 0:
            ix = 0
        if iy >= Gy:
            iy = Gy - 1
        if iy < 0:
            iy = 0
        if iz >= Gz:
            iz = Gz - 1
        if iz < 0:
            iz = 0

        c = ix + iy * Gx + iz * Gx * Gy

        # Push onto the head of the list
        cell_next[i] = cell_heads[c]
        cell_heads[c] = i

    return cell_heads, cell_next


# ============================================================================
# 2.5 jit_find_pairs_cell_list_zone - poisoned-zone subset rebuild kernel [pair-inc]
# ============================================================================

@njit
def jit_find_pairs_cell_list_zone(
    x, y, z, obj_ids, type_ids, owner_ranks,
    n,
    cell_heads, cell_next,
    neighbor_offsets, neighbor_counts,
    zone_mask,
    match_table, max_dist_table,
    box_size_x, box_size_y, box_size_z,
    periodic_x, periodic_y, periodic_z,
    my_rank,
    out_idx1, out_idx2, out_min_ids, out_max_ids,
    out_owners, out_distances, out_event_types,
    max_pairs
):
    """[pair-inc] Rebuild only pairs with at least one endpoint in a poisoned cell

    Line for line semantically identical to jit_find_pairs_cell_list (minimum
    image, 2D LUT, min-ID ownership, true-count overflow protocol); only the
    traversal and deduplication rules differ:
    - iterate over poisoned cells c only
    - neighbour c2 also poisoned: keep c2 >= c to deduplicate (half neighbourhood)
    - neighbour c2 outside the zone: always pair (such cells are never iterated as c)

    The resulting set == the subset of the full kernel output with any endpoint in
    the zone, matching the zone part of the kill predicate exactly (kill <=> rebuild).
    """
    total_cells = len(cell_heads)
    n_total = 0

    for c in range(total_cells):
        if not zone_mask[c]:
            continue

        nc = neighbor_counts[c]
        for k in range(nc):
            c2 = neighbor_offsets[c, k]

            if zone_mask[c2]:
                # Zone-zone: half neighbourhood deduplication
                if c2 < c:
                    continue
                same_cell = (c == c2)
            else:
                # Zone-outside: always pair (outside cells are not iterated)
                same_cell = False

            i = cell_heads[c]
            while i >= 0:
                j = cell_heads[c2]
                while j >= 0:
                    if same_cell and j <= i:
                        j = cell_next[j]
                        continue

                    if obj_ids[i] == obj_ids[j]:
                        j = cell_next[j]
                        continue

                    ti = type_ids[i]
                    tj = type_ids[j]
                    etype = match_table[ti, tj]
                    if etype < 0:
                        j = cell_next[j]
                        continue

                    dx = x[i] - x[j]
                    dy = y[i] - y[j]
                    dz = z[i] - z[j]
                    if periodic_x:
                        dx = dx - box_size_x * np.rint(dx / box_size_x)
                    if periodic_y:
                        dy = dy - box_size_y * np.rint(dy / box_size_y)
                    if periodic_z:
                        dz = dz - box_size_z * np.rint(dz / box_size_z)
                    dist = np.sqrt(dx * dx + dy * dy + dz * dz)

                    if dist > max_dist_table[ti, tj]:
                        j = cell_next[j]
                        continue

                    if obj_ids[i] < obj_ids[j]:
                        owner = owner_ranks[i]
                        min_id = obj_ids[i]
                        max_id = obj_ids[j]
                    else:
                        owner = owner_ranks[j]
                        min_id = obj_ids[j]
                        max_id = obj_ids[i]

                    if owner != my_rank:
                        j = cell_next[j]
                        continue

                    if n_total < max_pairs:
                        out_idx1[n_total] = i
                        out_idx2[n_total] = j
                        out_min_ids[n_total] = min_id
                        out_max_ids[n_total] = max_id
                        out_owners[n_total] = owner
                        out_distances[n_total] = dist
                        out_event_types[n_total] = etype
                    n_total += 1

                    j = cell_next[j]
                i = cell_next[i]

    return n_total


# ============================================================================
# 3. jit_find_pairs_cell_list - fused pair finding + filtering kernel
# ============================================================================

@njit
def jit_find_pairs_cell_list(
    x, y, z, obj_ids, type_ids, owner_ranks,
    n,
    cell_heads, cell_next,
    neighbor_offsets, neighbor_counts,
    match_table, max_dist_table,
    box_size_x, box_size_y, box_size_z,
    periodic_x, periodic_y, periodic_z,
    my_rank,
    out_idx1, out_idx2, out_min_ids, out_max_ids,
    out_owners, out_distances, out_event_types,
    max_pairs
):
    """Fused cell-list pair finding + filtering kernel

    Walks all cell pairs and, for each object pair, performs:
    - distance computation (minimum image convention along periodic axes)
    - type matching (2D LUT)
    - distance filtering
    - ownership decision
    - self-pair exclusion

    Args:
        x, y, z: (N,) object coordinates
        obj_ids: (N,) int64 object IDs
        type_ids: (N,) int32 type IDs
        owner_ranks: (N,) int32 owning ranks
        n: number of objects
        cell_heads, cell_next: linked-list cell bins
        neighbor_offsets, neighbor_counts: neighbour table
        match_table: (n_types, n_types) int32, -1 = no match
        max_dist_table: (n_types, n_types) float64
        box_size_x/y/z: box dimensions (for the minimum image)
        periodic_x/y/z: periodic boundary flags
        my_rank: int32
        out_*: preallocated output arrays
        max_pairs: capacity of the output arrays

    Returns:
        n_total: total number of qualifying pairs. May exceed max_pairs - the
                 excess is not written to the output, so the caller should test
                 for n_total > max_pairs, grow the arrays and rerun.
    """
    total_cells = len(cell_heads)
    n_total = 0

    for c in range(total_cells):
        # Walk all neighbour cells of cell c (including itself)
        nc = neighbor_counts[c]
        for k in range(nc):
            c2 = neighbor_offsets[c, k]

            # Avoid double counting: handle only cell pairs with c2 >= c
            if c2 < c:
                continue

            same_cell = (c == c2)

            # Walk all objects in cell c
            i = cell_heads[c]
            while i >= 0:
                # Walk all objects in cell c2
                j = cell_heads[c2]
                while j >= 0:
                    # Within one cell avoid duplicates: handle only j > i
                    if same_cell and j <= i:
                        j = cell_next[j]
                        continue

                    # Self-pair check
                    if obj_ids[i] == obj_ids[j]:
                        j = cell_next[j]
                        continue

                    # Type match (O(1) LUT)
                    ti = type_ids[i]
                    tj = type_ids[j]
                    etype = match_table[ti, tj]
                    if etype < 0:
                        j = cell_next[j]
                        continue

                    # Distance (minimum image along periodic axes)
                    dx = x[i] - x[j]
                    dy = y[i] - y[j]
                    dz = z[i] - z[j]
                    if periodic_x:
                        dx = dx - box_size_x * np.rint(dx / box_size_x)
                    if periodic_y:
                        dy = dy - box_size_y * np.rint(dy / box_size_y)
                    if periodic_z:
                        dz = dz - box_size_z * np.rint(dz / box_size_z)
                    dist = np.sqrt(dx * dx + dy * dy + dz * dz)

                    # Distance filter
                    if dist > max_dist_table[ti, tj]:
                        j = cell_next[j]
                        continue

                    # Ownership: the owner of min_id must be my_rank
                    if obj_ids[i] < obj_ids[j]:
                        owner = owner_ranks[i]
                        min_id = obj_ids[i]
                        max_id = obj_ids[j]
                    else:
                        owner = owner_ranks[j]
                        min_id = obj_ids[j]
                        max_id = obj_ids[i]

                    if owner != my_rank:
                        j = cell_next[j]
                        continue

                    # Write output (keep counting past capacity so the caller can grow)
                    if n_total < max_pairs:
                        out_idx1[n_total] = i
                        out_idx2[n_total] = j
                        out_min_ids[n_total] = min_id
                        out_max_ids[n_total] = max_id
                        out_owners[n_total] = owner
                        out_distances[n_total] = dist
                        out_event_types[n_total] = etype
                    n_total += 1

                    j = cell_next[j]
                i = cell_next[i]

    return n_total
