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
Vectorized event builder

Core pipeline:
NumPy arrays -> cKDTree.query_pairs -> flatten -> vectorized filtering -> bulk event creation

Performance notes:
1. One cKDTree: all local objects plus ghost objects are inserted at once
2. query_pairs: O(N log N) for every candidate pair
3. Flatten: reduce to an (N_pairs, 2) index array
4. Vectorized filtering: NumPy broadcasting filters every pair at once
5. Bulk creation: preallocated arrays, filled by vectorized assignment

Compared with the cell-based approach:
- cell-based: many mini KD-trees, processed cell by cell
- this approach: one build, global processing, better suited to GPU acceleration

"""

import numpy as np
from scipy.spatial import cKDTree
from typing import Dict, List, Tuple, Optional, Set
# [fix] hashlib removed; a fast integer hash replaces MD5

from .rma_object_pool import RMAObjectPool
from .rma_event_pool import (
    RMAEventPool,
    create_event,
    STATUS_TODO,
    STATUS_DONE,
    STATUS_SENT,
)
from .domain_decomposition import DomainDecomposition

# JIT accelerated kernels (the only filtering path)
from .jit_event_builder import (
    jit_compute_distances, jit_compute_distances_pbc, jit_filter_pairs_lut,
    build_allowed_pairs_table, build_type_to_int,
    build_match_tables,
    jit_filter_single_objects_lut, build_single_match_table,
    jit_count_single_events_lut, jit_fill_single_events_lut,
    jit_count_valid_event_rows, jit_compact_pair_rows,
    jit_scan_reusable_single_events,
    jit_mark_invalid_pair_events,
)

# JIT cell list - fused pair finding + filtering
from .jit_cell_list import (
    build_neighbor_map, build_cell_bins, jit_find_pairs_cell_list,
    jit_find_pairs_cell_list_zone,
)


class VectorizedPairEventBuilder:
    """
    Vectorized pair event builder

    Core pipeline:
    1. collect every object (local + large + ghost)
    2. build one cKDTree
    3. query_pairs for every pair within range
    4. vectorized filtering (type matching, ownership)
    5. bulk event creation
    """

    def __init__(self, domain_decomp: DomainDecomposition,
                 config: dict,
                 object_dtype: np.dtype,
                 event_dtype: np.dtype,
                 verbose: bool = False,
                 use_cell_list: bool = True):
        """
        Initialize the vectorized event builder

        Args:
            domain_decomp: domain decomposition object
            config: configuration dict
            object_dtype: object dtype
            event_dtype: event dtype
            verbose: whether to print debugging output
        """
        self.domain_decomp = domain_decomp
        self.comm = domain_decomp.comm
        self.rank = domain_decomp.rank
        self.size = domain_decomp.size
        self.config = config
        self.object_dtype = object_dtype
        self.event_dtype = event_dtype
        self.verbose = verbose

        # Maximum interaction distance
        self.max_interaction_distance = domain_decomp.max_interaction_distance

        # Event configuration
        self.pair_type_filters = config.get('pair_type_filters', {})
        self.pair_param_extractors = config.get('pair_param_extractors', {})
        self.event_weights = config.get('event_weights', {})
        self._global_event_name_to_int = config.get('_event_name_to_int', {})

        # Vectorized filter configuration
        # Format: {event_type: {'type_pairs': [(t1, t2), ...], 'max_distance': float}}
        self.vectorized_filters = config.get('vectorized_pair_filters', {})

        # Build the event type index mapping
        self.pair_event_types = {}
        for idx, event_name in enumerate(self.pair_type_filters.keys()):
            self.pair_event_types[event_name] = idx
        n_pair_types = len(self.pair_event_types)
        self._pair_local_to_global = np.full(max(n_pair_types, 1), -1, dtype=np.int32)
        self._pair_weights_by_local = np.ones(max(n_pair_types, 1), dtype=np.float64)
        for event_name, idx in self.pair_event_types.items():
            self._pair_local_to_global[idx] = self._global_event_name_to_int.get(event_name, -1)
            self._pair_weights_by_local[idx] = self.event_weights.get(event_name, 1.0)

        # Caches: object array and dict (the dict is lazy, see the property)
        self.all_objects = None
        self._all_objects_dict = {}

        # Prebuilt JIT lookup tables
        self._jit_type_to_int = {}
        self._jit_allowed_pairs = None
        self._jit_max_dists = None
        self._jit_event_type_for_pair = None
        self._jit_event_int_to_name = {}
        self._jit_match_table = None
        self._jit_max_dist_table = None

        # Cell list cache
        self.use_cell_list = use_cell_list
        self._cl_neighbor_map = None
        self._cl_neighbor_counts = None
        self._cl_grid = None  # (Gx, Gy, Gz) cache, rebuilt when the grid changes

        if self.vectorized_filters:
            self._precompute_jit_tables()

    @property
    def all_objects_dict(self):
        """id -> object dict (built lazily).

        [performance fix] Only the fallback path that embeds objects by id needs
        it; the vectorized path indexes directly. It used to be built unconditionally
        """
        if self._all_objects_dict is None:
            objs = self.all_objects
            self._all_objects_dict = (
                {int(o['id']): o for o in objs} if objs is not None else {}
            )
        return self._all_objects_dict

    @all_objects_dict.setter
    def all_objects_dict(self, value):
        self._all_objects_dict = value

    def _precompute_jit_tables(self):
        """Prebuild the lookup tables the JIT filter needs"""
        # Collect every type string
        all_types = set()
        for fconfig in self.vectorized_filters.values():
            for t1, t2 in fconfig.get('type_pairs', []):
                all_types.add(t1)
                all_types.add(t2)
        if not all_types:
            return

        # Prefer config['_type_to_int'] (consistent with the type_id field)
        config_type_to_int = self.config.get('_type_to_int', {})
        if config_type_to_int:
            self._jit_type_to_int = dict(config_type_to_int)
            # Add types that appear in type_pairs but are missing from config
            next_id = max(self._jit_type_to_int.values()) + 1 if self._jit_type_to_int else 0
            for t in all_types:
                if t not in self._jit_type_to_int:
                    self._jit_type_to_int[t] = next_id
                    next_id += 1
        else:
            self._jit_type_to_int = build_type_to_int(all_types)

        # Event type name -> int mapping (reusing pair_event_types)
        event_type_to_int = self.pair_event_types
        self._jit_event_int_to_name = {v: k for k, v in event_type_to_int.items()}

        self._jit_allowed_pairs, self._jit_max_dists, self._jit_event_type_for_pair = \
            build_allowed_pairs_table(self.vectorized_filters,
                                      self._jit_type_to_int,
                                      event_type_to_int)

        if len(self._jit_allowed_pairs) == 0:
            return

        # Build the 2D LUT (O(1) type matching)
        self._jit_match_table, self._jit_max_dist_table = \
            build_match_tables(self.vectorized_filters,
                               self._jit_type_to_int,
                               event_type_to_int)

        # [feat-2] Class-level rate increment LUT: global event_type_id -> incremental_rate
        self._rate_incremental_by_global = None
        registry = self.config.get('_event_registry')
        if registry is not None and self._global_event_name_to_int:
            gmax = max(self._global_event_name_to_int.values())
            lut = np.zeros(gmax + 1, dtype=np.bool_)
            for event_name, gid in self._global_event_name_to_int.items():
                event_class = registry.get(event_name)
                if event_class is not None and 0 <= gid <= gmax:
                    lut[gid] = bool(getattr(event_class, 'incremental_rate', False))
            self._rate_incremental_by_global = lut

    def build_pair_events(self, rma_object_pool: RMAObjectPool,
                          current_step: int = 0) -> np.ndarray:
        """
        Build every pair event (vectorized pipeline)

        Procedure:
        1. collect every object (local small objects + large objects + ghosts)
        2. build the cKDTree
        3. query_pairs for the candidates
        4. vectorized filtering
        5. bulk event creation

        Args:
            rma_object_pool: RMA object pool
            current_step: current step

        Returns:
            the event array (event_dtype)
        """
        # [performance] Early exit
        if len(self.pair_type_filters) == 0:
            return np.array([], dtype=self.event_dtype)

        # ===== Step 1: collect every object =====
        all_objects = self._collect_all_objects(rma_object_pool)

        n_total = len(all_objects)
        if n_total < 2:
            return np.array([], dtype=self.event_dtype)

        # Kept for the embedding stage below
        # [performance fix] The dict is now lazy (see the all_objects_dict property)
        self.all_objects = all_objects
        self._all_objects_dict = None

        if self.verbose:
            print(f"[Rank {self.rank}] VectorizedBuilder: {n_total} objects total")

        # ===== Step 2+3+4+5: Pair finding + filtering =====
        if not self.vectorized_filters:
            print(f"[Rank {self.rank}] Step {current_step}: No vectorized filters, returning 0 pair events")
            return np.array([], dtype=self.event_dtype)

        can_use_cell_list = (
            self.use_cell_list
            and self._jit_match_table is not None
            and hasattr(self.domain_decomp, 'box_size')
            and getattr(self.domain_decomp, 'grid_system', None) is not None
        )

        if can_use_cell_list:
            # ===== Fused cell list path =====
            result = self._build_pairs_cell_list(all_objects, n_total, current_step)
        else:
            # ===== cKDTree fallback path =====
            result = self._build_pairs_kdtree(all_objects, n_total, current_step)

        if self.verbose:
            print(f"[Rank {self.rank}] VectorizedBuilder: {len(result)} total pair events")

        return result

    def build_pair_events_incremental(self, rma_object_pool, rma_event_pool,
                                      changed_arr, current_step: int,
                                      allow_rate_keep: bool = False) -> bool:
        """[pair-inc] Cell-level poisoned-zone structural update of the pair event pool

        The zone = cells of the (old union new) coordinates in the state packet, plus the
        neighbour ring (identical on every rank). Kill predicate (JIT): any embedded object
        ID in the packet OR any embedded position inside the zone. Rebuild scope: pairs
        owned here with an endpoint in a poisoned cell (kill <=> rebuild, no comms).

        Returns:
            True  = the incremental update was applied (the pool is ready)
            False = the preconditions failed; the caller must rebuild in full
        """
        # ===== Precondition checks (any failure falls back to a full rebuild) =====
        if changed_arr is None:
            return False
        if not self.use_cell_list or self._jit_match_table is None:
            return False
        if not (hasattr(self.domain_decomp, 'box_size')
                and getattr(self.domain_decomp, 'grid_system', None) is not None):
            return False
        if ('objects' not in self.event_dtype.names
                or 'obj_ids' not in self.event_dtype.names
                or 'event_type_id' not in self.event_dtype.names):
            return False
        if 'old_valid' not in changed_arr.dtype.names:
            return False

        # The grid key genuinely changed (max_dist / box): the old structure used the old grid
        (Gx, Gy, Gz), (cell_sx, cell_sy, cell_sz), box_size, periodic = \
            self._pair_grid_params()
        grid_key = (Gx, Gy, Gz)
        if self._cl_grid is not None and self._cl_grid != grid_key:
            return False

        # [fix] A sparse rank (fewer than 2 objects) never ran a full build, so its grid
        # cache is uninitialized - yet it may host events moved in by the load balancer.
        # This used to fall back to a full rebuild, which produces an empty array on an
        # empty rank and replaces the whole pool, wiping the hosted copies (the owner does
        # not rebuild them outside its zone -> a global loss). Build the grid on demand.
        if self._cl_grid is None or self._cl_neighbor_map is None:
            self._cl_neighbor_map, self._cl_neighbor_counts = build_neighbor_map(
                Gx, Gy, Gz, self.max_interaction_distance,
                cell_sx, cell_sy, cell_sz,
                periodic[0], periodic[1], periodic[2]
            )
            self._cl_grid = grid_key

        # ===== Step 1: collect the current objects (as in the full path) =====
        all_objects = self._collect_all_objects(rma_object_pool)
        n_total = len(all_objects)
        self.all_objects = all_objects
        self._all_objects_dict = None

        # ===== Step 2: poisoned-zone mask (pure vectorization, identical on all ranks) =====
        poison_mask = self._compute_poison_mask(
            changed_arr, (Gx, Gy, Gz), (cell_sx, cell_sy, cell_sz),
            box_size, periodic)

        # ===== Step 3: kill decision (JIT, over the whole pool including SENT/hosted) =====
        n_ev = rma_event_pool.count_pair
        changed_ids_sorted = np.unique(
            np.ascontiguousarray(changed_arr['id'], dtype=np.int64))
        if n_ev > 0:
            ev = rma_event_pool.pair_obj_array[:n_ev]
            emb1 = ev['objects'][:, 0]
            emb2 = ev['objects'][:, 1]
            keep_i8 = np.empty(n_ev, dtype=np.int8)
            jit_mark_invalid_pair_events(
                np.ascontiguousarray(emb1['x'], dtype=np.float64),
                np.ascontiguousarray(emb1['y'], dtype=np.float64),
                np.ascontiguousarray(emb1['z'], dtype=np.float64),
                np.ascontiguousarray(emb2['x'], dtype=np.float64),
                np.ascontiguousarray(emb2['y'], dtype=np.float64),
                np.ascontiguousarray(emb2['z'], dtype=np.float64),
                np.ascontiguousarray(ev['obj_ids'][:, 0], dtype=np.int64),
                np.ascontiguousarray(ev['obj_ids'][:, 1], dtype=np.int64),
                changed_ids_sorted, poison_mask,
                np.int64(Gx), np.int64(Gy), np.int64(Gz),
                float(cell_sx), float(cell_sy), float(cell_sz),
                float(box_size[0]), float(box_size[1]), float(box_size[2]),
                bool(periodic[0]), bool(periodic[1]), bool(periodic[2]),
                keep_i8)
            keep_mask = keep_i8.astype(np.bool_)
        else:
            keep_mask = np.zeros(0, dtype=np.bool_)

        # ===== Step 4: rebuild the zone subset (from the owner's viewpoint) =====
        new_events = np.array([], dtype=self.event_dtype)
        if n_total >= 2 and np.any(poison_mask):
            new_events = self._build_pairs_zone(
                all_objects, n_total, current_step, poison_mask,
                (Gx, Gy, Gz), (cell_sx, cell_sy, cell_sz), box_size, periodic)

        # ===== Step 5: class-level rate retention mask (feat-2) =====
        keep_rate_mask = None
        if (allow_rate_keep and n_ev > 0
                and self._rate_incremental_by_global is not None):
            lut = self._rate_incremental_by_global
            gids = rma_event_pool.pair_obj_array[:n_ev]['event_type_id']
            keep_rate_mask = np.zeros(n_ev, dtype=np.bool_)
            in_range = (gids >= 0) & (gids < len(lut))
            keep_rate_mask[in_range] = lut[gids[in_range]]

        # ===== Step 6: update the pool in place =====
        rma_event_pool.update_pair_events_inplace(
            keep_mask, new_events, keep_rate_mask)

        # ===== Optional: paranoid verification (config['_pair_structure_verify']) =====
        if self.config.get('_pair_structure_verify', False):
            self._verify_pair_pool_against_full(
                rma_object_pool, rma_event_pool, current_step)

        return True

    def _compute_poison_mask(self, changed_arr, grid_dims, cell_sizes,
                             box_size, periodic) -> np.ndarray:
        """[pair-inc] Poisoned-zone mask = cells of the (old union new) coordinates, plus the neighbour ring

        - new-position side: x/y/z of the active records (a deletion has active=False, and
          its new coordinates serialize to zero, so they do not poison anything)
        - old-position side: old_x/old_y/old_z where old_valid is set (deletions and moves
          are filled in by the executing rank from the self-contained event snapshot)
        - the neighbour ring uses the cached pair grid neighbour table (periodic wrap included)
        - every input comes from the broadcast packet and the global grid parameters, so all ranks agree
        """
        Gx, Gy, Gz = grid_dims
        cell_sx, cell_sy, cell_sz = cell_sizes
        poison_mask = np.zeros(Gx * Gy * Gz, dtype=np.bool_)

        active = changed_arr['active'].astype(np.bool_) \
            if 'active' in changed_arr.dtype.names \
            else np.ones(len(changed_arr), dtype=np.bool_)
        old_valid = changed_arr['old_valid'].astype(np.bool_)

        xs = np.concatenate([changed_arr['x'][active],
                             changed_arr['old_x'][old_valid]])
        ys = np.concatenate([changed_arr['y'][active],
                             changed_arr['old_y'][old_valid]])
        zs = np.concatenate([changed_arr['z'][active],
                             changed_arr['old_z'][old_valid]])

        if len(xs) > 0:
            if periodic[0]:
                xs = np.mod(xs, box_size[0])
            if periodic[1]:
                ys = np.mod(ys, box_size[1])
            if periodic[2]:
                zs = np.mod(zs, box_size[2])
            ix = np.clip((xs / cell_sx).astype(np.int64), 0, Gx - 1)
            iy = np.clip((ys / cell_sy).astype(np.int64), 0, Gy - 1)
            iz = np.clip((zs / cell_sz).astype(np.int64), 0, Gz - 1)
            seed_cells = np.unique(ix + iy * Gx + iz * Gx * Gy)

            poison_mask[seed_cells] = True
            # Neighbour ring (periodic wrap included, from the cached table)
            nb = self._cl_neighbor_map[seed_cells]
            counts = self._cl_neighbor_counts[seed_cells]
            valid = np.arange(nb.shape[1])[None, :] < counts[:, None]
            poison_mask[nb[valid]] = True

        return poison_mask

    def _build_pairs_zone(self, all_objects, n_total, current_step,
                          poison_mask, grid_dims, cell_sizes, box_size, periodic):
        """Build the pairs of the poisoned cell subset (reusing the materialization of the full path)"""
        Gx, Gy, Gz = grid_dims
        cell_sx, cell_sy, cell_sz = cell_sizes

        x = np.ascontiguousarray(all_objects['x'], dtype=np.float64)
        y = np.ascontiguousarray(all_objects['y'], dtype=np.float64)
        z = np.ascontiguousarray(all_objects['z'], dtype=np.float64)
        obj_ids = np.ascontiguousarray(all_objects['id'], dtype=np.int64)
        owner_ranks = np.ascontiguousarray(all_objects['owner_rank'], dtype=np.int32)

        if 'type_id' in all_objects.dtype.names:
            type_ids = np.ascontiguousarray(all_objects['type_id'], dtype=np.int32)
        else:
            from .jit_event_builder import build_type_id_array
            type_ids = build_type_id_array(all_objects['type'], self._jit_type_to_int)

        cell_heads, cell_next = build_cell_bins(
            x, y, z, n_total,
            cell_sx, cell_sy, cell_sz,
            Gx, Gy, Gz,
            box_size[0], box_size[1], box_size[2],
            periodic[0], periodic[1], periodic[2]
        )

        max_pairs = 1024
        while True:
            out_idx1 = np.empty(max_pairs, dtype=np.int64)
            out_idx2 = np.empty(max_pairs, dtype=np.int64)
            out_min_ids = np.empty(max_pairs, dtype=np.int64)
            out_max_ids = np.empty(max_pairs, dtype=np.int64)
            out_owners = np.empty(max_pairs, dtype=np.int32)
            out_distances = np.empty(max_pairs, dtype=np.float64)
            out_event_types = np.empty(max_pairs, dtype=np.int32)

            n_found = jit_find_pairs_cell_list_zone(
                x, y, z, obj_ids, type_ids, owner_ranks,
                n_total,
                cell_heads, cell_next,
                self._cl_neighbor_map, self._cl_neighbor_counts,
                poison_mask,
                self._jit_match_table, self._jit_max_dist_table,
                float(box_size[0]), float(box_size[1]), float(box_size[2]),
                bool(periodic[0]), bool(periodic[1]), bool(periodic[2]),
                np.int32(self.rank),
                out_idx1, out_idx2, out_min_ids, out_max_ids,
                out_owners, out_distances, out_event_types,
                max_pairs
            )
            if n_found <= max_pairs:
                break
            max_pairs = n_found

        if n_found == 0:
            return np.array([], dtype=self.event_dtype)

        return self._create_events_from_compact_pairs(
            out_event_types[:n_found], out_min_ids[:n_found],
            out_max_ids[:n_found], out_owners[:n_found],
            out_distances[:n_found], out_idx1[:n_found], out_idx2[:n_found],
            all_objects, current_step
        )

    def _verify_pair_pool_against_full(self, rma_object_pool, rma_event_pool,
                                       current_step):
        """[pair-inc paranoid mode] Compare the incremental result against a full rebuild every step

        Multiple ranks: [fix] the check is global - predictive migration leaves a legitimate
        live copy hosted elsewhere (absent from the owner's pool), so any local comparison
        raises false alarms. The real invariant of the design is:
            the union of live copies (status != SENT) across ranks == the union of a full
            rebuild across ranks, with no duplicates (exactly one live copy of each pair).
        """
        n = rma_event_pool.count_pair
        pool_ev = rma_event_pool.pair_obj_array[:n]

        full = self.build_pair_events(rma_object_pool, current_step)
        full_keys = list(zip(full['obj_ids'][:, 0].tolist(),
                             full['obj_ids'][:, 1].tolist(),
                             full['event_type_id'].tolist())) if len(full) else []

        comm = getattr(self.domain_decomp, 'comm', None)
        if comm is not None and comm.Get_size() > 1:
            live = pool_ev[pool_ev['status'] != STATUS_SENT]
            live_keys = list(zip(live['obj_ids'][:, 0].tolist(),
                                 live['obj_ids'][:, 1].tolist(),
                                 live['event_type_id'].tolist()))
            all_live = [k for part in comm.allgather(live_keys) for k in part]
            all_full = [k for part in comm.allgather(full_keys) for k in part]

            n_dup = len(all_live) - len(set(all_live))
            missing = set(all_full) - set(all_live)
            extra = set(all_live) - set(all_full)
            if n_dup or missing or extra:
                raise RuntimeError(
                    f"[pair-inc verify] rank {self.rank} step {current_step}: "
                    f"global live-copy invariant violated "
                    f"(duplicates={n_dup}, missing={len(missing)}, "
                    f"extra={len(extra)}; e.g. missing={list(missing)[:3]}, "
                    f"extra={list(extra)[:3]})")
        else:
            own = pool_ev['owner_rank'] == self.rank
            pool_keys = set(zip(pool_ev['obj_ids'][own, 0].tolist(),
                                pool_ev['obj_ids'][own, 1].tolist(),
                                pool_ev['event_type_id'][own].tolist()))
            full_set = set(full_keys)
            if pool_keys != full_set:
                missing = full_set - pool_keys
                extra = pool_keys - full_set
                raise RuntimeError(
                    f"[pair-inc verify] rank {self.rank} step {current_step}: "
                    f"incremental pool diverged from full rebuild "
                    f"(missing={len(missing)}, extra={len(extra)}; "
                    f"e.g. missing={list(missing)[:3]}, extra={list(extra)[:3]})")

    def _collect_all_objects(self, rma_object_pool) -> np.ndarray:
        """Collect every object taking part in pair finding (local + large + ghost, deduplicated, active)"""
        # Local objects (center + boundary)
        local_objects = rma_object_pool.get_local_objects()

        # Large objects (replicated globally, present on every rank)
        if hasattr(rma_object_pool, 'large_objects') and rma_object_pool.large_count > 0:
            large_objects = rma_object_pool.large_objects[:rma_object_pool.large_count]
            large_objects = large_objects[large_objects['active']]
        else:
            large_objects = np.array([], dtype=self.object_dtype)

        # Ghost objects (from the cache)
        ghost_objects = rma_object_pool.get_ghost_objects()

        # Merge everything
        arrays_to_concat = [local_objects]
        if len(large_objects) > 0:
            arrays_to_concat.append(large_objects)
        if len(ghost_objects) > 0:
            arrays_to_concat.append(ghost_objects)

        if len(arrays_to_concat) == 1:
            all_objects = local_objects
        else:
            all_objects = np.concatenate(arrays_to_concat)

        # [fix] Deduplicate by ID: one object may sit in both the local and large arrays
        # A duplicate would make cKDTree return a self-pair (distance 0) and cause ghost clustering
        if len(all_objects) > 0:
            _, unique_idx = np.unique(all_objects['id'], return_index=True)
            if len(unique_idx) < len(all_objects):
                all_objects = all_objects[np.sort(unique_idx)]

        # Keep the active objects
        if 'active' in all_objects.dtype.names:
            all_objects = all_objects[all_objects['active']]

        return all_objects

    def _pair_grid_params(self):
        """Pair cell list grid parameters (identical to _build_pairs_cell_list)"""
        max_dist = self.max_interaction_distance
        box_size = self.domain_decomp.box_size
        periodic = self.domain_decomp.grid_system.periodic

        Gx = max(1, int(np.floor(box_size[0] / max_dist)))
        Gy = max(1, int(np.floor(box_size[1] / max_dist)))
        Gz = max(1, int(np.floor(box_size[2] / max_dist)))

        cell_sx = box_size[0] / Gx
        cell_sy = box_size[1] / Gy
        cell_sz = box_size[2] / Gz
        return (Gx, Gy, Gz), (cell_sx, cell_sy, cell_sz), box_size, periodic

    def _build_pairs_cell_list(self, all_objects, n_total, current_step):
        """Fused cell list path"""
        # Extract the SoA arrays
        x = np.ascontiguousarray(all_objects['x'], dtype=np.float64)
        y = np.ascontiguousarray(all_objects['y'], dtype=np.float64)
        z = np.ascontiguousarray(all_objects['z'], dtype=np.float64)
        obj_ids = np.ascontiguousarray(all_objects['id'], dtype=np.int64)
        owner_ranks = np.ascontiguousarray(all_objects['owner_rank'], dtype=np.int32)

        if 'type_id' in all_objects.dtype.names:
            type_ids = np.ascontiguousarray(all_objects['type_id'], dtype=np.int32)
        else:
            from .jit_event_builder import build_type_id_array
            type_ids = build_type_id_array(all_objects['type'], self._jit_type_to_int)

        # Compute the cell list grid
        max_dist = self.max_interaction_distance
        box_size = self.domain_decomp.box_size
        periodic = self.domain_decomp.grid_system.periodic

        Gx = max(1, int(np.floor(box_size[0] / max_dist)))
        Gy = max(1, int(np.floor(box_size[1] / max_dist)))
        Gz = max(1, int(np.floor(box_size[2] / max_dist)))

        cell_size_x = box_size[0] / Gx
        cell_size_y = box_size[1] / Gy
        cell_size_z = box_size[2] / Gz

        # Lazily initialize the neighbour map (first call, or after a grid change)
        grid_key = (Gx, Gy, Gz)
        if self._cl_grid != grid_key:
            self._cl_neighbor_map, self._cl_neighbor_counts = build_neighbor_map(
                Gx, Gy, Gz, max_dist,
                cell_size_x, cell_size_y, cell_size_z,
                periodic[0], periodic[1], periodic[2]
            )
            self._cl_grid = grid_key

        # Build the cell bins
        cell_heads, cell_next = build_cell_bins(
            x, y, z, n_total,
            cell_size_x, cell_size_y, cell_size_z,
            Gx, Gy, Gz,
            box_size[0], box_size[1], box_size[2],
            periodic[0], periodic[1], periodic[2]
        )

        # Preallocate the output (heuristic capacity; the kernel returns the true total and we grow on overflow)
        max_pairs = min(n_total * (n_total - 1) // 2, n_total * 50)
        max_pairs = max(max_pairs, 64)

        while True:
            out_idx1 = np.empty(max_pairs, dtype=np.int64)
            out_idx2 = np.empty(max_pairs, dtype=np.int64)
            out_min_ids = np.empty(max_pairs, dtype=np.int64)
            out_max_ids = np.empty(max_pairs, dtype=np.int64)
            out_owners = np.empty(max_pairs, dtype=np.int32)
            out_distances = np.empty(max_pairs, dtype=np.float64)
            out_event_types = np.empty(max_pairs, dtype=np.int32)

            # Fused kernel (returns the number of qualifying rows, possibly above max_pairs)
            n_found = jit_find_pairs_cell_list(
                x, y, z, obj_ids, type_ids, owner_ranks,
                n_total,
                cell_heads, cell_next,
                self._cl_neighbor_map, self._cl_neighbor_counts,
                self._jit_match_table, self._jit_max_dist_table,
                float(box_size[0]), float(box_size[1]), float(box_size[2]),
                bool(periodic[0]), bool(periodic[1]), bool(periodic[2]),
                np.int32(self.rank),
                out_idx1, out_idx2, out_min_ids, out_max_ids,
                out_owners, out_distances, out_event_types,
                max_pairs
            )

            if n_found <= max_pairs:
                break
            # Buffer overflow: grow to the true size and rerun (no pair is silently dropped)
            if self.verbose or self.rank == 0:
                print(f"[Rank {self.rank}] Cell list pair buffer overflow "
                      f"({n_found} > {max_pairs}), retrying with larger buffer")
            max_pairs = n_found

        if n_found == 0:
            return np.array([], dtype=self.event_dtype)

        # Trim to the valid results
        idx1 = out_idx1[:n_found]
        idx2 = out_idx2[:n_found]
        min_ids = out_min_ids[:n_found]
        max_ids = out_max_ids[:n_found]
        owners = out_owners[:n_found]
        distances = out_distances[:n_found]
        event_types = out_event_types[:n_found]

        if self.verbose:
            print(f"[Rank {self.rank}] Cell list found {n_found} valid pairs")

        return self._create_events_from_compact_pairs(
            event_types, min_ids, max_ids, owners, distances,
            idx1, idx2, all_objects, current_step
        )

    def _get_box_periodic(self):
        """Get (box_size, periodic); returns (None, None) when unavailable"""
        box_size = getattr(self.domain_decomp, 'box_size', None)
        grid = getattr(self.domain_decomp, 'grid_system', None)
        periodic = getattr(grid, 'periodic', None) if grid is not None else \
            getattr(self.domain_decomp, 'periodic', None)
        if box_size is None or periodic is None:
            return None, None
        return box_size, periodic

    def _build_pairs_kdtree(self, all_objects, n_total, current_step):
        """cKDTree fallback path (periodic-boundary aware)"""
        positions = np.column_stack([
            all_objects['x'],
            all_objects['y'],
            all_objects['z']
        ])

        box_size, periodic = self._get_box_periodic()
        use_pbc = box_size is not None and periodic is not None and any(periodic)

        if use_pbc and all(periodic):
            # Fully periodic: wrap the coordinates and use the toroidal topology of cKDTree, so cross-boundary pairs are found
            box_arr = np.asarray(box_size, dtype=np.float64)
            wrapped = np.mod(positions, box_arr)
            # Numerical guard: the mod result may equal box (floating point), so wrap to 0
            wrapped = np.where(wrapped >= box_arr, 0.0, wrapped)
            tree = cKDTree(wrapped, boxsize=box_arr)
        else:
            # Mixed or non-periodic: build the tree on the raw coordinates (a mixed setup may
            # miss cross-boundary candidates; the cell list path handles them fully).
            tree = cKDTree(positions)
        pairs_set = tree.query_pairs(r=self.max_interaction_distance, output_type='set')

        if len(pairs_set) == 0:
            return np.array([], dtype=self.event_dtype)

        pairs_array = np.array(list(pairs_set), dtype=np.int64)
        n_pairs = len(pairs_array)

        if self.verbose:
            print(f"[Rank {self.rank}] cKDTree query_pairs found {n_pairs} candidate pairs")

        idx1 = pairs_array[:, 0]
        idx2 = pairs_array[:, 1]

        ids1 = all_objects['id'][idx1]
        ids2 = all_objects['id'][idx2]
        owners1 = all_objects['owner_rank'][idx1]
        owners2 = all_objects['owner_rank'][idx2]
        x1, y1, z1 = all_objects['x'][idx1], all_objects['y'][idx1], all_objects['z'][idx1]
        x2, y2, z2 = all_objects['x'][idx2], all_objects['y'][idx2], all_objects['z'][idx2]

        distances = np.empty(n_pairs, dtype=np.float64)
        min_ids = np.empty(n_pairs, dtype=np.int64)
        max_ids = np.empty(n_pairs, dtype=np.int64)

        if use_pbc:
            jit_compute_distances_pbc(
                x1, y1, z1, x2, y2, z2, ids1, ids2,
                float(box_size[0]), float(box_size[1]), float(box_size[2]),
                bool(periodic[0]), bool(periodic[1]), bool(periodic[2]),
                distances, min_ids, max_ids)
        else:
            jit_compute_distances(x1, y1, z1, x2, y2, z2, ids1, ids2,
                                  distances, min_ids, max_ids)

        if 'type_id' in all_objects.dtype.names:
            type_ids1 = all_objects['type_id'][idx1]
            type_ids2 = all_objects['type_id'][idx2]
        else:
            from .jit_event_builder import build_type_id_array
            type_ids1 = build_type_id_array(all_objects['type'][idx1], self._jit_type_to_int)
            type_ids2 = build_type_id_array(all_objects['type'][idx2], self._jit_type_to_int)

        out_event_types = np.empty(n_pairs, dtype=np.int32)
        jit_filter_pairs_lut(type_ids1, type_ids2, distances,
                             owners1, owners2, ids1, ids2,
                             self._jit_match_table,
                             self._jit_max_dist_table,
                             np.int32(self.rank), out_event_types)

        # Owners must be prepared for the compacted event creation
        # Derived from ids1/ids2 and min_ids
        eo_all = np.where(ids1 == min_ids, owners1, owners2)

        counts = np.empty(n_pairs, dtype=np.int32)
        jit_count_valid_event_rows(out_event_types, counts)
        total_valid = int(np.sum(counts, dtype=np.int64))
        if total_valid == 0:
            return np.array([], dtype=self.event_dtype)

        offsets = np.empty(n_pairs, dtype=np.int64)
        if n_pairs > 0:
            offsets[0] = 0
            if n_pairs > 1:
                offsets[1:] = np.cumsum(counts[:-1], dtype=np.int64)

        c_event_types = np.empty(total_valid, dtype=np.int32)
        c_idx1 = np.empty(total_valid, dtype=np.int64)
        c_idx2 = np.empty(total_valid, dtype=np.int64)
        c_min_ids = np.empty(total_valid, dtype=np.int64)
        c_max_ids = np.empty(total_valid, dtype=np.int64)
        c_owners = np.empty(total_valid, dtype=np.int32)
        c_distances = np.empty(total_valid, dtype=np.float64)

        jit_compact_pair_rows(
            out_event_types, idx1, idx2, min_ids, max_ids, eo_all, distances,
            offsets,
            c_event_types, c_idx1, c_idx2,
            c_min_ids, c_max_ids, c_owners, c_distances
        )

        return self._create_events_from_compact_pairs(
            c_event_types, c_min_ids, c_max_ids, c_owners, c_distances,
            c_idx1, c_idx2, all_objects, current_step
        )

    def _compute_pair_swap_mask(self, event_types, idx1, idx2, all_objects):
        """Compute the swap mask from the type_pairs definition so objects[:, 0] matches the first type of each pair.

        Swapping the indices before embedding (rather than swapping the NumPy fancy-index
        copies afterwards) keeps objects / obj_ids / params in the normalized order.
        """
        n = len(event_types)
        swap = np.zeros(n, dtype=bool)
        if n == 0 or 'type_id' not in all_objects.dtype.names:
            return swap

        type_to_int = self._jit_type_to_int or self.config.get('_type_to_int', {})
        if not type_to_int:
            return swap

        tid1 = all_objects['type_id'][idx1]
        tid2 = all_objects['type_id'][idx2]

        for etype_int in np.unique(event_types):
            event_name = self._jit_event_int_to_name.get(int(etype_int))
            if event_name is None:
                continue
            fconfig = self.vectorized_filters.get(event_name, {})
            etype_mask = (event_types == etype_int)
            for t1, t2 in fconfig.get('type_pairs', []):
                if t1 == t2:
                    continue
                a = type_to_int.get(t1, -1)
                b = type_to_int.get(t2, -1)
                if a < 0 or b < 0:
                    continue
                swap |= etype_mask & (tid1 == b) & (tid2 == a)

        return swap

    def _create_events_from_compact_pairs(self, event_types, min_ids, max_ids,
                                          owners, distances, idx1, idx2,
                                          all_objects, current_step):
        """Materialize the events from the compact pair arrays at once, avoiding a Python group-by-type."""
        n_events = len(event_types)
        if n_events == 0:
            return np.array([], dtype=self.event_dtype)

        event_types = np.ascontiguousarray(event_types, dtype=np.int32)

        # [fix] Normalize obj1/obj2 at the index level before embedding (heterogeneous pairs: objects[0] = first type of type_pairs)
        swap = self._compute_pair_swap_mask(event_types, idx1, idx2, all_objects)
        if np.any(swap):
            idx1_orig = idx1
            idx1 = np.where(swap, idx2, idx1)
            idx2 = np.where(swap, idx1_orig, idx2)

        events = np.zeros(n_events, dtype=self.event_dtype)

        events['event_type_id'] = self._pair_local_to_global[event_types]
        events['status'] = STATUS_TODO
        events['owner_rank'] = owners
        events['rate'] = 0.0
        events['weight'] = self._pair_weights_by_local[event_types]
        events['obj_ids'][:, 0] = min_ids
        events['obj_ids'][:, 1] = max_ids
        if events['obj_ids'].shape[1] > 2:
            events['obj_ids'][:, 2:] = -1

        prefix = 1
        id_high = ((prefix << 61)
                   | (current_step << 26)
                   | (self.rank << 10)
                   | event_types.astype(np.int64))
        if 'id_high' in self.event_dtype.names:
            events['id_high'] = id_high
        if 'id_low' in self.event_dtype.names:
            events['id_low'] = np.arange(n_events, dtype=np.int64) + 1
        events['id'] = 0

        if 'params' in self.event_dtype.names:
            events['params'][:, 0] = distances

        if 'objects' in self.event_dtype.names:
            events['objects'][:, 0] = all_objects[idx1]
            events['objects'][:, 1] = all_objects[idx2]
            n_slots = events['objects'].shape[1]
            if n_slots > 2:
                events['objects'][:, 2:]['id'] = -1
                events['objects'][:, 2:]['active'] = False

        if not hasattr(self, '_param_extractor_warned'):
            self._param_extractor_warned = set()

        for etype_int in np.unique(event_types):
            event_name = self._jit_event_int_to_name.get(int(etype_int))
            if event_name is None:
                continue

            valid_mask = (event_types == etype_int)
            valid_indices = np.where(valid_mask)[0]
            if len(valid_indices) == 0:
                continue

            if self.verbose:
                print(f"[Rank {self.rank}] Event type '{event_name}': {len(valid_indices)} events")

            param_extractor = self.pair_param_extractors.get(event_name)
            if 'params' in self.event_dtype.names and param_extractor is not None:
                max_params = self.event_dtype.fields['params'][0].shape[0]
                for i in valid_indices:
                    obj1 = all_objects[idx1[i]]
                    obj2 = all_objects[idx2[i]]
                    try:
                        params = param_extractor(obj1, obj2)
                        n_params = min(len(params), max_params)
                        events[i]['params'][:n_params] = params[:n_params]
                    except Exception as exc:
                        # Parameter extraction failed: keep the kernel distance, zero the rest,
                        # and warn once (no longer completely silent)
                        if event_name not in self._param_extractor_warned:
                            self._param_extractor_warned.add(event_name)
                            print(f"[Rank {self.rank}] Warning: param extractor for "
                                  f"'{event_name}' raised {type(exc).__name__}: {exc}; "
                                  f"falling back to distance-only params.")
                        events[i]['params'][0] = distances[i]

        return events

    def _batch_create_events(self, event_type: str,
                              min_ids: np.ndarray,
                              max_ids: np.ndarray,
                              owners: np.ndarray,
                              distances: np.ndarray,
                              idx1: np.ndarray,
                              idx2: np.ndarray,
                              all_objects: np.ndarray,
                              current_step: int) -> np.ndarray:
        """
        Create the events in bulk (vectorized)

        Args:
            event_type: event type
            min_ids, max_ids: normalized object IDs
            owners: rank owning the event
            distances: array of distances
            idx1, idx2: indices of the objects inside all_objects
            all_objects: array of every object
            current_step: current step

        Returns:
            the event array
        """
        n_events = len(min_ids)

        # Allocate the event array at once
        events = np.zeros(n_events, dtype=self.event_dtype)

        # Event weight
        weight = self.event_weights.get(event_type, 1.0)

        # Parameter extractor
        param_extractor = self.pair_param_extractors.get(event_type, None)

        # Event type index
        type_index = self.pair_event_types.get(event_type, 0)

        # ===== Fill the basic fields (vectorized) =====
        if 'event_type_id' in self.event_dtype.names:
            events['event_type_id'] = self._global_event_name_to_int.get(event_type, -1)
        events['status'] = STATUS_TODO
        events['owner_rank'] = owners
        events['rate'] = 0.0
        events['weight'] = weight
        events['obj_ids'][:, 0] = min_ids
        events['obj_ids'][:, 1] = max_ids

        # ===== Generate the 128-bit ID (vectorized) =====
        # Prefix = 01 (pair event)
        prefix = 1
        id_high = (prefix << 61) | (current_step << 26) | (self.rank << 10) | type_index

        if 'id_high' in self.event_dtype.names:
            events['id_high'] = id_high

        if 'id_low' in self.event_dtype.names:
            # [fix] The sequence number is id_low, unique under one id_high
            # id_high already encodes (step, rank, type); id_low only separates events of one type
            events['id_low'] = np.arange(n_events, dtype=np.int64) + 1

        events['id'] = 0  # 128-bit ID mode

        # ===== Fill the parameters =====
        if 'params' in self.event_dtype.names:
            max_params = self.event_dtype.fields['params'][0].shape[0]

            if param_extractor is not None:
                # Call the parameter extractor pair by pair
                for i in range(n_events):
                    obj1 = all_objects[idx1[i]]
                    obj2 = all_objects[idx2[i]]
                    try:
                        params = param_extractor(obj1, obj2)
                        n_params = min(len(params), max_params)
                        events[i]['params'][:n_params] = params[:n_params]
                    except Exception:
                        events[i]['params'][0] = distances[i]
            else:
                # Default: the first parameter is the distance
                events['params'][:, 0] = distances

        # ===== Embed the object data (self-contained mode) =====
        if 'objects' in self.event_dtype.names:
            # Vectorized embedding: NumPy fancy indexing replaces the per-event loop
            events['objects'][:, 0] = all_objects[idx1]
            events['objects'][:, 1] = all_objects[idx2]
            # Mark the remaining slots invalid
            n_slots = events['objects'].shape[1]
            if n_slots > 2:
                events['objects'][:, 2:]['id'] = -1
                events['objects'][:, 2:]['active'] = False

            # Normalize the object order: objects[:, 0] matches the first type of type_pairs
            fconfig = self.vectorized_filters.get(event_type, {})
            type_pairs = fconfig.get('type_pairs', [])
            if type_pairs and n_events > 0:
                self._normalize_pair_order(events, type_pairs)

        return events

    def _normalize_pair_order(self, events, type_pairs):
        """Normalize objects[:, 0] and objects[:, 1] to the type_pairs order

        Guarantees that the type of objects[:, 0] is the first type of each type_pairs entry.
        With type_pairs=[('H', 'V')], objects[:, 0] is always H and objects[:, 1] always V.
        Symmetric pairs such as ('H', 'H') need no swap.
        """
        n = len(events)
        if n == 0:
            return

        type_to_int = self.config.get('_type_to_int', {})
        needs_swap = np.zeros(n, dtype=bool)

        if 'type_id' in events['objects'].dtype.names and type_to_int:
            obj0_tids = events['objects'][:, 0]['type_id']
            obj1_tids = events['objects'][:, 1]['type_id']

            for type1, type2 in type_pairs:
                if type1 == type2:
                    continue
                tid1 = type_to_int.get(type1, -1)
                tid2 = type_to_int.get(type2, -1)
                needs_swap |= (obj0_tids == tid2) & (obj1_tids == tid1)
        elif 'type' in events['objects'].dtype.names:
            obj0_types = events['objects'][:, 0]['type']
            obj1_types = events['objects'][:, 1]['type']
            for type1, type2 in type_pairs:
                if type1 == type2:
                    continue
                needs_swap |= (obj0_types == type2) & (obj1_types == type1)

        if np.any(needs_swap):
            temp = events['objects'][needs_swap, 0].copy()
            events['objects'][needs_swap, 0] = events['objects'][needs_swap, 1]
            events['objects'][needs_swap, 1] = temp
            if 'obj_ids' in events.dtype.names:
                temp_ids = events['obj_ids'][needs_swap, 0].copy()
                events['obj_ids'][needs_swap, 0] = events['obj_ids'][needs_swap, 1]
                events['obj_ids'][needs_swap, 1] = temp_ids


class VectorizedSingleEventBuilder:
    """
    Vectorized single-object event builder

    Follows the MPI+OpenMP pattern of VectorizedPairEventBuilder:
    1. take the single-object event classes and their object_types from EventRegistry
    2. prebuild the 2D lookup table (type_id -> event_type_ids)
    3. classify in parallel in a JIT kernel (Numba prange)
    4. create the events in bulk with NumPy
    5. support the incremental mode (only changed_objects are processed)
    """

    def __init__(self, domain_decomp: DomainDecomposition,
                 config: dict,
                 object_dtype: np.dtype,
                 event_dtype: np.dtype,
                 verbose: bool = False):
        self.domain_decomp = domain_decomp
        self.comm = domain_decomp.comm
        self.rank = domain_decomp.rank
        self.size = domain_decomp.size
        self.config = config
        self.object_dtype = object_dtype
        self.event_dtype = event_dtype
        self.verbose = verbose

        # Event weights
        self.event_weights = config.get('event_weights', {})

        # Build the lookup table from EventRegistry
        self._event_registry = config.get('_event_registry')
        self._type_to_int = config.get('_type_to_int', {})
        self._global_event_name_to_int = config.get('_event_name_to_int', {})

        # Event type mapping
        self._single_event_names = []
        self._event_name_to_int = {}
        self._event_int_to_name = {}
        self._match_table_2d = None
        self._event_classes = {}  # event_name → event_class
        self._single_local_to_global = np.empty(0, dtype=np.int32)
        self._single_weights_by_local = np.empty(0, dtype=np.float64)

        # Incremental mode: snapshot of the old events
        self.prev_single_events = None
        self.last_reused_rate_sum = 0.0

        if self._event_registry and self._type_to_int:
            self._precompute_tables()

    def _precompute_tables(self):
        """Prebuild the JIT lookup table"""
        self._single_event_names = self._event_registry.get_single_events()
        if not self._single_event_names:
            return

        # Cache the event classes
        for name in self._single_event_names:
            self._event_classes[name] = self._event_registry.get(name)

        # Build the 2D LUT
        self._match_table_2d, self._event_int_to_name, self._event_name_to_int = \
            build_single_match_table(self._event_registry, self._type_to_int)
        n_types = len(self._event_name_to_int)
        self._single_local_to_global = np.full(max(n_types, 1), -1, dtype=np.int32)
        self._single_weights_by_local = np.ones(max(n_types, 1), dtype=np.float64)
        for event_name, local_id in self._event_name_to_int.items():
            self._single_local_to_global[local_id] = self._global_event_name_to_int.get(event_name, -1)
            self._single_weights_by_local[local_id] = self.event_weights.get(event_name, 1.0)

        # [feat-2] Class-level rate increment LUT: global event_type_id -> incremental_rate
        self._rate_incremental_by_global = None
        if self._global_event_name_to_int:
            gmax = max(self._global_event_name_to_int.values())
            if gmax >= 0:
                lut = np.zeros(gmax + 1, dtype=np.bool_)
                for event_name, gid in self._global_event_name_to_int.items():
                    event_class = self._event_registry.get(event_name)
                    if event_class is not None and 0 <= gid <= gmax:
                        lut[gid] = bool(getattr(event_class, 'incremental_rate', False))
                self._rate_incremental_by_global = lut

        if self.verbose and self.rank == 0:
            print(f"[VectorizedSingleBuilder] {len(self._single_event_names)} event types, "
                  f"match_table shape: {self._match_table_2d.shape}")

    def build_single_events(self, objects: np.ndarray,
                             changed_objects: Optional[Set[int]] = None,
                             current_step: int = 0,
                             migrated_obj_ids: Optional[Set[int]] = None) -> np.ndarray:
        """
        Build the single-object events (vectorized pipeline)

        Args:
            objects: array of objects owned locally (owner_rank == self.rank)
            changed_objects: set of changed object IDs (None = full, set = incremental)
            current_step: current step

        Returns:
            the event array (event_dtype)
        """
        if (self._match_table_2d is None or
                len(self._single_event_names) == 0 or len(objects) == 0):
            self.last_reused_rate_sum = 0.0
            return np.array([], dtype=self.event_dtype)

        # ===== Step 1: build the eligible mask =====
        self.last_reused_rate_sum = 0.0
        changed_obj_ids = set()
        if changed_objects is None:
            eligible_mask = np.ones(len(objects), dtype=np.bool_)
        elif 'new' in objects.dtype.names and 'changed' in objects.dtype.names:
            eligible_mask = np.ascontiguousarray(objects['new'] | objects['changed'], dtype=np.bool_)
            if np.any(eligible_mask):
                changed_obj_ids = set(objects['id'][eligible_mask].tolist())
            else:
                eligible_mask = np.ones(len(objects), dtype=np.bool_)
                changed_objects = None
        elif len(changed_objects) > 0:
            eligible_mask = np.isin(objects['id'], list(changed_objects))
            changed_obj_ids = set(objects['id'][eligible_mask].tolist())
        else:
            eligible_mask = np.ones(len(objects), dtype=np.bool_)
            changed_objects = None

        # ===== Step 2: two-pass JIT count + prefix sum =====
        new_events_list = []
        if len(objects) > 0:
            if 'type_id' in objects.dtype.names:
                type_ids = np.ascontiguousarray(objects['type_id'], dtype=np.int32)
            else:
                from .jit_event_builder import build_type_id_array
                type_ids = build_type_id_array(
                    np.array([str(o['type']).strip() for o in objects]),
                    self._type_to_int)

            owner_ranks = np.ascontiguousarray(objects['owner_rank'], dtype=np.int32)
            eligible_mask = np.ascontiguousarray(eligible_mask, dtype=np.bool_)

            counts = np.empty(len(objects), dtype=np.int32)
            jit_count_single_events_lut(
                type_ids, owner_ranks, eligible_mask, np.int32(self.rank),
                self._match_table_2d, counts
            )

            total_events = int(np.sum(counts, dtype=np.int64))
            if total_events > 0:
                offsets = np.empty(len(objects), dtype=np.int64)
                offsets[0] = 0
                if len(objects) > 1:
                    offsets[1:] = np.cumsum(counts[:-1], dtype=np.int64)

                compact_event_types = np.empty(total_events, dtype=np.int32)
                compact_object_indices = np.empty(total_events, dtype=np.int64)
                jit_fill_single_events_lut(
                    type_ids, owner_ranks, eligible_mask, np.int32(self.rank),
                    self._match_table_2d, offsets,
                    compact_event_types, compact_object_indices
                )

                events = self._materialize_single_events(
                    objects,
                    compact_object_indices,
                    compact_event_types,
                    current_step,
                )
                if len(events) > 0:
                    new_events_list.append(events)
        # ===== Step 4: reuse the old events in incremental mode =====
        if changed_objects is not None and self.prev_single_events is not None:
            valid_obj_ids = objects['id']
            if migrated_obj_ids:
                valid_obj_ids = np.concatenate([
                    valid_obj_ids,
                    np.fromiter(migrated_obj_ids, dtype=np.int64, count=len(migrated_obj_ids))
                ])
            valid_obj_ids_sorted = np.unique(np.asarray(valid_obj_ids, dtype=np.int64))

            if changed_obj_ids:
                changed_obj_ids_sorted = np.array(sorted(changed_obj_ids), dtype=np.int64)
            else:
                changed_obj_ids_sorted = np.empty(0, dtype=np.int64)

            prev_events = self.prev_single_events
            prev_obj_ids = np.ascontiguousarray(prev_events['obj_ids'][:, 0], dtype=np.int64)
            prev_status = np.ascontiguousarray(prev_events['status'], dtype=np.int32)
            actions = np.empty(len(prev_events), dtype=np.int8)

            jit_scan_reusable_single_events(
                prev_obj_ids,
                prev_status,
                valid_obj_ids_sorted,
                changed_obj_ids_sorted,
                np.int32(STATUS_DONE),
                np.int32(STATUS_SENT),
                actions
            )

            reuse_mask = actions > 0
            if np.any(reuse_mask):
                reused_events = prev_events[reuse_mask].copy()
                reused_actions = actions[reuse_mask].copy()

                # [feat-2] Class-level rate increment gate: a class without incremental_rate
                # has its rate recomputed every step even when the object is unchanged (DONE -> TODO)
                lut = getattr(self, '_rate_incremental_by_global', None)
                if lut is not None and 'event_type_id' in reused_events.dtype.names:
                    gids = reused_events['event_type_id']
                    allowed = np.zeros(len(gids), dtype=np.bool_)
                    in_range = (gids >= 0) & (gids < len(lut))
                    allowed[in_range] = lut[gids[in_range]]
                    reused_actions[(reused_actions == 1) & ~allowed] = 2

                todo_mask = reused_actions == 2
                if np.any(todo_mask):
                    reused_events['status'][todo_mask] = STATUS_TODO

                done_mask = reused_actions == 1
                if np.any(done_mask):
                    reused_rates = reused_events['rate'][done_mask]
                    valid_rate_mask = reused_rates >= 0.0
                    if np.any(valid_rate_mask):
                        self.last_reused_rate_sum += float(np.sum(
                            reused_rates[valid_rate_mask], dtype=np.float64
                        ))

                new_events_list.append(reused_events)

        # ===== Merge the results =====
        if len(new_events_list) == 0:
            return np.array([], dtype=self.event_dtype)

        result = np.concatenate(new_events_list)

        if self.verbose:
            print(f"[Rank {self.rank}] VectorizedSingleBuilder: {len(result)} events")

        return result

    def _materialize_single_events(self, objects: np.ndarray,
                                   object_indices: np.ndarray,
                                   local_event_types: np.ndarray,
                                   current_step: int) -> np.ndarray:
        """Materialize the single-object events at once from the compact two-pass JIT output."""
        n_events = len(object_indices)
        if n_events == 0:
            return np.array([], dtype=self.event_dtype)

        matched_objects = objects[object_indices]
        local_event_types = np.ascontiguousarray(local_event_types, dtype=np.int32)
        events = np.zeros(n_events, dtype=self.event_dtype)

        events['event_type_id'] = self._single_local_to_global[local_event_types]
        events['status'] = STATUS_TODO
        events['owner_rank'] = self.rank
        events['rate'] = 0.0
        events['weight'] = self._single_weights_by_local[local_event_types]
        events['obj_ids'][:, 0] = matched_objects['id']
        if events['obj_ids'].shape[1] > 1:
            events['obj_ids'][:, 1:] = -1

        prefix = 0
        id_high = ((prefix << 61)
                   | (current_step << 26)
                   | (self.rank << 10)
                   | local_event_types.astype(np.int64))
        if 'id_high' in self.event_dtype.names:
            events['id_high'] = id_high
        if 'id_low' in self.event_dtype.names:
            events['id_low'] = matched_objects['id']
        events['id'] = 0

        if 'objects' in self.event_dtype.names:
            events['objects'][:, 0] = matched_objects
            n_slots = events['objects'].shape[1]
            if n_slots > 1:
                events['objects'][:, 1:]['id'] = -1
                events['objects'][:, 1:]['active'] = False

        if 'params' in self.event_dtype.names:
            max_params = self.event_dtype.fields['params'][0].shape[0]
            int_to_type = self.config.get('_int_to_type', {})
            for local_id in np.unique(local_event_types):
                event_name = self._event_int_to_name.get(int(local_id))
                event_class = self._event_classes.get(event_name)
                param_names = tuple(getattr(event_class, 'params', ()) or ())
                if event_class is None or len(param_names) == 0:
                    continue

                valid_indices = np.where(local_event_types == local_id)[0]
                props_mapping = self.config.get('_props_mapping', {}).get('float_props', {})
                for i in valid_indices:
                    obj = matched_objects[i]
                    obj_dict = {name: obj[name] for name in obj.dtype.names}
                    obj_dict['type'] = int_to_type.get(int(obj['type_id']), '')
                    # [fix] Expand the named float_props (such as 'E_m', 'n'), otherwise
                    # extract_params cannot find the named parameters carried by the object -> 0.0
                    if 'float_props' in obj.dtype.names:
                        mapping = props_mapping.get(obj_dict['type'], {})
                        fp = obj['float_props']
                        for prop_name, slot_idx in mapping.items():
                            if isinstance(slot_idx, (int, np.integer)) and slot_idx < len(fp):
                                obj_dict.setdefault(prop_name, float(fp[slot_idx]))
                    try:
                        params_dict = event_class.extract_params(
                            [obj_dict], self.config, event_name=event_name)
                        params_list = [params_dict.get(p, 0.0) for p in param_names]
                        n_params = min(len(params_list), max_params)
                        events[i]['params'][:n_params] = params_list[:n_params]
                    except Exception:
                        continue

        return events

    def save_snapshot(self, events: np.ndarray):
        """Save the event snapshot (reused by the incremental mode)"""
        self.prev_single_events = events.copy() if len(events) > 0 else None


class VectorizedZeroEventBuilder:
    """
    Vectorized zero-object event builder

    Two spatial modes are supported:
    - global: one event per rank at most (the existing logic, handled by zero_obj_builders)
    - per_cell: one event per local cell (e.g. dislocation nucleation), built vectorized

    MPI+OpenMP strategy:
    - MPI: every rank creates events only for domain_decomp.get_local_cells()
    - OpenMP: jit_build_per_cell_zero_events fills them with Numba prange
    """

    def __init__(self, domain_decomp: DomainDecomposition,
                 config: dict,
                 event_dtype: np.dtype,
                 verbose: bool = False):
        self.domain_decomp = domain_decomp
        self.comm = domain_decomp.comm
        self.rank = domain_decomp.rank
        self.size = domain_decomp.size
        self.config = config
        self.event_dtype = event_dtype
        self.verbose = verbose
        self._global_event_name_to_int = config.get('_event_name_to_int', {})

        # Take the zero-object event classes from EventRegistry
        self._event_registry = config.get('_event_registry')

        # Classify the events: global versus per_cell
        self._global_events = []   # event_name list
        self._per_cell_events = [] # event_name list
        self._event_classes = {}   # event_name → event_class

        if self._event_registry:
            for name in self._event_registry.get_zero_events():
                event_class = self._event_registry.get(name)
                self._event_classes[name] = event_class
                if getattr(event_class, 'spatial_mode', 'global') == 'per_cell':
                    self._per_cell_events.append(name)
                else:
                    self._global_events.append(name)

        # Precompute the local cell positions (N_cells, 3)
        self._local_cell_ids = None
        self._local_cell_positions = None
        if self._per_cell_events:
            self._precompute_cell_positions()

    def _precompute_cell_positions(self):
        """Precompute the IDs and centre coordinates of the local cells"""
        local_cells = sorted(self.domain_decomp.get_local_cells())
        n_cells = len(local_cells)
        if n_cells == 0:
            self._local_cell_ids = np.array([], dtype=np.int64)
            self._local_cell_positions = np.empty((0, 3), dtype=np.float64)
            return

        self._local_cell_ids = np.array(local_cells, dtype=np.int64)
        positions = np.empty((n_cells, 3), dtype=np.float64)
        grid = self.domain_decomp.grid_system
        for i, cid in enumerate(local_cells):
            cx, cy, cz = grid.get_cell_center(cid)
            positions[i, 0] = cx
            positions[i, 1] = cy
            positions[i, 2] = cz
        self._local_cell_positions = positions

        if self.verbose and self.rank == 0:
            print(f"[VectorizedZeroBuilder] {n_cells} local cells, "
                  f"{len(self._per_cell_events)} per_cell events")

    def build_zero_events(self, config: dict, current_step: int = 0) -> np.ndarray:
        """
        Build every zero-object event

        For each event type:
        - global mode: handled by zero_obj_builders (not built here)
        - per_cell mode: _build_per_cell_events()

        Returns:
            the per_cell event array (event_dtype); global events are not included
        """
        if not self._per_cell_events:
            return np.array([], dtype=self.event_dtype)

        # Check whether local_cells changed (load balancing moves them)
        current_local_cells = sorted(self.domain_decomp.get_local_cells())
        if (self._local_cell_ids is None or
                len(current_local_cells) != len(self._local_cell_ids) or
                not np.array_equal(np.array(current_local_cells, dtype=np.int64),
                                   self._local_cell_ids)):
            self._precompute_cell_positions()

        all_events = []
        for event_name in self._per_cell_events:
            event_class = self._event_classes[event_name]
            events = self._build_per_cell_events(
                event_name, event_class, config, current_step)
            if len(events) > 0:
                all_events.append(events)

        if not all_events:
            return np.array([], dtype=self.event_dtype)

        return np.concatenate(all_events)

    def _build_per_cell_events(self, event_name: str, event_class,
                                config: dict, current_step: int) -> np.ndarray:
        """
        Build the event array of the per_cell zero-object events

        Step 1: get the local cell positions (precomputed)
        Step 2: interpolate the field values in bulk
        Step 3: preallocate the event array
        Step 4: vectorized filling + JIT kernel
        Step 5: vectorized 128-bit ID generation
        """
        cell_ids = self._local_cell_ids
        cell_positions = self._local_cell_positions
        n_cells = len(cell_ids)
        if n_cells == 0:
            return np.array([], dtype=self.event_dtype)

        field_deps = getattr(event_class, 'field_deps', [])
        n_field_deps = len(field_deps)

        # Check the params capacity: n_field_deps + 3 (cell_x, cell_y, cell_z) are needed
        max_params = self.event_dtype.fields['params'][0].shape[0] if 'params' in self.event_dtype.names else 8
        required_params = n_field_deps + 3
        if required_params > max_params:
            raise ValueError(
                f"per_cell event '{event_name}' needs {required_params} param slots "
                f"({n_field_deps} fields + 3 coords), but max_params={max_params}. "
                f"Increase event_dtype params size."
            )

        # Step 2: interpolate the field values in bulk
        if n_field_deps > 0:
            field_values_2d = np.zeros((n_cells, n_field_deps), dtype=np.float64)
            physics = config.get('physics')
            for f_idx, field_name in enumerate(field_deps):
                if physics is not None and hasattr(physics, 'interpolate_field_batch'):
                    vals = physics.interpolate_field_batch(field_name, cell_positions)
                    if vals is not None:
                        field_values_2d[:, f_idx] = vals
        else:
            field_values_2d = np.zeros((n_cells, 1), dtype=np.float64)

        # Step 3: preallocate the event array
        events = np.zeros(n_cells, dtype=self.event_dtype)
        max_params = self.event_dtype.fields['params'][0].shape[0] if 'params' in self.event_dtype.names else 8

        # Step 4: vectorized filling
        if 'event_type_id' in self.event_dtype.names:
            events['event_type_id'] = self._global_event_name_to_int.get(event_name, -1)
        events['status'] = STATUS_TODO
        events['owner_rank'] = self.rank

        weight = getattr(event_class, 'weight', 1.0)
        events['weight'] = weight

        # obj_ids: [0]=cell_id, every other slot set to -1
        # [fix] Only slot 1 used to be set; slots 2+ kept the 0 from np.zeros and the
        # non-negative counting rule misfiled the event as a pair event
        events['obj_ids'][:, 1:] = -1

        # The JIT kernel fills obj_ids[:, 0] and params
        out_obj_ids_0 = events['obj_ids'][:, 0]
        out_params = events['params']

        from .jit_event_builder import jit_build_per_cell_zero_events
        jit_build_per_cell_zero_events(
            cell_ids,
            cell_positions[:, 0], cell_positions[:, 1], cell_positions[:, 2],
            field_values_2d,
            out_obj_ids_0, out_params,
            n_field_deps
        )

        # Step 5: vectorized 128-bit ID generation
        # Index of the zero-object event type
        zero_events_list = self._event_registry.get_zero_events()
        type_index = zero_events_list.index(event_name) if event_name in zero_events_list else 0

        prefix = 2  # binary 10 = zero-object event
        id_high = (prefix << 61) | (current_step << 26) | (self.rank << 10) | type_index

        if 'id_high' in self.event_dtype.names:
            events['id_high'] = id_high
        if 'id_low' in self.event_dtype.names:
            # id_low = cell_id (unique per cell)
            events['id_low'] = cell_ids
        events['id'] = 0  # 128-bit ID mode

        if self.verbose:
            print(f"[Rank {self.rank}] VectorizedZeroBuilder: "
                  f"{n_cells} per_cell events for '{event_name}' "
                  f"({n_field_deps} field deps)")

        return events

    def has_per_cell_events(self) -> bool:
        """Whether any per_cell zero-object event exists"""
        return len(self._per_cell_events) > 0


def create_vectorized_pair_event_builder(domain_decomp: DomainDecomposition,
                                          config: dict,
                                          object_dtype: np.dtype,
                                          event_dtype: np.dtype,
                                          verbose: bool = False,
                                          use_cell_list: bool = True) -> VectorizedPairEventBuilder:
    """
    Factory: create a vectorized event builder
    """
    return VectorizedPairEventBuilder(
        domain_decomp, config, object_dtype, event_dtype, verbose,
        use_cell_list=use_cell_list
    )


# ===== Test code =====

if __name__ == "__main__":
    from mpi4py import MPI
    import time

    comm = MPI.COMM_WORLD
    rank = comm.rank
    size = comm.size

    print(f"[Rank {rank}] VectorizedEventBuilder test")

    # Create the test data
    n_objects = 1000
    np.random.seed(42 + rank)

    # A simple object dtype
    obj_dtype = np.dtype([
        ('id', np.int64),
        ('type', 'U16'),
        ('x', np.float64),
        ('y', np.float64),
        ('z', np.float64),
        ('owner_rank', np.int32),
        ('active', np.bool_),
    ])

    # Create random objects
    objects = np.zeros(n_objects, dtype=obj_dtype)
    objects['id'] = np.arange(rank * n_objects, (rank + 1) * n_objects)
    objects['type'] = np.where(np.random.random(n_objects) < 0.5, 'H', 'V')
    objects['x'] = np.random.random(n_objects) * 100
    objects['y'] = np.random.random(n_objects) * 100
    objects['z'] = np.random.random(n_objects) * 100
    objects['owner_rank'] = rank
    objects['active'] = True

    # Benchmark cKDTree + query_pairs
    positions = np.column_stack([objects['x'], objects['y'], objects['z']])

    t0 = time.perf_counter()
    tree = cKDTree(positions)
    t_build = time.perf_counter() - t0

    t0 = time.perf_counter()
    pairs = tree.query_pairs(r=5.0, output_type='set')
    t_query = time.perf_counter() - t0

    print(f"[Rank {rank}] {n_objects} objects:")
    print(f"  - cKDTree build: {t_build*1000:.2f} ms")
    print(f"  - query_pairs(r=5.0): {t_query*1000:.2f} ms, found {len(pairs)} pairs")

    # Test the vectorized filtering
    if len(pairs) > 0:
        pairs_array = np.array(list(pairs), dtype=np.int64)
        idx1 = pairs_array[:, 0]
        idx2 = pairs_array[:, 1]

        types1 = objects['type'][idx1]
        types2 = objects['type'][idx2]

        t0 = time.perf_counter()
        # Vectorized type filter: H + V
        mask = ((types1 == 'H') & (types2 == 'V')) | ((types1 == 'V') & (types2 == 'H'))
        n_filtered = np.sum(mask)
        t_filter = time.perf_counter() - t0

        print(f"  - Vectorized filter: {t_filter*1000:.2f} ms, {n_filtered} H-V pairs")

    print(f"[Rank {rank}] Test completed")
