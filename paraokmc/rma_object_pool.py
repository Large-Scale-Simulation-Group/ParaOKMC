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
RMA object pool module (three-way storage architecture)

Key features:
1. Windows are always created over a byte view (view('B')), so the MPI addressing unit is exactly one byte
2. Objects are stored in three classes:
   - center_objects: central objects (plain local allocation, outside the RMA window)
   - boundary_objects: boundary objects (inside the RMA window, reachable by neighbours)
   - large_objects: large objects (replicated on every rank and kept in sync by broadcast)
3. Only boundary objects are pulled, cutting the traffic by 50-80%
4. Large objects are detected automatically and broadcast to every rank

Design note: docs/ARCHITECTURE.md (object classification and boundary exchange)
"""

import numpy as np
from mpi4py import MPI
from typing import List, Dict, Optional, Tuple, Set

try:
    from numba import njit, prange
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


ACTION_DELETE = 0
ACTION_NEW = 1
ACTION_UPDATE = 2

CATEGORY_NONE = -1
CATEGORY_CENTER = 0
CATEGORY_BOUNDARY = 1
CATEGORY_LARGE = 2


@njit(parallel=True, cache=True)
def _classify_small_objects_region_batch_jit(
    x_arr, y_arr, z_arr, size_arr,
    bound_x_min, bound_x_max,
    bound_y_min, bound_y_max,
    bound_z_min, bound_z_max,
    box_x, box_y, box_z,
    periodic_x, periodic_y, periodic_z,
    r_cut,
    out_is_boundary
):
    """Decide in bulk whether small objects are center or boundary."""
    n = len(x_arr)
    for i in prange(n):
        x = x_arr[i]
        y = y_arr[i]
        z = z_arr[i]

        obj_radius = 0.0
        if size_arr[i] > 0.0:
            obj_radius = size_arr[i] / 2.0

        dx_min = abs(x - bound_x_min)
        dx_max = abs(x - bound_x_max)
        dy_min = abs(y - bound_y_min)
        dy_max = abs(y - bound_y_max)
        dz_min = abs(z - bound_z_min)
        dz_max = abs(z - bound_z_max)

        if periodic_x:
            dx_wrap = box_x - max(dx_min, dx_max)
            dx_min = min(dx_min, dx_wrap)
            dx_max = min(dx_max, dx_wrap)

        if periodic_y:
            dy_wrap = box_y - max(dy_min, dy_max)
            dy_min = min(dy_min, dy_wrap)
            dy_max = min(dy_max, dy_wrap)

        if periodic_z:
            dz_wrap = box_z - max(dz_min, dz_max)
            dz_min = min(dz_min, dz_wrap)
            dz_max = min(dz_max, dz_wrap)

        dist_to_boundary = min(dx_min, dx_max, dy_min, dy_max, dz_min, dz_max)
        out_is_boundary[i] = dist_to_boundary < (r_cut + obj_radius)


@njit(parallel=True, cache=True)
def _plan_reclassify_moves_jit(
    x_arr, y_arr, z_arr, size_arr, active_arr, affected_cell_mask,
    gx, gy, gz,
    box_x, box_y, box_z,
    cell_x, cell_y, cell_z,
    periodic_x, periodic_y, periodic_z,
    bound_x_min, bound_x_max,
    bound_y_min, bound_y_max,
    bound_z_min, bound_z_max,
    r_cut, current_region_is_boundary,
    out_move_mask
):
    """Build the move mask for the reclassification."""
    n = len(x_arr)
    for i in prange(n):
        out_move_mask[i] = False
        if not active_arr[i]:
            continue

        x = x_arr[i]
        y = y_arr[i]
        z = z_arr[i]

        if periodic_x:
            x = x % box_x
        elif x < 0.0 or x >= box_x:
            continue

        if periodic_y:
            y = y % box_y
        elif y < 0.0 or y >= box_y:
            continue

        if periodic_z:
            z = z % box_z
        elif z < 0.0 or z >= box_z:
            continue

        ix = int(np.floor(x / cell_x))
        iy = int(np.floor(y / cell_y))
        iz = int(np.floor(z / cell_z))

        if ix >= gx:
            ix = gx - 1
        if iy >= gy:
            iy = gy - 1
        if iz >= gz:
            iz = gz - 1
        if ix < 0:
            ix = 0
        if iy < 0:
            iy = 0
        if iz < 0:
            iz = 0

        cell_id = ix + iy * gx + iz * gx * gy
        if not affected_cell_mask[cell_id]:
            continue

        obj_radius = 0.0
        if size_arr[i] > 0.0:
            obj_radius = size_arr[i] / 2.0

        dx_min = abs(x - bound_x_min)
        dx_max = abs(x - bound_x_max)
        dy_min = abs(y - bound_y_min)
        dy_max = abs(y - bound_y_max)
        dz_min = abs(z - bound_z_min)
        dz_max = abs(z - bound_z_max)

        if periodic_x:
            dx_wrap = box_x - max(dx_min, dx_max)
            dx_min = min(dx_min, dx_wrap)
            dx_max = min(dx_max, dx_wrap)

        if periodic_y:
            dy_wrap = box_y - max(dy_min, dy_max)
            dy_min = min(dy_min, dy_wrap)
            dy_max = min(dy_max, dy_wrap)

        if periodic_z:
            dz_wrap = box_z - max(dz_min, dz_max)
            dz_min = min(dz_min, dz_wrap)
            dz_max = min(dz_max, dz_wrap)

        dist_to_boundary = min(dx_min, dx_max, dy_min, dy_max, dz_min, dz_max)
        is_boundary = dist_to_boundary < (r_cut + obj_radius)
        out_move_mask[i] = (is_boundary != current_region_is_boundary)


# [perf] Serial JIT: this kernel only handles changed_arr each step (delta ~ 1-3 records),
# and with parallel=True the cost of waking the thread team dwarfs the work - an empty
# parallel kernel measured ~12 ms per call when MPI oversubscribed the cores (the main
# cause of the 7.9 ms/step apply phase); serial execution brings it down to microseconds
@njit(cache=True)
def _plan_changed_actions_jit(
    x_arr, y_arr, z_arr, size_arr, active_arr, new_arr,
    ownership_map, gx, gy, gz,
    box_x, box_y, box_z,
    cell_x, cell_y, cell_z,
    periodic_x, periodic_y, periodic_z,
    bound_x_min, bound_x_max,
    bound_y_min, bound_y_max,
    bound_z_min, bound_z_max,
    r_cut, large_threshold, my_rank,
    out_actions, out_owners, out_categories, out_valid
):
    """Plan the actions for changed_arr: operation kind, owner and target class."""
    n = len(x_arr)
    for i in range(n):
        if not active_arr[i]:
            out_actions[i] = ACTION_DELETE
            out_owners[i] = -1
            out_categories[i] = CATEGORY_NONE
            out_valid[i] = True
            continue

        out_actions[i] = ACTION_NEW if new_arr[i] else ACTION_UPDATE

        x = x_arr[i]
        y = y_arr[i]
        z = z_arr[i]
        valid = True

        if periodic_x:
            x = x % box_x
        elif x < 0.0 or x >= box_x:
            valid = False

        if periodic_y:
            y = y % box_y
        elif y < 0.0 or y >= box_y:
            valid = False

        if periodic_z:
            z = z % box_z
        elif z < 0.0 or z >= box_z:
            valid = False

        if not valid:
            out_owners[i] = -1
            out_categories[i] = CATEGORY_NONE
            out_valid[i] = False
            continue

        ix = int(np.floor(x / cell_x))
        iy = int(np.floor(y / cell_y))
        iz = int(np.floor(z / cell_z))

        if ix >= gx:
            ix = gx - 1
        if iy >= gy:
            iy = gy - 1
        if iz >= gz:
            iz = gz - 1
        if ix < 0:
            ix = 0
        if iy < 0:
            iy = 0
        if iz < 0:
            iz = 0

        cell_id = ix + iy * gx + iz * gx * gy
        owner = int(ownership_map[cell_id])
        out_owners[i] = owner
        out_valid[i] = True

        obj_radius = 0.5 * size_arr[i] if size_arr[i] > 0.0 else 0.0
        influence_range = obj_radius + r_cut
        if influence_range >= large_threshold:
            out_categories[i] = CATEGORY_LARGE
            continue

        if owner != my_rank:
            out_categories[i] = CATEGORY_NONE
            continue

        dx_min = abs(x - bound_x_min)
        dx_max = abs(x - bound_x_max)
        dy_min = abs(y - bound_y_min)
        dy_max = abs(y - bound_y_max)
        dz_min = abs(z - bound_z_min)
        dz_max = abs(z - bound_z_max)

        if periodic_x:
            dx_wrap = box_x - max(dx_min, dx_max)
            if dx_wrap < dx_min:
                dx_min = dx_wrap
            if dx_wrap < dx_max:
                dx_max = dx_wrap
        if periodic_y:
            dy_wrap = box_y - max(dy_min, dy_max)
            if dy_wrap < dy_min:
                dy_min = dy_wrap
            if dy_wrap < dy_max:
                dy_max = dy_wrap
        if periodic_z:
            dz_wrap = box_z - max(dz_min, dz_max)
            if dz_wrap < dz_min:
                dz_min = dz_wrap
            if dz_wrap < dz_max:
                dz_max = dz_wrap

        dist_to_boundary = min(dx_min, dx_max, dy_min, dy_max, dz_min, dz_max)
        boundary_threshold = r_cut + obj_radius
        if dist_to_boundary < boundary_threshold:
            out_categories[i] = CATEGORY_BOUNDARY
        else:
            out_categories[i] = CATEGORY_CENTER


def create_default_object_dtype() -> np.dtype:
    """Create the default object dtype"""
    MAX_FLOAT_PROPS = 20

    return np.dtype([
        ('id', np.int64),
        ('type_id', np.int32),           # integer type ID
        ('x', np.float64),
        ('y', np.float64),
        ('z', np.float64),
        ('size', np.float64),        # physical size of the object (radius/scale)
        ('owner_rank', np.int32),
        ('active', np.bool_),        # whether the object is alive (False = deleted, the slot is reusable)
        ('new', np.bool_),           # new-object flag (an event must be built)
        ('changed', np.bool_),       # changed-object flag (the event must be rebuilt)
        ('float_props', np.float64, (MAX_FLOAT_PROPS,)),
    ], align=True)


OBJECT_DTYPE = create_default_object_dtype()


class RMAObjectPool:
    """
    RMA object pool (three-way storage architecture)

    Objects are stored in three classes:
    1. center_objects: central objects (distance to the boundary >= r_cut + obj_radius)
    2. boundary_objects: boundary objects (distance to the boundary < r_cut + obj_radius)
    3. large_objects: large objects (reach >= ghost_capacity x multiplier)
    """

    def __init__(self, comm: MPI.Comm, dtype: Optional[np.dtype] = None,
                 initial_capacity: int = 1000,
                 auto_expand: bool = True, verbose: bool = False,
                 domain_decomposition=None,
                 large_object_multiplier: float = 5.0,
                 max_large_objects: int = 1000):
        """
        Initialize the RMA object pool

        Args:
            comm: MPI communicator
            dtype: object dtype (OBJECT_DTYPE by default)
            initial_capacity: initial capacity
            auto_expand: whether to grow automatically
            verbose: whether to print debugging output
            domain_decomposition: domain decomposition object (required)
            large_object_multiplier: multiplier of the large-object test (default 5.0)
            max_large_objects: upper bound on the number of large objects (default 1000)
        """
        self.comm = comm
        self.rank = comm.rank
        self.size = comm.size
        self.auto_expand = auto_expand
        self.verbose = verbose

        if dtype is None:
            self.dtype = OBJECT_DTYPE
        else:
            self.dtype = dtype

        # Configuration parameters
        self.large_object_multiplier = large_object_multiplier
        self.max_large_objects = max_large_objects
        self.domain_decomposition = domain_decomposition

        # ===== Three-way storage =====
        # Heuristic estimate of the share of each class
        center_ratio = 0.7
        boundary_ratio = 0.29
        large_ratio = 0.01

        # 1. Central objects (a plain local array, outside the RMA window)
        self.center_capacity = max(int(initial_capacity * center_ratio), 100)
        self.center_count = 0
        self.center_objects = np.zeros(self.center_capacity, dtype=self.dtype)

        # 2. Boundary objects (inside the RMA window)
        self.boundary_capacity = max(int(initial_capacity * boundary_ratio), 100)
        self.boundary_count = 0
        self.boundary_objects = np.zeros(self.boundary_capacity, dtype=self.dtype)
        # With a single rank no RMA window is created: there is no neighbour, so boundary
        # objects need no one-sided exchange. This saves the overhead and sidesteps MPI
        # builds that cannot create a one-rank window (MPI_ERR_WIN). Every use site guards for None.
        if comm.size > 1:
            self.win_boundary = MPI.Win.Create(self.boundary_objects.view('B'), comm=comm)
        else:
            self.win_boundary = None

        # 3. Large objects (a local array kept in sync by broadcast)
        self.large_capacity = max(int(initial_capacity * large_ratio), max_large_objects)
        self.large_count = 0
        self.large_objects = np.zeros(self.large_capacity, dtype=self.dtype)

        # Auxiliary structures
        # ID to location map: {obj_id: ('center'|'boundary'|'large', index)}
        self.obj_id_to_location: Dict[int, Tuple[str, int]] = {}

        # Set of large object IDs (for a fast test)
        self.large_object_ids: Set[int] = set()

        # Cell index (maintained per class)
        self.center_cell_index = None
        self.boundary_cell_index = None
        self.large_cell_index = None

        # Growth flags
        self.expansion_needed = False

        # [fix] Deferred growth flag (prevents an MPI deadlock)
        # Problem: MPI.Win.Create/Free are collectives, so a single rank calling them deadlocks
        # Fix: growth only enlarges the local array; the window is rebuilt collectively at a sync point
        self._boundary_window_dirty = False

        # [fix Bug 1] Keep a reference to the old array so the GC cannot cause a segfault
        # Problem: after _expand_boundary_array() creates a new array the old one loses its
        # last reference, yet win_boundary still points at the old memory. If Python's GC
        # collects it before sync_window_rebuild(), a remote RMA read touches freed memory
        # and segfaults.
        # Fix: keep the old array alive until the window has been rebuilt
        self._old_boundary_arrays = []

        # Ghost cache (removes the per-step RMA traffic)
        # Principle:
        # 1. the per-step Bcast(changed_objects) already tells every rank what changed
        # 2. as long as the domain split is unchanged, the ghost region is fixed
        # 3. updating the cache incrementally avoids an RMA Get on every step
        self.ghost_cache: Dict[int, np.ndarray] = {}  # obj_id -> obj_data
        self._ghost_cache_initialized = False
        self.__ghost_cache_dirty = True  # a build is needed initially (double underscore blocks direct access)
        self._invalidate_count = 0  # debugging counter
        self._type_to_int: Dict[str, int] = {}
        self._int_to_type: Dict[int, str] = {}

    @property
    def _ghost_cache_dirty(self):
        return self.__ghost_cache_dirty

    @_ghost_cache_dirty.setter
    def _ghost_cache_dirty(self, value):
        self.__ghost_cache_dirty = value

        # Window rebuild check optimization
        # Principle: growth is rare, so most steps need no allreduce check
        # The steps since the last rebuild are counted and the check is skipped for N steps
        self._steps_since_last_rebuild = 0
        self._window_check_interval = 10  # check every 10 steps (when the local state is clean)

        if self.verbose and self.rank == 0:
            print(f"[RMAObjectPool] Initialized (three-way storage)")
            print(f"  center_capacity: {self.center_capacity}")
            print(f"  boundary_capacity: {self.boundary_capacity}")
            print(f"  large_capacity: {self.large_capacity}")

    # ===== Object classification =====

    def _classify_object_region(self, obj: np.ndarray) -> str:
        """
        Decide whether an object is central or on the boundary

        Algorithm:
        1. compute the minimum distance from the object to this rank's subdomain boundary
        2. distance < (r_cut + obj_radius) makes it a boundary object
        3. otherwise it is a central object

        Complexity: O(1)
        """
        if self.domain_decomposition is None:
            return 'boundary'

        x, y, z = obj['x'], obj['y'], obj['z']
        obj_radius = (obj['size'] / 2.0) if ('size' in obj.dtype.names and obj['size'] > 0) else 0.0

        # Bounds of this rank's subdomain
        bounds = self.domain_decomposition.subdomain_bounds

        # Distance to each boundary
        dx_min = abs(x - bounds['x_min'])
        dx_max = abs(x - bounds['x_max'])
        dy_min = abs(y - bounds['y_min'])
        dy_max = abs(y - bounds['y_max'])
        dz_min = abs(z - bounds['z_min'])
        dz_max = abs(z - bounds['z_max'])

        # Periodic boundary handling
        box_size = self.domain_decomposition.box_size
        periodic = self.domain_decomposition.periodic

        if periodic[0]:
            dx_wrap = box_size[0] - max(dx_min, dx_max)
            dx_min = min(dx_min, dx_wrap)
            dx_max = min(dx_max, dx_wrap)

        if periodic[1]:
            dy_wrap = box_size[1] - max(dy_min, dy_max)
            dy_min = min(dy_min, dy_wrap)
            dy_max = min(dy_max, dy_wrap)

        if periodic[2]:
            dz_wrap = box_size[2] - max(dz_min, dz_max)
            dz_min = min(dz_min, dz_wrap)
            dz_max = min(dz_max, dz_wrap)

        # Minimum distance to a boundary (in any direction)
        dist_to_boundary = min(dx_min, dx_max, dy_min, dy_max, dz_min, dz_max)

        # Decision threshold
        r_cut = self.domain_decomposition.max_interaction_distance
        threshold = r_cut + obj_radius

        if dist_to_boundary < threshold:
            return 'boundary'
        else:
            return 'center'

    def _is_large_object(self, obj: np.ndarray) -> bool:
        """
        Decide whether this is a large object

        Algorithm:
        1. reach of the object = obj_radius + r_cut
        2. collection capability of the current ghost region = N_layers x L_min_cell
        3. reach >= capability x multiplier makes it a large object

        Complexity: O(1)
        """
        if self.domain_decomposition is None:
            return False

        obj_radius = (obj['size'] / 2.0) if ('size' in obj.dtype.names and obj['size'] > 0) else 0.0
        r_cut = self.domain_decomposition.max_interaction_distance

        # Reach of the object
        influence_range = obj_radius + r_cut

        # Collection capability of the current ghost region
        grid_system = self.domain_decomposition.grid_system
        min_cell_size = min(grid_system.cell_size)
        n_layers = int(np.ceil(r_cut / min_cell_size))
        ghost_capacity = n_layers * min_cell_size

        # Decision (the multiplier keeps false positives away)
        threshold = ghost_capacity * self.large_object_multiplier

        return influence_range >= threshold

    def is_large_object_by_id(self, obj_id: int) -> bool:
        """Test whether an object ID belongs to a large object"""
        return obj_id in self.large_object_ids

    # ===== Object insertion =====

    def add_object(self, obj: np.ndarray) -> int:
        """
        Add an object (classified automatically)

        Procedure:
        1. decide the class (large / boundary / center)
        2. append to the matching array
        3. update the ID map

        Complexity: O(1)
        """
        obj_id = int(obj['id'])

        # Decide the class
        is_large = self._is_large_object(obj)

        if is_large:
            return self._add_to_large_array(obj)

        # Small object: central or boundary?
        region = self._classify_object_region(obj)

        if region == 'boundary':
            return self._add_to_boundary_array(obj)
        else:
            return self._add_to_center_array(obj)

    def _add_to_center_array(self, obj: np.ndarray) -> int:
        """Append to the central object array"""
        if self.center_count >= self.center_capacity:
            if self.auto_expand:
                self._expand_center_array()
            else:
                raise RuntimeError(f"Center object pool full (capacity={self.center_capacity})")

        idx = self.center_count
        self.center_objects[idx] = obj
        self.center_count += 1

        obj_id = int(obj['id'])
        self.obj_id_to_location[obj_id] = ('center', idx)

        return idx

    def _add_to_boundary_array(self, obj: np.ndarray) -> int:
        """Append to the boundary object array"""
        if self.boundary_count >= self.boundary_capacity:
            if self.auto_expand:
                self._expand_boundary_array()
            else:
                raise RuntimeError(f"Boundary object pool full (capacity={self.boundary_capacity})")

        idx = self.boundary_count
        self.boundary_objects[idx] = obj
        self.boundary_count += 1

        obj_id = int(obj['id'])
        self.obj_id_to_location[obj_id] = ('boundary', idx)

        return idx

    def _add_to_large_array(self, obj: np.ndarray) -> int:
        """Append to the large object array"""
        if self.large_count >= self.large_capacity:
            if self.auto_expand:
                self._expand_large_array()
            else:
                raise RuntimeError(f"Large object pool full (capacity={self.large_capacity})")

        idx = self.large_count
        self.large_objects[idx] = obj
        self.large_count += 1

        obj_id = int(obj['id'])
        self.obj_id_to_location[obj_id] = ('large', idx)
        self.large_object_ids.add(obj_id)

        return idx

    def _add_to_category_code(self, obj: np.ndarray, category_code: int) -> int:
        """Append using a preplanned class code, so the test is not repeated."""
        if category_code == CATEGORY_CENTER:
            return self._add_to_center_array(obj)
        if category_code == CATEGORY_BOUNDARY:
            return self._add_to_boundary_array(obj)
        if category_code == CATEGORY_LARGE:
            return self._add_to_large_array(obj)
        return self.add_object(obj)

    def _ensure_center_capacity_for(self, additional: int):
        """Make sure the central array can still take `additional` objects."""
        if additional <= 0:
            return
        while self.center_count + additional > self.center_capacity:
            if self.auto_expand:
                self._expand_center_array()
            else:
                raise RuntimeError(
                    f"Center object pool full (capacity={self.center_capacity}, "
                    f"need_additional={additional})"
                )

    def _ensure_boundary_capacity_for(self, additional: int):
        """Make sure the boundary array can still take `additional` objects."""
        if additional <= 0:
            return
        while self.boundary_count + additional > self.boundary_capacity:
            if self.auto_expand:
                self._expand_boundary_array()
            else:
                raise RuntimeError(
                    f"Boundary object pool full (capacity={self.boundary_capacity}, "
                    f"need_additional={additional})"
                )

    def add_received_small_objects_batch(self, recv_buf: np.ndarray, owner_rank: int) -> Tuple[int, int]:
        """
        Insert the small objects received by a migration in bulk.

        Large objects never take part in object redistribution, so the received objects
        only have to be classified as center/boundary and written in bulk.

        Returns:
            (n_center, n_boundary)
        """
        n_recv = len(recv_buf)
        if n_recv == 0:
            return 0, 0

        if self.domain_decomposition is None:
            raise RuntimeError("domain_decomposition is required for batch receive insertion")

        recv_local = np.array(recv_buf, copy=True, dtype=self.dtype)
        recv_local['owner_rank'] = owner_rank

        bounds = self.domain_decomposition.subdomain_bounds
        box_size = self.domain_decomposition.box_size
        periodic = self.domain_decomposition.periodic
        r_cut = float(self.domain_decomposition.max_interaction_distance)

        size_arr = recv_local['size'] if 'size' in recv_local.dtype.names else np.zeros(n_recv, dtype=np.float64)
        is_boundary = np.empty(n_recv, dtype=np.bool_)
        _classify_small_objects_region_batch_jit(
            np.ascontiguousarray(recv_local['x'], dtype=np.float64),
            np.ascontiguousarray(recv_local['y'], dtype=np.float64),
            np.ascontiguousarray(recv_local['z'], dtype=np.float64),
            np.ascontiguousarray(size_arr, dtype=np.float64),
            float(bounds['x_min']), float(bounds['x_max']),
            float(bounds['y_min']), float(bounds['y_max']),
            float(bounds['z_min']), float(bounds['z_max']),
            float(box_size[0]), float(box_size[1]), float(box_size[2]),
            bool(periodic[0]), bool(periodic[1]), bool(periodic[2]),
            r_cut,
            is_boundary
        )

        boundary_indices = np.flatnonzero(is_boundary)
        center_indices = np.flatnonzero(~is_boundary)

        n_boundary = len(boundary_indices)
        n_center = len(center_indices)

        self._ensure_boundary_capacity_for(n_boundary)
        self._ensure_center_capacity_for(n_center)

        if n_boundary > 0:
            start = self.boundary_count
            end = start + n_boundary
            self.boundary_objects[start:end] = recv_local[boundary_indices]
            self.boundary_count = end
            for offset, obj_id in enumerate(np.ascontiguousarray(recv_local['id'][boundary_indices], dtype=np.int64)):
                self.obj_id_to_location[int(obj_id)] = ('boundary', start + offset)

        if n_center > 0:
            start = self.center_count
            end = start + n_center
            self.center_objects[start:end] = recv_local[center_indices]
            self.center_count = end
            for offset, obj_id in enumerate(np.ascontiguousarray(recv_local['id'][center_indices], dtype=np.int64)):
                self.obj_id_to_location[int(obj_id)] = ('center', start + offset)

        return n_center, n_boundary

    def _split_received_small_objects_batch(
        self,
        recv_buf: np.ndarray,
        owner_rank: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Classify the received small objects as center/boundary without inserting them yet."""
        n_recv = len(recv_buf)
        if n_recv == 0:
            return np.array([], dtype=self.dtype), np.array([], dtype=self.dtype)

        if self.domain_decomposition is None:
            raise RuntimeError("domain_decomposition is required for batch receive classification")

        recv_local = np.array(recv_buf, copy=True, dtype=self.dtype)
        recv_local['owner_rank'] = owner_rank

        bounds = self.domain_decomposition.subdomain_bounds
        box_size = self.domain_decomposition.box_size
        periodic = self.domain_decomposition.periodic
        r_cut = float(self.domain_decomposition.max_interaction_distance)

        size_arr = recv_local['size'] if 'size' in recv_local.dtype.names else np.zeros(n_recv, dtype=np.float64)
        is_boundary = np.empty(n_recv, dtype=np.bool_)
        _classify_small_objects_region_batch_jit(
            np.ascontiguousarray(recv_local['x'], dtype=np.float64),
            np.ascontiguousarray(recv_local['y'], dtype=np.float64),
            np.ascontiguousarray(recv_local['z'], dtype=np.float64),
            np.ascontiguousarray(size_arr, dtype=np.float64),
            float(bounds['x_min']), float(bounds['x_max']),
            float(bounds['y_min']), float(bounds['y_max']),
            float(bounds['z_min']), float(bounds['z_max']),
            float(box_size[0]), float(box_size[1]), float(box_size[2]),
            bool(periodic[0]), bool(periodic[1]), bool(periodic[2]),
            r_cut,
            is_boundary
        )

        boundary_indices = np.flatnonzero(is_boundary)
        center_indices = np.flatnonzero(~is_boundary)
        recv_center = recv_local[center_indices].copy() if len(center_indices) > 0 else np.array([], dtype=self.dtype)
        recv_boundary = recv_local[boundary_indices].copy() if len(boundary_indices) > 0 else np.array([], dtype=self.dtype)
        return recv_center, recv_boundary

    # ===== Growth =====

    def _expand_center_array(self):
        """Grow the central object array"""
        new_capacity = self.center_capacity * 2
        new_array = np.zeros(new_capacity, dtype=self.dtype)
        new_array[:self.center_count] = self.center_objects[:self.center_count]
        self.center_objects = new_array
        self.center_capacity = new_capacity
        if self.verbose:
            print(f"[Rank {self.rank}] Center array expanded to {new_capacity}")

    def _expand_boundary_array(self):
        """
        [fix] Grow the boundary object array (with a deferred RMA window rebuild)

        Problem: MPI.Win.Create/Free are collectives, so a single rank calling them deadlocks
        Fix: only the local array grows; the state is marked dirty and the window is rebuilt collectively

        Deadlock scenario:
        - only the winning rank adds an object while executing an event
        - if that triggers growth, the winner calls Win.Create alone
        - the other ranks know nothing about it and the program hangs

        Procedure after the fix:
        1. this method only grows the local NumPy array
        2. it sets _boundary_window_dirty = True
        3. the main loop calls sync_window_rebuild() collectively at a sync point

        [fix Bug 1] Keep a reference to the old array to avoid a segfault
        Problem: after growth the old array loses its last reference and may be collected,
              yet win_boundary still points at the old memory, so an RMA read segfaults
        Fix: keep the old array in the _old_boundary_arrays list and clear it inside
              sync_window_rebuild()
        """
        new_capacity = self.boundary_capacity * 2

        # 1. Only grow the local array (no MPI call)
        new_array = np.zeros(new_capacity, dtype=self.dtype)
        new_array[:self.boundary_count] = self.boundary_objects[:self.boundary_count]

        # [fix Bug 1] Keep the old array alive against the GC
        # win_boundary still references it, so it must stay valid until the rebuild
        self._old_boundary_arrays.append(self.boundary_objects)

        self.boundary_objects = new_array
        self.boundary_capacity = new_capacity

        # 2. Mark the window for rebuilding (deferred to a sync point)
        self._boundary_window_dirty = True

        if self.verbose:
            print(f"[Rank {self.rank}] Boundary array expanded to {new_capacity} (window rebuild pending)")

    def _expand_large_array(self):
        """Grow the large object array"""
        new_capacity = self.large_capacity * 2
        new_array = np.zeros(new_capacity, dtype=self.dtype)
        new_array[:self.large_count] = self.large_objects[:self.large_count]
        self.large_objects = new_array
        self.large_capacity = new_capacity
        if self.verbose:
            print(f"[Rank {self.rank}] Large array expanded to {new_capacity}")

    # ===== Object update and deletion =====

    def update_object(self, obj: np.ndarray):
        """
        Update an object (which may reclassify it)
        """
        obj_id = int(obj['id'])

        if obj_id not in self.obj_id_to_location:
            # A new object: just add it
            self.add_object(obj)
            return

        old_location = self.obj_id_to_location[obj_id]
        old_category, old_idx = old_location

        # Decide the new class
        is_large = self._is_large_object(obj)
        if is_large:
            new_category = 'large'
        else:
            new_category = self._classify_object_region(obj)

        if old_category == new_category:
            # The class is unchanged: update in place
            if old_category == 'center':
                self.center_objects[old_idx] = obj
            elif old_category == 'boundary':
                self.boundary_objects[old_idx] = obj
            else:  # large
                self.large_objects[old_idx] = obj
        else:
            # The class changed: delete the old entry and add a new one
            self._delete_from_category(obj_id, old_category, old_idx)
            self.add_object(obj)

    def _update_local_small_object_planned(self, obj: np.ndarray, target_category: int):
        """Update a local small object using the preplanned center/boundary class."""
        obj_id = int(obj['id'])
        if obj_id not in self.obj_id_to_location:
            self._add_to_category_code(obj, target_category)
            return

        old_category, old_idx = self.obj_id_to_location[obj_id]
        if old_category == 'center' and target_category == CATEGORY_CENTER:
            self.center_objects[old_idx] = obj
            return
        if old_category == 'boundary' and target_category == CATEGORY_BOUNDARY:
            self.boundary_objects[old_idx] = obj
            return

        self._delete_from_category(obj_id, old_category, old_idx)
        self._add_to_category_code(obj, target_category)

    def _upsert_large_object_planned(self, obj: np.ndarray):
        """Update or add a large object replica."""
        obj_id = int(obj['id'])
        if obj_id in self.obj_id_to_location:
            category, idx = self.obj_id_to_location[obj_id]
            if category == 'large':
                self.large_objects[idx] = obj
                return
            self._delete_from_category(obj_id, category, idx)
        self._add_to_large_array(obj)

    def delete_object(self, obj_id: int):
        """Delete an object"""
        if obj_id not in self.obj_id_to_location:
            return

        category, idx = self.obj_id_to_location[obj_id]
        self._delete_from_category(obj_id, category, idx)

    def _delete_from_category(self, obj_id: int, category: str, idx: int):
        """Delete an object from one class (swap-remove)"""
        if category == 'center':
            if idx < self.center_count - 1:
                # Fill the hole with the last object
                self.center_objects[idx] = self.center_objects[self.center_count - 1]
                moved_id = int(self.center_objects[idx]['id'])
                self.obj_id_to_location[moved_id] = ('center', idx)
            self.center_count -= 1

        elif category == 'boundary':
            if idx < self.boundary_count - 1:
                self.boundary_objects[idx] = self.boundary_objects[self.boundary_count - 1]
                moved_id = int(self.boundary_objects[idx]['id'])
                self.obj_id_to_location[moved_id] = ('boundary', idx)
            self.boundary_count -= 1

        else:  # large
            if idx < self.large_count - 1:
                self.large_objects[idx] = self.large_objects[self.large_count - 1]
                moved_id = int(self.large_objects[idx]['id'])
                self.obj_id_to_location[moved_id] = ('large', idx)
            self.large_count -= 1
            self.large_object_ids.discard(obj_id)

        del self.obj_id_to_location[obj_id]

    def remove_large_object(self, obj_id: int):
        """Delete a large object (synchronized across ranks)"""
        if obj_id in self.obj_id_to_location:
            category, idx = self.obj_id_to_location[obj_id]
            if category == 'large':
                self._delete_from_category(obj_id, category, idx)

    # ===== Object queries =====

    def has_object(self, obj_id: int) -> bool:
        """Test whether an object exists"""
        return obj_id in self.obj_id_to_location

    def get_object_by_id(self, obj_id: int) -> Optional[np.ndarray]:
        """Get an object by ID"""
        if obj_id not in self.obj_id_to_location:
            return None

        category, idx = self.obj_id_to_location[obj_id]

        if category == 'center':
            return self.center_objects[idx].copy()
        elif category == 'boundary':
            return self.boundary_objects[idx].copy()
        else:  # large
            return self.large_objects[idx].copy()

    def get_local_objects(self, active_only: bool = True) -> np.ndarray:
        """
        Get every local object (owner_rank == self.rank)

        Returns center union boundary union large(owner=self)
        """
        arrays = []

        if self.center_count > 0:
            center = self.center_objects[:self.center_count]
            if active_only:
                center = center[center['active']]
            center = center[center['owner_rank'] == self.rank]
            if len(center) > 0:
                arrays.append(center)

        if self.boundary_count > 0:
            boundary = self.boundary_objects[:self.boundary_count]
            if active_only:
                boundary = boundary[boundary['active']]
            boundary = boundary[boundary['owner_rank'] == self.rank]
            if len(boundary) > 0:
                arrays.append(boundary)

        if self.large_count > 0:
            large = self.large_objects[:self.large_count]
            if active_only:
                large = large[large['active']]
            large = large[large['owner_rank'] == self.rank]
            if len(large) > 0:
                arrays.append(large)

        if not arrays:
            return np.array([], dtype=self.dtype)

        return np.concatenate(arrays)

    def get_type_name(self, obj) -> str:
        """Resolve the user-visible type name from an object record."""
        if isinstance(obj, dict):
            return str(obj.get('type', ''))
        if hasattr(obj, 'dtype') and obj.dtype.names is not None:
            if 'type' in obj.dtype.names:
                return str(obj['type']).strip()
            if 'type_id' in obj.dtype.names:
                return self._int_to_type.get(int(obj['type_id']), '')
        return ''

    def get_all_objects_including_large(self, active_only: bool = True) -> np.ndarray:
        """
        Get every object including the large-object replicas (used by event construction)

        Returns center(owner=self) union boundary(owner=self) union large(all)
        """
        arrays = []

        if self.center_count > 0:
            center = self.center_objects[:self.center_count]
            if active_only:
                center = center[center['active']]
            center = center[center['owner_rank'] == self.rank]
            if len(center) > 0:
                arrays.append(center)

        if self.boundary_count > 0:
            boundary = self.boundary_objects[:self.boundary_count]
            if active_only:
                boundary = boundary[boundary['active']]
            boundary = boundary[boundary['owner_rank'] == self.rank]
            if len(boundary) > 0:
                arrays.append(boundary)

        # Large objects: every replica (owner_rank is not filtered)
        if self.large_count > 0:
            large = self.large_objects[:self.large_count]
            if active_only:
                large = large[large['active']]
            if len(large) > 0:
                arrays.append(large)

        if not arrays:
            return np.array([], dtype=self.dtype)

        return np.concatenate(arrays)

    def get_total_count(self) -> int:
        """Total number of local objects (center + boundary + large)"""
        return self.center_count + self.boundary_count + self.large_count

    def get_local_count(self) -> int:
        """Number of objects owned locally (owner_rank == self.rank)"""
        count = 0

        # Central objects
        if self.center_count > 0:
            mask = self.center_objects[:self.center_count]['owner_rank'] == self.rank
            count += np.sum(mask)

        # Boundary objects
        if self.boundary_count > 0:
            mask = self.boundary_objects[:self.boundary_count]['owner_rank'] == self.rank
            count += np.sum(mask)

        # Large objects (only those we own)
        if self.large_count > 0:
            mask = self.large_objects[:self.large_count]['owner_rank'] == self.rank
            count += np.sum(mask)

        return int(count)

    def sync(self):
        """Synchronize the RMA window (make sure every write landed)"""
        self.comm.Barrier()

    def sync_window_rebuild(self):
        """
        [fix] Rebuild the RMA window collectively (prevents an MPI deadlock)

        Must be called at a sync point of the main loop so every rank runs it together.

        Procedure:
        1. Allreduce to see whether any rank needs a rebuild
        2. if so, every rank performs Free + Create collectively
        3. reset the dirty flag
        4. [fix Bug 1] release the old array references

        When to call:
        - at the end of every main-loop step
        - after executing an event
        - at the sync point following anything that may trigger growth

        Returns:
            True if a window rebuild took place

        [fix] The old 'deferred check' optimization was removed
        Reason: it broke the consistency of the MPI collective and deadlocked
        Scenario: rank A grew (dirty=True) and waited inside allreduce while rank B had
              not grown (dirty=False, steps < interval) and returned immediately
              -> rank A blocked forever
        Fix: every rank must enter the allreduce unconditionally
        """
        if self.win_boundary is None:
            return False

        # [fix] Every rank must enter the allreduce unconditionally
        # Returning early on local state would break the collective
        local_dirty = 1 if self._boundary_window_dirty else 0
        global_dirty = self.comm.allreduce(local_dirty, op=MPI.MAX)

        if global_dirty == 0:
            return False  # no rank needs a rebuild

        # 2. Every rank frees the old window collectively
        self.win_boundary.Free()

        # 3. Every rank creates the new window collectively
        # Note: each rank uses its own boundary_objects (capacities may differ)
        self.win_boundary = MPI.Win.Create(self.boundary_objects.view('B'), comm=self.comm)

        # 4. Reset the dirty flag
        self._boundary_window_dirty = False

        # [fix Bug 1] Release the old array references
        # The window has been rebuilt, so the old arrays are no longer reachable by RMA
        self._old_boundary_arrays.clear()

        if self.verbose and self.rank == 0:
            print(f"[RMA] Boundary windows rebuilt globally")

        return True

    # ===== Ghost cache management =====

    def _is_in_my_ghost_zone(self, x: float, y: float, z: float) -> bool:
        """
        Test whether a position lies in this rank's ghost region

        Ghost conditions:
        1. it is outside this rank's subdomain (not a local object)
        2. it is inside this rank's extended domain (distance to the boundary < r_cut)
        """
        if self.domain_decomposition is None:
            return False

        bounds = self.domain_decomposition.subdomain_bounds
        r_cut = self.domain_decomposition.max_interaction_distance
        box_size = self.domain_decomposition.box_size
        periodic = self.domain_decomposition.periodic

        # Inside this rank's subdomain? (then it is not a ghost)
        in_subdomain = (
            bounds['x_min'] <= x < bounds['x_max'] and
            bounds['y_min'] <= y < bounds['y_max'] and
            bounds['z_min'] <= z < bounds['z_max']
        )
        if in_subdomain:
            return False

        # Distance to this rank's boundary (periodic boundaries included)
        def min_dist_to_range(val, rmin, rmax, box_len, is_periodic):
            if rmin <= val < rmax:
                return 0.0
            d1 = abs(val - rmin)
            d2 = abs(val - rmax)
            d = min(d1, d2)
            if is_periodic:
                # Periodic boundary: consider the wrapped distance too
                d_wrap = box_len - max(d1, d2)
                d = min(d, d_wrap)
            return d

        dx = min_dist_to_range(x, bounds['x_min'], bounds['x_max'], box_size[0], periodic[0])
        dy = min_dist_to_range(y, bounds['y_min'], bounds['y_max'], box_size[1], periodic[1])
        dz = min_dist_to_range(z, bounds['z_min'], bounds['z_max'], box_size[2], periodic[2])

        # Ghost condition: the distance is < r_cut along any dimension
        return dx < r_cut or dy < r_cut or dz < r_cut

    def rebuild_ghost_cache(self):
        """
        Rebuild the ghost cache completely

        When to call:
        1. after initialization
        2. after load balancing (the domain split changed)
        3. after a large number of objects changed

        Implementation: fetch the boundary objects of every neighbour rank over RMA
        Optimization: batch the RMA Gets in Fence mode instead of serial Lock/Unlock
        """
        import time
        t0 = time.perf_counter()

        self.ghost_cache.clear()

        # Single rank: no neighbour, no RMA window, so the ghost cache is always empty.
        # [fix] An empty cache is a legitimate initialized state, so the flags must be set
        # before returning; otherwise _ghost_cache_initialized stays False at np=1 and the
        # pair structural increment never passes its precondition (a full rebuild every step)
        if self.win_boundary is None:
            self._ghost_cache_initialized = True
            self._ghost_cache_dirty = False
            return

        if self.domain_decomposition is None:
            self._ghost_cache_initialized = True
            self._ghost_cache_dirty = False
            return

        # Fetch the boundary objects of every neighbour rank
        ghost_ranks = self.domain_decomposition.ghost_ranks

        # One allgather gets the boundary_count of every rank
        t1 = time.perf_counter()
        all_boundary_counts = self.comm.allgather(self.boundary_count)
        t_allgather = time.perf_counter() - t1

        # [perf] Prepare the buffers of every RMA Get
        recv_buffers = {}
        target_ranks_to_fetch = []

        for target_rank in ghost_ranks:
            if target_rank == self.rank:
                continue

            remote_count = all_boundary_counts[target_rank]
            if remote_count == 0:
                continue

            target_ranks_to_fetch.append(target_rank)
            total_bytes = remote_count * self.dtype.itemsize
            recv_buffers[target_rank] = np.empty(total_bytes, dtype='B')

        # [perf] Issue the RMA Gets in bulk in Fence mode
        t_fence1 = time.perf_counter()
        self.win_boundary.Fence(0)  # start the RMA epoch
        t_fence1_elapsed = time.perf_counter() - t_fence1

        t_get = time.perf_counter()
        for target_rank in target_ranks_to_fetch:
            remote_count = all_boundary_counts[target_rank]
            total_bytes = remote_count * self.dtype.itemsize
            recv_buffer = recv_buffers[target_rank]

            # Issue the Gets in bulk (non-blocking)
            self.win_boundary.Get(
                [recv_buffer, total_bytes, MPI.BYTE],
                target_rank,
                (0, total_bytes, MPI.BYTE)
            )
        t_get_elapsed = time.perf_counter() - t_get

        # Wait for every Get to finish
        t_fence2 = time.perf_counter()
        self.win_boundary.Fence(0)  # end the RMA epoch, synchronizing every operation
        t_fence2_elapsed = time.perf_counter() - t_fence2

        # Parse every received object
        t_parse = time.perf_counter()
        for target_rank in target_ranks_to_fetch:
            remote_count = all_boundary_counts[target_rank]
            recv_buffer = recv_buffers[target_rank]
            remote_objects = recv_buffer.view(self.dtype)[:remote_count]

            for obj in remote_objects:
                if obj['active']:
                    obj_id = int(obj['id'])
                    # Is it inside my ghost region?
                    if self._is_in_my_ghost_zone(obj['x'], obj['y'], obj['z']):
                        self.ghost_cache[obj_id] = obj.copy()
        t_parse_elapsed = time.perf_counter() - t_parse

        # Add the large objects to the cache as well (every rank holds them)
        t_large = time.perf_counter()
        for i in range(self.large_count):
            obj = self.large_objects[i]
            if obj['active'] and obj['owner_rank'] != self.rank:
                obj_id = int(obj['id'])
                self.ghost_cache[obj_id] = obj.copy()
        t_large_elapsed = time.perf_counter() - t_large

        self._ghost_cache_initialized = True
        self._ghost_cache_dirty = False

        elapsed = time.perf_counter() - t0

        if self.verbose and self.rank == 0:
            print(f"[perf] Ghost cache rebuilt: {len(self.ghost_cache)} objects in {elapsed*1e3:.1f}ms")
            print(f"  allgather: {t_allgather*1e3:.1f}ms, fence1: {t_fence1_elapsed*1e3:.1f}ms, "
                  f"get: {t_get_elapsed*1e3:.1f}ms, fence2: {t_fence2_elapsed*1e3:.1f}ms, "
                  f"parse: {t_parse_elapsed*1e3:.1f}ms, large: {t_large_elapsed*1e3:.1f}ms")

    def update_ghost_cache_incremental(self, changed_objects: np.ndarray):
        """
        Update the ghost cache incrementally

        Args:
            changed_objects: array of changed objects (from the broadcast changed_arr)

        Logic:
        1. a deleted object is removed from the cache
        2. an object inside my ghost region is updated or added
        3. an object outside my ghost region is removed (it may have migrated away)
        """
        if not self._ghost_cache_initialized:
            # The first call needs a full build
            self.rebuild_ghost_cache()
            return

        for obj in changed_objects:
            obj_id = int(obj['id'])
            # Fix: test the active field rather than the deleted field
            # active=False means the object has been deleted
            is_deleted = not bool(obj['active']) if 'active' in obj.dtype.names else (bool(obj['deleted']) if 'deleted' in obj.dtype.names else False)
            is_mine = (int(obj['owner_rank']) == self.rank) if 'owner_rank' in obj.dtype.names else False

            if is_deleted:
            # Deletion: remove from the cache
                if obj_id in self.ghost_cache:
                    self.ghost_cache.pop(obj_id)
                    if self.verbose:
                        print(f"Rank {self.rank}: Removed deleted obj {obj_id} from ghost_cache")
            elif is_mine:
            # A local object has no place in the ghost cache
                self.ghost_cache.pop(obj_id, None)
            elif self._is_in_my_ghost_zone(obj['x'], obj['y'], obj['z']):
            # Inside my ghost region: update or add
            # [fix] Convert to the right dtype, dropping extra fields such as 'new'/'deleted'
                if hasattr(obj, 'dtype') and obj.dtype != self.dtype:
                    converted_obj = np.zeros(1, dtype=self.dtype)[0]
                    for field in self.dtype.names:
                        if field in obj.dtype.names:
                            converted_obj[field] = obj[field]
                    self.ghost_cache[obj_id] = converted_obj
                else:
                    self.ghost_cache[obj_id] = obj.copy()
            else:
            # Outside my ghost region: remove (it may have migrated away)
                self.ghost_cache.pop(obj_id, None)

    def get_ghost_objects(self) -> np.ndarray:
        """
        Get the cached ghost objects

        Returns:
            the array of ghost objects (active ones only)
        """
        need_rebuild = not self._ghost_cache_initialized or self._ghost_cache_dirty
        if need_rebuild:
            self.rebuild_ghost_cache()

        if not self.ghost_cache:
            return np.array([], dtype=self.dtype)

        ghost_list = list(self.ghost_cache.values())

        # [fix] Handle a dtype mismatch
        # Objects in ghost_cache may come from changed_object_dtype (with 'new'/'deleted')
        # so only the fields present in self.dtype are extracted
        first_obj = ghost_list[0]
        if hasattr(first_obj, 'dtype') and first_obj.dtype != self.dtype:
            # The dtypes differ, so copy field by field
            result = np.zeros(len(ghost_list), dtype=self.dtype)
            for i, obj in enumerate(ghost_list):
                for field in self.dtype.names:
                    if field in obj.dtype.names:
                        result[i][field] = obj[field]
        else:
            result = np.array(ghost_list, dtype=self.dtype)

        # Drop the inactive objects (a defensive check)
        if 'active' in result.dtype.names and len(result) > 0:
            inactive_count = np.sum(~result['active'])
            if inactive_count > 0 and self.verbose:
                inactive_ids = result['id'][~result['active']]
                print(f"Rank {self.rank}: Filtering {inactive_count} inactive ghost objects: {inactive_ids[:5]}")
            result = result[result['active']]

        return result

    def invalidate_ghost_cache(self):
        """
        Mark the ghost cache for rebuilding

        When to call:
        1. after load balancing
        2. after the domain split changed
        """
        self._invalidate_count += 1
        self._ghost_cache_dirty = True

    def destroy(self):
        """Destroy the RMA object pool and release its resources"""
        self.free()

    @property
    def count(self) -> int:
        """Compatibility property: number of boundary objects (used by RMA communication)"""
        return self.boundary_count

    # ===== RMA operations =====

    def rma_get_boundary_objects(self, target_rank: int,
                                  needed_cell_ids: Optional[List[int]] = None) -> np.ndarray:
        """
        Pull only the boundary objects of the target rank

        Procedure:
        1. get the boundary_count of the target rank through allgather
        2. pull the boundary_objects array of the target rank
        3. filter locally: keep only the objects in needed_cell_ids

        Complexity: O(N_boundary)
        """
        # Single rank: there is no remote rank to pull from
        if self.win_boundary is None:
            return np.array([], dtype=self.dtype)

        # Get the boundary_count of every rank
        all_boundary_counts = self.comm.allgather(self.boundary_count)
        remote_boundary_count = all_boundary_counts[target_rank]

        if remote_boundary_count == 0:
            return np.array([], dtype=self.dtype)

        # Pull the boundary_objects array
        total_bytes = remote_boundary_count * self.dtype.itemsize
        recv_buffer = np.empty(total_bytes, dtype='B')

        self.win_boundary.Lock(target_rank, MPI.LOCK_SHARED)
        try:
            self.win_boundary.Get(
                [recv_buffer, total_bytes, MPI.BYTE],
                target_rank,
                (0, total_bytes, MPI.BYTE)
            )
        finally:
            self.win_boundary.Unlock(target_rank)

        boundary_objs = np.frombuffer(recv_buffer, dtype=self.dtype).copy()

        # Filter locally
        if needed_cell_ids is None:
            return boundary_objs

        if self.domain_decomposition is None:
            return boundary_objs

        # Keep the objects of the requested cells
        dd = self.domain_decomposition
        needed_cells_set = set(needed_cell_ids)
        filtered = []

        for obj in boundary_objs:
            if not obj['active']:
                continue
            try:
                cid = dd.pos_to_cell_id(obj['x'], obj['y'], obj['z'])
                if cid in needed_cells_set:
                    filtered.append(obj)
            except ValueError:
                continue

        if filtered:
            return np.array(filtered, dtype=self.dtype)
        else:
            return np.array([], dtype=self.dtype)

    def rma_get_cell_data_batch(self, requests: List[Tuple[int, List[int]]]) -> Dict[int, Dict[int, np.ndarray]]:
        """
        Pull the cell data of several ranks in bulk

        Args:
            requests: [(target_rank, [cell_ids]), ...]

        Returns:
            {rank: {cell_id: objects_array}}
        """
        result = {}

        for target_rank, cell_ids in requests:
            boundary_objs = self.rma_get_boundary_objects(target_rank, cell_ids)

            # Group by cell
            cell_data = {cid: [] for cid in cell_ids}
            cell_ids_set = set(cell_ids)

            if self.domain_decomposition is not None:
                dd = self.domain_decomposition
                for obj in boundary_objs:
                    try:
                        cid = dd.pos_to_cell_id(obj['x'], obj['y'], obj['z'])
                        if cid in cell_ids_set:
                            cell_data[cid].append(obj)
                    except ValueError:
                        continue

            # Convert to a NumPy array
            final_cell_data = {}
            for cid, obj_list in cell_data.items():
                if obj_list:
                    final_cell_data[cid] = np.array(obj_list, dtype=self.dtype)
                else:
                    final_cell_data[cid] = np.array([], dtype=self.dtype)

            result[target_rank] = final_cell_data

        return result

    # ===== Cell-related methods =====

    def get_objects_in_cell(self, cell_id: int) -> np.ndarray:
        """Get every object in one cell"""
        if self.domain_decomposition is None:
            return np.array([], dtype=self.dtype)

        dd = self.domain_decomposition
        all_objs = []

        # Walk the three arrays
        for arr, count in [(self.center_objects, self.center_count),
                           (self.boundary_objects, self.boundary_count),
                           (self.large_objects, self.large_count)]:
            for i in range(count):
                obj = arr[i]
                if not obj['active']:
                    continue
                try:
                    cid = dd.pos_to_cell_id(obj['x'], obj['y'], obj['z'])
                    if cid == cell_id:
                        all_objs.append(obj)
                except ValueError:
                    continue

        if all_objs:
            return np.array(all_objs, dtype=self.dtype)
        else:
            return np.array([], dtype=self.dtype)

    def get_non_empty_cells(self) -> List[int]:
        """Get the IDs of every non-empty cell"""
        if self.domain_decomposition is None:
            return []

        dd = self.domain_decomposition
        cell_ids = set()

        for arr, count in [(self.center_objects, self.center_count),
                           (self.boundary_objects, self.boundary_count),
                           (self.large_objects, self.large_count)]:
            for i in range(count):
                obj = arr[i]
                if not obj['active']:
                    continue
                if obj['owner_rank'] != self.rank:
                    continue
                try:
                    cid = dd.pos_to_cell_id(obj['x'], obj['y'], obj['z'])
                    cell_ids.add(cid)
                except ValueError:
                    continue

        return list(cell_ids)

    def extract_cells(self, cell_ids: List[int]) -> np.ndarray:
        """
        [fix] Extract and delete every object in the given cells (used by the RCB redistribution)

        Unlike get_objects_in_cell, this method removes the objects from the local arrays

        Args:
            cell_ids: list of cell IDs to extract

        Returns:
            the array of extracted objects (already removed locally)

        Note:
            - only objects owned by this rank are extracted (owner_rank == self.rank)
            - large objects are not extracted (they are broadcast, never migrated)
            - the internal index (obj_id_to_location) is updated
        """
        if self.domain_decomposition is None or len(cell_ids) == 0:
            return np.array([], dtype=self.dtype)

        dd = self.domain_decomposition
        cell_ids_set = set(cell_ids)

        # Collect the objects to extract
        extracted_objs = []
        obj_ids_to_delete = []

        # Walk the center and boundary arrays (large_objects is skipped)
        for arr, count, category in [
            (self.center_objects, self.center_count, 'center'),
            (self.boundary_objects, self.boundary_count, 'boundary')
        ]:
            for i in range(count):
                obj = arr[i]
                if not obj['active']:
                    continue
                # Extract only the objects owned by this rank
                if obj['owner_rank'] != self.rank:
                    continue

                try:
                    cid = dd.pos_to_cell_id(obj['x'], obj['y'], obj['z'])
                    if cid in cell_ids_set:
                        extracted_objs.append(obj.copy())
                        obj_ids_to_delete.append(int(obj['id']))
                except ValueError:
                    continue

        # Delete those objects
        for obj_id in obj_ids_to_delete:
            self.delete_object(obj_id)

        if extracted_objs:
            return np.array(extracted_objs, dtype=self.dtype)
        else:
            return np.array([], dtype=self.dtype)

    def update_large_objects_ownership(self):
        """
        [fix] Refresh owner_rank of every large object (after an RCB redistribution)

        Problem:
        - an RCB redistribution moved the domain boundaries
        - ordinary objects migrate through extract_cells and load_balancer updates owner_rank
        - large objects never migrate (every rank holds a replica), so owner_rank goes stale

        Fix:
        - walk every large object
        - recompute owner_rank from the current domain decomposition
        - every rank updates in step (the replicas stay identical)

        Returns:
            the number of large objects updated
        """
        if self.domain_decomposition is None:
            return 0

        dd = self.domain_decomposition
        updated_count = 0

        for i in range(self.large_count):
            obj = self.large_objects[i]
            if not obj['active']:
                continue

            # Compute the new owner from the position
            pos = [float(obj['x']), float(obj['y']), float(obj['z'])]
            try:
                new_owner = dd.get_owner_rank(pos)
            except (ValueError, KeyError):
                # Out of bounds: keep the old owner
                continue

            old_owner = int(obj['owner_rank'])
            if new_owner != old_owner:
                self.large_objects[i]['owner_rank'] = new_owner
                updated_count += 1

        return updated_count

    def _move_center_to_boundary(self, center_idx: int):
        """Move a center object into the boundary array"""
        obj = self.center_objects[center_idx].copy()
        obj_id = int(obj['id'])

        # Remove from center (swap-remove)
        self._delete_from_category(obj_id, 'center', center_idx)

        # Append to boundary
        self._add_to_boundary_array(obj)

    def _move_boundary_to_center(self, boundary_idx: int):
        """Move a boundary object into the center array"""
        obj = self.boundary_objects[boundary_idx].copy()
        obj_id = int(obj['id'])

        # Remove from boundary (swap-remove)
        self._delete_from_category(obj_id, 'boundary', boundary_idx)

        # Append to center
        self._add_to_center_array(obj)

    def _rebuild_small_object_locations(self):
        """Rebuild the location map of the center/boundary objects, keeping the large ones."""
        new_locations = {}
        for obj_id, location in self.obj_id_to_location.items():
            if location[0] == 'large':
                new_locations[obj_id] = location

        for i in range(self.center_count):
            obj_id = int(self.center_objects[i]['id'])
            new_locations[obj_id] = ('center', i)

        for i in range(self.boundary_count):
            obj_id = int(self.boundary_objects[i]['id'])
            new_locations[obj_id] = ('boundary', i)

        self.obj_id_to_location = new_locations

    def apply_migration_transaction(
        self,
        center_targets: np.ndarray,
        boundary_targets: np.ndarray,
        center_move_mask: np.ndarray,
        boundary_move_mask: np.ndarray,
        recv_buf: np.ndarray,
        owner_rank: int,
    ) -> Tuple[int, int, int]:
        """
        Apply an object migration transaction in one go:
        - delete the objects leaving this rank
        - reclassify local objects between center and boundary
        - insert the objects received

        Returns:
            (recv_center_count, recv_boundary_count, reclassified_count)
        """
        center_view = self.center_objects[:self.center_count]
        boundary_view = self.boundary_objects[:self.boundary_count]

        center_send_mask = center_targets >= 0 if len(center_targets) > 0 else np.empty(0, dtype=np.bool_)
        boundary_send_mask = boundary_targets >= 0 if len(boundary_targets) > 0 else np.empty(0, dtype=np.bool_)

        keep_center_mask = ~(center_send_mask | center_move_mask) if len(center_view) > 0 else np.empty(0, dtype=np.bool_)
        keep_boundary_mask = ~(boundary_send_mask | boundary_move_mask) if len(boundary_view) > 0 else np.empty(0, dtype=np.bool_)

        center_to_boundary = center_view[center_move_mask].copy() if np.any(center_move_mask) else np.array([], dtype=self.dtype)
        boundary_to_center = boundary_view[boundary_move_mask].copy() if np.any(boundary_move_mask) else np.array([], dtype=self.dtype)
        kept_center = center_view[keep_center_mask].copy() if len(center_view) > 0 else np.array([], dtype=self.dtype)
        kept_boundary = boundary_view[keep_boundary_mask].copy() if len(boundary_view) > 0 else np.array([], dtype=self.dtype)

        recv_center, recv_boundary = self._split_received_small_objects_batch(recv_buf, owner_rank)

        recv_center_count = len(recv_center)
        recv_boundary_count = len(recv_boundary)
        reclassified_count = len(center_to_boundary) + len(boundary_to_center)

        new_center_count = len(kept_center) + len(boundary_to_center) + recv_center_count
        new_boundary_count = len(kept_boundary) + len(center_to_boundary) + recv_boundary_count

        self._ensure_center_capacity_for(max(0, new_center_count - self.center_count))
        self._ensure_boundary_capacity_for(max(0, new_boundary_count - self.boundary_count))

        pos = 0
        if len(kept_center) > 0:
            self.center_objects[pos:pos + len(kept_center)] = kept_center
            pos += len(kept_center)
        if len(boundary_to_center) > 0:
            self.center_objects[pos:pos + len(boundary_to_center)] = boundary_to_center
            pos += len(boundary_to_center)
        if recv_center_count > 0:
            self.center_objects[pos:pos + recv_center_count] = recv_center
            pos += recv_center_count
        self.center_count = new_center_count

        pos = 0
        if len(kept_boundary) > 0:
            self.boundary_objects[pos:pos + len(kept_boundary)] = kept_boundary
            pos += len(kept_boundary)
        if len(center_to_boundary) > 0:
            self.boundary_objects[pos:pos + len(center_to_boundary)] = center_to_boundary
            pos += len(center_to_boundary)
        if recv_boundary_count > 0:
            self.boundary_objects[pos:pos + recv_boundary_count] = recv_boundary
            pos += recv_boundary_count
        self.boundary_count = new_boundary_count

        self._rebuild_small_object_locations()
        return recv_center_count, recv_boundary_count, reclassified_count

    def incremental_reclassify(self, old_local_cells: set = None,
                                new_local_cells: set = None) -> int:
        """
        After a boundary change, reclassify only the affected objects instead of rebuilding.

        Scans the center and boundary arrays and calls _classify_object_region() again for
        every object, moving it when the class changed.

        Returns: the number of objects reclassified
        """
        reclassified = 0

        if self.domain_decomposition is None:
            return 0

        # Nothing to do when the boundaries did not move.
        if old_local_cells is not None and new_local_cells is not None:
            if old_local_cells == new_local_cells:
                return 0

        affected_cells = None
        if old_local_cells is not None and new_local_cells is not None:
            changed_cells = set(old_local_cells) ^ set(new_local_cells)
            if not changed_cells:
                return 0

            dd = self.domain_decomposition
            grid = dd.grid_system
            min_cell_size = min(grid.cell_size)
            r_cut = float(dd.max_interaction_distance)
            expand_dist = r_cut + min_cell_size

            affected_cells = set(changed_cells)
            for cell_id in list(changed_cells):
                affected_cells.update(
                    grid.get_neighbor_cells(cell_id, max_distance=expand_dist, include_self=True)
                )
        dd = self.domain_decomposition
        grid = dd.grid_system
        total_cells = int(grid.grid_dim[0] * grid.grid_dim[1] * grid.grid_dim[2])
        if affected_cells is None:
            affected_cell_mask = np.ones(total_cells, dtype=np.bool_)
        else:
            affected_cell_mask = np.zeros(total_cells, dtype=np.bool_)
            if affected_cells:
                affected_indices = np.fromiter(affected_cells, dtype=np.int64)
                affected_cell_mask[affected_indices] = True

        bounds = dd.subdomain_bounds
        box_size = dd.box_size
        periodic = dd.periodic
        r_cut = float(dd.max_interaction_distance)

        center_view = self.center_objects[:self.center_count]
        boundary_view = self.boundary_objects[:self.boundary_count]

        center_move_mask = np.zeros(self.center_count, dtype=np.bool_)
        boundary_move_mask = np.zeros(self.boundary_count, dtype=np.bool_)

        if self.center_count > 0:
            center_sizes = center_view['size'] if 'size' in center_view.dtype.names else np.zeros(self.center_count, dtype=np.float64)
            _plan_reclassify_moves_jit(
                np.ascontiguousarray(center_view['x'], dtype=np.float64),
                np.ascontiguousarray(center_view['y'], dtype=np.float64),
                np.ascontiguousarray(center_view['z'], dtype=np.float64),
                np.ascontiguousarray(center_sizes, dtype=np.float64),
                np.ascontiguousarray(center_view['active'], dtype=np.bool_),
                affected_cell_mask,
                np.int32(grid.grid_dim[0]), np.int32(grid.grid_dim[1]), np.int32(grid.grid_dim[2]),
                float(box_size[0]), float(box_size[1]), float(box_size[2]),
                float(grid.cell_size[0]), float(grid.cell_size[1]), float(grid.cell_size[2]),
                bool(periodic[0]), bool(periodic[1]), bool(periodic[2]),
                float(bounds['x_min']), float(bounds['x_max']),
                float(bounds['y_min']), float(bounds['y_max']),
                float(bounds['z_min']), float(bounds['z_max']),
                r_cut, False, center_move_mask
            )

        if self.boundary_count > 0:
            boundary_sizes = boundary_view['size'] if 'size' in boundary_view.dtype.names else np.zeros(self.boundary_count, dtype=np.float64)
            _plan_reclassify_moves_jit(
                np.ascontiguousarray(boundary_view['x'], dtype=np.float64),
                np.ascontiguousarray(boundary_view['y'], dtype=np.float64),
                np.ascontiguousarray(boundary_view['z'], dtype=np.float64),
                np.ascontiguousarray(boundary_sizes, dtype=np.float64),
                np.ascontiguousarray(boundary_view['active'], dtype=np.bool_),
                affected_cell_mask,
                np.int32(grid.grid_dim[0]), np.int32(grid.grid_dim[1]), np.int32(grid.grid_dim[2]),
                float(box_size[0]), float(box_size[1]), float(box_size[2]),
                float(grid.cell_size[0]), float(grid.cell_size[1]), float(grid.cell_size[2]),
                bool(periodic[0]), bool(periodic[1]), bool(periodic[2]),
                float(bounds['x_min']), float(bounds['x_max']),
                float(bounds['y_min']), float(bounds['y_max']),
                float(bounds['z_min']), float(bounds['z_max']),
                r_cut, True, boundary_move_mask
            )

        center_move_count = int(np.count_nonzero(center_move_mask))
        boundary_move_count = int(np.count_nonzero(boundary_move_mask))
        reclassified = center_move_count + boundary_move_count
        if reclassified == 0:
            return 0

        center_to_boundary = center_view[center_move_mask].copy() if center_move_count > 0 else np.array([], dtype=self.dtype)
        center_remain = center_view[~center_move_mask].copy()
        boundary_to_center = boundary_view[boundary_move_mask].copy() if boundary_move_count > 0 else np.array([], dtype=self.dtype)
        boundary_remain = boundary_view[~boundary_move_mask].copy()

        new_center_count = len(center_remain) + len(boundary_to_center)
        new_boundary_count = len(boundary_remain) + len(center_to_boundary)

        self._ensure_center_capacity_for(max(0, new_center_count - self.center_count))
        self._ensure_boundary_capacity_for(max(0, new_boundary_count - self.boundary_count))

        if len(center_remain) > 0:
            self.center_objects[:len(center_remain)] = center_remain
        if len(boundary_to_center) > 0:
            self.center_objects[len(center_remain):new_center_count] = boundary_to_center
        self.center_count = new_center_count

        if len(boundary_remain) > 0:
            self.boundary_objects[:len(boundary_remain)] = boundary_remain
        if len(center_to_boundary) > 0:
            self.boundary_objects[len(boundary_remain):new_boundary_count] = center_to_boundary
        self.boundary_count = new_boundary_count

        self._rebuild_small_object_locations()
        return reclassified

    # ===== Synchronization =====

    def sync_large_objects(self, changed_objects: List[dict]):
        """
        Synchronize the large objects across ranks

        Called after an event executes; the changed_objects broadcast does the work
        """
        for obj_diff in changed_objects:
            obj_id = obj_diff['id']

            # Deletion is signalled by active = False
            if obj_diff.get('active', True) == False:
                # Delete the large object
                if obj_id in self.large_object_ids:
                    self.remove_large_object(obj_id)

            elif obj_diff.get('new', False) or obj_id in self.large_object_ids:
                # A new or updated large object
                # Build the full object record
                obj = np.zeros(1, dtype=self.dtype)[0]
                obj['id'] = obj_id
                if 'type_id' in self.dtype.names:
                    type_name = str(obj_diff.get('type', ''))
                    obj['type_id'] = self._type_to_int.get(type_name, obj_diff.get('type_id', -1))
                obj['x'] = obj_diff.get('x', 0.0)
                obj['y'] = obj_diff.get('y', 0.0)
                obj['z'] = obj_diff.get('z', 0.0)
                obj['size'] = obj_diff.get('size', 0.0)
                obj['owner_rank'] = obj_diff.get('owner_rank', 0)
                obj['active'] = True

                # Is it a large object?
                if self._is_large_object(obj):
                    if obj_id in self.obj_id_to_location:
                        self.update_object(obj)
                    else:
                        self._add_to_large_array(obj)

    def sort_and_sync(self):
        """
        Sort the objects and synchronize (initialization only)

        Guarantees:
        1. objects are classified by position (center/boundary/large)
        2. the large object array is identical on every rank

        Note: at runtime apply_changed_objects() updates incrementally and this is not called
        """
        # Reclassify every object
        all_local = self.get_local_objects(active_only=True)

        # Empty every array
        self.center_count = 0
        self.boundary_count = 0
        self.large_count = 0
        self.obj_id_to_location.clear()
        self.large_object_ids.clear()

        # Add them again (classification happens automatically)
        for obj in all_local:
            self.add_object(obj)

        # Synchronize the large objects across ranks
        self._sync_all_large_objects()

    def _create_object_from_changed_record(self, obj_rec, owner_rank: int) -> np.ndarray:
        """
        Build an object_dtype record from a changed_arr record

        Args:
            obj_rec: a changed_object_dtype record (with the new/deleted flags)
            owner_rank: rank owning the object

        Returns:
            an object_dtype record
        """
        obj = np.zeros(1, dtype=self.dtype)[0]
        # Copy every shared field
        for field in self.dtype.names:
            if field in obj_rec.dtype.names:
                obj[field] = obj_rec[field]
        obj['owner_rank'] = owner_rank
        return obj

    def _build_changed_action_plan(self, changed_arr: np.ndarray, domain_decomp):
        """Use the JIT planner to precompute the action, owner and class of changed_arr."""
        n = len(changed_arr)
        actions = np.empty(n, dtype=np.int32)
        owners = np.empty(n, dtype=np.int32)
        categories = np.empty(n, dtype=np.int32)
        valid = np.empty(n, dtype=np.bool_)

        if n == 0:
            return actions, owners, categories, valid

        grid = domain_decomp.grid_system
        bounds = self.domain_decomposition.subdomain_bounds
        min_cell_size = min(grid.cell_size)
        r_cut = float(self.domain_decomposition.max_interaction_distance)
        ghost_capacity = int(np.ceil(r_cut / min_cell_size)) * min_cell_size
        large_threshold = float(ghost_capacity * self.large_object_multiplier)

        size_arr = changed_arr['size'] if 'size' in changed_arr.dtype.names else np.zeros(n, dtype=np.float64)

        _plan_changed_actions_jit(
            np.ascontiguousarray(changed_arr['x'], dtype=np.float64),
            np.ascontiguousarray(changed_arr['y'], dtype=np.float64),
            np.ascontiguousarray(changed_arr['z'], dtype=np.float64),
            np.ascontiguousarray(size_arr, dtype=np.float64),
            np.ascontiguousarray(changed_arr['active'], dtype=np.bool_),
            np.ascontiguousarray(changed_arr['new'], dtype=np.bool_),
            np.ascontiguousarray(domain_decomp.ownership_map, dtype=np.int32),
            np.int32(grid.grid_dim[0]), np.int32(grid.grid_dim[1]), np.int32(grid.grid_dim[2]),
            float(domain_decomp.box_size[0]), float(domain_decomp.box_size[1]), float(domain_decomp.box_size[2]),
            float(grid.cell_size[0]), float(grid.cell_size[1]), float(grid.cell_size[2]),
            bool(domain_decomp.periodic[0]), bool(domain_decomp.periodic[1]), bool(domain_decomp.periodic[2]),
            float(bounds['x_min']), float(bounds['x_max']),
            float(bounds['y_min']), float(bounds['y_max']),
            float(bounds['z_min']), float(bounds['z_max']),
            r_cut, large_threshold, np.int32(self.rank),
            actions, owners, categories, valid
        )

        return actions, owners, categories, valid

    def apply_changed_objects(self, changed_arr: np.ndarray, domain_decomp) -> list:
        """
        Update the object pool incrementally (only the changed objects)

        The runtime replacement for sort_and_sync().
        Wraps the object update logic of _execute_event step 3.

        Procedure:
        1. walk changed_arr (the broadcast array of changed objects)
        2. classify each record as delete / insert / modify
        3. for a modification decide the owner and class (center/boundary/large)
        4. move the data

        Args:
            changed_arr: the broadcast array of changed objects (changed_object_dtype, with new/deleted)
            domain_decomp: domain decomposition object (for get_owner_rank)

        Returns:
            changed_ids: list of changed object IDs
        """
        changed_ids = []
        _n_deleted = 0
        _n_added = 0
        _n_migrate_in = 0
        _n_migrate_out = 0
        _n_phantom_add = 0  # objects that belong to no rank yet get added

        _count_before = self.get_local_count()
        actions, owners, categories, valid_plan = self._build_changed_action_plan(changed_arr, domain_decomp)

        for i in range(len(changed_arr)):
            obj_rec = changed_arr[i]
            obj_id = int(obj_rec['id'])
            changed_ids.append(obj_id)

            # The active flag signals deletion
            action = int(actions[i])
            is_deleted = (action == ACTION_DELETE)
            is_new = (action == ACTION_NEW)

            # Does this involve a large-object conversion?
            was_large = self.is_large_object_by_id(obj_id)

            new_owner = int(owners[i])
            target_category = int(categories[i])

            if not is_deleted and not bool(valid_plan[i]):
                pos = [float(obj_rec['x']), float(obj_rec['y']), float(obj_rec['z'])]
                new_owner = domain_decomp.get_owner_rank(pos)
                fallback_obj = self._create_object_from_changed_record(obj_rec, new_owner)
                if self._is_large_object(fallback_obj):
                    target_category = CATEGORY_LARGE
                elif new_owner == self.rank:
                    region = self._classify_object_region(fallback_obj)
                    target_category = CATEGORY_BOUNDARY if region == 'boundary' else CATEGORY_CENTER
                else:
                    target_category = CATEGORY_NONE

            if is_deleted:
                # Delete the object
                if was_large:
                    self.remove_large_object(obj_id)
                    _n_deleted += 1
                elif self.has_object(obj_id):
                    self.delete_object(obj_id)
                    _n_deleted += 1

            elif is_new:
                # A new object: use the owner and class planned by the JIT kernel
                new_obj = self._create_object_from_changed_record(obj_rec, new_owner)

                # Set the new-object flag
                new_obj['new'] = True
                new_obj['changed'] = False

                if target_category == CATEGORY_LARGE:
                    self._add_to_large_array(new_obj)
                    if new_owner == self.rank:
                        _n_added += 1
                elif new_owner == self.rank:
                    self._add_to_category_code(new_obj, target_category)
                    _n_added += 1

            else:
                # A modified object: use the owner and class planned by the JIT kernel
                updated_obj = self._create_object_from_changed_record(obj_rec, new_owner)

                # Set the changed flag
                updated_obj['new'] = False
                updated_obj['changed'] = True

                is_now_large = (target_category == CATEGORY_LARGE)

                # Case 1: small -> large
                if not was_large and is_now_large:
                    if self.has_object(obj_id):
                        self.delete_object(obj_id)
                    self._add_to_large_array(updated_obj)

                # Case 2: large -> small
                elif was_large and not is_now_large:
                    self.remove_large_object(obj_id)
                    if new_owner == self.rank:
                        self._add_to_category_code(updated_obj, target_category)

                # Case 3: large -> large
                elif was_large and is_now_large:
                    self._upsert_large_object_planned(updated_obj)

                # Case 4: small -> small
                else:
                    if self.has_object(obj_id):
                        if new_owner == self.rank:
                            self._update_local_small_object_planned(updated_obj, target_category)
                        else:
                            # Migrating to another rank: drop the local copy
                            self.delete_object(obj_id)
                            _n_migrate_out += 1
                    elif new_owner == self.rank:
                        # Migrating in from another rank: add it with the planned class
                        self._add_to_category_code(updated_obj, target_category)
                        _n_migrate_in += 1
                    elif new_owner != self.rank and not self.has_object(obj_id):
                        # The object is on neither this rank nor becomes ours -> normal, skip
                        pass

        _count_after = self.get_local_count()
        _actual_delta = _count_after - _count_before
        _expected_delta = _n_added - _n_deleted + _n_migrate_in - _n_migrate_out

        # Report only on an anomaly (actual changes != expected changes)
        if _actual_delta != _expected_delta:
            print(
                f"[DBG] *** MISMATCH *** rank={self.rank} "
                f"new=+{_n_added} del=-{_n_deleted} "
                f"mig_in=+{_n_migrate_in} mig_out=-{_n_migrate_out} "
                f"expected={_expected_delta:+d} actual={_actual_delta:+d} "
                f"count={_count_before}->{_count_after} "
                f"center={self.center_count} boundary={self.boundary_count} "
                f"large={self.large_count} id_map_size={len(self.obj_id_to_location)}",
                flush=True
            )

        return changed_ids

    def _sync_all_large_objects(self):
        """
        Synchronize every large object across ranks

        [optimization] Large objects are kept in sync by the changed_objects broadcast,
        so this method runs only at initialization and needs no allgather/allreduce later
        """
        # Collect the large objects of every rank (only those we own)
        local_large = []
        if self.large_count > 0:
            for i in range(self.large_count):
                obj = self.large_objects[i]
                if obj['active'] and obj['owner_rank'] == self.rank:
                    local_large.append(obj.copy())

        # With no local large object, take part in the allgather with an empty list
        # The allreduce pre-check was removed (this runs once, at initialization)
        all_large_lists = self.comm.allgather(local_large)

        # Merge into the local array
        for rank_large in all_large_lists:
            for obj in rank_large:
                obj_id = int(obj['id'])
                if obj_id not in self.obj_id_to_location:
                    self._add_to_large_array(obj)

    # ===== Cleanup =====

    def free(self):
        """Release the MPI window"""
        if self.win_boundary is not None:
            self.win_boundary.Free()
            self.win_boundary = None

    def __del__(self):
        """Destructor"""
        try:
            self.free()
        except:
            pass


# ===== Factory function =====

def create_rma_object_pool(comm: MPI.Comm,
                            dtype: Optional[np.dtype] = None,
                            initial_capacity: int = 1000,
                            auto_expand: bool = True,
                            verbose: bool = False,
                            domain_decomposition=None,
                            large_object_multiplier: float = 5.0,
                            max_large_objects: int = 1000) -> RMAObjectPool:
    """
    Factory: create an RMA object pool (three-way storage architecture)

    Args:
        comm: MPI communicator
        dtype: object dtype (OBJECT_DTYPE by default)
        initial_capacity: initial capacity
        auto_expand: whether to grow automatically
        verbose: whether to print debugging output
        domain_decomposition: domain decomposition object (used to classify objects)
        large_object_multiplier: multiplier of the large-object test (default 5.0)
        max_large_objects: upper bound on the number of large objects (default 1000)

    Returns:
        an RMAObjectPool instance
    """
    return RMAObjectPool(
        comm=comm,
        dtype=dtype,
        initial_capacity=initial_capacity,
        auto_expand=auto_expand,
        verbose=verbose,
        domain_decomposition=domain_decomposition,
        large_object_multiplier=large_object_multiplier,
        max_large_objects=max_large_objects
    )


def create_object(obj_id: int, obj_type: str, position: List[float],
                  owner_rank: int, float_props: List[float] = None,
                  size: float = 0.0, config: dict = None) -> np.ndarray:
    """
    Create a single object

    Args:
        obj_id: object ID
        obj_type: object type
        position: position [x, y, z]
        owner_rank: owning rank
        float_props: list of float properties
        size: physical size of the object (radius/scale)
        config: configuration dict (used to get the dynamic dtype)

    Returns:
        the object record
    """
    # Get the dtype
    if config is not None and '_object_dtype' in config:
        dtype = config['_object_dtype']
    else:
        dtype = OBJECT_DTYPE

    obj = np.zeros(1, dtype=dtype)[0]
    obj['id'] = obj_id
    obj['x'] = position[0]
    obj['y'] = position[1]
    obj['z'] = position[2]
    obj['owner_rank'] = owner_rank
    obj['active'] = True

    # Initialize the flags (a new object)
    if 'new' in dtype.names:
        obj['new'] = True
        obj['changed'] = False

    # Set type_id (the integer type ID that speeds up the hot path)
    type_to_int = config.get('_type_to_int', {}) if config else {}
    obj['type_id'] = type_to_int.get(str(obj_type), -1)

    if 'size' in dtype.names:
        obj['size'] = size

    if float_props and 'float_props' in dtype.names:
        max_float = dtype.fields['float_props'][0].shape[0]
        for i, val in enumerate(float_props[:max_float]):
            obj['float_props'][i] = val

    return obj
