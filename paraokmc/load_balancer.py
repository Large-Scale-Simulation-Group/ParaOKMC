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
Dynamic object load balancer

Redistributes objects with an RCB (recursive coordinate bisection) algorithm.
Cures the memory and compute bottleneck of badly skewed particle distributions

Core design:
1. Rectangular topology: the region owned by a rank must be a 3D rectangular prism
2. Fine-grid alignment: cuts must fall on the cell boundaries defined by GridSystem
3. No loss: objects are conserved and their IDs never change

Workflow:
Step 1: profiling - build the global load density map over the fine grid
Step 2: decision making - rank 0 runs the RCB algorithm
Step 3: broadcast and update - refresh ownership_map
Step 4: migration - exchange the objects with Alltoallv

"""

import numpy as np
from mpi4py import MPI
from typing import Dict, List, Tuple, Optional
import time
from copy import deepcopy

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


TAG_OBJECT_REBALANCE = 31001


class Bounds:
    """
    Bounds of a 3D rectangular region (range of grid indices)

    Attributes:
        x_start, x_end: range along X, [x_start, x_end)
        y_start, y_end: range along Y, [y_start, y_end)
        z_start, z_end: range along Z, [z_start, z_end)
    """
    def __init__(self, x_start: int, x_end: int,
                 y_start: int, y_end: int,
                 z_start: int, z_end: int):
        self.x_start = x_start
        self.x_end = x_end
        self.y_start = y_start
        self.y_end = y_end
        self.z_start = z_start
        self.z_end = z_end

    def to_dict(self) -> Dict:
        """Convert to a dict (for MPI transfer)"""
        return {
            'x_start': self.x_start,
            'x_end': self.x_end,
            'y_start': self.y_start,
            'y_end': self.y_end,
            'z_start': self.z_start,
            'z_end': self.z_end,
        }

    @staticmethod
    def from_dict(d: Dict) -> 'Bounds':
        """Build from a dict (for MPI transfer)"""
        return Bounds(
            d['x_start'], d['x_end'],
            d['y_start'], d['y_end'],
            d['z_start'], d['z_end']
        )

    def volume(self) -> int:
        """Volume of the region (number of cells)"""
        return (self.x_end - self.x_start) * \
               (self.y_end - self.y_start) * \
               (self.z_end - self.z_start)

    def __repr__(self) -> str:
        return f"Bounds(x=[{self.x_start}, {self.x_end}), " \
               f"y=[{self.y_start}, {self.y_end}), " \
               f"z=[{self.z_start}, {self.z_end}))"


@njit(parallel=True, cache=True)
def _compute_send_targets_jit(
    x_arr, y_arr, z_arr, active_arr, owner_arr,
    my_rank, leaving_owner_map,
    gx, gy, gz,
    box_x, box_y, box_z,
    cell_x, cell_y, cell_z,
    periodic_x, periodic_y, periodic_z
):
    """Decide which objects must migrate and to which rank."""
    n = len(x_arr)
    out_targets = np.full(n, -1, dtype=np.int32)

    for i in prange(n):
        if not active_arr[i]:
            continue
        if int(owner_arr[i]) != my_rank:
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

        cell_id = ix + iy * gx + iz * gx * gy
        target = int(leaving_owner_map[cell_id])
        if target >= 0 and target != my_rank:
            out_targets[i] = target

    return out_targets


@njit(parallel=True, cache=True)
def _plan_migration_and_reclassify_jit(
    x_arr, y_arr, z_arr, size_arr, active_arr, owner_arr,
    my_rank, leaving_owner_map,
    gx, gy, gz,
    box_x, box_y, box_z,
    cell_x, cell_y, cell_z,
    periodic_x, periodic_y, periodic_z,
    bound_x_min, bound_x_max,
    bound_y_min, bound_y_max,
    bound_z_min, bound_z_max,
    r_cut, current_region_is_boundary,
    out_targets, out_move_mask
):
    """Plan the migration targets and the local reclassification in one scan."""
    n = len(x_arr)
    for i in prange(n):
        out_targets[i] = -1
        out_move_mask[i] = False

        if not active_arr[i]:
            continue
        if int(owner_arr[i]) != my_rank:
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
        target = int(leaving_owner_map[cell_id])
        if target >= 0 and target != my_rank:
            out_targets[i] = target
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


@njit(cache=True)
def _fill_send_plan_jit(targets, source_kind, positions, out_kind, out_idx):
    """Pack the send plan compactly, grouped by target rank."""
    for i in range(len(targets)):
        target = int(targets[i])
        if target < 0:
            continue
        pos = positions[target]
        out_kind[pos] = source_kind
        out_idx[pos] = i
        positions[target] = pos + 1


@njit(cache=True)
def _fill_send_buffer_jit(targets, objects, positions, out_buf, out_ids):
    """Fill the final send buffer directly, grouped by target rank."""
    for i in range(len(targets)):
        target = int(targets[i])
        if target < 0:
            continue
        pos = positions[target]
        out_buf[pos] = objects[i]
        out_ids[pos] = int(objects[i]['id'])
        positions[target] = pos + 1


def _boxes_intersect(a: Dict, b: Dict) -> bool:
    """Test whether two half-open boxes overlap in volume."""
    return (
        a['x_start'] < b['x_end'] and b['x_start'] < a['x_end'] and
        a['y_start'] < b['y_end'] and b['y_start'] < a['y_end'] and
        a['z_start'] < b['z_end'] and b['z_start'] < a['z_end']
    )


def _pack_migration_message(obj_chunk: np.ndarray, obj_dtype) -> np.ndarray:
    """Pack count + object payload into a single BYTE message."""
    count = int(len(obj_chunk))
    header = np.array([count], dtype=np.int64).view(np.uint8)
    payload_nbytes = count * obj_dtype.itemsize
    msg = np.empty(len(header) + payload_nbytes, dtype=np.uint8)
    msg[:len(header)] = header
    if payload_nbytes > 0:
        msg[len(header):] = np.ascontiguousarray(obj_chunk).view(np.uint8)
    return msg


def _unpack_migration_message(msg_bytes: np.ndarray, obj_dtype) -> np.ndarray:
    """Parse a single count + object payload message."""
    header_nbytes = np.dtype(np.int64).itemsize
    if len(msg_bytes) < header_nbytes:
        raise ValueError("Migration message shorter than header")

    count = int(msg_bytes[:header_nbytes].view(np.int64)[0])
    if count == 0:
        return np.array([], dtype=obj_dtype)

    payload = msg_bytes[header_nbytes:]
    expected = count * obj_dtype.itemsize
    if len(payload) != expected:
        raise ValueError(
            f"Migration payload size mismatch: got {len(payload)} bytes, expected {expected}"
        )
    return payload.view(obj_dtype).copy()


def compute_imbalance(loads: List[int]) -> float:
    """
    Compute the load imbalance

    Args:
        loads: list of loads per rank

    Returns:
        imbalance: max_load / avg_load - 1.0
                   0.0 means perfectly balanced, > 0.2 means badly imbalanced
    """
    if len(loads) == 0 or sum(loads) == 0:
        return 0.0

    max_load = max(loads)
    avg_load = sum(loads) / len(loads)

    if avg_load == 0:
        return 0.0

    return max_load / avg_load - 1.0


def recursive_bisection(grid_load: np.ndarray,
                       ranks: List[int],
                       current_box: Bounds,
                       verbose: bool = False) -> Dict[int, Bounds]:
    """
    Recursive coordinate bisection (RCB)

    Args:
        grid_load: 3D numpy array of weights (the global load density map)
        ranks: list of rank IDs available for this region
        current_box: bounds of the current region
        verbose: whether to print debugging output

    Returns:
        a {rank_id: Bounds} dict describing the region owned by each rank
    """
    # Recursion stops with a single rank left
    if len(ranks) == 1:
        return {ranks[0]: current_box}

    # 1. Choose the split axis (the longest one)
    lengths = [
        current_box.x_end - current_box.x_start,
        current_box.y_end - current_box.y_start,
        current_box.z_end - current_box.z_start
    ]
    axis = int(np.argmax(lengths))

    if verbose:
        print(f"[RCB] Splitting {len(ranks)} ranks along axis {axis}, box={current_box}")

    # 2. Extract the load submatrix of this region
    sub_load = grid_load[
        current_box.x_start:current_box.x_end,
        current_box.y_start:current_box.y_end,
        current_box.z_start:current_box.z_end
    ]

    # 3. Project the load along the chosen axis
    other_axes = [i for i in range(3) if i != axis]
    projection = np.sum(sub_load, axis=tuple(other_axes))

    # 4. Find the split point
    # Goal: load on the left ~ total load * (ranks on the left / total ranks)
    total_load = np.sum(projection)
    mid = len(ranks) // 2
    target_load = total_load * (mid / len(ranks))

    cumsum = np.cumsum(projection)

    # searchsorted finds the first prefix reaching the target.
    # Note: a legal split boundary s means the left load is projection[:s], so the
    # boundary candidate for prefix index idx is idx+1.
    if total_load > 0:
        split_idx_local = int(np.searchsorted(cumsum, target_load)) + 1
        # Keep the split point inside the valid range [1, len-1]
        split_idx_local = max(1, min(split_idx_local, len(projection) - 1))
    else:
        # Split evenly if the region carries no load
        split_idx_local = len(projection) // 2

    # Convert to a global grid index
    axis_start = [current_box.x_start, current_box.y_start, current_box.z_start][axis]
    split_idx_global = axis_start + split_idx_local

    if verbose:
        left_load = np.sum(projection[:split_idx_local])
        right_load = np.sum(projection[split_idx_local:])
        print(f"  Split at idx={split_idx_global} (local={split_idx_local}), "
              f"left_load={left_load}, right_load={right_load}")

    # 5. Split the rank set
    left_ranks = ranks[:mid]
    right_ranks = ranks[mid:]

    # 6. Build the bounds of the sub-regions
    if axis == 0:  # split along X
        left_box = Bounds(
            current_box.x_start, split_idx_global,
            current_box.y_start, current_box.y_end,
            current_box.z_start, current_box.z_end
        )
        right_box = Bounds(
            split_idx_global, current_box.x_end,
            current_box.y_start, current_box.y_end,
            current_box.z_start, current_box.z_end
        )
    elif axis == 1:  # split along Y
        left_box = Bounds(
            current_box.x_start, current_box.x_end,
            current_box.y_start, split_idx_global,
            current_box.z_start, current_box.z_end
        )
        right_box = Bounds(
            current_box.x_start, current_box.x_end,
            split_idx_global, current_box.y_end,
            current_box.z_start, current_box.z_end
        )
    else:  # split along Z
        left_box = Bounds(
            current_box.x_start, current_box.x_end,
            current_box.y_start, current_box.y_end,
            current_box.z_start, split_idx_global
        )
        right_box = Bounds(
            current_box.x_start, current_box.x_end,
            current_box.y_start, current_box.y_end,
            split_idx_global, current_box.z_end
        )

    # 7. Recurse into the left and right sub-regions
    result = {}
    result.update(recursive_bisection(grid_load, left_ranks, left_box, verbose))
    result.update(recursive_bisection(grid_load, right_ranks, right_box, verbose))

    return result


def distributed_rcb(comm: MPI.Comm,
                    global_load_3d: np.ndarray,
                    global_box: Bounds,
                    all_ranks: List[int],
                    verbose: bool = False) -> Dict[int, Bounds]:
    """
    Distributed recursive coordinate bisection (distributed RCB)

    Uses the same mathematics as the centralized recursive_bisection()
    (argmax to pick the axis, then cumsum + searchsorted), but differently executed:
    - every rank holds an identical global_load_3d up front
    - the sub-communicators only drive the recursion; no sub-map is reduced again
    - no root has to hold and solve the whole RCB tree serially

    Args:
        comm: MPI communicator of the current subgroup
        global_load_3d: the 3D global load map, identical on every rank
        global_box: bounds of the current recursion region
        all_ranks: global rank IDs in the current subgroup
        verbose: debugging output

    Returns:
        a {rank_id: Bounds} dict
    """
    if comm.size != len(all_ranks):
        raise ValueError(
            f"distributed_rcb requires comm.size == len(all_ranks), "
            f"got {comm.size} and {len(all_ranks)}"
        )

    if len(all_ranks) == 1:
        return {all_ranks[0]: global_box}

    # 1. Choose the split axis (the longest one)
    lengths = [
        global_box.x_end - global_box.x_start,
        global_box.y_end - global_box.y_start,
        global_box.z_end - global_box.z_start
    ]
    axis = int(np.argmax(lengths))

    # 2. Extract the sub-load of the current box
    sub_load = global_load_3d[
        global_box.x_start:global_box.x_end,
        global_box.y_start:global_box.y_end,
        global_box.z_start:global_box.z_end
    ]

    # 3. Project to 1D straight from the consistent global load map
    other_axes = tuple(i for i in range(3) if i != axis)
    global_proj = np.sum(sub_load, axis=other_axes, dtype=np.int64)

    # 5. Every rank computes the same split point
    total_load = np.sum(global_proj)
    mid = len(all_ranks) // 2
    target_load = total_load * (mid / len(all_ranks))

    cumsum = np.cumsum(global_proj)

    if total_load > 0:
        # A split s means the left load is global_proj[:s], so the prefix index
        # returned by searchsorted becomes the boundary candidate idx+1.
        split_idx_local = int(np.searchsorted(cumsum, target_load)) + 1
        split_idx_local = max(1, min(split_idx_local, len(global_proj) - 1))
    else:
        split_idx_local = len(global_proj) // 2

    # Convert to a global grid index
    axis_start = [global_box.x_start, global_box.y_start, global_box.z_start][axis]
    split_idx_global = axis_start + split_idx_local

    if verbose and comm.rank == 0:
        left_load = int(np.sum(global_proj[:split_idx_local]))
        right_load = int(np.sum(global_proj[split_idx_local:]))
        print(f"[Distributed RCB] axis={axis}, split={split_idx_global}, "
              f"left_load={left_load}, right_load={right_load}, ranks={len(all_ranks)}")

    # 6. Split the rank set
    left_ranks = all_ranks[:mid]
    right_ranks = all_ranks[mid:]

    # 7. Build the bounds of the sub-regions
    if axis == 0:
        left_box = Bounds(
            global_box.x_start, split_idx_global,
            global_box.y_start, global_box.y_end,
            global_box.z_start, global_box.z_end
        )
        right_box = Bounds(
            split_idx_global, global_box.x_end,
            global_box.y_start, global_box.y_end,
            global_box.z_start, global_box.z_end
        )
    elif axis == 1:
        left_box = Bounds(
            global_box.x_start, global_box.x_end,
            global_box.y_start, split_idx_global,
            global_box.z_start, global_box.z_end
        )
        right_box = Bounds(
            global_box.x_start, global_box.x_end,
            split_idx_global, global_box.y_end,
            global_box.z_start, global_box.z_end
        )
    else:
        left_box = Bounds(
            global_box.x_start, global_box.x_end,
            global_box.y_start, global_box.y_end,
            global_box.z_start, split_idx_global
        )
        right_box = Bounds(
            global_box.x_start, global_box.x_end,
            global_box.y_start, global_box.y_end,
            split_idx_global, global_box.z_end
        )

    # 8. Comm.Split groups by colour
    color = 0 if comm.rank < mid else 1
    sub_comm = comm.Split(color, comm.rank)

    # 9. Recurse
    if color == 0:
        sub_result = distributed_rcb(sub_comm, global_load_3d, left_box, left_ranks, verbose)
    else:
        sub_result = distributed_rcb(sub_comm, global_load_3d, right_box, right_ranks, verbose)

    if sub_comm.rank == 0:
        local_bounds = list(sub_result.items())
    else:
        local_bounds = []
    sub_comm.Free()

    # 10. Allgather collects the complete bounds_list
    all_bounds_lists = comm.allgather(local_bounds)

    result = {}
    for bounds_list in all_bounds_lists:
        for rank_id, bounds in bounds_list:
            result[rank_id] = bounds

    return result


class ObjectLoadBalancer:
    """
    Object load balancer

    Features:
    1. Measure the global load distribution
    2. Run the RCB algorithm for a new partition
    3. Migrate the objects

    Use cases:
    - cascade simulations: objects crowd around the irradiation source
    - precipitation simulations: objects grow quickly in a local region
    """

    def __init__(self, comm: MPI.Comm, domain_decomp, object_pool, config: Dict):
        """
        Initialize the load balancer

        Args:
            comm: MPI communicator
            domain_decomp: DomainDecomposition instance
            object_pool: RMAObjectPool instance
            config: configuration dict
        """
        self.comm = comm
        self.rank = comm.Get_rank()
        self.size = comm.Get_size()

        self.domain_decomp = domain_decomp
        self.object_pool = object_pool

        # Configuration parameters
        self.enabled = config.get('enabled', False)
        self.balance_interval = config.get('balance_interval', 500)
        self.imbalance_threshold = config.get('imbalance_threshold', 0.2)
        self.min_improvement = config.get('min_improvement', 0.05)
        self.verbose = config.get('verbose', False)

        # Statistics
        self.stats = {
            'total_rebalances': 0,
            'total_objects_migrated': 0,
            'total_time': 0.0,
        }

        if self.verbose and self.rank == 0:
            print(f"[LoadBalancer] Initialized:")
            print(f"  enabled={self.enabled}")
            print(f"  balance_interval={self.balance_interval}")
            print(f"  imbalance_threshold={self.imbalance_threshold}")

    def should_rebalance(self, current_step: int) -> bool:
        """
        Decide whether a redistribution is due

        Args:
            current_step: current step

        Returns:
            True if a redistribution is needed
        """
        if not self.enabled:
            return False

        # 1. Check the interval
        if current_step % self.balance_interval != 0:
            return False

        # 2. Collect the object count of every rank
        local_count = self.object_pool.get_local_count()
        all_counts = self.comm.allgather(local_count)

        # 3. Compute the imbalance
        imbalance = compute_imbalance(all_counts)

        # 4. Compare against the threshold
        # [fix] With very few objects the imbalance easily exceeds the threshold
        # while redistribution is pointless: 5 objects over 4 ranks is [1,1,1,2] at best,
        # an imbalance of 0.6. So skip redistribution when the total is below ranks*2
        total_objects = sum(all_counts)
        min_objects_for_rebalance = self.comm.size * 2

        if total_objects < min_objects_for_rebalance:
            return False

        return imbalance > self.imbalance_threshold

    def collect_global_load(self) -> Optional[np.ndarray]:
        """
        Collect the global load density map

        Procedure:
        1. walk the local objects and count the particles per cell
        2. MPI_Reduce onto rank 0 to form the global density map

        Returns:
            global_load_map: a (GridDim_X, GridDim_Y, GridDim_Z) 3D array (rank 0 only)
            None: on the other ranks
        """
        # Fine grid dimensions
        grid_dim = self.domain_decomp.grid_system.grid_dim

        # Initialize the local load map
        local_load = np.zeros(grid_dim, dtype=np.int32)

        # Count the local objects
        local_objs = self.object_pool.get_local_objects()

        for obj in local_objs:
            # Cell ID holding the object
            x, y, z = obj['x'], obj['y'], obj['z']
            cell_id = self.domain_decomp.grid_system.pos_to_cell_id(x, y, z)
            ix, iy, iz = self.domain_decomp.grid_system.cell_id_to_grid_index(cell_id)

            # Accumulate into the local load map
            local_load[ix, iy, iz] += 1

        # Global reduction onto rank 0
        if self.rank == 0:
            global_load = np.zeros(grid_dim, dtype=np.int32)
        else:
            global_load = None

        self.comm.Reduce(local_load, global_load, op=MPI.SUM, root=0)

        return global_load

    def collect_local_load(self) -> np.ndarray:
        """
        Collect this rank's load density map (without a reduction)

        Returns:
            local_load: a (GridDim_X, GridDim_Y, GridDim_Z) 3D array
        """
        grid_dim = self.domain_decomp.grid_system.grid_dim
        local_load = np.zeros(grid_dim, dtype=np.int32)

        local_objs = self.object_pool.get_local_objects()
        gs = self.domain_decomp.grid_system

        for obj in local_objs:
            x, y, z = obj['x'], obj['y'], obj['z']
            cell_id = gs.pos_to_cell_id(x, y, z)
            ix, iy, iz = gs.cell_id_to_grid_index(cell_id)
            local_load[ix, iy, iz] += 1

        return local_load

    def compute_new_bounds_distributed(self) -> List[Dict]:
        """
        Find the new bounds with a distributed RCB (replacing the centralized rank-0 version)

        Every rank takes part; the communicator is split recursively with Comm.Split.
        Each level performs one 1D Allreduce; no root has to hold the full 3D map.

        Returns:
            new_bounds_list: [Bounds.to_dict(), ...] (length = size)
        """
        # 1. Collect the local load map
        local_load = self.collect_local_load()

        # 2. Every rank holds an identical global load map, so a sub-communicator
        # cannot miss objects inside the current box whose old owner left the subgroup.
        global_load = np.zeros_like(local_load, dtype=np.int64)
        self.comm.Allreduce(local_load.astype(np.int64), global_load, op=MPI.SUM)

        # 3. Define the global bounds
        grid_dim = self.domain_decomp.grid_system.grid_dim
        global_box = Bounds(0, grid_dim[0], 0, grid_dim[1], 0, grid_dim[2])

        # 4. Run the distributed RCB
        all_ranks = list(range(self.size))
        partition = distributed_rcb(
            self.comm, global_load, global_box, all_ranks, self.verbose
        )

        # 5. Convert to a list (ordered by rank ID)
        new_bounds_list = []
        for rank_id in range(self.size):
            new_bounds_list.append(partition[rank_id].to_dict())

        return new_bounds_list

    def compute_new_partition_rcb(self, global_load_map: np.ndarray) -> List[Dict]:
        """
        (rank 0 only) Run the RCB algorithm for a new partition

        Args:
            global_load_map: the global load density map (3D numpy array)

        Returns:
            new_bounds_list: [Bounds.to_dict(), ...] (length = size)
        """
        assert self.rank == 0, "compute_new_partition_rcb should only be called by Rank 0"

        # Debug: inspect global_load_map
        total_load = np.sum(global_load_map)
        nonzero_cells = np.count_nonzero(global_load_map)

        # 1. Define the global bounds
        grid_dim = self.domain_decomp.grid_system.grid_dim
        global_box = Bounds(0, grid_dim[0], 0, grid_dim[1], 0, grid_dim[2])

        # 2. Run the RCB algorithm
        ranks = list(range(self.size))
        partition = recursive_bisection(global_load_map, ranks, global_box, self.verbose)

        # 3. Convert to a list (ordered by rank ID)
        new_bounds_list = []
        for rank_id in range(self.size):
            bounds = partition[rank_id]
            new_bounds_list.append(bounds.to_dict())

        return new_bounds_list

    def check_improvement(self, old_loads: List[int], new_bounds_list: List[Dict],
                         global_load_map: np.ndarray) -> Tuple[bool, List[int]]:
        """
        (rank 0 only) Check whether the new partition improves things enough

        Args:
            old_loads: list of the old loads per rank
            new_bounds_list: list of the new bounds
            global_load_map: the global load density map

        Returns:
            (should_migrate, new_loads)
            should_migrate: True if the improvement >= min_improvement
            new_loads: list of the new loads per rank
        """
        assert self.rank == 0

        # Load of the new partition
        new_loads = []
        for bounds_dict in new_bounds_list:
            bounds = Bounds.from_dict(bounds_dict)
            load = np.sum(global_load_map[
                bounds.x_start:bounds.x_end,
                bounds.y_start:bounds.y_end,
                bounds.z_start:bounds.z_end
            ])
            new_loads.append(int(load))

        # Imbalance before and after
        old_imbalance = compute_imbalance(old_loads)
        new_imbalance = compute_imbalance(new_loads)

        improvement = old_imbalance - new_imbalance

        should_migrate = improvement >= self.min_improvement

        return should_migrate, new_loads

    def _build_leaving_owner_map(self, old_local_cells: set) -> np.ndarray:
        """Build the map of cells leaving this rank -> target_rank."""
        grid_dim = self.domain_decomp.grid_system.grid_dim
        total_cells = int(grid_dim[0] * grid_dim[1] * grid_dim[2])
        leaving_owner_map = np.full(total_cells, -1, dtype=np.int32)

        for cell_id in old_local_cells:
            new_owner = int(self.domain_decomp.get_cell_owner(cell_id))
            if new_owner != self.rank:
                leaving_owner_map[int(cell_id)] = new_owner

        return leaving_owner_map

    def _compute_send_targets(self, objects: np.ndarray, leaving_owner_map: np.ndarray) -> np.ndarray:
        """JIT: target rank of a migrating object, or -1 when it stays."""
        if len(objects) == 0:
            return np.empty(0, dtype=np.int32)

        grid = self.domain_decomp.grid_system
        return _compute_send_targets_jit(
            np.ascontiguousarray(objects['x'], dtype=np.float64),
            np.ascontiguousarray(objects['y'], dtype=np.float64),
            np.ascontiguousarray(objects['z'], dtype=np.float64),
            np.ascontiguousarray(objects['active'], dtype=np.bool_),
            np.ascontiguousarray(objects['owner_rank'], dtype=np.int32),
            np.int32(self.rank),
            np.ascontiguousarray(leaving_owner_map, dtype=np.int32),
            np.int32(grid.grid_dim[0]), np.int32(grid.grid_dim[1]), np.int32(grid.grid_dim[2]),
            float(grid.box_size[0]), float(grid.box_size[1]), float(grid.box_size[2]),
            float(grid.cell_size[0]), float(grid.cell_size[1]), float(grid.cell_size[2]),
            bool(grid.periodic[0]), bool(grid.periodic[1]), bool(grid.periodic[2])
        )

    def _plan_migration_actions(
        self,
        objects: np.ndarray,
        leaving_owner_map: np.ndarray,
        current_region_is_boundary: bool,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """One scan yields both the send targets and the local reclassification mask."""
        if len(objects) == 0:
            return np.empty(0, dtype=np.int32), np.empty(0, dtype=np.bool_)

        grid = self.domain_decomp.grid_system
        bounds = self.domain_decomp.subdomain_bounds
        targets = np.empty(len(objects), dtype=np.int32)
        move_mask = np.empty(len(objects), dtype=np.bool_)
        size_arr = objects['size'] if 'size' in objects.dtype.names else np.zeros(len(objects), dtype=np.float64)

        _plan_migration_and_reclassify_jit(
            np.ascontiguousarray(objects['x'], dtype=np.float64),
            np.ascontiguousarray(objects['y'], dtype=np.float64),
            np.ascontiguousarray(objects['z'], dtype=np.float64),
            np.ascontiguousarray(size_arr, dtype=np.float64),
            np.ascontiguousarray(objects['active'], dtype=np.bool_),
            np.ascontiguousarray(objects['owner_rank'], dtype=np.int32),
            np.int32(self.rank),
            np.ascontiguousarray(leaving_owner_map, dtype=np.int32),
            np.int32(grid.grid_dim[0]), np.int32(grid.grid_dim[1]), np.int32(grid.grid_dim[2]),
            float(grid.box_size[0]), float(grid.box_size[1]), float(grid.box_size[2]),
            float(grid.cell_size[0]), float(grid.cell_size[1]), float(grid.cell_size[2]),
            bool(grid.periodic[0]), bool(grid.periodic[1]), bool(grid.periodic[2]),
            float(bounds['x_min']), float(bounds['x_max']),
            float(bounds['y_min']), float(bounds['y_max']),
            float(bounds['z_min']), float(bounds['z_max']),
            float(self.domain_decomp.max_interaction_distance),
            bool(current_region_is_boundary),
            targets,
            move_mask,
        )
        return targets, move_mask

    def _build_planned_send_buffers(self, leaving_owner_map: np.ndarray):
        """Build the compact send buffer from leaving_owner_map with a two-pass JIT scan."""
        pool = self.object_pool
        center_view = pool.center_objects[:pool.center_count]
        boundary_view = pool.boundary_objects[:pool.boundary_count]

        center_targets, center_move_mask = self._plan_migration_actions(
            center_view, leaving_owner_map, current_region_is_boundary=False
        )
        boundary_targets, boundary_move_mask = self._plan_migration_actions(
            boundary_view, leaving_owner_map, current_region_is_boundary=True
        )

        send_counts = np.zeros(self.size, dtype=np.int32)
        if len(center_targets) > 0:
            valid = center_targets >= 0
            if np.any(valid):
                send_counts += np.bincount(center_targets[valid], minlength=self.size).astype(np.int32)
        if len(boundary_targets) > 0:
            valid = boundary_targets >= 0
            if np.any(valid):
                send_counts += np.bincount(boundary_targets[valid], minlength=self.size).astype(np.int32)

        send_displs = np.zeros(self.size, dtype=np.int32)
        if self.size > 1:
            send_displs[1:] = np.cumsum(send_counts[:-1], dtype=np.int32)

        total_send = int(np.sum(send_counts))
        if total_send == 0:
            empty_buf = np.array([], dtype=pool.dtype)
            return (
                send_counts,
                send_displs,
                empty_buf,
                np.empty(0, dtype=np.int64),
                center_targets,
                boundary_targets,
                center_move_mask,
                boundary_move_mask,
            )

        send_buf = np.empty(total_send, dtype=pool.dtype)
        send_obj_ids = np.empty(total_send, dtype=np.int64)
        positions = send_displs.copy()

        _fill_send_buffer_jit(center_targets, center_view, positions, send_buf, send_obj_ids)
        _fill_send_buffer_jit(boundary_targets, boundary_view, positions, send_buf, send_obj_ids)
        return (
            send_counts,
            send_displs,
            send_buf,
            send_obj_ids,
            center_targets,
            boundary_targets,
            center_move_mask,
            boundary_move_mask,
        )

    def _get_current_bounds_list(self) -> List[Dict]:
        """Return the bounds list currently held by the domain decomposition."""
        bounds_list = getattr(self.domain_decomp, 'current_bounds_list', None)
        if bounds_list is None:
            raise RuntimeError("DomainDecomposition.current_bounds_list is not initialized")
        return [dict(b) for b in bounds_list]

    def _compute_sparse_migration_graph(
        self,
        old_bounds_list: List[Dict],
        new_bounds_list: List[Dict],
    ) -> Tuple[List[int], List[int]]:
        """Derive this rank's sparse send and receive neighbours from the old/new bounds."""
        my_old = old_bounds_list[self.rank]
        my_new = new_bounds_list[self.rank]

        send_targets = [
            target for target in range(self.size)
            if target != self.rank and _boxes_intersect(my_old, new_bounds_list[target])
        ]
        recv_sources = [
            source for source in range(self.size)
            if source != self.rank and _boxes_intersect(old_bounds_list[source], my_new)
        ]
        return send_targets, recv_sources

    def _post_sparse_migration_messages(
        self,
        send_targets: List[int],
        send_counts: np.ndarray,
        send_displs: np.ndarray,
        send_buf: np.ndarray,
    ):
        """Send one round of count+payload messages over the sparse communication graph."""
        requests = []
        outgoing_buffers = []

        for target in send_targets:
            count = int(send_counts[target])
            disp = int(send_displs[target])
            chunk = send_buf[disp:disp + count] if count > 0 else np.array([], dtype=self.object_pool.dtype)
            msg = _pack_migration_message(chunk, self.object_pool.dtype)
            outgoing_buffers.append(msg)
            requests.append(self.comm.Isend([msg, MPI.BYTE], dest=target, tag=TAG_OBJECT_REBALANCE))

        return requests, outgoing_buffers

    def _recv_sparse_migration_messages(
        self,
        recv_sources: List[int],
    ) -> np.ndarray:
        """Receive one round of count+payload messages from the sparse source set."""
        received_chunks = []

        for source in recv_sources:
            status = MPI.Status()
            self.comm.Probe(source=source, tag=TAG_OBJECT_REBALANCE, status=status)
            nbytes = status.Get_count(MPI.BYTE)
            msg = np.empty(nbytes, dtype=np.uint8)
            self.comm.Recv([msg, MPI.BYTE], source=source, tag=TAG_OBJECT_REBALANCE, status=status)
            chunk = _unpack_migration_message(msg, self.object_pool.dtype)
            if len(chunk) > 0:
                received_chunks.append(chunk)

        if not received_chunks:
            return np.array([], dtype=self.object_pool.dtype)
        if len(received_chunks) == 1:
            return received_chunks[0]
        return np.concatenate(received_chunks)

    def execute_migration(self, new_bounds_list: List[Dict]) -> int:
        """
        Migrate the objects

        Procedure:
        1. update the ownership_map of domain_decomp
        2. identify the cells that must leave
        3. pack the data
        4. exchange the objects with MPI_Alltoallv
        5. receive and merge
        6. clean up and reorder

        Args:
            new_bounds_list: list of the new bounds

        Returns:
            migrated_count: number of objects this rank sent away
        """
        t0 = time.perf_counter()

        # Step 1: keep the old local_cells
        old_local_cells = set(self.domain_decomp.local_cells)

        # Step 2: update ownership_map
        self.domain_decomp.update_layout_from_bounds(new_bounds_list)

        # Step 3: identify the cells that must leave
        send_plan = {}  # {target_rank: [cell_id_1, cell_id_2, ...]}

        for cell_id in old_local_cells:
            new_owner = self.domain_decomp.get_cell_owner(cell_id)
            if new_owner != self.rank:
                if new_owner not in send_plan:
                    send_plan[new_owner] = []
                send_plan[new_owner].append(cell_id)

        # Step 4: pack the data (grouped by target rank)
        send_data = {}
        migrated_count = 0

        for target_rank, cell_ids in send_plan.items():
            # Extract the objects of those cells
            objs = self.object_pool.extract_cells(cell_ids)
            send_data[target_rank] = objs
            migrated_count += len(objs)

        # Step 5: prepare the Alltoallv arguments
        send_counts = np.zeros(self.size, dtype=np.int32)
        for target_rank, objs in send_data.items():
            send_counts[target_rank] = len(objs)

        # Exchange the counts
        recv_counts = np.zeros(self.size, dtype=np.int32)
        self.comm.Alltoall(send_counts, recv_counts)

        # [fix] Prepare the send and receive buffers (bulk allocation)
        # Problem: a Python list plus extend() and tolist() tripled or quadrupled the peak memory
        # Fix: compute the total size up front, allocate NumPy arrays and assign by slice

        send_displs = np.zeros(self.size, dtype=np.int32)
        total_send = np.sum(send_counts)

        # Allocate the whole send buffer directly (no intermediate Python list)
        if total_send > 0:
            send_buf = np.empty(total_send, dtype=self.object_pool.dtype)
            offset = 0
            for i in range(self.size):
                send_displs[i] = offset
                if i in send_data:
                    count = len(send_data[i])
                    # Direct NumPy slice assignment (no copy)
                    send_buf[offset:offset+count] = send_data[i]
                    offset += count
        else:
            send_buf = np.array([], dtype=self.object_pool.dtype)

        recv_displs = np.zeros(self.size, dtype=np.int32)
        for i in range(1, self.size):
            recv_displs[i] = recv_displs[i-1] + recv_counts[i-1]

        total_recv = np.sum(recv_counts)

        recv_buf = np.empty(total_recv, dtype=self.object_pool.dtype)

        # [Bug 1 fix] Step 6: exchange the objects with Alltoallv
        # Problem: send_counts and recv_counts hold object counts, but MPI.BYTE needs bytes
        # Fix: multiply counts and displs by the object size (itemsize)
        itemsize = self.object_pool.dtype.itemsize

        send_counts_bytes = send_counts * itemsize
        send_displs_bytes = send_displs * itemsize
        recv_counts_bytes = recv_counts * itemsize
        recv_displs_bytes = recv_displs * itemsize

        self.comm.Alltoallv(
            [send_buf, send_counts_bytes, send_displs_bytes, MPI.BYTE],
            [recv_buf, recv_counts_bytes, recv_displs_bytes, MPI.BYTE]
        )

        # Step 7: receive and merge
        # [fix] Set owner_rank to this rank (fixes the loss of migrated objects)
        for obj in recv_buf:
            obj['owner_rank'] = self.rank  # after migration the object belongs to this rank
            self.object_pool.add_object(obj)

        # Step 8: clean up and reorder
        self.object_pool.sort_and_sync()

        # [fix] Step 9: refresh owner_rank of the large objects
        # Problem: large objects do not migrate, but the domain bounds moved and owner_rank went stale
        # Fix: every rank recomputes owner_rank of the large objects from the new bounds
        large_updated = self.object_pool.update_large_objects_ownership()

        # Statistics
        elapsed = time.perf_counter() - t0
        self.stats['total_rebalances'] += 1
        self.stats['total_objects_migrated'] += migrated_count
        self.stats['total_time'] += elapsed

        if self.verbose:
            print(f"[LoadBalancer] Rank {self.rank}: migrated_out={migrated_count}, "
                  f"migrated_in={total_recv}, large_updated={large_updated}, elapsed={elapsed:.3f}s")

        return migrated_count

    def execute_migration_planned(self, new_bounds_list: List[Dict]) -> int:
        """
        Plan first, then migrate, with direct classification (replaces execute_migration)

        Improvements:
        1. the new owner follows from the object position (not by extracting cells)
        2. the receiver classifies straight into center/boundary (skipping the large test in add_object)
        3. incremental_reclassify replaces the full sort_and_sync

        Args:
            new_bounds_list: list of the new bounds

        Returns:
            migrated_count: number of objects this rank sent away
        """
        t0 = time.perf_counter()

        # Step 1: keep the old local_cells
        old_local_cells = set(self.domain_decomp.local_cells)
        old_bounds_list = self._get_current_bounds_list()

        # Step 2: update ownership_map
        self.domain_decomp.update_layout_from_bounds(new_bounds_list)
        new_local_cells = set(self.domain_decomp.local_cells)

        # Step 3: build the send plan from the leaving-cell map with a two-pass JIT scan
        pool = self.object_pool
        leaving_owner_map = self._build_leaving_owner_map(old_local_cells)
        (
            send_counts,
            send_displs,
            send_buf,
            send_obj_ids,
            center_targets,
            boundary_targets,
            center_move_mask,
            boundary_move_mask,
        ) = self._build_planned_send_buffers(
            leaving_owner_map
        )
        migrated_count = len(send_obj_ids)

        # Step 5: build the sparse communication graph from the old/new bounds and send one round
        send_targets, recv_sources = self._compute_sparse_migration_graph(old_bounds_list, new_bounds_list)
        send_requests, _outgoing_buffers = self._post_sparse_migration_messages(
            send_targets, send_counts, send_displs, send_buf
        )
        recv_buf = self._recv_sparse_migration_messages(recv_sources)
        if send_requests:
            MPI.Request.Waitall(send_requests)

        total_recv = len(recv_buf)

        # Step 6: apply migration and local reclassification in one transaction, replacing delete + batch_insert + incremental_reclassify
        recv_center, recv_boundary, reclassified = pool.apply_migration_transaction(
            center_targets,
            boundary_targets,
            center_move_mask,
            boundary_move_mask,
            recv_buf,
            self.rank,
        )

        # Step 8: refresh the large object owners
        large_updated = pool.update_large_objects_ownership()

        # Statistics
        elapsed = time.perf_counter() - t0
        self.stats['total_rebalances'] += 1
        self.stats['total_objects_migrated'] += migrated_count
        self.stats['total_time'] += elapsed

        if self.verbose:
            print(f"[LoadBalancer] Rank {self.rank}: migrated_out={migrated_count}, "
                  f"migrated_in={total_recv} (center={recv_center}, boundary={recv_boundary}), "
                  f"reclassified={reclassified}, "
                  f"large_updated={large_updated}, elapsed={elapsed:.3f}s")

        return migrated_count

    def rebalance(self) -> bool:
        """
        Run the full redistribution (distributed RCB + planned migration)

        Returns:
            True if a redistribution took place
        """
        t_start = time.perf_counter()

        # Step 1: collect the current load (for the improvement measure)
        local_count = self.object_pool.get_local_count()
        all_counts = self.comm.allgather(local_count)
        old_imbalance = compute_imbalance(all_counts)

        # Step 2: find the new bounds with a distributed RCB (every rank takes part)
        new_bounds_list = self.compute_new_bounds_distributed()

        # Step 3: every rank computes the new load and checks the improvement identically
        # Collect the local load map to score the new partition
        local_load = self.collect_local_load()
        # Allreduce gives the global load map (for scoring; smaller than the objects themselves)
        grid_dim = self.domain_decomp.grid_system.grid_dim
        global_load_map = np.zeros(grid_dim, dtype=np.int64)
        self.comm.Allreduce(local_load.astype(np.int64), global_load_map, op=MPI.SUM)

        new_loads = []
        for bounds_dict in new_bounds_list:
            b = Bounds.from_dict(bounds_dict)
            load = int(np.sum(global_load_map[
                b.x_start:b.x_end, b.y_start:b.y_end, b.z_start:b.z_end
            ]))
            new_loads.append(load)

        new_imbalance = compute_imbalance(new_loads)
        improvement = old_imbalance - new_imbalance
        should_migrate = improvement >= self.min_improvement

        if not should_migrate:
            return False

        # Step 4: run the planned migration
        self.execute_migration_planned(new_bounds_list)

        # Measure and report the elapsed time
        elapsed = time.perf_counter() - t_start
        max_elapsed = self.comm.allreduce(elapsed, op=MPI.MAX)
        if self.rank == 0:
            print(f"[Object LB] imbalance {old_imbalance:.2%}->{new_imbalance:.2%}, "
                  f"local objects={local_count}, CPU time={max_elapsed:.3f}s")

        return True

    def rebalance_legacy(self) -> bool:
        """
        [Deprecated] The old redistribution path (centralized RCB + sort_and_sync)
        Kept for regression comparison.
        """
        t_start = time.perf_counter()

        # Step 1: collect the global load
        global_load_map = self.collect_global_load()

        # Collect the current load (for the improvement measure)
        local_count = self.object_pool.get_local_count()
        old_loads = self.comm.gather(local_count, root=0)

        # Step 2: rank 0 decides
        if self.rank == 0:
            old_imbalance = compute_imbalance(old_loads)
            new_bounds_list = self.compute_new_partition_rcb(global_load_map)
            should_migrate, new_loads = self.check_improvement(
                old_loads, new_bounds_list, global_load_map
            )
            new_imbalance = compute_imbalance(new_loads) if should_migrate else old_imbalance
        else:
            should_migrate = None
            new_bounds_list = None
            old_imbalance = 0.0
            new_imbalance = 0.0

        # Step 3: broadcast the decision
        should_migrate = self.comm.bcast(should_migrate, root=0)
        old_imbalance = self.comm.bcast(old_imbalance, root=0)
        new_imbalance = self.comm.bcast(new_imbalance, root=0)

        if not should_migrate:
            return False

        # Step 4: broadcast the new bounds
        new_bounds_list = self.comm.bcast(new_bounds_list, root=0)

        # Step 5: migrate (the old way, including sort_and_sync)
        self.execute_migration(new_bounds_list)

        elapsed = time.perf_counter() - t_start
        max_elapsed = self.comm.allreduce(elapsed, op=MPI.MAX)
        if self.rank == 0:
            print(f"[Object LB Legacy] imbalance {old_imbalance:.2%}->{new_imbalance:.2%}, "
                  f"local objects={local_count}, CPU time={max_elapsed:.3f}s")

        return True

    def print_summary(self):
        """Print the statistics summary"""
        if self.rank == 0:
            print(f"\n[LoadBalancer] Summary:")
            print(f"  Total rebalances: {self.stats['total_rebalances']}")
            print(f"  Total objects migrated: {self.stats['total_objects_migrated']}")
            print(f"  Total time: {self.stats['total_time']:.3f}s")
            if self.stats['total_rebalances'] > 0:
                avg_time = self.stats['total_time'] / self.stats['total_rebalances']
                print(f"  Avg time per rebalance: {avg_time:.3f}s")


def create_object_load_balancer(comm: MPI.Comm, domain_decomp, object_pool,
                                config: Dict) -> ObjectLoadBalancer:
    """
    Factory: create an object load balancer

    Args:
        comm: MPI communicator
        domain_decomp: DomainDecomposition instance
        object_pool: RMAObjectPool instance
        config: configuration dict

    Returns:
        an ObjectLoadBalancer instance
    """
    lb_config = {
        'enabled': config.get('object_load_balancer', {}).get('enabled', False),
        'balance_interval': config.get('object_load_balancer', {}).get('balance_interval', 500),
        'imbalance_threshold': config.get('object_load_balancer', {}).get('imbalance_threshold', 0.2),
        'min_improvement': config.get('object_load_balancer', {}).get('min_improvement', 0.05),
        'verbose': config.get('object_load_balancer', {}).get('verbose', False),
    }

    return ObjectLoadBalancer(comm, domain_decomp, object_pool, lb_config)
