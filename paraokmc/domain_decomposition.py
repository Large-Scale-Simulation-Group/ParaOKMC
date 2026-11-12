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
Domain decomposition module (fully automatic version)

Key features:
1. [Auto-Topology] picks the optimal nx/ny/nz so subdomains are nearly cubic.
2. [Auto-Grid] derives grid_dim and forces it to a multiple of nx/ny/nz.
3. [Flat-Boundary] a Cartesian split algorithm keeps rank boundaries perfectly flat.

Usage:
    config = {
        'box_size': [100, 100, 100],
        'max_interaction_distance': 3.0,
        # nx, ny, nz and grid_dim are all derived automatically
    }
    dd = create_domain_decomposition(comm, config)
"""

import numpy as np
from mpi4py import MPI
from typing import Tuple, List, Dict, Set, Optional
import math

from numba import njit, prange

try:
    from .grid_system import GridSystem
    GRID_SYSTEM_AVAILABLE = True
except ImportError:
    GRID_SYSTEM_AVAILABLE = False


# ============================================================================
# [performance fix] JIT parallel kernels for the standard ghost region
#
# The old implementation looped over cells in pure Python: on a fine grid (one cell
# per lattice site, ~1e7 cells) initialization stalled for minutes with no output.
# Both kernels are OMP parallel (prange); NUMBA_NUM_THREADS sets the thread count.
# ============================================================================

@njit(cache=True, parallel=True)
def _jit_local_bbox(ownership_map, rank, gx, gxgy):
    """Scan ownership_map and return the index bounding box of this rank's cells.

    Returns: (count, x_min, x_max, y_min, y_max, z_min, z_max)
    The bounding box is invalid when count == 0.
    """
    big = np.int64(1) << 60
    x_min = big
    y_min = big
    z_min = big
    x_max = np.int64(-1)
    y_max = np.int64(-1)
    z_max = np.int64(-1)
    count = np.int64(0)

    for i in prange(ownership_map.shape[0]):
        if ownership_map[i] == rank:
            count += 1
            iz = i // gxgy
            rem = i % gxgy
            iy = rem // gx
            ix = rem % gx
            x_min = min(x_min, ix)
            x_max = max(x_max, ix)
            y_min = min(y_min, iy)
            y_max = max(y_max, iy)
            z_min = min(z_min, iz)
            z_max = max(z_max, iz)

    return count, x_min, x_max, y_min, y_max, z_min, z_max


@njit(cache=True, parallel=True)
def _jit_fill_slab(out_ids, out_owners, ownership_map, base,
                   x0, x1, y0, y1, z0, z1, gx, gxgy):
    """Generate the ids and owners of every cell in a rectangular face region, in parallel.

    The region [x0,x1) x [y0,y1) x [z0,z1) is flattened in (x, y, z) lexicographic
    order and written into out_ids/out_owners over [base, base+n).
    """
    ny = y1 - y0
    nz = z1 - z0
    n = (x1 - x0) * ny * nz
    for k in prange(n):
        x = x0 + k // (ny * nz)
        r = k % (ny * nz)
        y = y0 + r // nz
        z = z0 + r % nz
        cid = x + y * gx + z * gxgy
        out_ids[base + k] = cid
        out_owners[base + k] = ownership_map[cid]


def _min_bounds_distance(b1: Dict[str, float], b2: Dict[str, float],
                         box_size, periodic) -> float:
    """Minimum distance between two physical bounding boxes (wrapped along periodic axes)."""
    def axis_dist(min1, max1, min2, max2, L, p):
        if min2 > max1:
            direct = min2 - max1
        elif min1 > max2:
            direct = min1 - max2
        else:
            direct = 0.0
        if p:
            wrap = L - (max(max1, max2) - min(min1, min2))
            return max(0.0, min(direct, wrap))
        return direct

    dx = axis_dist(b1['x_min'], b1['x_max'], b2['x_min'], b2['x_max'],
                   float(box_size[0]), bool(periodic[0]))
    dy = axis_dist(b1['y_min'], b1['y_max'], b2['y_min'], b2['y_max'],
                   float(box_size[1]), bool(periodic[1]))
    dz = axis_dist(b1['z_min'], b1['z_max'], b2['z_min'], b2['z_max'],
                   float(box_size[2]), bool(periodic[2]))
    return float(np.sqrt(dx * dx + dy * dy + dz * dz))


def compute_ghost_ranks_from_bounds(rank: int,
                                    physical_bounds_list: List[Dict[str, float]],
                                    box_size, periodic, r_cut: float) -> List[int]:
    """[fix] Derive the ghost adjacency from the minimum distance between rank regions.

    Runs a periodic-aware minimum distance test over the bounding boxes of all ranks
    (< r_cut means neighbour). This fixes two problems at once:
    1. the old implementation hardcoded [-1, 0, 1]^3 and searched a single shell, so a
       subdomain narrower than r_cut in any direction lost legitimate neighbours beyond
       that shell (the _update_interaction_distance limit the comment claimed never existed);
    2. after an RCB repartition the topology changes, and the old implementation relied on
       the initial regular grid; this function uses only the current bounding boxes.
    """
    ghost_ranks = []
    my_bounds = physical_bounds_list[rank]
    for r, bounds in enumerate(physical_bounds_list):
        if r == rank:
            continue
        if _min_bounds_distance(my_bounds, bounds, box_size, periodic) < r_cut:
            ghost_ranks.append(r)
    return sorted(ghost_ranks)


class DomainDecomposition:
    """
    Domain decomposition manager (lattice-based configuration)

    Configuration parameters:
        lattice_constant: lattice constant (A)
        lattice_grid: (Nx, Ny, Nz) number of lattice sites
        h_factor: subdivision factor (each lattice site becomes h^3 cells), default 1
        periodic: [bool, bool, bool] periodic boundary conditions
        max_interaction_distance: maximum interaction distance (optional, default cell_size * 2)

    Derived automatically:
        box_size = lattice_constant * lattice_grid
        cell_size = lattice_constant / h_factor
        grid_dim = lattice_grid * h_factor
    """

    def __init__(self, comm: MPI.Comm, config: dict, verbose: bool = False):
        self.comm = comm
        self.rank = comm.rank
        self.size = comm.size
        self.verbose = verbose

        # 1. Basic physical parameters (lattice-based configuration)
        if 'lattice_constant' not in config or 'lattice_grid' not in config:
            raise ValueError("Config must contain 'lattice_constant' and 'lattice_grid'")

        lc = config['lattice_constant']
        lg = config['lattice_grid']  # (Nx, Ny, Nz)
        h = config.get('h_factor', 1)  # subdivision factor, default 1 (no subdivision)

        self.lattice_constant = lc
        self.lattice_grid = tuple(lg)
        self.h_factor = h

        # Derive box_size and grid_dim
        self.box_size = np.array([lc * lg[i] for i in range(3)], dtype=np.float64)
        self.grid_dim = tuple(lg[i] * h for i in range(3))

        # cell_size and max_interaction_distance
        cell_size = lc / h
        self.max_interaction_distance = config.get('max_interaction_distance', cell_size * 2.0)

        self.periodic = config.get('periodic', [True, True, True])

        # v1.6.5: the fine-grid mode is always on
        self.use_fine_mesh = True

        # 2. Compute the process split (from the fixed grid_dim)
        result = self._setup_process_topology()
        self.Px, self.Py, self.Pz = result['Px'], result['Py'], result['Pz']
        self.nx, self.ny, self.nz = self.grid_dim

        # 4. Compute the subdomain parameters (from the process split Px, Py, Pz)
        self.subdomain_size = self.box_size / np.array([self.Px, self.Py, self.Pz])
        self.subdomain_idx = self._rank_to_subdomain(self.rank)
        self.subdomain_bounds = self._compute_subdomain_bounds(self.subdomain_idx)

        self.ghost_ranks = self._compute_ghost_ranks()

        # 5. Initialize the fine grid system
        self.grid_system = None
        self.ownership_map = None
        self.local_cells = None
        self.current_bounds_list = None

        self._initialize_fine_mesh()

        # 6. [performance] Precompute the standard ghost cells (assuming small objects)
        # This avoids recomputing them at a cost of 3 s on every step
        self.standard_ghost_cells = self._compute_standard_ghost_cells()

        if self.verbose and self.rank == 0:
            self._print_info()

    # ===== Process topology =====

    def _setup_process_topology(self) -> Dict:
        """
        Compute the process split from the fixed grid_dim

        Goals:
        1. Px * Py * Pz = size
        2. every rank's subdomain is as close to a cube as possible

        Returns:
            {'Px': Px, 'Py': Py, 'Pz': Pz}
        """
        Lx, Ly, Lz = self.box_size
        Px, Py, Pz = self._choose_dims_cubic(self.size, Lx, Ly, Lz)

        if self.verbose and self.rank == 0:
            gx, gy, gz = self.grid_dim
            cell_size = [self.box_size[i] / self.grid_dim[i] for i in range(3)]
            print(f"[Domain decomposition]")
            print(f"  lattice constant: {self.lattice_constant:.3f} A")
            print(f"  lattice sites: {self.lattice_grid[0]} x {self.lattice_grid[1]} x {self.lattice_grid[2]}")
            print(f"  subdivision factor h: {self.h_factor}")
            print(f"  physical box: {Lx:.1f} x {Ly:.1f} x {Lz:.1f} A")
            print(f"  cell grid: {gx} x {gy} x {gz} ({gx*gy*gz} cells in total)")
            print(f"  cell size: {cell_size[0]:.3f} x {cell_size[1]:.3f} x {cell_size[2]:.3f} A")
            print(f"  interaction distance: {self.max_interaction_distance:.2f} A")
            print(f"  process split: {Px} x {Py} x {Pz} = {Px*Py*Pz}")
            print(f"  per rank: {gx/Px:.1f} x {gy/Py:.1f} x {gz/Pz:.1f} cells")
            print(f"  subdomain size: {Lx/Px:.1f} x {Ly/Py:.1f} x {Lz/Pz:.1f} A")

        return {'Px': Px, 'Py': Py, 'Pz': Pz}

    def _factorize(self, N: int) -> List[int]:
        """Prime factorization of N (descending)"""
        factors = []
        x = N

        # Handle the factor 2
        while x % 2 == 0:
            factors.append(2)
            x //= 2

        # Handle the odd factors
        p = 3
        while p * p <= x:
            while x % p == 0:
                factors.append(p)
                x //= p
            p += 2

        if x > 1:
            factors.append(x)  # the remaining prime

        factors.sort(reverse=True)  # descending
        return factors

    def _anisotropy(self, lx: float, ly: float, lz: float) -> float:
        """
        Cost function measuring how cubic a split is

        The closer to a cube, the smaller the value
        """
        m = (lx + ly + lz) / 3.0
        if m < 1e-12:
            return 1e300
        ax = lx / m - 1.0
        ay = ly / m - 1.0
        az = lz / m - 1.0
        return ax*ax + ay*ay + az*az

    def _choose_dims_cubic(self, N: int, Lx: float, Ly: float, Lz: float) -> Tuple[int, int, int]:
        """
        Choose the process split (Px, Py, Pz) that keeps subdomains cubic

        Greedy algorithm: hand the prime factors of N, largest first, to the three
        dimensions, each time picking the dimension that keeps subdomains most cubic
        """
        factors = self._factorize(N)

        Px, Py, Pz = 1, 1, 1

        for f in factors:
            # Current subdomain size
            lx = Lx / Px
            ly = Ly / Py
            lz = Lz / Pz

            # Try all three directions
            cost_x = self._anisotropy(Lx / (Px * f), ly, lz)
            cost_y = self._anisotropy(lx, Ly / (Py * f), lz)
            cost_z = self._anisotropy(lx, ly, Lz / (Pz * f))

            # Pick the direction with the lowest cost
            min_cost = min(cost_x, cost_y, cost_z)
            if min_cost == cost_x:
                Px *= f
            elif min_cost == cost_y:
                Py *= f
            else:
                Pz *= f

        return (Px, Py, Pz)

    # ===== Grid system initialization =====

    def _initialize_fine_mesh(self):
        """Initialize the fine grid system and the ownership map"""
        if not GRID_SYSTEM_AVAILABLE:
            raise ImportError("GridSystem not found.")

        self.grid_system = GridSystem(
            grid_dim=self.grid_dim,
            box_size=self.box_size.tolist(),
            periodic=self.periodic
        )

        # Use the Cartesian split
        self._initialize_ownership_map_cartesian()
        self._update_local_cells()

    def _initialize_ownership_map_cartesian(self):
        """
        3D Cartesian grid split (nearest-assignment strategy)

        Strategy:
        - base share: every process gets gx // Px cells
        - remainder: the last process absorbs the leftover cells (keeping boundaries flat)
        - benefits: simple, robust, flat boundaries, easy ghost region computation
        """
        gx, gy, gz = self.grid_dim
        nx, ny, nz = self.nx, self.ny, self.nz

        self.ownership_map = np.empty(self.grid_system.total_cells, dtype=np.int32)
        self.ownership_map.fill(-1)

        # Base number of cells per rank along each dimension (from Px, Py, Pz)
        cells_per_rank_x = gx // self.Px
        cells_per_rank_y = gy // self.Py
        cells_per_rank_z = gz // self.Pz

        bounds_list = []
        for r in range(self.size):
            rx, ry, rz = self._rank_to_subdomain(r)

            # Grid index range of this rank (nearest assignment: the last rank takes the rest)
            x_start = rx * cells_per_rank_x
            if rx == self.Px - 1:  # the last process
                x_end = gx  # extend to the end, absorbing the leftover cells
            else:
                x_end = (rx + 1) * cells_per_rank_x

            y_start = ry * cells_per_rank_y
            if ry == self.Py - 1:
                y_end = gy
            else:
                y_end = (ry + 1) * cells_per_rank_y

            z_start = rz * cells_per_rank_z
            if rz == self.Pz - 1:
                z_end = gz
            else:
                z_end = (rz + 1) * cells_per_rank_z

            bounds_list.append({
                'x_start': int(x_start),
                'x_end': int(x_end),
                'y_start': int(y_start),
                'y_end': int(y_end),
                'z_start': int(z_start),
                'z_end': int(z_end),
            })

            # Fill the map (bulk assignment through slicing)
            for iz in range(z_start, z_end):
                z_offset = iz * gx * gy
                for iy in range(y_start, y_end):
                    y_offset = iy * gx

                    start_cell_id = z_offset + y_offset + x_start
                    end_cell_id = z_offset + y_offset + x_end

                    self.ownership_map[start_cell_id : end_cell_id] = r

        if np.any(self.ownership_map == -1):
            raise RuntimeError("Domain Decomposition failed: some cells unassigned!")

        self.current_bounds_list = bounds_list
        self.subdomain_bounds = self._grid_bounds_to_physical_bounds(bounds_list[self.rank])

    # ===== Helper methods =====

    def _rank_to_subdomain(self, rank: int) -> Tuple[int, int, int]:
        """Map a rank to its subdomain indices (ix, iy, iz) (using Px, Py, Pz)"""
        iz = rank // (self.Px * self.Py)
        iy = (rank % (self.Px * self.Py)) // self.Px
        ix = rank % self.Px
        return (ix, iy, iz)

    def _subdomain_to_rank(self, ix: int, iy: int, iz: int) -> int:
        """Map subdomain indices to a rank, handling periodic boundaries (using Px, Py, Pz)"""
        if self.periodic[0]:
            ix = ix % self.Px
        elif ix < 0 or ix >= self.Px:
            return -1

        if self.periodic[1]:
            iy = iy % self.Py
        elif iy < 0 or iy >= self.Py:
            return -1

        if self.periodic[2]:
            iz = iz % self.Pz
        elif iz < 0 or iz >= self.Pz:
            return -1

        return ix + iy * self.Px + iz * self.Px * self.Py

    def _compute_subdomain_bounds(self, subdomain_idx: Tuple[int, int, int]) -> Dict[str, float]:
        """Compute the subdomain bounds"""
        ix, iy, iz = subdomain_idx
        dx, dy, dz = self.subdomain_size
        return {
            'x_min': ix * dx, 'x_max': (ix + 1) * dx,
            'y_min': iy * dy, 'y_max': (iy + 1) * dy,
            'z_min': iz * dz, 'z_max': (iz + 1) * dz,
        }

    def _grid_bounds_to_physical_bounds(self, bounds: Dict[str, int]) -> Dict[str, float]:
        """Convert grid index bounds into physical coordinate bounds."""
        dx, dy, dz = self.grid_system.cell_size
        return {
            'x_min': float(bounds['x_start']) * dx,
            'x_max': float(bounds['x_end']) * dx,
            'y_min': float(bounds['y_start']) * dy,
            'y_max': float(bounds['y_end']) * dy,
            'z_min': float(bounds['z_start']) * dz,
            'z_max': float(bounds['z_end']) * dz,
        }

    def _rank_physical_bounds_list(self) -> List[Dict[str, float]]:
        """Physical bounding boxes of all ranks.

        After an RCB repartition current_bounds_list is used (grid index bounds -> physical);
        otherwise the boxes follow from the initial regular split of the subdomain grid.
        """
        bounds_list = getattr(self, 'current_bounds_list', None)
        if bounds_list:
            return [self._grid_bounds_to_physical_bounds(b) for b in bounds_list]
        return [self._compute_subdomain_bounds(self._rank_to_subdomain(r))
                for r in range(self.size)]

    def _compute_ghost_ranks(self) -> List[int]:
        """
        Compute the list of ghost ranks (coarse, rank level)

        [fix] Now based on the periodic minimum distance between rank regions:
        - the old implementation hardcoded [-1, 0, 1]^3 and searched a single shell, so a
          subdomain narrower than r_cut lost legitimate neighbours further out (the
          _update_interaction_distance limit the comment claimed was never implemented);
        - after an RCB repartition the current bounds must be used (see
          update_layout_from_bounds / invalidate_ghost_cache), which the old code could not do.
        """
        return compute_ghost_ranks_from_bounds(
            self.rank, self._rank_physical_bounds_list(),
            self.box_size, self.periodic, self.max_interaction_distance)

    def _compute_subdomain_distance(self, idx1, idx2) -> float:
        """Minimum distance between two subdomains"""
        bounds1 = self._compute_subdomain_bounds(idx1)
        bounds2 = self._compute_subdomain_bounds(idx2)

        def axis_dist(min1, max1, min2, max2, L, p):
            if p:  # periodic boundary
                direct = max(0, min2 - max1) if min2 > max1 else max(0, min1 - max2)
                wrap = L - (max(max1, max2) - min(min1, min2))
                return min(direct, wrap)
            else:  # non-periodic boundary
                return max(0, min2 - max1) if min2 > max1 else max(0, min1 - max2)

        dx = axis_dist(bounds1['x_min'], bounds1['x_max'], bounds2['x_min'], bounds2['x_max'],
                      self.box_size[0], self.periodic[0])
        dy = axis_dist(bounds1['y_min'], bounds1['y_max'], bounds2['y_min'], bounds2['y_max'],
                      self.box_size[1], self.periodic[1])
        dz = axis_dist(bounds1['z_min'], bounds1['z_max'], bounds2['z_min'], bounds2['z_max'],
                      self.box_size[2], self.periodic[2])

        return np.sqrt(dx**2 + dy**2 + dz**2)

    def _update_local_cells(self):
        """Update the set of local cells"""
        if self.ownership_map is None:
            return
        self.local_cells = set(np.where(self.ownership_map == self.rank)[0])

    # ===== Public interface =====

    def get_cell_owner(self, cell_id: int) -> int:
        """Look up the owner of a cell"""
        return int(self.ownership_map[cell_id])

    def _compute_standard_ghost_cells(self) -> Dict[int, List[int]]:
        """
        Precompute the standard ghost cells (assuming every object is small)

        [optimization] Directly produces the ghost cell list grouped by rank

        Principle:
        - each rank owns a rectangular block (guaranteed by the RCB algorithm)
        - the ghost zone is the region within r_cut of the boundary
        - all required ghost cells are computed at once and grouped by rank
        - no per-local-cell computation (avoiding the O(n_local * n_ghost) cost)

        Returns:
            {rank: [ghost_cell_ids]} - the standard ghost cells grouped by rank
        """
        import time

        t0 = time.perf_counter()

        # 1. Rectangular bounds of this rank (range of grid indices)
        gx, gy, gz = self.grid_system.grid_dim
        cell_size = self.grid_system.cell_size

        # Number of cell layers in the ghost zone (rounded up)
        r_cut = self.max_interaction_distance
        ghost_layers_x = int(np.ceil(r_cut / cell_size[0]))
        ghost_layers_y = int(np.ceil(r_cut / cell_size[1]))
        ghost_layers_z = int(np.ceil(r_cut / cell_size[2]))

        # 2. Find the rectangular bounds of this rank
        # [performance fix] A JIT parallel kernel scans ownership_map for the box.
        # The old implementation called the Python-level cell_id_to_grid_index() once per
        # local cell, stalling initialization for minutes on a fine grid (~1e7 cells).
        # local_cells always comes from ownership_map == rank (_update_local_cells), so
        # using ownership_map as the source is equivalent to the old implementation.
        gxgy = gx * gy
        n_local, my_x_min, my_x_max, my_y_min, my_y_max, my_z_min, my_z_max = \
            _jit_local_bbox(self.ownership_map, self.rank, gx, gxgy)
        if n_local == 0:
            return {}
        my_x_min = int(my_x_min); my_x_max = int(my_x_max)
        my_y_min = int(my_y_min); my_y_max = int(my_y_max)
        my_z_min = int(my_z_min); my_z_max = int(my_z_max)

        # Extend to the region including the ghost zone
        ext_x_min = max(0, my_x_min - ghost_layers_x)
        ext_x_max = min(gx, my_x_max + ghost_layers_x + 1)
        ext_y_min = max(0, my_y_min - ghost_layers_y)
        ext_y_max = min(gy, my_y_max + ghost_layers_y + 1)
        ext_z_min = max(0, my_z_min - ghost_layers_z)
        ext_z_max = min(gz, my_z_max + ghost_layers_z + 1)

        # 3. Find every candidate ghost cell (inside the extended region, outside this rank)
        # [performance fix] The six faces are generated in bulk by a JIT parallel
        # kernel and grouped by rank in NumPy. The enumeration matches the old code
        # exactly and the regions do not overlap: the two X faces cover the full extended
        # cross-section; the Y faces are limited to the local x range; the Z faces to x and y.
        slabs = [
            # The two faces along X
            (ext_x_min, my_x_min, ext_y_min, ext_y_max, ext_z_min, ext_z_max),
            (my_x_max + 1, ext_x_max, ext_y_min, ext_y_max, ext_z_min, ext_z_max),
            # The two faces along Y (excluding the X boundary already covered)
            (my_x_min, my_x_max + 1, ext_y_min, my_y_min, ext_z_min, ext_z_max),
            (my_x_min, my_x_max + 1, my_y_max + 1, ext_y_max, ext_z_min, ext_z_max),
            # The two faces along Z (excluding the X and Y boundaries already covered)
            (my_x_min, my_x_max + 1, my_y_min, my_y_max + 1, ext_z_min, my_z_min),
            (my_x_min, my_x_max + 1, my_y_min, my_y_max + 1, my_z_max + 1, ext_z_max),
        ]
        slabs = [s for s in slabs if s[1] > s[0] and s[3] > s[2] and s[5] > s[4]]
        sizes = [(s[1] - s[0]) * (s[3] - s[2]) * (s[5] - s[4]) for s in slabs]
        total = int(sum(sizes))

        ghost_cells_by_rank = {}
        if total > 0:
            candidate_ids = np.empty(total, dtype=np.int64)
            candidate_owners = np.empty(total, dtype=np.int64)
            base = 0
            for s, n in zip(slabs, sizes):
                _jit_fill_slab(candidate_ids, candidate_owners, self.ownership_map,
                               base, s[0], s[1], s[2], s[3], s[4], s[5], gx, gxgy)
                base += int(n)

            mask = candidate_owners != self.rank
            candidate_ids = candidate_ids[mask]
            candidate_owners = candidate_owners[mask]
            for owner in np.unique(candidate_owners):
                ghost_cells_by_rank[int(owner)] = \
                    candidate_ids[candidate_owners == owner].tolist()

        elapsed = time.perf_counter() - t0
        if self.verbose:
            total_ghost = sum(len(ghosts) for ghosts in ghost_cells_by_rank.values())
            print(f"[Rank {self.rank}] Precomputed standard ghost cells: {total_ghost} ghost cells for {len(self.local_cells)} local cells in {elapsed:.3f}s")

        return ghost_cells_by_rank

    def invalidate_ghost_cache(self):
        """
        Invalidate and recompute the ghost cell cache

        This must be called whenever max_interaction_distance grows, otherwise
        standard_ghost_cells keeps using the old interaction distance and may
        miss distant neighbour cells.

        [fix] ghost_ranks is recomputed as well: both a change of r_cut and an RCB
        repartition alter the rank adjacency, so refreshing only the cell cache
        """
        if self.verbose and self.rank == 0:
            print(f"[DomainDecomp] Invalidating ghost cache, recomputing with max_dist={self.max_interaction_distance}")

        self.ghost_ranks = self._compute_ghost_ranks()
        self.standard_ghost_cells = self._compute_standard_ghost_cells()

    def get_ghost_cells_v164(self, rma_object_pool=None) -> Dict[int, List[int]]:
        """
        Get the ghost cells (adaptive, optimized version)

        Strategy:
        1. in most cases return the precomputed standard_ghost_cells (an O(1) lookup)
        2. if a cell holds a large object (>= 0.5 of the shortest cell axis), recompute all ghost cells

        This avoids spending 3 s per step recomputing every ghost cell.

        standard_ghost_cells now returns {rank: [ghost_cell_ids]} directly

        Args:
            rma_object_pool: object pool (to inspect object sizes); None returns the standard result

        Returns:
            {rank: [ghost_cell_ids]} - ghost cells grouped by rank
        """
        # Fast path: without an object pool return the standard result
        if rma_object_pool is None:
            return self.standard_ghost_cells

        # Threshold on the shortest cell axis (half of it)
        min_cell_size = min(self.grid_system.cell_size)
        large_object_threshold = 0.5 * min_cell_size

        # Look for large objects
        has_large_objects = False
        for cell_id in self.local_cells:
            if self._has_large_objects_in_cell(cell_id, rma_object_pool, large_object_threshold):
                has_large_objects = True
                break

        # Without large objects return the standard result
        if not has_large_objects:
            return self.standard_ghost_cells

        # There is a large object, so every ghost cell must be recomputed
        # using grid_system.get_neighbor_cells()
        adaptive_ghost_map = {}

        for cell_id in self.local_cells:
            neighbor_ids = self.grid_system.get_neighbor_cells(
                cell_id,
                max_distance=self.max_interaction_distance,
                include_self=False
            )
            ghost_ids = [nid for nid in neighbor_ids if self.ownership_map[nid] != self.rank]
            adaptive_ghost_map[cell_id] = ghost_ids

        # Group by rank
        return self._group_by_rank(adaptive_ghost_map)

    def _has_large_objects_in_cell(self, cell_id: int, rma_object_pool, threshold: float) -> bool:
        """
        Test whether a cell holds a large object

        Args:
            cell_id: Cell ID
            rma_object_pool: object pool
            threshold: large-object threshold (radius)

        Returns:
            True if a large object is present, False otherwise
        """
        # [fix] The use_binning check was removed
        # v1.6.6+ uses the three-way storage architecture and no longer has use_binning
        # Simply test whether the object pool supports cell queries
        if not hasattr(rma_object_pool, 'get_objects_in_cell'):
            return False  # without that method, assume there is no large object

        # Fetch the objects in this cell
        objects = rma_object_pool.get_objects_in_cell(cell_id)

        if len(objects) == 0:
            return False

        # [Bug 2 fix] Test whether any object has radius >= threshold
        # v1.6.5: prefer the size field, falling back to float_props[0]

        # Route 1: prefer the size field (added in v1.6.5)
        if 'size' in objects.dtype.names:
            # The size field usually holds a diameter or physical scale
            # radius = size / 2 (consistent with rma_object_pool._is_large_object)
            radii = objects['size'] / 2.0
            if np.any(radii >= threshold):
                return True

        # Route 2: fall back to the old float_props[0] convention (backward compatible)
        if 'float_props' in objects.dtype.names:
            radii = objects['float_props'][:, 0]
            if np.any(radii >= threshold):
                return True

        return False

    def _group_by_rank(self, cell_ghost_map: Dict[int, List[int]]) -> Dict[int, List[int]]:
        """
        Convert {cell_id: [ghost_cell_ids]} into {rank: [ghost_cell_ids]}

        Args:
            cell_ghost_map: ghost cells of each local cell

        Returns:
            the ghost cells grouped by rank (deduplicated)
        """
        ghost_cells_by_rank = {}

        for cell_id, ghost_ids in cell_ghost_map.items():
            for nid in ghost_ids:
                owner = self.ownership_map[nid]
                if owner not in ghost_cells_by_rank:
                    ghost_cells_by_rank[owner] = []
                ghost_cells_by_rank[owner].append(nid)

        # Deduplicate
        for r in ghost_cells_by_rank:
            ghost_cells_by_rank[r] = list(set(ghost_cells_by_rank[r]))

        return ghost_cells_by_rank

    def get_local_cells(self) -> Set[int]:
        """Get the set of local cells"""
        return self.local_cells.copy()

    def is_local_cell(self, cell_id: int) -> bool:
        """Test whether a cell belongs to this rank"""
        return cell_id in self.local_cells

    def pos_to_cell_id(self, x, y, z) -> int:
        """Coordinate -> cell ID"""
        return self.grid_system.pos_to_cell_id(x, y, z)

    def get_cell_owner_by_pos(self, x, y, z) -> int:
        """Coordinate -> owning rank of the cell"""
        cell_id = self.grid_system.pos_to_cell_id(x, y, z)
        return self.get_cell_owner(cell_id)

    def get_owner_rank(self, position: np.ndarray) -> int:
        """Get the owning rank of an object"""
        return self.get_cell_owner_by_pos(position[0], position[1], position[2])

    def is_local(self, position: np.ndarray) -> bool:
        """Test whether an object belongs to this rank"""
        return self.get_owner_rank(position) == self.rank

    def get_subdomain_center(self) -> np.ndarray:
        """Get the centre coordinates of this rank's subdomain"""
        b = self.subdomain_bounds
        return np.array([
            (b['x_min'] + b['x_max']) / 2,
            (b['y_min'] + b['y_max']) / 2,
            (b['z_min'] + b['z_max']) / 2
        ])

    def get_all_ghost_ranks_stats(self) -> Dict[str, float]:
        """Count the ghost ranks (diagnostic)"""
        count = len(self.ghost_ranks)
        all_counts = self.comm.allgather(count)
        return {
            'min': min(all_counts),
            'max': max(all_counts),
            'avg': np.mean(all_counts)
        }

    def _print_info(self):
        """Print the domain decomposition configuration"""
        print("=" * 60)
        print("Domain Decomposition Configuration (Auto-Generated)")
        print("=" * 60)
        print(f"Total ranks: {self.size}")
        print(f"Topology: {self.nx} x {self.ny} x {self.nz}")
        print(f"Box Size: {self.box_size}")
        print(f"Grid dimensions: {self.grid_dim} (Fine Mesh)")
        print(f"Cell size: {self.grid_system.cell_size}")
        print(f"Max interaction dist: {self.max_interaction_distance}")
        print(f"Cells per rank: {len(self.local_cells)} (perfect balance: {self.grid_system.total_cells // self.size})")
        print("=" * 60)

    def update_ownership_map(self, new_map):
        """Update the ownership map (used by dynamic load balancing)"""
        self.ownership_map = new_map.astype(np.int32)
        self._update_local_cells()

    def update_layout_from_bounds(self, bounds_list):
        """Update the layout from a list of bounds (used by dynamic load balancing)"""
        self.current_bounds_list = [dict(b) for b in bounds_list]
        self.ownership_map[:] = -1

        for r, b in enumerate(bounds_list):
            for iz in range(b['z_start'], b['z_end']):
                for iy in range(b['y_start'], b['y_end']):
                    start = iz * self.grid_dim[0] * self.grid_dim[1] + iy * self.grid_dim[0] + b['x_start']
                    end = start + (b['x_end'] - b['x_start'])
                    self.ownership_map[start:end] = r

        self.subdomain_bounds = self._grid_bounds_to_physical_bounds(self.current_bounds_list[self.rank])
        self._update_local_cells()

        # [fix] After RCB the rank adjacency has changed, so ghost_ranks must be
        # recomputed from the new bounds, or new physical neighbours are never pulled
        self.ghost_ranks = self._compute_ghost_ranks()


def create_domain_decomposition(comm: MPI.Comm, config: dict, verbose: bool = False) -> DomainDecomposition:
    """Factory: create a domain decomposition object"""
    return DomainDecomposition(comm, config, verbose)
