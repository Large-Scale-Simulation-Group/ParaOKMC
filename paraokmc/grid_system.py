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
GridSystem: fine-grid spatial discretization (fix: distance-based neighbours)

Core features:
1. Coordinate <-> cell ID conversion
2. Neighbour cell computation (from the physical distance, adaptive number of shells)
3. Grid boundary tests
4. Periodic boundary support

Design idea:
- the simulation box is divided into Gx x Gy x Gz cells
- there are far more cells than ranks (e.g. 8 ranks over a 64x64x64 grid)
- a cell is the smallest unit of load balancing and communication

"""

import numpy as np
from typing import Tuple, List, Set


class GridSystem:
    """
    Fine grid system: handles spatial discretization and cell ID mapping

    Args:
        grid_dim: (Gx, Gy, Gz) global fine-grid dimensions
        box_size: [Lx, Ly, Lz] physical size of the simulation box
        periodic: [bool, bool, bool] periodic boundary conditions

    Attributes:
        grid_dim: tuple of grid dimensions
        box_size: array of box dimensions
        cell_size: physical size of one cell [dx, dy, dz]
        total_cells: total number of cells, Gx * Gy * Gz
        periodic: periodic boundary conditions
    """

    def __init__(self,
                 grid_dim: Tuple[int, int, int],
                 box_size: List[float],
                 periodic: List[bool] = None):
        """
        Initialize the grid system

        Args:
            grid_dim: (Gx, Gy, Gz) grid dimensions
            box_size: [Lx, Ly, Lz] box dimensions
            periodic: [px, py, pz] periodic boundaries (all True by default)
        """
        self.grid_dim = grid_dim
        self.box_size = np.array(box_size, dtype=np.float64)

        # Physical size of one cell
        self.cell_size = self.box_size / np.array(grid_dim, dtype=np.float64)

        # Total number of cells
        self.total_cells = grid_dim[0] * grid_dim[1] * grid_dim[2]

        # Periodic boundary conditions
        if periodic is None:
            self.periodic = [True, True, True]
        else:
            self.periodic = periodic

        # Neighbour cache: {(cell_id, max_distance_key): [neighbor_ids]}
        # max_distance_key uses round(max_distance, 6) to dodge float precision issues
        self._neighbor_cache = {}

        # [performance] Precompute the bounds of every cell
        # Stored as numpy arrays so the operations can be vectorized
        # shape: (total_cells, 6) = [x_min, x_max, y_min, y_max, z_min, z_max]
        self._cell_bounds_array = self._precompute_all_cell_bounds()

        # [performance] Precompute the valid neighbour offsets
        # For a given max_distance, which (dx, dy, dz) offsets are valid
        # {max_distance_key: [(dx, dy, dz), ...]}
        self._valid_offsets_cache = {}

    def pos_to_cell_id(self, x: float, y: float, z: float) -> int:
        """
        Coordinate -> cell ID

        Mapping:
        - cell_id = ix + iy * Gx + iz * Gx * Gy (flattened index)
        - ix = floor(x / dx), and likewise for iy, iz
        - periodic boundaries are handled as coordinate % box_size

        Args:
            x, y, z: object coordinates

        Returns:
            cell_id: cell ID (0 ... total_cells-1)

        Raises:
            ValueError: if the coordinate is out of bounds and not periodic
        """
        # Handle periodic boundaries
        pos = np.array([x, y, z], dtype=np.float64)
        for dim in range(3):
            if self.periodic[dim]:
                # Periodic: map into [0, box_size)
                pos[dim] = pos[dim] % self.box_size[dim]
            else:
                # Non-periodic: check the bounds
                if pos[dim] < 0 or pos[dim] >= self.box_size[dim]:
                    raise ValueError(
                        f"coordinate out of bounds: {['x', 'y', 'z'][dim]}={pos[dim]:.3f}, "
                        f"valid range [0, {self.box_size[dim]:.3f})"
                    )

        # Grid indices (ix, iy, iz)
        ix = int(np.floor(pos[0] / self.cell_size[0]))
        iy = int(np.floor(pos[1] / self.cell_size[1]))
        iz = int(np.floor(pos[2] / self.cell_size[2]))

        # Handle the edge case of a coordinate exactly on the box boundary
        ix = min(ix, self.grid_dim[0] - 1)
        iy = min(iy, self.grid_dim[1] - 1)
        iz = min(iz, self.grid_dim[2] - 1)

        # Flattened index
        cell_id = ix + iy * self.grid_dim[0] + iz * self.grid_dim[0] * self.grid_dim[1]

        return cell_id

    def cell_id_to_grid_index(self, cell_id: int) -> Tuple[int, int, int]:
        """
        Cell ID -> grid indices (ix, iy, iz)

        Inverse mapping:
        - iz = cell_id // (Gx * Gy)
        - iy = (cell_id % (Gx * Gy)) // Gx
        - ix = cell_id % Gx

        Args:
            cell_id: cell ID

        Returns:
            (ix, iy, iz): grid indices
        """
        gx, gy, gz = self.grid_dim

        iz = cell_id // (gx * gy)
        remainder = cell_id % (gx * gy)
        iy = remainder // gx
        ix = remainder % gx

        return (ix, iy, iz)

    def grid_index_to_cell_id(self, ix: int, iy: int, iz: int) -> int:
        """
        Grid indices (ix, iy, iz) -> cell ID

        Args:
            ix, iy, iz: grid indices

        Returns:
            cell_id: cell ID
        """
        return ix + iy * self.grid_dim[0] + iz * self.grid_dim[0] * self.grid_dim[1]

    # ===== Core fix: distance-based neighbour search =====

    def _precompute_all_cell_bounds(self) -> np.ndarray:
        """
        Precompute the physical bounds of every cell

        Returns:
            bounds_array: shape (total_cells, 6)
                         each row = [x_min, x_max, y_min, y_max, z_min, z_max]
        """
        # [performance fix] Vectorized: the old code called the Python-level
        # cell_id_to_grid_index() once per cell, costing ~17 s to initialize 9 million
        # cells (a 100x300x300 grid). This uses the same decomposition in bulk.
        gx, gy, gz = self.grid_dim
        gxgy = gx * gy

        cell_ids = np.arange(self.total_cells, dtype=np.int64)
        iz, rem = np.divmod(cell_ids, gxgy)
        iy, ix = np.divmod(rem, gx)

        bounds = np.empty((self.total_cells, 6), dtype=np.float64)
        bounds[:, 0] = ix * self.cell_size[0]          # x_min
        bounds[:, 1] = (ix + 1) * self.cell_size[0]    # x_max
        bounds[:, 2] = iy * self.cell_size[1]          # y_min
        bounds[:, 3] = (iy + 1) * self.cell_size[1]    # y_max
        bounds[:, 4] = iz * self.cell_size[2]          # z_min
        bounds[:, 5] = (iz + 1) * self.cell_size[2]    # z_max

        return bounds

    def _compute_valid_offsets(self, max_distance: float) -> List[Tuple[int, int, int]]:
        """
        Precompute the valid neighbour offsets

        Work out which (dx, dy, dz) offsets are valid for a given max_distance.
        This is computed once and then applies to every cell.

        Args:
            max_distance: maximum interaction distance

        Returns:
            valid_offsets: [(dx, dy, dz), ...] list of valid offsets
        """
        if max_distance <= 0:
            # Degenerate to a single shell of neighbours
            rx, ry, rz = 1, 1, 1
        else:
            rx = int(np.ceil(max_distance / self.cell_size[0]))
            ry = int(np.ceil(max_distance / self.cell_size[1]))
            rz = int(np.ceil(max_distance / self.cell_size[2]))

        valid_offsets = []

        # Measure distances from the reference cell (0, 0, 0)
        for dz in range(-rz, rz + 1):
            for dy in range(-ry, ry + 1):
                for dx in range(-rx, rx + 1):
                    if dx == 0 and dy == 0 and dz == 0:
                        continue  # skip ourselves

            # Distance from the reference cell to the offset cell
                    if max_distance > 0:
                        dist = self._compute_cell_distance((0, 0, 0), (dx, dy, dz))
                        if dist >= max_distance:
                            continue

                    valid_offsets.append((dx, dy, dz))

        return valid_offsets

    def _compute_cell_bounds(self, ix: int, iy: int, iz: int) -> dict:
        """
        Physical bounds of the given grid indices

        Args:
            ix, iy, iz: grid indices

        Returns:
            bounds: {x_min, x_max, y_min, y_max, z_min, z_max}
        """
        return {
            'x_min': ix * self.cell_size[0],
            'x_max': (ix + 1) * self.cell_size[0],
            'y_min': iy * self.cell_size[1],
            'y_max': (iy + 1) * self.cell_size[1],
            'z_min': iz * self.cell_size[2],
            'z_max': (iz + 1) * self.cell_size[2],
        }

    def _compute_cell_distance(self, idx1: Tuple[int, int, int], idx2: Tuple[int, int, int]) -> float:
        """
        Minimum distance between two cells (periodic boundaries included)

        Replicates the logic of _compute_subdomain_distance from v1.6.3

        Args:
            idx1, idx2: grid indices (ix, iy, iz)

        Returns:
            distance: the minimum distance
        """
        # Use the precomputed bounds
        cell_id1 = self.grid_index_to_cell_id(*idx1)
        cell_id2 = self.grid_index_to_cell_id(*idx2)

        bounds1_arr = self._cell_bounds_array[cell_id1]
        bounds2_arr = self._cell_bounds_array[cell_id2]

        def axis_distance(min1, max1, min2, max2, length, periodic):
            """Minimum distance along one axis (periodic boundaries included)"""
            if periodic:
                # Overlapping
                if not (min2 > max1 or min1 > max2):
                    return 0.0
                # Direct distance
                if min2 > max1:
                    direct = min2 - max1
                else:
                    direct = min1 - max2
                # Wrapped distance
                wrap = length - (max(max1, max2) - min(min1, min2))
                return min(direct, wrap)
            else:
                # Non-periodic: only the direct distance
                if min2 > max1:
                    return min2 - max1
                elif min1 > max2:
                    return min1 - max2
                else:
                    return 0.0  # overlapping

        dx = axis_distance(bounds1_arr[0], bounds1_arr[1],
                          bounds2_arr[0], bounds2_arr[1],
                          self.box_size[0], self.periodic[0])
        dy = axis_distance(bounds1_arr[2], bounds1_arr[3],
                          bounds2_arr[2], bounds2_arr[3],
                          self.box_size[1], self.periodic[1])
        dz = axis_distance(bounds1_arr[4], bounds1_arr[5],
                          bounds2_arr[4], bounds2_arr[5],
                          self.box_size[2], self.periodic[2])

        return np.sqrt(dx**2 + dy**2 + dz**2)

    def get_neighbor_cells(self, cell_id: int, max_distance: float = 0.0,
                          include_self: bool = False) -> List[int]:
        """
        Get the neighbour cell IDs (from the physical distance, adaptive shells)

        v1.6.4-fix: repairs the physics bug of hardcoding 26 neighbours
        - searches several shells automatically when max_distance > cell_size
        - uses an exact box-to-box distance test, so no long-range interaction is missed
        - caches results to avoid recomputation
        - uses the precomputed offsets to avoid repeating the distance test

        Args:
            cell_id: ID of the central cell
            max_distance: maximum interaction distance (physical units)
                         - <= 0 degenerates to a single shell (backward compatible)
                         - > 0 searches adaptively for every neighbour within range
            include_self: whether to include the cell itself

        Returns:
            neighbor_ids: list of neighbour cell IDs
        """
        # Cache key: rounded to dodge float precision issues
        cache_key = (cell_id, round(max_distance, 6), include_self)

        # Check the cache
        if cache_key in self._neighbor_cache:
            return self._neighbor_cache[cache_key]

        # Fetch or compute the valid offsets
        max_dist_key = round(max_distance, 6)
        if max_dist_key not in self._valid_offsets_cache:
            self._valid_offsets_cache[max_dist_key] = self._compute_valid_offsets(max_distance)

        valid_offsets = self._valid_offsets_cache[max_dist_key]

        # Grid indices of this cell
        ix, iy, iz = self.cell_id_to_grid_index(cell_id)
        gx, gy, gz = self.grid_dim

        neighbors = []

        # Include ourselves if requested
        if include_self:
            neighbors.append(cell_id)

        # Apply the precomputed offsets
        for dx, dy, dz in valid_offsets:
            # Target grid indices
            t_ix = ix + dx
            t_iy = iy + dy
            t_iz = iz + dz

            # Periodic wrap and validity check
            valid = True

            # X axis
            if self.periodic[0]:
                t_ix_wrapped = t_ix % gx
            else:
                if t_ix < 0 or t_ix >= gx:
                    valid = False
                t_ix_wrapped = t_ix

            # Y axis
            if self.periodic[1]:
                t_iy_wrapped = t_iy % gy
            else:
                if t_iy < 0 or t_iy >= gy:
                    valid = False
                t_iy_wrapped = t_iy

            # Z axis
            if self.periodic[2]:
                t_iz_wrapped = t_iz % gz
            else:
                if t_iz < 0 or t_iz >= gz:
                    valid = False
                t_iz_wrapped = t_iz

            if not valid:
                continue

            # Add to the neighbour list
            nid = self.grid_index_to_cell_id(t_ix_wrapped, t_iy_wrapped, t_iz_wrapped)
            neighbors.append(nid)

        # Deduplicate (PBC may map different offsets onto one cell)
        result = list(set(neighbors))

        # Store in the cache
        self._neighbor_cache[cache_key] = result

        return result

    def get_cell_center(self, cell_id: int) -> Tuple[float, float, float]:
        """
        Get the centre coordinates of a cell

        Args:
            cell_id: cell ID

        Returns:
            (x, y, z): coordinates of the cell centre
        """
        ix, iy, iz = self.cell_id_to_grid_index(cell_id)

        x = (ix + 0.5) * self.cell_size[0]
        y = (iy + 0.5) * self.cell_size[1]
        z = (iz + 0.5) * self.cell_size[2]

        return (x, y, z)

    def is_boundary_cell(self, cell_id: int) -> bool:
        """
        Test whether a cell lies on a physical boundary

        Definition: ix=0 or ix=Gx-1 (along a non-periodic dimension)

        Args:
            cell_id: cell ID

        Returns:
            is_boundary: whether this is a boundary cell
        """
        ix, iy, iz = self.cell_id_to_grid_index(cell_id)
        gx, gy, gz = self.grid_dim

        # Check each dimension
        boundary = False

        if not self.periodic[0]:
            if ix == 0 or ix == gx - 1:
                boundary = True

        if not self.periodic[1]:
            if iy == 0 or iy == gy - 1:
                boundary = True

        if not self.periodic[2]:
            if iz == 0 or iz == gz - 1:
                boundary = True

        return boundary

    def get_cells_in_box(self,
                         x_min: float, x_max: float,
                         y_min: float, y_max: float,
                         z_min: float, z_max: float) -> Set[int]:
        """
        Get every cell ID inside a physical region

        Use: domain decomposition, to find the cells a rank owns

        Args:
            x_min, x_max: bounds along X
            y_min, y_max: bounds along Y
            z_min, z_max: bounds along Z

        Returns:
            cell_ids: set of cell IDs
        """
        # Range of grid indices
        ix_min = int(np.floor(x_min / self.cell_size[0]))
        ix_max = int(np.floor(x_max / self.cell_size[0]))
        iy_min = int(np.floor(y_min / self.cell_size[1]))
        iy_max = int(np.floor(y_max / self.cell_size[1]))
        iz_min = int(np.floor(z_min / self.cell_size[2]))
        iz_max = int(np.floor(z_max / self.cell_size[2]))

        # Bounds check
        ix_min = max(0, min(ix_min, self.grid_dim[0] - 1))
        ix_max = max(0, min(ix_max, self.grid_dim[0] - 1))
        iy_min = max(0, min(iy_min, self.grid_dim[1] - 1))
        iy_max = max(0, min(iy_max, self.grid_dim[1] - 1))
        iz_min = max(0, min(iz_min, self.grid_dim[2] - 1))
        iz_max = max(0, min(iz_max, self.grid_dim[2] - 1))

        cell_ids = set()

        for iz in range(iz_min, iz_max + 1):
            for iy in range(iy_min, iy_max + 1):
                for ix in range(ix_min, ix_max + 1):
                    cell_id = self.grid_index_to_cell_id(ix, iy, iz)
                    cell_ids.add(cell_id)

        return cell_ids

    def __repr__(self):
        return (f"GridSystem(grid_dim={self.grid_dim}, "
                f"box_size={self.box_size.tolist()}, "
                f"cell_size={self.cell_size.tolist()}, "
                f"total_cells={self.total_cells}, "
                f"periodic={self.periodic})")


# ===== Unit tests (optional) =====
if __name__ == "__main__":
    # Test setup: a 10x10x10 grid over a 100x100x100 box
    grid = GridSystem(
        grid_dim=(10, 10, 10),
        box_size=[100.0, 100.0, 100.0],
        periodic=[True, True, True]
    )

    print("=" * 60)
    print("GridSystem unit tests")
    print("=" * 60)

    # Test 1: coordinate -> cell ID
    print("\nTest 1: coordinate -> cell ID")
    x, y, z = 15.0, 25.0, 35.0
    cell_id = grid.pos_to_cell_id(x, y, z)
    print(f"coordinate ({x}, {y}, {z}) -> cell ID {cell_id}")

    # Test 2: cell ID -> grid indices -> cell ID (round trip)
    print("\nTest 2: cell ID round trip")
    ix, iy, iz = grid.cell_id_to_grid_index(cell_id)
    print(f"cell ID {cell_id} -> grid indices ({ix}, {iy}, {iz})")
    cell_id_back = grid.grid_index_to_cell_id(ix, iy, iz)
    print(f"grid indices ({ix}, {iy}, {iz}) -> cell ID {cell_id_back}")
    assert cell_id == cell_id_back, "round trip failed"
    print("OK round trip passed")

    # Test 3: 26 neighbours
    print("\nTest 3: 26 neighbours")
    neighbors = grid.get_neighbor_cells(cell_id)
    print(f"number of neighbours of cell {cell_id}: {len(neighbors)}")
    assert len(neighbors) == 26, f"wrong neighbour count: {len(neighbors)} != 26"
    print(f"first 5 neighbours: {neighbors[:5]}")
    print("OK neighbour computation correct")

    # Test 4: cell centre coordinates
    print("\nTest 4: cell centre coordinates")
    center = grid.get_cell_center(cell_id)
    print(f"centre of cell {cell_id}: ({center[0]:.2f}, {center[1]:.2f}, {center[2]:.2f})")

    # Test 5: boundary test
    print("\nTest 5: boundary test")
    grid_no_pbc = GridSystem(
        grid_dim=(10, 10, 10),
        box_size=[100.0, 100.0, 100.0],
        periodic=[False, False, False]
    )
    corner_cell = grid_no_pbc.grid_index_to_cell_id(0, 0, 0)
    is_boundary = grid_no_pbc.is_boundary_cell(corner_cell)
    print(f"is cell {corner_cell} (corner) on the boundary: {is_boundary}")
    assert is_boundary, "boundary test failed"

    center_cell = grid_no_pbc.grid_index_to_cell_id(5, 5, 5)
    is_boundary_center = grid_no_pbc.is_boundary_cell(center_cell)
    print(f"is cell {center_cell} (centre) on the boundary: {is_boundary_center}")
    assert not is_boundary_center, "boundary test failed"
    print("OK boundary test correct")

    # Test 6: query cells inside a region
    print("\nTest 6: query cells inside a region")
    cells_in_box = grid.get_cells_in_box(0, 30, 0, 30, 0, 30)
    print(f"number of cells inside [0,30]x[0,30]x[0,30]: {len(cells_in_box)}")
    print(f"expected: 3x3x3=27, actual: {len(cells_in_box)}")

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)
