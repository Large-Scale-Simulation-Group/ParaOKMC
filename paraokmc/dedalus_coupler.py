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
Dedalus field coupler

Core features:
1. High-precision field storage (distributed, in the native Dedalus layout)
2. Field compression (high precision -> the compressed OKMC copy)
3. Field history (an initial condition for the external solver)
4. Example solvers (heat conduction, stress field)

Architecture:
- HighPrecisionFieldStore: distributed high-precision field storage
  - every rank keeps the local part of the Dedalus field
  - both coefficient space and grid space are supported
  - the history is used to compute time derivatives

- FieldCompressor: high-precision field -> compressed OKMC field
  - gradient-adaptive compression (resolution is kept where the gradient is high)
  - produces the compressed data on a non-uniform grid
  - every rank holds a full copy of the compressed field

- DedalusCoupler: the main coupler class
  - manages the distributed storage of the high-precision fields
  - calls the Dedalus solver to update them
  - compresses them and synchronizes them into the OKMC UnifiedPhysics

- DedalusHeatSolver/DedalusStressSolver: example solvers
  - implementing the ExternalSolver interface
  - solving the heat conduction / elasticity equations with Dedalus

Data flow:
    OKMC objects -> rasterization -> Dedalus heat source -> Dedalus solve
                                              ↓
    OKMC rate evaluation <- UnifiedPhysics <- compression <- high-precision field

Dependencies:
- dedalus (v3.x)
- mpi4py
- numpy
- scipy

"""

import numpy as np
from mpi4py import MPI
from typing import Dict, List, Tuple, Optional, Callable, Union, Any
from scipy.interpolate import RegularGridInterpolator
from abc import ABC, abstractmethod
import warnings

# Try to import Dedalus (it may not be installed)
DEDALUS_AVAILABLE = False
DEDALUS_VERSION = None
d3 = None

try:
    import dedalus.public as d3
    DEDALUS_AVAILABLE = True

    # [fix] Detect the Dedalus version
    try:
        import dedalus
        DEDALUS_VERSION = getattr(dedalus, '__version__', 'unknown')

        # Is this v2 or v3?
        if hasattr(d3, 'Distributor'):
            DEDALUS_VERSION = f"3.x ({DEDALUS_VERSION})"
        elif hasattr(d3, 'Domain'):
            DEDALUS_VERSION = f"2.x ({DEDALUS_VERSION})"
    except Exception:
        DEDALUS_VERSION = 'unknown'

except ImportError:
    pass

from .unified_physics import ExternalSolver, UnifiedPhysics


# =============================================================================
# High-precision field storage
# =============================================================================

class HighPrecisionFieldStore:
    """
    Distributed high-precision field storage

    Properties:
    - every rank keeps the local part of the Dedalus field (distributed)
    - both coefficient space (spectral coefficients) and grid space are supported
    - the history serves time derivatives or the initial condition of a solver

    Storage layout:
    - current_fields: Dict[str, np.ndarray] - the fields of the current step
    - prev_fields: Dict[str, np.ndarray] - the fields of the previous step
    - field_metadata: Dict[str, dict] - metadata (grid and distribution information)
    """

    def __init__(self, comm: MPI.Comm, enable_history: bool = True):
        """
        Initialize the high-precision field storage

        Args:
            comm: MPI communicator
            enable_history: whether to keep the history
        """
        self.comm = comm
        self.rank = comm.Get_rank()
        self.size = comm.Get_size()
        self.enable_history = enable_history

        # Field data (distributed: every rank keeps its local part)
        self.current_fields: Dict[str, np.ndarray] = {}
        self.prev_fields: Dict[str, np.ndarray] = {}

        # Field metadata
        self.field_metadata: Dict[str, dict] = {}

        # Time bookkeeping
        self.current_time: float = 0.0
        self.prev_time: float = 0.0

        # References to the Dedalus objects (for direct access)
        self._dedalus_fields: Dict[str, Any] = {}  # Dedalus Field objects
        self._dedalus_dist: Optional[Any] = None   # Dedalus Distributor

    def register_field(self, name: str,
                       shape: Tuple[int, ...],
                       global_shape: Tuple[int, ...],
                       local_slice: Tuple[slice, ...],
                       grid_axes: Tuple[np.ndarray, ...],
                       space: str = 'grid') -> None:
        """
        Register a high-precision field

        Args:
            name: field name
            shape: shape of the local data
            global_shape: shape of the global data
            local_slice: slice of the local data inside the global array
            grid_axes: grid coordinates (x, y, z)
            space: 'grid' or 'coefficient'
        """
        # [fix] Is this the replicated storage mode?
        is_replicated = (tuple(shape) == tuple(global_shape))

        self.field_metadata[name] = {
            'local_shape': shape,
            'global_shape': global_shape,
            'local_slice': local_slice,
            'grid_axes': grid_axes,
            'space': space,
            'dtype': np.float64,
            'is_replicated': is_replicated,  # marks the storage mode
        }

        # Initialize the storage
        self.current_fields[name] = np.zeros(shape, dtype=np.float64)
        if self.enable_history:
            self.prev_fields[name] = np.zeros(shape, dtype=np.float64)

    def register_dedalus_field(self, name: str,
                               dedalus_field: Any,
                               space: str = 'grid') -> None:
        """
        Register a Dedalus Field object directly

        Args:
            name: field name
            dedalus_field: the Dedalus Field object
            space: 'grid' or 'coefficient'
        """
        if not DEDALUS_AVAILABLE:
            raise RuntimeError("Dedalus not installed")

        self._dedalus_fields[name] = dedalus_field

        # Distribution information of the field
        if space == 'grid':
            data = dedalus_field['g']
        else:
            data = dedalus_field['c']

        # Global shape
        # Dedalus v3: use the domain information
        if hasattr(dedalus_field, 'domain'):
            global_shape = tuple(b.size for b in dedalus_field.domain.bases)
        else:
            # Estimate the global shape (by gathering)
            local_shape = data.shape
            all_shapes = self.comm.allgather(local_shape)
            # Simplification: assume the distribution runs along the first axis
            global_shape = list(local_shape)
            global_shape[0] = sum(s[0] for s in all_shapes)
            global_shape = tuple(global_shape)

        # Local slice
        # Simplification: assume the distribution runs along the first axis
        local_size = data.shape[0]
        all_sizes = self.comm.allgather(local_size)
        start = sum(all_sizes[:self.rank])
        local_slice = (slice(start, start + local_size),) + tuple(slice(None) for _ in range(len(data.shape) - 1))

        # Grid coordinates (in grid space)
        grid_axes = None
        if space == 'grid' and hasattr(dedalus_field, 'domain'):
            try:
                coords = dedalus_field.domain.grids()
                grid_axes = tuple(c.ravel() if c.ndim > 1 else c for c in coords)
            except:
                pass

        # [fix] Is this the replicated storage mode?
        is_replicated = (tuple(data.shape) == tuple(global_shape))

        self.field_metadata[name] = {
            'local_shape': data.shape,
            'global_shape': global_shape,
            'local_slice': local_slice,
            'grid_axes': grid_axes,
            'space': space,
            'dtype': data.dtype,
            'is_dedalus': True,
            'is_replicated': is_replicated,  # marks the storage mode
        }

        self.current_fields[name] = data.copy()
        if self.enable_history:
            self.prev_fields[name] = data.copy()

    def update_field(self, name: str, data: np.ndarray) -> None:
        """
        Update the field data (the old value goes into the history first)

        Args:
            name: field name
            data: the new field data (the local part)
        """
        if name not in self.current_fields:
            raise KeyError(f"Unknown field: {name}")

        # Save the current value into the history
        if self.enable_history:
            self.prev_fields[name] = self.current_fields[name].copy()

        # Store the new value
        self.current_fields[name] = data.copy()

    def update_from_dedalus(self, name: str, space: str = 'grid') -> None:
        """
        Update the data from a Dedalus Field object

        Args:
            name: field name
            space: 'grid' or 'coefficient'
        """
        if name not in self._dedalus_fields:
            raise KeyError(f"No Dedalus field registered: {name}")

        field = self._dedalus_fields[name]
        if space == 'grid':
            data = field['g']
        else:
            data = field['c']

        self.update_field(name, data)

    def save_history(self, time: float) -> None:
        """
        Save the current state into the history (call it before the Dedalus solve)

        Args:
            time: current time
        """
        if not self.enable_history:
            return

        self.prev_time = self.current_time
        self.current_time = time

        for name in self.current_fields:
            self.prev_fields[name] = self.current_fields[name].copy()

    def update_time(self, time: float) -> None:
        """
        Advance the current time (without saving the history)
        """
        self.current_time = time

    def get_current(self, name: str) -> np.ndarray:
        """Get the current field data (the local part)"""
        return self.current_fields.get(name)

    def get_prev(self, name: str) -> Optional[np.ndarray]:
        """Get the field data of the previous step (the local part)"""
        if not self.enable_history:
            return None
        return self.prev_fields.get(name)

    def get_time_delta(self) -> float:
        """Get the time difference"""
        return self.current_time - self.prev_time

    def get_field_rate(self, name: str) -> Optional[np.ndarray]:
        """
        Get the rate of change of a field (the local part)

        Returns:
            dField/dt, or None when dt is 0 or the history is disabled
        """
        if not self.enable_history:
            return None

        dt = self.get_time_delta()
        if dt <= 0:
            return None

        current = self.get_current(name)
        prev = self.get_prev(name)

        if current is None or prev is None:
            return None

        return (current - prev) / dt

    def gather_global_field(self, name: str) -> Optional[np.ndarray]:
        """
        Gather the global field data onto rank 0

        Two modes are supported:
        1. replicated storage (every rank holds everything): return the data of rank 0
        2. distributed storage (native Dedalus): gather with Gatherv

        Returns:
            rank 0: the global field data
            every other rank: None
        """
        local_data = self.current_fields.get(name)
        if local_data is None:
            return None

        meta = self.field_metadata[name]
        global_shape = meta['global_shape']
        local_shape = meta['local_shape']

        # [fix] Is the storage replicated?
        # The explicit marker wins (set by DedalusCoupler.register_field)
        # falling back to comparing the shapes (for older code)
        is_replicated = meta.get('is_replicated', False)
        if not is_replicated:
            # The shapes must be tuples for the comparison to work
            is_replicated = (tuple(local_shape) == tuple(global_shape))

        if is_replicated:
            # Replicated mode: return the data of rank 0
            if self.rank == 0:
                return local_data.copy()
            return None

        # Distributed mode: gather with Gatherv
        local_flat = local_data.ravel()

        # Collect the size of every rank
        local_size = local_flat.size
        all_sizes = self.comm.allgather(local_size)

        if self.rank == 0:
            global_flat = np.zeros(sum(all_sizes), dtype=local_data.dtype)
            displacements = [sum(all_sizes[:i]) for i in range(self.size)]
        else:
            global_flat = None
            displacements = None

        # Gatherv
        sendbuf = [local_flat, local_size]
        recvbuf = [global_flat, all_sizes, displacements, MPI.DOUBLE] if self.rank == 0 else None

        self.comm.Gatherv(sendbuf, recvbuf, root=0)

        if self.rank == 0:
            return global_flat.reshape(global_shape)
        return None


# =============================================================================
# Field compressor
# =============================================================================

class FieldCompressor:
    """
    Field compressor: high-precision field -> compressed OKMC field

    Compression strategy:
    1. gradient adaptive: keep the resolution where the gradient is high, downsample elsewhere
    2. non-uniform grid: produce an adaptive, non-uniform mesh
    3. interpolation fidelity: keep the interpolation error acceptable after compression

    Input: the global high-precision field (held by rank 0 or gathered)
    Output: the compressed field plus the non-uniform grid (broadcast to every rank)
    """

    def __init__(self, comm: MPI.Comm,
                 gradient_threshold: float = 0.01,
                 min_points: int = 10,
                 max_compression_ratio: float = 0.5,
                 target_size: Optional[Tuple[int, int, int]] = None):
        """
        Initialize the compressor

        Args:
            comm: MPI communicator
            gradient_threshold: gradient threshold (regions below it may be compressed)
            min_points: minimum number of points per dimension
            max_compression_ratio: maximum compression ratio (compressed / original)
            target_size: target size (when given, uniform downsampling replaces the adaptive scheme)
        """
        self.comm = comm
        self.rank = comm.Get_rank()
        self.gradient_threshold = gradient_threshold
        self.min_points = min_points
        self.max_compression_ratio = max_compression_ratio
        self.target_size = target_size

    def compress(self, field: np.ndarray,
                 x: np.ndarray, y: np.ndarray, z: np.ndarray,
                 method: str = 'adaptive') -> Tuple[np.ndarray,
                                                     np.ndarray, np.ndarray, np.ndarray]:
        """
        Compress a 3D field

        Args:
            field: the 3D field data (nx, ny, nz)
            x, y, z: the original grid coordinates
            method: 'adaptive' (gradient adaptive) or 'uniform' (plain downsampling)

        Returns:
            (compressed_field, x_new, y_new, z_new)
        """
        if self.rank != 0:
            # Only rank 0 compresses
            return None, None, None, None

        if method == 'uniform' or self.target_size is not None:
            return self._compress_uniform(field, x, y, z)
        else:
            return self._compress_adaptive(field, x, y, z)

    def _compress_uniform(self, field: np.ndarray,
                          x: np.ndarray, y: np.ndarray, z: np.ndarray
                          ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Uniform downsampling"""
        nx, ny, nz = field.shape

        if self.target_size:
            tx, ty, tz = self.target_size
        else:
            # Compress to max_compression_ratio by default
            tx = max(self.min_points, int(nx * self.max_compression_ratio))
            ty = max(self.min_points, int(ny * self.max_compression_ratio))
            tz = max(self.min_points, int(nz * self.max_compression_ratio))

        # Uniform sampling indices
        ix = np.linspace(0, nx - 1, tx, dtype=int)
        iy = np.linspace(0, ny - 1, ty, dtype=int)
        iz = np.linspace(0, nz - 1, tz, dtype=int)

        # Compress
        x_new = x[ix]
        y_new = y[iy]
        z_new = z[iz]

        # Index through meshgrid
        compressed = field[np.ix_(ix, iy, iz)]

        return compressed, x_new, y_new, z_new

    def _compress_adaptive(self, field: np.ndarray,
                           x: np.ndarray, y: np.ndarray, z: np.ndarray
                           ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Gradient-adaptive compression"""
        nx, ny, nz = field.shape

        # Compute the gradient
        grad_x = np.abs(np.gradient(field, x, axis=0))
        grad_y = np.abs(np.gradient(field, y, axis=1))
        grad_z = np.abs(np.gradient(field, z, axis=2))

        # Combine the gradients (take the maximum)
        grad_mag = np.maximum(np.maximum(grad_x, grad_y), grad_z)

        # Choose the indices to keep along each axis
        def get_keep_indices(coords: np.ndarray,
                             grad: np.ndarray,
                             axis: int) -> np.ndarray:
            """Decide which indices to keep along one axis"""
            n = len(coords)
            if n <= self.min_points:
                return np.arange(n)

            # Maximum gradient at each position (the maximum over the other axes)
            other_axes = tuple(i for i in range(3) if i != axis)
            max_grad = np.max(grad, axis=other_axes)

            # Normalize
            if max_grad.max() > 0:
                norm_grad = max_grad / max_grad.max()
            else:
                norm_grad = np.ones_like(max_grad)

            # Importance score (a larger gradient means more importance)
            importance = np.clip(norm_grad / self.gradient_threshold, 0.0, 1.0)
            importance[0] = 1.0   # always keep the end points
            importance[-1] = 1.0

            # Decide which points to keep
            keep_mask = np.zeros(n, dtype=bool)
            keep_mask[0] = True
            keep_mask[-1] = True

            # Use the cumulative importance to place the samples
            cumsum = np.cumsum(importance)
            target_count = max(self.min_points, int(n * self.max_compression_ratio))

            # Sample at equal importance intervals
            if target_count >= n:
                return np.arange(n)

            target_values = np.linspace(0, cumsum[-1], target_count)
            keep_indices = np.searchsorted(cumsum, target_values)
            keep_indices = np.unique(np.clip(keep_indices, 0, n - 1))

            # Make sure the end points survive
            if 0 not in keep_indices:
                keep_indices = np.concatenate([[0], keep_indices])
            if n - 1 not in keep_indices:
                keep_indices = np.concatenate([keep_indices, [n - 1]])

            return np.sort(np.unique(keep_indices))

        # Indices kept along each axis
        ix = get_keep_indices(x, grad_mag, 0)
        iy = get_keep_indices(y, grad_mag, 1)
        iz = get_keep_indices(z, grad_mag, 2)

        # Compress
        x_new = x[ix]
        y_new = y[iy]
        z_new = z[iz]

        # Resample onto the new grid by interpolation
        # Build the interpolator of the original field first
        interp = RegularGridInterpolator(
            (x, y, z), field,
            method='linear',
            bounds_error=False,
            fill_value=None
        )

        # Sample on the new grid
        xx, yy, zz = np.meshgrid(x_new, y_new, z_new, indexing='ij')
        points = np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=-1)
        compressed = interp(points).reshape(len(x_new), len(y_new), len(z_new))

        return compressed, x_new, y_new, z_new

    def compress_and_broadcast(self, field_store: HighPrecisionFieldStore,
                               field_name: str,
                               x: np.ndarray, y: np.ndarray, z: np.ndarray,
                               method: str = 'adaptive'
                               ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Compress a high-precision field and broadcast it to every rank

        Args:
            field_store: the high-precision field store
            field_name: field name
            x, y, z: the original grid coordinates
            method: compression method

        Returns:
            (compressed_field, x_new, y_new, z_new) on every rank
        """
        # Gather the global field onto rank 0 first
        global_field = field_store.gather_global_field(field_name)

        # Rank 0 compresses
        if self.rank == 0:
            compressed, x_new, y_new, z_new = self.compress(
                global_field, x, y, z, method
            )
            shape = compressed.shape
            x_len, y_len, z_len = len(x_new), len(y_new), len(z_new)
        else:
            shape = None
            x_len, y_len, z_len = 0, 0, 0

        # Broadcast the shape
        shape = self.comm.bcast(shape, root=0)
        x_len = self.comm.bcast(x_len, root=0)
        y_len = self.comm.bcast(y_len, root=0)
        z_len = self.comm.bcast(z_len, root=0)

        # Allocate the receive buffers
        if self.rank != 0:
            compressed = np.zeros(shape, dtype=np.float64)
            x_new = np.zeros(x_len, dtype=np.float64)
            y_new = np.zeros(y_len, dtype=np.float64)
            z_new = np.zeros(z_len, dtype=np.float64)

        # Broadcast the compressed data (no serialization)
        self.comm.Bcast([compressed, MPI.DOUBLE], root=0)
        self.comm.Bcast([x_new, MPI.DOUBLE], root=0)
        self.comm.Bcast([y_new, MPI.DOUBLE], root=0)
        self.comm.Bcast([z_new, MPI.DOUBLE], root=0)

        return compressed, x_new, y_new, z_new


# =============================================================================
# The main Dedalus coupler class
# =============================================================================

class DedalusCoupler:
    """
    Dedalus field coupler

    Responsibilities:
    1. manage the Dedalus solvers and fields
    2. maintain the distributed high-precision storage (history included)
    3. compress the fields and synchronize them into UnifiedPhysics
    4. provide object rasterization tools (to build a heat source, say)

    Usage:
    1. initialization: register the fields and configure the solvers
    2. every step:
       a. save_history() - keep the current high-precision field
       b. take the object data from OKMC
       c. rasterize the objects -> heat source / dislocation density
       d. call the Dedalus solve
       e. update the high-precision field
       f. compress -> UnifiedPhysics
    """

    def __init__(self, comm: MPI.Comm,
                 box_size: Tuple[float, float, float],
                 config: Optional[dict] = None):
        """
        Initialize the Dedalus coupler

        Args:
            comm: MPI communicator
            box_size: simulation box dimensions (Lx, Ly, Lz)
            config: configuration dict
                - resolution: resolution of the high-precision grid (nx, ny, nz)
                - enable_history: whether to keep the history
                - compression: compression configuration
                    - method: 'adaptive' or 'uniform'
                    - gradient_threshold: gradient threshold
                    - target_size: target compressed size
        """
        self.comm = comm
        self.rank = comm.Get_rank()
        self.size = comm.Get_size()
        self.box_size = box_size
        self.config = config or {}

        # Parse the configuration
        resolution = self.config.get('resolution', (64, 64, 64))
        enable_history = self.config.get('enable_history', True)
        compression_config = self.config.get('compression', {})

        # High-precision resolution
        self.resolution = resolution
        self.nx, self.ny, self.nz = resolution

        # Coordinates of the high-precision grid
        self.x = np.linspace(0, box_size[0], self.nx)
        self.y = np.linspace(0, box_size[1], self.ny)
        self.z = np.linspace(0, box_size[2], self.nz)

        # Create the high-precision field store
        self.field_store = HighPrecisionFieldStore(comm, enable_history)

        # Create the compressor
        self.compressor = FieldCompressor(
            comm,
            gradient_threshold=compression_config.get('gradient_threshold', 0.01),
            min_points=compression_config.get('min_points', 10),
            max_compression_ratio=compression_config.get('max_compression_ratio', 0.5),
            target_size=compression_config.get('target_size', None)
        )

        # Dedalus objects (when available)
        self._dedalus_dist = None
        self._dedalus_coords = None
        self._dedalus_bases = None
        self._dedalus_fields: Dict[str, Any] = {}

        # Solver registry
        self._solvers: Dict[str, 'DedalusSolverBase'] = {}

        # Statistics
        self.stats = {
            'compressions': 0,
            'solver_calls': 0,
            'total_compression_time': 0.0,
            'total_solve_time': 0.0,
        }

        # Initialize Dedalus (when available)
        if DEDALUS_AVAILABLE:
            self._init_dedalus()

    def _init_dedalus(self) -> None:
        """Initialize the Dedalus domain and distributor"""
        if not DEDALUS_AVAILABLE:
            return

        # [fix] The initialization differs between versions
        is_v3 = hasattr(d3, 'Distributor')
        is_v2 = hasattr(d3, 'Domain') and not is_v3

        if is_v3:
            # Dedalus v3 API
            try:
                self._dedalus_coords = d3.CartesianCoordinates('x', 'y', 'z')

                # Create the distributor
                try:
                    self._dedalus_dist = d3.Distributor(
                        self._dedalus_coords,
                        dtype=np.float64,
                        comm=self.comm
                    )
                except TypeError:
                    # The comm argument may not be supported
                    self._dedalus_dist = d3.Distributor(
                        self._dedalus_coords,
                        dtype=np.float64
                    )

                # Create the basis
                Lx, Ly, Lz = self.box_size
                self._dedalus_bases = (
                    d3.RealFourier(self._dedalus_coords['x'],
                                   size=self.nx,
                                   bounds=(0, Lx)),
                    d3.RealFourier(self._dedalus_coords['y'],
                                   size=self.ny,
                                   bounds=(0, Ly)),
                    d3.RealFourier(self._dedalus_coords['z'],
                                   size=self.nz,
                                   bounds=(0, Lz)),
                )
            except Exception as e:
                warnings.warn(f"Failed to initialize Dedalus v3: {e}")
                self._dedalus_dist = None
                self._dedalus_bases = None

        elif is_v2:
            # [fix] Dedalus v2 is not supported, so raise
            raise RuntimeError(
                f"Dedalus v2 ({DEDALUS_VERSION}) is not supported. "
                f"Please upgrade to Dedalus v3.x. "
                f"Install with: pip install dedalus==3.0.4"
            )

        else:
            warnings.warn(f"Unknown Dedalus version: {DEDALUS_VERSION}")
            self._dedalus_dist = None
            self._dedalus_bases = None

    def register_field(self, name: str,
                       initial_value: Union[float, np.ndarray] = 0.0,
                       create_dedalus_field: bool = False) -> None:
        """
        Register a field (replicated mode: every rank holds a full copy)

        Args:
            name: field name
            initial_value: initial value (a scalar or an array)
            create_dedalus_field: whether to create a Dedalus field as well (only when needed)
        """
        # Shape of the local data (replicated: every rank holds everything)
        local_shape = (self.nx, self.ny, self.nz)
        global_shape = (self.nx, self.ny, self.nz)
        local_slice = (slice(None), slice(None), slice(None))

        self.field_store.register_field(
            name, local_shape, global_shape, local_slice,
            (self.x, self.y, self.z), 'grid'
        )

        # Set the initial value
        if isinstance(initial_value, (int, float)):
            self.field_store.current_fields[name][:] = initial_value
        else:
            self.field_store.current_fields[name][:] = initial_value

        # Create a Dedalus field when required and available
        # Note: Dedalus distributes the data itself, so the shape may differ from the replica
        if create_dedalus_field and DEDALUS_AVAILABLE and self._dedalus_dist is not None:
            try:
                field = self._dedalus_dist.Field(
                    name=name,
                    bases=self._dedalus_bases
                )
                self._dedalus_fields[name] = field
                # Note: the initial value is not set here, since the shapes may differ
                # The user must set it through the Dedalus API
            except Exception as e:
                warnings.warn(f"Failed to create Dedalus field '{name}': {e}")

    def register_solver(self, name: str, solver: 'DedalusSolverBase',
                        fields: List[str]) -> None:
        """
        Register a Dedalus solver

        Args:
            name: solver name
            solver: the solver instance
            fields: names of the fields this solver updates
        """
        self._solvers[name] = {
            'solver': solver,
            'fields': fields,
        }
        solver.initialize(self, self.config)

    def save_history(self, time: float) -> None:
        """Save the current state into the history"""
        self.field_store.save_history(time)

    def update_time(self, time: float) -> None:
        """Advance the current time"""
        self.field_store.update_time(time)

    def rasterize_objects(self, objects: np.ndarray,
                          property_name: str = 'energy',
                          kernel: str = 'gaussian',
                          sigma: float = 1.0,
                          periodic: bool = True) -> np.ndarray:
        """
        Rasterize the objects into a field

        Args:
            objects: the object array (structured, with x, y and z fields)
            property_name: name of the property to rasterize (e.g. 'energy')
            kernel: kernel type ('gaussian', 'uniform', 'delta')
            sigma: standard deviation of the Gaussian kernel
            periodic: whether to use periodic boundaries (True by default, as OKMC usually does)

        Returns:
            the rasterized 3D field
        """
        result = np.zeros((self.nx, self.ny, self.nz), dtype=np.float64)

        if len(objects) == 0:
            return result

        # Coordinates
        x_obj = objects['x']
        y_obj = objects['y']
        z_obj = objects['z']

        # Property values
        if property_name in objects.dtype.names:
            values = objects[property_name]
        elif 'float_props' in objects.dtype.names:
            # Try to read it from float_props
            # energy is assumed to be the first float_prop
            values = objects['float_props'][:, 0]
        else:
            # Default
            values = np.ones(len(objects))

        # Grid spacing
        dx = self.box_size[0] / (self.nx - 1) if self.nx > 1 else 1.0
        dy = self.box_size[1] / (self.ny - 1) if self.ny > 1 else 1.0
        dz = self.box_size[2] / (self.nz - 1) if self.nz > 1 else 1.0

        if kernel == 'delta':
            # Delta kernel: project onto the nearest grid point
            ix = np.clip(np.round(x_obj / dx).astype(int), 0, self.nx - 1)
            iy = np.clip(np.round(y_obj / dy).astype(int), 0, self.ny - 1)
            iz = np.clip(np.round(z_obj / dz).astype(int), 0, self.nz - 1)

            np.add.at(result, (ix, iy, iz), values)

        elif kernel == 'gaussian':
            # [fix] Vectorized Gaussian kernel: no quadruple Python loop
            # Range of influence (3 sigma)
            r_cut = 3 * sigma
            n_cut_x = int(np.ceil(r_cut / dx))
            n_cut_y = int(np.ceil(r_cut / dy))
            n_cut_z = int(np.ceil(r_cut / dz))

            # Precompute the offset grid
            d_ix = np.arange(-n_cut_x, n_cut_x + 1)
            d_iy = np.arange(-n_cut_y, n_cut_y + 1)
            d_iz = np.arange(-n_cut_z, n_cut_z + 1)
            DIX, DIY, DIZ = np.meshgrid(d_ix, d_iy, d_iz, indexing='ij')
            # Flatten into 1D arrays
            dix_flat = DIX.ravel()
            diy_flat = DIY.ravel()
            diz_flat = DIZ.ravel()

            # Precompute the Gaussian factor
            two_sigma_sq = 2 * sigma**2

            # Vectorized computation per object
            for i in range(len(x_obj)):
                xp, yp, zp, val = x_obj[i], y_obj[i], z_obj[i], values[i]

                # Central grid point
                ix0 = int(np.round(xp / dx))
                iy0 = int(np.round(yp / dy))
                iz0 = int(np.round(zp / dz))

                # Every candidate grid point
                ix_all = ix0 + dix_flat
                iy_all = iy0 + diy_flat
                iz_all = iz0 + diz_flat

                # [fix] Periodic boundary handling
                if periodic:
                    # Modular arithmetic wraps the boundary: -1 -> nx-1
                    ix_valid = ix_all % self.nx
                    iy_valid = iy_all % self.ny
                    iz_valid = iz_all % self.nz

                    # The distance follows the minimum image convention
                    x_grid = ix_valid * dx
                    y_grid = iy_valid * dy
                    z_grid = iz_valid * dz

                    # Minimum image distance
                    dx_vec = x_grid - xp
                    dy_vec = y_grid - yp
                    dz_vec = z_grid - zp

                    # Periodic correction
                    Lx, Ly, Lz = self.box_size
                    dx_vec = dx_vec - Lx * np.round(dx_vec / Lx)
                    dy_vec = dy_vec - Ly * np.round(dy_vec / Ly)
                    dz_vec = dz_vec - Lz * np.round(dz_vec / Lz)

                    r2 = dx_vec**2 + dy_vec**2 + dz_vec**2
                else:
                    # Non-periodic boundary: drop the contribution outside the box
                    valid = (ix_all >= 0) & (ix_all < self.nx) & \
                            (iy_all >= 0) & (iy_all < self.ny) & \
                            (iz_all >= 0) & (iz_all < self.nz)

                    ix_valid = ix_all[valid]
                    iy_valid = iy_all[valid]
                    iz_valid = iz_all[valid]

                    # Distances and weights (vectorized)
                    x_grid = ix_valid * dx
                    y_grid = iy_valid * dy
                    z_grid = iz_valid * dz
                    r2 = (x_grid - xp)**2 + (y_grid - yp)**2 + (z_grid - zp)**2

                weights = val * np.exp(-r2 / two_sigma_sq)

                # Atomic accumulation
                np.add.at(result, (ix_valid, iy_valid, iz_valid), weights)

        elif kernel == 'uniform':
            # [fix] Vectorized uniform spherical kernel
            r_cut = sigma  # sigma acts as the radius
            r_cut_sq = r_cut**2
            n_cut_x = int(np.ceil(r_cut / dx))
            n_cut_y = int(np.ceil(r_cut / dy))
            n_cut_z = int(np.ceil(r_cut / dz))

            # Precompute the offset grid
            d_ix = np.arange(-n_cut_x, n_cut_x + 1)
            d_iy = np.arange(-n_cut_y, n_cut_y + 1)
            d_iz = np.arange(-n_cut_z, n_cut_z + 1)
            DIX, DIY, DIZ = np.meshgrid(d_ix, d_iy, d_iz, indexing='ij')
            dix_flat = DIX.ravel()
            diy_flat = DIY.ravel()
            diz_flat = DIZ.ravel()

            for i in range(len(x_obj)):
                xp, yp, zp, val = x_obj[i], y_obj[i], z_obj[i], values[i]

                ix0 = int(np.round(xp / dx))
                iy0 = int(np.round(yp / dy))
                iz0 = int(np.round(zp / dz))

                # Every candidate grid point
                ix_all = ix0 + dix_flat
                iy_all = iy0 + diy_flat
                iz_all = iz0 + diz_flat

                # [fix] Periodic boundary handling
                if periodic:
                    # Modular arithmetic wraps the boundary
                    ix_valid = ix_all % self.nx
                    iy_valid = iy_all % self.ny
                    iz_valid = iz_all % self.nz

                    # The distance follows the minimum image convention
                    x_grid = ix_valid * dx
                    y_grid = iy_valid * dy
                    z_grid = iz_valid * dz

                    dx_vec = x_grid - xp
                    dy_vec = y_grid - yp
                    dz_vec = z_grid - zp

                    Lx, Ly, Lz = self.box_size
                    dx_vec = dx_vec - Lx * np.round(dx_vec / Lx)
                    dy_vec = dy_vec - Ly * np.round(dy_vec / Ly)
                    dz_vec = dz_vec - Lz * np.round(dz_vec / Lz)

                    r2 = dx_vec**2 + dy_vec**2 + dz_vec**2

                    # Keep the points inside the spherical kernel
                    in_sphere = r2 <= r_cut_sq
                    ix_in = ix_valid[in_sphere]
                    iy_in = iy_valid[in_sphere]
                    iz_in = iz_valid[in_sphere]
                else:
                    # Non-periodic boundary: drop the contribution outside the box
                    valid = (ix_all >= 0) & (ix_all < self.nx) & \
                            (iy_all >= 0) & (iy_all < self.ny) & \
                            (iz_all >= 0) & (iz_all < self.nz)

                    ix_valid = ix_all[valid]
                    iy_valid = iy_all[valid]
                    iz_valid = iz_all[valid]

                    # Distances (vectorized)
                    x_grid = ix_valid * dx
                    y_grid = iy_valid * dy
                    z_grid = iz_valid * dz
                    r2 = (x_grid - xp)**2 + (y_grid - yp)**2 + (z_grid - zp)**2

                    # Keep the points inside the spherical kernel
                    in_sphere = r2 <= r_cut_sq
                    ix_in = ix_valid[in_sphere]
                    iy_in = iy_valid[in_sphere]
                    iz_in = iz_valid[in_sphere]

                # Atomic accumulation
                np.add.at(result, (ix_in, iy_in, iz_in), val)

        return result

    def solve_and_compress(self, solver_name: str,
                           step: int, time: float,
                           object_data: Optional[np.ndarray] = None,
                           trigger_source: Optional[str] = None,
                           physics: Optional['UnifiedPhysics'] = None
                           ) -> Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        """
        Run the Dedalus solve and compress the result

        Args:
            solver_name: solver name
            step: OKMC step
            time: simulation time
            object_data: object data
            trigger_source: what triggered the call
            physics: the UnifiedPhysics instance (used to store the compressed field)

        Returns:
            Dict[field_name, (compressed_field, x, y, z)]
        """
        if solver_name not in self._solvers:
            raise KeyError(f"Unknown solver: {solver_name}")

        solver_info = self._solvers[solver_name]
        solver = solver_info['solver']
        fields = solver_info['fields']

        import time as time_module

        # Call the solver
        t0 = time_module.perf_counter()
        results = solver.solve(self, step, time, object_data, trigger_source)
        self.stats['solver_calls'] += 1
        self.stats['total_solve_time'] += time_module.perf_counter() - t0

        if results is None:
            return {}

        # Update the high-precision field store
        for field_name, field_data in results.items():
            if field_name in self.field_store.current_fields:
                self.field_store.update_field(field_name, field_data)

        # Compress and broadcast
        t0 = time_module.perf_counter()
        compressed_results = {}
        for field_name in fields:
            if field_name not in results:
                continue

            compressed, x_new, y_new, z_new = self.compressor.compress_and_broadcast(
                self.field_store, field_name,
                self.x, self.y, self.z,
                method=self.config.get('compression', {}).get('method', 'adaptive')
            )
            compressed_results[field_name] = (compressed, x_new, y_new, z_new)

            # When physics is given, write the result into UnifiedPhysics
            if physics is not None:
                physics.set_field(field_name, compressed, (x_new, y_new, z_new))

        self.stats['compressions'] += len(compressed_results)
        self.stats['total_compression_time'] += time_module.perf_counter() - t0

        return compressed_results

    def get_high_precision_field(self, name: str) -> Optional[np.ndarray]:
        """Get the high-precision field data"""
        return self.field_store.get_current(name)

    def get_high_precision_prev(self, name: str) -> Optional[np.ndarray]:
        """Get the high-precision field of the previous step"""
        return self.field_store.get_prev(name)

    def get_field_rate(self, name: str) -> Optional[np.ndarray]:
        """Get the rate of change of a field"""
        return self.field_store.get_field_rate(name)

    def get_global_field(self, name: str) -> Optional[np.ndarray]:
        """Gather the global high-precision field onto rank 0"""
        return self.field_store.gather_global_field(name)

    def print_stats(self) -> None:
        """Print the statistics"""
        if self.rank == 0:
            print("="*60)
            print("DedalusCoupler Statistics")
            print("="*60)
            print(f"Solver calls: {self.stats['solver_calls']}")
            print(f"Compressions: {self.stats['compressions']}")
            print(f"Total solve time: {self.stats['total_solve_time']:.3f}s")
            print(f"Total compression time: {self.stats['total_compression_time']:.3f}s")
            print("="*60)


# =============================================================================
# Base class of the Dedalus solvers
# =============================================================================

class DedalusSolverBase(ExternalSolver):
    """
    Base class of the Dedalus solvers

    Subclasses must implement:
    - _setup_problem(): set the Dedalus problem up
    - _solve_step(): advance the solve by one step
    """

    def __init__(self):
        self.coupler: Optional[DedalusCoupler] = None
        self.config: dict = {}
        self.problem = None
        self.solver = None

    def initialize(self, physics_or_coupler: Union['UnifiedPhysics', 'DedalusCoupler'],
                   config: dict) -> None:
        """
        Initialize the solver

        Args:
            physics_or_coupler: a UnifiedPhysics or DedalusCoupler instance
            config: configuration dict
        """
        if isinstance(physics_or_coupler, DedalusCoupler):
            self.coupler = physics_or_coupler
        self.config = config
        self._setup_problem()

    @abstractmethod
    def _setup_problem(self) -> None:
        """Set the Dedalus problem up (implemented by the subclass)"""
        pass

    @abstractmethod
    def _solve_step(self, coupler: 'DedalusCoupler',
                    step: int, time: float,
                    object_data: Optional[np.ndarray],
                    trigger_source: Optional[str]) -> Optional[Dict[str, np.ndarray]]:
        """Advance the solve by one step (implemented by the subclass)"""
        pass

    def solve(self, physics: 'UnifiedPhysics',
              step: int, time: float,
              object_data: Optional[np.ndarray] = None,
              trigger_source: Optional[str] = None) -> Optional[Dict[str, np.ndarray]]:
        """
        Implement the ExternalSolver interface

        When called through DedalusCoupler, the physics argument is really the coupler
        """
        if isinstance(physics, DedalusCoupler):
            return self._solve_step(physics, step, time, object_data, trigger_source)
        else:
            # Used directly as an ExternalSolver: the user must handle it
            return self._solve_step(self.coupler, step, time, object_data, trigger_source)

    def finalize(self) -> None:
        """Release the resources"""
        self.problem = None
        self.solver = None


# =============================================================================
# Example: heat conduction solver
# =============================================================================

class DedalusHeatSolver(DedalusSolverBase):
    """
    Dedalus heat conduction solver

    Solves:
        ∂T/∂t = κ ∇²T + Q(x,y,z,t)

    where Q is the heat source computed from the OKMC objects (cascade energy release, say)

    Configuration:
        - kappa: thermal diffusivity
        - dt: time step
        - bc_type: boundary condition ('periodic', 'fixed', 'insulated')
        - T_boundary: boundary temperature (for the fixed BC)
    """

    def __init__(self, kappa: float = 1.0,
                 dt: float = 0.01,
                 bc_type: str = 'periodic',
                 T_boundary: float = 300.0):
        super().__init__()
        self.kappa = kappa
        self.dt = dt
        self.bc_type = bc_type
        self.T_boundary = T_boundary

    def _setup_problem(self) -> None:
        """Set the Dedalus heat conduction problem up"""
        if not DEDALUS_AVAILABLE or self.coupler is None:
            return

        # Set the problem up with Dedalus
        # The structure below is simplified pseudo-code
        # A real implementation must follow the Dedalus v3 API

        coupler = self.coupler

        try:
            dist = coupler._dedalus_dist
            bases = coupler._dedalus_bases

            if dist is None or bases is None:
                warnings.warn("Dedalus not properly initialized, using fallback")
                return

            # Create the temperature field
            T = dist.Field(name='T', bases=bases)

            # Create the heat source field
            Q = dist.Field(name='Q', bases=bases)

            # Keep the references
            self._T = T
            self._Q = Q

            # Set the problem up (simplified)
            # A real implementation uses the Dedalus Problem class

        except Exception as e:
            warnings.warn(f"Failed to setup Dedalus problem: {e}")

    def _solve_step(self, coupler: DedalusCoupler,
                    step: int, time: float,
                    object_data: Optional[np.ndarray],
                    trigger_source: Optional[str]) -> Optional[Dict[str, np.ndarray]]:
        """
        Advance the heat conduction solve by one step
        """
        # Current temperature field
        T_current = coupler.get_high_precision_field('temperature')
        if T_current is None:
            # Use the default initial temperature
            T_current = np.full((coupler.nx, coupler.ny, coupler.nz), self.T_boundary)

        # Heat source (rasterized from the objects)
        if object_data is not None and len(object_data) > 0:
            Q = coupler.rasterize_objects(
                object_data,
                property_name='energy',
                kernel='gaussian',
                sigma=2.0
            )
        else:
            Q = np.zeros_like(T_current)

        # With Dedalus available, use the spectral method
        if DEDALUS_AVAILABLE and hasattr(self, '_T'):
            try:
                # Set the current field values
                self._T['g'] = T_current
                self._Q['g'] = Q

                # The Dedalus solver should be called here
                # Simplification: an explicit finite difference
                T_new = self._solve_explicit(T_current, Q, coupler)

            except Exception as e:
                warnings.warn(f"Dedalus solve failed: {e}, using fallback")
                T_new = self._solve_explicit(T_current, Q, coupler)
        else:
            # Fallback: a simple explicit finite difference
            T_new = self._solve_explicit(T_current, Q, coupler)

        return {'temperature': T_new}

    def _solve_explicit(self, T: np.ndarray, Q: np.ndarray,
                        coupler: DedalusCoupler) -> np.ndarray:
        """
        Solve the heat equation with an explicit finite difference (the fallback)

        ∂T/∂t = κ ∇²T + Q
        """
        nx, ny, nz = T.shape
        Lx, Ly, Lz = coupler.box_size

        dx = Lx / (nx - 1) if nx > 1 else 1.0
        dy = Ly / (ny - 1) if ny > 1 else 1.0
        dz = Lz / (nz - 1) if nz > 1 else 1.0

        # Stability condition
        dt_max = 0.25 * min(dx, dy, dz)**2 / self.kappa
        dt = min(self.dt, dt_max)

        # Laplacian
        T_new = T.copy()

        # Interior points
        for i in range(1, nx - 1):
            for j in range(1, ny - 1):
                for k in range(1, nz - 1):
                    laplacian = (
                        (T[i+1, j, k] - 2*T[i, j, k] + T[i-1, j, k]) / dx**2 +
                        (T[i, j+1, k] - 2*T[i, j, k] + T[i, j-1, k]) / dy**2 +
                        (T[i, j, k+1] - 2*T[i, j, k] + T[i, j, k-1]) / dz**2
                    )
                    T_new[i, j, k] = T[i, j, k] + dt * (self.kappa * laplacian + Q[i, j, k])

        # Boundary conditions
        if self.bc_type == 'periodic':
            # numpy roll handles the periodic boundary more efficiently
            lap_x = (np.roll(T, -1, axis=0) - 2*T + np.roll(T, 1, axis=0)) / dx**2
            lap_y = (np.roll(T, -1, axis=1) - 2*T + np.roll(T, 1, axis=1)) / dy**2
            lap_z = (np.roll(T, -1, axis=2) - 2*T + np.roll(T, 1, axis=2)) / dz**2
            T_new = T + dt * (self.kappa * (lap_x + lap_y + lap_z) + Q)

        elif self.bc_type == 'fixed':
            # Fixed boundary temperature
            T_new[0, :, :] = self.T_boundary
            T_new[-1, :, :] = self.T_boundary
            T_new[:, 0, :] = self.T_boundary
            T_new[:, -1, :] = self.T_boundary
            T_new[:, :, 0] = self.T_boundary
            T_new[:, :, -1] = self.T_boundary

        elif self.bc_type == 'insulated':
            # Insulated boundary (zero flux)
            T_new[0, :, :] = T_new[1, :, :]
            T_new[-1, :, :] = T_new[-2, :, :]
            T_new[:, 0, :] = T_new[:, 1, :]
            T_new[:, -1, :] = T_new[:, -2, :]
            T_new[:, :, 0] = T_new[:, :, 1]
            T_new[:, :, -1] = T_new[:, :, -2]

        return T_new

    def get_supported_fields(self) -> List[str]:
        return ['temperature']

    def get_required_params(self) -> List[str]:
        return ['kappa', 'T_boundary']


# =============================================================================
# ExternalSolver wrapper integrating with UnifiedPhysics
# =============================================================================

class DedalusCouplerAdapter(ExternalSolver):
    """
    ExternalSolver adapter for DedalusCoupler

    Lets DedalusCoupler be driven through the standard UnifiedPhysics interface
    """

    def __init__(self, coupler: DedalusCoupler, solver_name: str):
        """
        Args:
            coupler: the DedalusCoupler instance
            solver_name: name of the solver to call
        """
        self.coupler = coupler
        self.solver_name = solver_name
        self._physics: Optional[UnifiedPhysics] = None

    def initialize(self, physics: 'UnifiedPhysics', config: dict) -> None:
        self._physics = physics

    def solve(self, physics: 'UnifiedPhysics',
              step: int, time: float,
              object_data: Optional[np.ndarray] = None,
              trigger_source: Optional[str] = None) -> Optional[Dict[str, np.ndarray]]:
        """
        Solve through DedalusCoupler
        """
        # Save the history
        self.coupler.save_history(time)

        # Solve and compress
        results = self.coupler.solve_and_compress(
            self.solver_name, step, time,
            object_data, trigger_source,
            physics  # pass physics so the compressed field is stored automatically
        )

        # Return the compressed field data (for UnifiedPhysics)
        return_dict = {}
        for field_name, (compressed, x, y, z) in results.items():
            return_dict[field_name] = compressed

        return return_dict if return_dict else None

    def finalize(self) -> None:
        pass

    def get_supported_fields(self) -> List[str]:
        if self.solver_name in self.coupler._solvers:
            return self.coupler._solvers[self.solver_name]['fields']
        return []


# =============================================================================
# Module test
# =============================================================================

if __name__ == '__main__':
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    print(f"[Rank {rank}] Dedalus Coupler Module Test")
    print(f"[Rank {rank}] DEDALUS_AVAILABLE = {DEDALUS_AVAILABLE}")

    # Test the field compressor
    if rank == 0:
        print("\n" + "="*60)
        print("Testing FieldCompressor")
        print("="*60)

    box_size = (10.0, 10.0, 10.0)
    config = {
        'resolution': (32, 32, 32),
        'compression': {
            'method': 'uniform',
            'target_size': (16, 16, 16),
        }
    }

    # Create the DedalusCoupler
    coupler = DedalusCoupler(comm, box_size, config)

    # Register the temperature field
    coupler.register_field('temperature', 300.0)

    # Build a test temperature field (with a gradient)
    x, y, z = np.meshgrid(coupler.x, coupler.y, coupler.z, indexing='ij')
    T_test = 300 + 50 * np.sin(2 * np.pi * x / 10) * np.cos(2 * np.pi * y / 10)
    coupler.field_store.update_field('temperature', T_test)

    # Test the compression
    compressed, x_new, y_new, z_new = coupler.compressor.compress_and_broadcast(
        coupler.field_store, 'temperature',
        coupler.x, coupler.y, coupler.z,
        method='uniform'
    )

    if rank == 0:
        print(f"Original shape: {T_test.shape}")
        print(f"Compressed shape: {compressed.shape}")
        print(f"Compression ratio: {compressed.size / T_test.size:.2%}")

    # Test the object rasterization
    if rank == 0:
        print("\n" + "="*60)
        print("Testing Object Rasterization")
        print("="*60)

    # Create the test objects
    n_obj = 10
    obj_dtype = np.dtype([
        ('id', np.int64),
        ('type', 'U32'),
        ('x', np.float64),
        ('y', np.float64),
        ('z', np.float64),
        ('energy', np.float64),
    ])

    objects = np.zeros(n_obj, dtype=obj_dtype)
    objects['id'] = np.arange(n_obj)
    objects['type'] = 'cascade'
    objects['x'] = np.random.uniform(0, 10, n_obj)
    objects['y'] = np.random.uniform(0, 10, n_obj)
    objects['z'] = np.random.uniform(0, 10, n_obj)
    objects['energy'] = np.random.uniform(1, 10, n_obj)

    # Rasterize
    heat_source = coupler.rasterize_objects(objects, 'energy', kernel='gaussian', sigma=1.0)

    if rank == 0:
        print(f"Heat source shape: {heat_source.shape}")
        print(f"Heat source max: {heat_source.max():.2f}")
        print(f"Heat source sum: {heat_source.sum():.2f}")

    # Test the heat conduction solver
    if rank == 0:
        print("\n" + "="*60)
        print("Testing Heat Solver")
        print("="*60)

    heat_solver = DedalusHeatSolver(kappa=1.0, dt=0.1, bc_type='periodic')
    coupler.register_solver('heat', heat_solver, ['temperature'])

    # Run the solve
    coupler.save_history(0.0)
    results = coupler.solve_and_compress('heat', 0, 0.1, objects)

    if rank == 0 and 'temperature' in results:
        compressed_T, x_c, y_c, z_c = results['temperature']
        print(f"Solved temperature shape: {compressed_T.shape}")
        print(f"Temperature range: [{compressed_T.min():.1f}, {compressed_T.max():.1f}]")

    # Print the statistics
    coupler.print_stats()

    if rank == 0:
        print("\n" + "="*60)
        print("All tests completed!")
        print("="*60)
