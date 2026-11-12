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
Unified physics module

Key improvements:
1. Parameters (constant physical quantities) support three coupling modes
   - TWO_WAY: objects -> parameter collector -> external solver -> parameter update
   - OUTWARD: objects -> parameter collector -> external solver (no update)
   - INWARD: external solver -> parameter update (independent of the objects)
2. Parameter collectors (computing a parameter value from the objects)
3. An extended external solver interface (which may return parameters)

Inherited features:
- the external field coupler is folded in
- three coupling modes: TWO_WAY, OUTWARD, INWARD
- event-triggered field updates
- field compression
- an abstract external solver interface
- physics can be modified while an event executes

Data structures:
- params: 1D float array holding every scalar parameter
- fields: list of 3D arrays holding every discretized spatial field
- field_axes: axis definition of each field [(x_arr, y_arr, z_arr), ...]

Coupling modes (available to both fields and parameters):
- TWO_WAY:  objects -> field/parameter -> rate (bidirectional coupling)
- OUTWARD: objects -> field/parameter (output only, e.g. VTK visualization)
- INWARD:  field/parameter -> rate (input only, e.g. a time-dependent boundary condition)

"""

import inspect
import numpy as np
from mpi4py import MPI
from typing import Dict, List, Tuple, Optional, Callable, Union, Set, Any
from scipy.interpolate import RegularGridInterpolator
from abc import ABC, abstractmethod
from functools import lru_cache
import warnings

try:
    from numba import njit
except ImportError:
    def njit(*args, **kwargs):
        if args and callable(args[0]):
            return args[0]

        def wrapper(fn):
            return fn
        return wrapper


# ===== Coupling mode constants =====
MODE_TWO_WAY = 'two_way'   # bidirectional: objects -> field -> rate
MODE_OUTWARD = 'outward'   # outward: objects -> field (nothing comes back)
MODE_INWARD  = 'inward'    # inward: field -> rate (independent of the objects)


@njit(cache=True)
def _match_type_ids_jit(type_ids: np.ndarray, allowed_ids: np.ndarray) -> np.ndarray:
    """JIT type filter: returns the boolean mask of the matching type_ids."""
    mask = np.zeros(len(type_ids), dtype=np.bool_)
    for i in range(len(type_ids)):
        tid = type_ids[i]
        for j in range(len(allowed_ids)):
            if tid == allowed_ids[j]:
                mask[i] = True
                break
    return mask


class PhysicsUpdateContext:
    """Execution context handed to the update_func of a field."""

    def __init__(self,
                 comm: MPI.Comm,
                 step: int,
                 time: float,
                 field_name: Optional[str] = None,
                 trigger_source: Optional[str] = None,
                 trigger_reason: Optional[str] = None,
                 changed_summary: Optional[dict] = None):
        self.comm = comm
        self.rank = comm.rank
        self.size = comm.size
        self.step = step
        self.time = time
        self.field_name = field_name
        self.trigger_source = trigger_source
        self.trigger_reason = trigger_reason
        self.changed_summary = changed_summary or {}


class ObjectQueryEngine:
    """
    Object query and snapshot service for the field updaters.

    The user only asks for 'every object' or 'objects of these types';
    this class filters locally and forms the global snapshot with Allgatherv.
    """

    def __init__(self, comm: MPI.Comm, rma_object_pool):
        self.comm = comm
        self.rank = comm.rank
        self.size = comm.size
        self.pool = rma_object_pool
        self.dtype = rma_object_pool.dtype

    def _allgatherv_zero_copy(self, local_objs: np.ndarray) -> np.ndarray:
        """Serialization-free Allgatherv, giving every rank the same global object snapshot."""
        local_count = len(local_objs)
        all_counts = self.comm.allgather(local_count)
        total_count = sum(all_counts)
        if total_count == 0:
            return np.array([], dtype=self.dtype)

        itemsize = self.dtype.itemsize
        byte_counts = [c * itemsize for c in all_counts]
        byte_displs = [sum(byte_counts[:i]) for i in range(self.size)]

        recv_buf = np.zeros(total_count, dtype=self.dtype)
        send_buf = np.ascontiguousarray(local_objs)
        send_count = local_count * itemsize

        self.comm.Allgatherv(
            [send_buf.view(np.uint8), send_count, MPI.BYTE],
            [recv_buf.view(np.uint8), byte_counts, byte_displs, MPI.BYTE]
        )
        return recv_buf

    def _filter_local_objects(self, local_objs: np.ndarray, query_spec: dict) -> np.ndarray:
        """Filter locally. The first version supports only all=True and types=[...]."""
        if not query_spec or query_spec.get('all', False):
            return local_objs

        if 'types' in query_spec:
            type_names = query_spec['types']
            if len(local_objs) == 0:
                return local_objs

            if 'type_id' in local_objs.dtype.names and hasattr(self.pool, '_type_to_int'):
                allowed_ids = np.array(
                    [self.pool._type_to_int[t] for t in type_names if t in self.pool._type_to_int],
                    dtype=np.int32
                )
                if len(allowed_ids) == 0:
                    return np.array([], dtype=self.dtype)
                mask = _match_type_ids_jit(local_objs['type_id'], allowed_ids)
                return local_objs[mask]

            # Fallback: without type_id, filter in Python
            filtered = [obj for obj in local_objs if self.pool.get_type_name(obj) in type_names]
            return np.array(filtered, dtype=self.dtype) if filtered else np.array([], dtype=self.dtype)

        return local_objs

    def build_snapshots(self, query_specs: Dict[str, dict]) -> Dict[str, np.ndarray]:
        """
        Build the object snapshot of a set of named queries.

        Returns:
            {query_name: global_snapshot_array}
        """
        if not query_specs:
            return {}

        local_objs = self.pool.get_local_objects(active_only=True)
        snapshots = {}
        for query_name, query_spec in query_specs.items():
            local_filtered = self._filter_local_objects(local_objs, query_spec or {})
            snapshots[query_name] = self._allgatherv_zero_copy(local_filtered)
        return snapshots


# ===== Abstract external solver interface =====

class ExternalSolver(ABC):
    """
    Abstract base class of an external solver

    Users subclass it to implement a concrete solver (Dedalus, FEniCSX, ...)

    Example 1 (a Dedalus heat conduction solver returning a field):
        class DedalusHeatSolver(ExternalSolver):
            def __init__(self, domain_size, resolution):
                self.problem = self._setup_dedalus_problem(domain_size, resolution)

            def solve(self, physics, step, time, object_data, trigger_source=None):
                # read the heat source distribution from physics
                heat_source = self._compute_heat_source(physics)

                # solve the heat equation
                T_field = self.problem.solve(heat_source, time)

                return {'temperature': T_field}

    Example 2 (a FEniCSX stress solver returning a field):
        class FEniCSXStressSolver(ExternalSolver):
            def __init__(self, mesh, material_props):
                self.mesh = mesh
                self.material = material_props

            def solve(self, physics, step, time, object_data, trigger_source=None):
                # read the defect distribution from physics
                defect_density = self._get_defect_density(physics)

                # solve the elasticity equations
                stress = self.solver.solve(defect_density)

                return {'stress_xx': stress[0], 'stress_yy': stress[1], ...}

    Example 3 (returning both a field and a parameter):
        class ThermalSolver(ExternalSolver):
            def solve(self, physics, step, time, object_data, trigger_source=None):
                # compute the temperature field
                T_field = self._solve_heat_equation(object_data)

                # compute the mean temperature (a scalar parameter)
                avg_temp = np.mean(T_field)

                # both may be returned together
                return {
                    'temperature': T_field,      # a field (np.ndarray)
                    'avg_temperature': avg_temp  # a parameter (float)
                }
    """

    @abstractmethod
    def initialize(self, physics: 'UnifiedPhysics', config: dict) -> None:
        """
        Initialize the solver

        Called before the OKMC simulation starts, to:
        - create the mesh/domain
        - set the boundary conditions
        - preallocate memory

        Args:
            physics: the unified physics module instance
            config: configuration dict
        """
        pass

    @abstractmethod
    def solve(self, physics: 'UnifiedPhysics',
              step: int,
              time: float,
              object_data: Optional[np.ndarray] = None,
              trigger_source: Optional[str] = None) -> Optional[Dict[str, Union[np.ndarray, float, int]]]:
        """
        Solve and return the updated fields and/or parameters

        Args:
            physics: the unified physics module (parameters and fields are readable)
            step: current OKMC step
            time: current simulation time
            object_data: object data array (TWO_WAY/OUTWARD modes)
            trigger_source: what triggered the call ('cascade', 'precipitation', ...);
                           None means an ordinary update

        Returns:
            a data dict supporting two kinds of entry:
            - field data: {'field_name': np.ndarray, ...}
            - scalar parameters: {'param_name': float, ...}
            they may be mixed: {'temperature': np.ndarray, 'avg_temp': 300.5}
            None means nothing is updated

        Note:
            - every MPI rank may call this simultaneously (parallel solvers are supported)
            - only the return value of rank 0 is used and then broadcast to every rank
        """
        pass

    @abstractmethod
    def finalize(self) -> None:
        """
        Release the resources

        Called when the OKMC simulation ends
        """
        pass

    def get_supported_fields(self) -> List[str]:
        """
        Return the names of the fields this solver can update

        The default returns an empty list; subclasses may override it
        """
        return []

    def get_required_params(self) -> List[str]:
        """
        Return the physics parameters this solver needs

        The default returns an empty list; subclasses may override it
        """
        return []

    def get_supported_params(self) -> List[str]:
        """
        Return the scalar parameters this solver can update

        The default returns an empty list; subclasses may override it
        """
        return []


# ===== Interpolation cache =====

class FieldCache:
    """
    Interpolation cache

    Cuts the cost of repeated queries: for a static object the field value at one
    """

    def __init__(self, precision: int = 2, max_size: int = 10000):
        """
        Initialize the cache

        Args:
            precision: coordinate precision (decimal places kept)
            max_size: maximum number of cache entries
        """
        self.precision = precision
        self.max_size = max_size
        self.cache: Dict[Tuple, Any] = {}
        self.hits = 0
        self.misses = 0

    def _make_key(self, field_name: str, x: float, y: float, z: float) -> Tuple:
        """Build the cache key (rounded coordinates)"""
        return (
            field_name,
            round(x, self.precision),
            round(y, self.precision),
            round(z, self.precision)
        )

    def get(self, field_name: str, field_func: Callable, x: float, y: float, z: float):
        """Get a field value (through the cache)"""
        key = self._make_key(field_name, x, y, z)

        if key in self.cache:
            self.hits += 1
            return self.cache[key]

        # Cache miss: compute the field value
        value = field_func(x, y, z)
        self.misses += 1

        # Cache management
        if len(self.cache) < self.max_size:
            self.cache[key] = value
        else:
            # FIFO policy
            first_key = next(iter(self.cache))
            del self.cache[first_key]
            self.cache[key] = value

        return value

    def clear(self):
        """Empty the cache"""
        self.cache.clear()
        self.hits = 0
        self.misses = 0

    def get_stats(self) -> Dict[str, Any]:
        """Get the cache statistics"""
        total = self.hits + self.misses
        hit_rate = self.hits / total if total > 0 else 0.0
        return {
            'hits': self.hits,
            'misses': self.misses,
            'total': total,
            'hit_rate': hit_rate,
            'size': len(self.cache),
        }


# ===== Unified physics module =====

class UnifiedPhysics:
    """
    Unified physics module

    Brings physics parameters, spatial fields and external coupling together

    Core features:
    1. physics parameter management (temperature, pressure and other scalars)
    2. spatial field management (stress, concentration and other 3D data)
    3. external field coupling (interaction with an external solver)
    4. event-triggered updates (a given event forces a field update)
    5. field compression (adaptive mesh compression)
    """

    def __init__(self, comm: MPI.Comm, config: dict):
        """
        Initialize the physics module

        Args:
            comm: MPI communicator
            config: configuration dict
        """
        self.comm = comm
        self.rank = comm.rank
        self.size = comm.size
        self.config = config

        # Physical domain configuration
        self.box_size = np.array(config.get('box_size', [100.0, 100.0, 100.0]))

        # ===== Parameter management =====
        self._param_name_to_idx: Dict[str, int] = {}
        self._param_idx_to_name: Dict[int, str] = {}

        num_params = config.get('num_params', 64)
        self.params = np.zeros(num_params, dtype=np.float64)
        self._param_count = 0

        # Predefined parameters
        for name in config.get('param_names', []):
            self.define_param(name)

        # ===== Field management =====
        self._field_name_to_idx: Dict[str, int] = {}
        self._field_idx_to_name: Dict[int, str] = {}
        self.fields: List[np.ndarray] = []
        self.field_axes: List[Tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        self._field_interpolators: Dict[int, RegularGridInterpolator] = {}

        # Predefined fields
        for i, shape in enumerate(config.get('field_shapes', [])):
            name = config.get('field_names', [])[i] if i < len(config.get('field_names', [])) else f'field_{i}'
            self.define_field(name, shape)

        # ===== External field coupling =====
        coupling_config = config.get('field_coupling', {})
        self.coupling_mode = coupling_config.get('mode', MODE_INWARD)
        self.distributed_mode = coupling_config.get('distributed_mode', False)

        # External solvers
        self._external_solvers: Dict[str, ExternalSolver] = {}
        self._solver_to_fields: Dict[str, List[str]] = {}  # solver_name -> [field_names]
        self._solver_to_params: Dict[str, List[str]] = {}  # solver_name -> [param_names]

        # ===== Parameter collectors =====
        # Compute a parameter value from the objects (TWO_WAY/OUTWARD modes)
        # Format: {param_name: collector_func}
        # collector_func(objects: np.ndarray, physics: UnifiedPhysics) -> float
        self._param_collectors: Dict[str, Callable] = {}

        # Interpolation cache
        self.enable_cache = coupling_config.get('enable_cache', False)
        cache_precision = coupling_config.get('cache_precision', 2)
        cache_max_size = coupling_config.get('cache_max_size', 10000)
        self._field_cache = FieldCache(cache_precision, cache_max_size) if self.enable_cache else None

        # ===== History storage =====
        # Keeps the fields and parameters of the previous step so an external solver
        self.enable_history = coupling_config.get('enable_history', False)
        self._prev_params: Optional[np.ndarray] = None  # parameters of the previous step
        self._prev_fields: List[Optional[np.ndarray]] = []  # fields of the previous step
        self._prev_time: float = 0.0  # time of the previous step
        self._current_time: float = 0.0  # current time
        self._history_initialized: bool = False

        # ===== Event trigger mechanism =====
        # Format: {event_type: [field_names]} or {event_type: {'fields': [...], 'solver': 'name'}}
        self._event_triggers: Dict[str, Dict] = {}
        self._pending_triggered_fields: Set[str] = set()  # fields waiting to be updated
        self._pending_trigger_source: Optional[str] = None  # what triggered them
        self._pending_trigger_reason: Optional[str] = None
        self._pending_event_data: Optional[dict] = None

        # ===== Independent update control =====
        self._param_update_config: Dict[str, dict] = {}
        self._field_update_config: Dict[str, dict] = {}
        self._skip_scheduled_fields_once: Set[str] = set()

        # ===== Object query and snapshot service =====
        self._object_query_engine: Optional[ObjectQueryEngine] = None

        # ===== Lifecycle functions =====
        self.ic_func: Optional[Callable] = config.get('ic_func', None)
        self.update_func: Optional[Callable] = config.get('update_func', None)

        # ===== State tracking =====
        self._initialized = False
        self._last_update_step = -1
        self._params_changed = False
        self._fields_changed = False

        # ===== Statistics =====
        self.stats = {
            'total_updates': 0,
            'triggered_rebuilds': 0,
            'event_triggered_updates': 0,
            'field_compressions': 0,
        }

        # ===== Verbose =====
        self.verbose = config.get('verbose', False)

        # ===== Pending changes made while an event executes =====
        # Collects the physics changes made inside execute_func so they travel with changed_objects
        # Parameter changes: [(param_idx, new_value), ...]
        self._pending_param_changes: List[Tuple[int, float]] = []
        # Field deltas: [(field_idx, x, y, z, delta, radius), ...]
        # radius > 0 means a Gaussian distribution, radius = 0 the nearest grid point
        self._pending_field_deltas: List[Tuple[int, float, float, float, float, float]] = []
        # Whether we are inside an event execution context (only then may changes be collected)
        self._in_event_execution = False

    # =========================================================================
    # Parameter management interface
    # =========================================================================

    def define_param(self, name: str, initial_value: float = 0.0) -> int:
        """Declare a named parameter"""
        if name in self._param_name_to_idx:
            return self._param_name_to_idx[name]

        idx = self._param_count
        if idx >= len(self.params):
            # Grow automatically
            new_params = np.zeros(len(self.params) * 2, dtype=np.float64)
            new_params[:len(self.params)] = self.params
            self.params = new_params

        self._param_name_to_idx[name] = idx
        self._param_idx_to_name[idx] = name
        self.params[idx] = initial_value
        self._param_count += 1

        return idx

    def get_param(self, name_or_idx: Union[str, int]) -> float:
        """Get a parameter value"""
        if isinstance(name_or_idx, str):
            idx = self._param_name_to_idx.get(name_or_idx)
            if idx is None:
                raise KeyError(f"Unknown parameter: {name_or_idx}")
        else:
            idx = name_or_idx
        return float(self.params[idx])

    def get_param_name(self, idx: int) -> Optional[str]:
        """Name of the parameter at index idx (None if undefined)"""
        for name, i in self._param_name_to_idx.items():
            if i == idx:
                return name
        return None

    def get_param_names(self) -> List[str]:
        """Names of every defined parameter"""
        return list(self._param_name_to_idx.keys())

    def get_field_name(self, idx: int) -> Optional[str]:
        """Name of the field at index idx (None if undefined)"""
        for name, i in self._field_name_to_idx.items():
            if i == idx:
                return name
        return None

    def get_field_names(self) -> List[str]:
        """Names of every defined field"""
        return list(self._field_name_to_idx.keys())

    def get_pending_trigger_fields(self) -> Set[str]:
        """Fields waiting for an event-triggered update (before process_pending_triggers)"""
        return set(self._pending_triggered_fields)

    def set_param(self, name_or_idx: Union[str, int], value: float,
                  tolerance: float = 1e-10) -> bool:
        """Set a parameter value; returns whether it changed"""
        if isinstance(name_or_idx, str):
            idx = self._param_name_to_idx.get(name_or_idx)
            if idx is None:
                idx = self.define_param(name_or_idx, value)
                self._params_changed = True
                return True
        else:
            idx = name_or_idx

        old_value = self.params[idx]
        if abs(value - old_value) > tolerance:
            self.params[idx] = value
            self._params_changed = True
            return True
        return False

    def get_param_index(self, name: str) -> int:
        """Get the index of a parameter"""
        idx = self._param_name_to_idx.get(name)
        if idx is None:
            raise KeyError(f"Unknown parameter: {name}")
        return idx

    def list_params(self) -> List[Tuple[str, int, float]]:
        """List every parameter"""
        result = []
        for name, idx in self._param_name_to_idx.items():
            result.append((name, idx, float(self.params[idx])))
        return sorted(result, key=lambda x: x[1])

    def set_param_update(self, name: str,
                         update_func: Callable,
                         step_interval: Optional[int] = None,
                         time_interval: Optional[float] = None,
                         enabled: bool = True):
        """Configure the update policy of a parameter"""
        if name not in self._param_name_to_idx:
            raise KeyError(f"Parameter '{name}' not defined")

        self._param_update_config[name] = {
            'enabled': enabled,
            'update_func': update_func,
            'step_interval': step_interval,
            'time_interval': time_interval,
            'last_update_step': 0,
            'last_update_time': 0.0,
        }

    # =========================================================================
    # Parameter collector interface (computing a parameter from the objects)
    # =========================================================================

    def set_param_collector(self, name: str, collector_func: Callable):
        """
        Install a parameter collector (used by the TWO_WAY/OUTWARD modes)

        A collector computes a parameter value from the object data, for:
        - TWO_WAY mode: objects -> parameter -> external solver -> parameter update
        - OUTWARD mode: objects -> parameter -> external solver (no update)

        Args:
            name: parameter name
            collector_func: the collector
                signature: collector_func(objects: np.ndarray, physics: UnifiedPhysics) -> float
                - objects: object array (all or local objects, depending on distributed_mode)
                - physics: the physics module (other parameters and fields are readable)
                - returns: the computed parameter value

        Example:
            # mean temperature (from the energy property of the objects)
            def avg_temperature_collector(objects, physics):
                if len(objects) == 0:
                    return 300.0  # default
                energies = objects['float_props'][:, 0]  # property 0 is assumed to be the energy
                k_B = physics.get_param('k_B')
                return np.mean(energies) / (1.5 * k_B)

            physics.set_param_collector('avg_temperature', avg_temperature_collector)

            # defect density
            def defect_density_collector(objects, physics):
                box_size = physics.box_size
                volume = box_size[0] * box_size[1] * box_size[2]
                return len(objects) / volume

            physics.set_param_collector('defect_density', defect_density_collector)
        """
        if name not in self._param_name_to_idx:
            self.define_param(name)

        self._param_collectors[name] = collector_func

    def remove_param_collector(self, name: str):
        """Remove a parameter collector"""
        if name in self._param_collectors:
            del self._param_collectors[name]

    def get_param_collectors(self) -> List[str]:
        """Get the names of every registered parameter collector"""
        return list(self._param_collectors.keys())

    def collect_params_from_objects(self, objects: np.ndarray) -> Dict[str, float]:
        """
        Collect every parameter value from the objects

        Args:
            objects: object array

        Returns:
            a {param_name: collected_value} dict
        """
        collected = {}
        for name, collector_func in self._param_collectors.items():
            try:
                value = collector_func(objects, self)
                collected[name] = float(value)
            except Exception as e:
                if self.verbose:
                    print(f"[UnifiedPhysics] Warning: collector for '{name}' failed: {e}")
                # Keep the old value
                collected[name] = self.get_param(name)
        return collected

    # =========================================================================
    # Change interface used while an event executes
    # =========================================================================

    def begin_event_execution(self):
        """
        Enter the event execution context

        Called by the engine before execute_func, switching on the collection of pending changes
        """
        self._in_event_execution = True
        self._pending_param_changes.clear()
        self._pending_field_deltas.clear()

    def end_event_execution(self) -> Tuple[List[Tuple[int, float]], List[Tuple[int, float, float, float, float, float]]]:
        """
        Leave the event execution context and return the collected changes

        Returns:
            (param_changes, field_deltas)
            - param_changes: [(param_idx, new_value), ...]
            - field_deltas: [(field_idx, x, y, z, delta, radius), ...]
        """
        self._in_event_execution = False
        param_changes = self._pending_param_changes.copy()
        field_deltas = self._pending_field_deltas.copy()
        self._pending_param_changes.clear()
        self._pending_field_deltas.clear()
        return param_changes, field_deltas

    def set_param_in_event(self, name: str, value: float):
        """
        Set a parameter while an event executes (the change is collected and broadcast)

        Args:
            name: parameter name
            value: the new value

        Note:
            - may only be called inside execute_func
            - it does not take effect at once but travels with changed_objects
            - every rank applies it after receiving the broadcast
        """
        if not self._in_event_execution:
            # Outside an event execution context, set it directly (useful during initialization)
            self.set_param(name, value)
            return

        idx = self._param_name_to_idx.get(name)
        if idx is None:
            # Declare the new parameter automatically
            idx = self.define_param(name, value)

        self._pending_param_changes.append((idx, float(value)))

    def add_field_delta_in_event(self, name: str,
                                  position: Tuple[float, float, float],
                                  delta: float,
                                  radius: float = 0.0):
        """
        Add a field delta while an event executes (the change is collected and broadcast)

        Args:
            name: field name
            position: the (x, y, z) position
            delta: the increment
            radius: radius of influence
                    - radius = 0: only the nearest grid point is affected
                    - radius > 0: a Gaussian with sigma = radius

        Example:
            # a cascade event produces a local temperature spike
            physics.add_field_delta_in_event('temperature',
                                             position=(x, y, z),
                                             delta=+500.0,  # 500 K hotter
                                             radius=2.0)    # a 2 nm Gaussian

        Note:
            - may only be called inside execute_func
            - it does not take effect at once but travels with changed_objects
        """
        if not self._in_event_execution:
            # Outside an event execution context, apply it directly
            self._apply_field_delta(name, position, delta, radius)
            return

        idx = self._field_name_to_idx.get(name)
        if idx is None:
            raise KeyError(f"Unknown field: {name}. Define it first with define_field().")

        x, y, z = position
        self._pending_field_deltas.append((idx, float(x), float(y), float(z), float(delta), float(radius)))

    def apply_pending_changes(self, param_changes: List[Tuple[int, float]],
                               field_deltas: List[Tuple[int, float, float, float, float, float]]) -> bool:
        """
        Apply the pending changes received by broadcast

        Args:
            param_changes: [(param_idx, new_value), ...]
            field_deltas: [(field_idx, x, y, z, delta, radius), ...]

        Returns:
            whether any change was applied
        """
        changed = False

        # Apply the parameter changes
        for param_idx, value in param_changes:
            if 0 <= param_idx < len(self.params):
                old_value = self.params[param_idx]
                if abs(value - old_value) > 1e-10:
                    self.params[param_idx] = value
                    self._params_changed = True
                    changed = True

        # Apply the field deltas
        for field_idx, x, y, z, delta, radius in field_deltas:
            if 0 <= field_idx < len(self.fields):
                self._apply_field_delta_by_idx(field_idx, (x, y, z), delta, radius)
                changed = True

        return changed

    def _apply_field_delta(self, name: str,
                           position: Tuple[float, float, float],
                           delta: float,
                           radius: float):
        """Internal: apply a field delta directly (by name)"""
        idx = self._field_name_to_idx.get(name)
        if idx is None:
            raise KeyError(f"Unknown field: {name}")
        self._apply_field_delta_by_idx(idx, position, delta, radius)

    def _apply_field_delta_by_idx(self, field_idx: int,
                                   position: Tuple[float, float, float],
                                   delta: float,
                                   radius: float):
        """Internal: apply a field delta directly (by index)"""
        field = self.fields[field_idx]
        x_arr, y_arr, z_arr = self.field_axes[field_idx]
        x, y, z = position

        if radius <= 0:
            # Nearest grid point mode
            ix = np.searchsorted(x_arr, x)
            iy = np.searchsorted(y_arr, y)
            iz = np.searchsorted(z_arr, z)

            # Boundary handling
            ix = min(max(ix, 0), len(x_arr) - 1)
            iy = min(max(iy, 0), len(y_arr) - 1)
            iz = min(max(iz, 0), len(z_arr) - 1)

            field[ix, iy, iz] += delta
        else:
            # Gaussian mode
            # Range of influence (3 sigma)
            x_min_idx = max(0, np.searchsorted(x_arr, x - 3*radius) - 1)
            x_max_idx = min(len(x_arr), np.searchsorted(x_arr, x + 3*radius) + 1)
            y_min_idx = max(0, np.searchsorted(y_arr, y - 3*radius) - 1)
            y_max_idx = min(len(y_arr), np.searchsorted(y_arr, y + 3*radius) + 1)
            z_min_idx = max(0, np.searchsorted(z_arr, z - 3*radius) - 1)
            z_max_idx = min(len(z_arr), np.searchsorted(z_arr, z + 3*radius) + 1)

            # Build the grid coordinates
            xx, yy, zz = np.meshgrid(
                x_arr[x_min_idx:x_max_idx],
                y_arr[y_min_idx:y_max_idx],
                z_arr[z_min_idx:z_max_idx],
                indexing='ij'
            )

            # Gaussian weights
            dist_sq = (xx - x)**2 + (yy - y)**2 + (zz - z)**2
            weights = np.exp(-dist_sq / (2 * radius**2))

            # Normalize so the total increment equals delta
            weights /= np.sum(weights)
            weights *= delta

            # Apply the increment
            field[x_min_idx:x_max_idx, y_min_idx:y_max_idx, z_min_idx:z_max_idx] += weights

        # Mark the field changed so the interpolator cache is invalidated
        self._fields_changed = True
        self._invalidate_interpolator(field_idx)

    # =========================================================================
    # Field management interface
    # =========================================================================

    def define_adaptive_field(self, name: str,
                              x_coords: np.ndarray,
                              y_coords: np.ndarray,
                              z_coords: np.ndarray,
                              initial_value: float = 0.0) -> int:
        """Declare an adaptive (non-uniform) grid field"""
        if name in self._field_name_to_idx:
            return self._field_name_to_idx[name]

        # Validate the coordinate arrays
        x_coords = np.asarray(x_coords, dtype=np.float64)
        y_coords = np.asarray(y_coords, dtype=np.float64)
        z_coords = np.asarray(z_coords, dtype=np.float64)

        if len(x_coords) < 2 or not np.all(np.diff(x_coords) > 0):
            raise ValueError("x_coords must be strictly increasing with at least 2 points")
        if len(y_coords) < 2 or not np.all(np.diff(y_coords) > 0):
            raise ValueError("y_coords must be strictly increasing with at least 2 points")
        if len(z_coords) < 2 or not np.all(np.diff(z_coords) > 0):
            raise ValueError("z_coords must be strictly increasing with at least 2 points")

        shape = (len(x_coords), len(y_coords), len(z_coords))
        idx = len(self.fields)
        field = np.full(shape, initial_value, dtype=np.float64)
        self.fields.append(field)
        self.field_axes.append((x_coords.copy(), y_coords.copy(), z_coords.copy()))

        self._field_name_to_idx[name] = idx
        self._field_idx_to_name[idx] = name

        return idx

    def define_field(self, name: str, shape: Tuple[int, int, int],
                     initial_value: float = 0.0) -> int:
        """Declare a uniform grid field"""
        if name in self._field_name_to_idx:
            return self._field_name_to_idx[name]

        lx, ly, lz = self.box_size
        nx, ny, nz = shape

        x_coords = np.linspace(0, lx, nx)
        y_coords = np.linspace(0, ly, ny)
        z_coords = np.linspace(0, lz, nz)

        return self.define_adaptive_field(name, x_coords, y_coords, z_coords, initial_value)

    def get_field(self, name_or_idx: Union[str, int]) -> np.ndarray:
        """Get the field data"""
        if isinstance(name_or_idx, str):
            idx = self._field_name_to_idx.get(name_or_idx)
            if idx is None:
                raise KeyError(f"Unknown field: {name_or_idx}")
        else:
            idx = name_or_idx
        return self.fields[idx]

    def set_field(self, name_or_idx: Union[str, int], data: np.ndarray,
                  tolerance: float = 1e-10) -> bool:
        """Set the field data; returns whether it changed"""
        if isinstance(name_or_idx, str):
            idx = self._field_name_to_idx.get(name_or_idx)
            if idx is None:
                idx = self.define_field(name_or_idx, data.shape)
        else:
            idx = name_or_idx

        old_field = self.fields[idx]

        if old_field.shape != data.shape:
            self.fields[idx] = data.copy()
            self._fields_changed = True
            self._invalidate_interpolator(idx)
            return True

        max_diff = np.max(np.abs(data - old_field))
        if max_diff > tolerance:
            self.fields[idx] = data.copy()
            self._fields_changed = True
            self._invalidate_interpolator(idx)
            return True

        return False

    def get_field_axes(self, name_or_idx: Union[str, int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get the axes of a field"""
        if isinstance(name_or_idx, str):
            idx = self._field_name_to_idx.get(name_or_idx)
            if idx is None:
                raise KeyError(f"Unknown field: {name_or_idx}")
        else:
            idx = name_or_idx
        return self.field_axes[idx]

    def list_fields(self) -> List[Tuple[str, int, Tuple, bool]]:
        """List every field"""
        result = []
        for name, idx in self._field_name_to_idx.items():
            shape = self.fields[idx].shape
            x, y, z = self.field_axes[idx]
            is_adaptive = not (
                np.allclose(np.diff(x), np.diff(x)[0]) and
                np.allclose(np.diff(y), np.diff(y)[0]) and
                np.allclose(np.diff(z), np.diff(z)[0])
            )
            result.append((name, idx, shape, is_adaptive))
        return sorted(result, key=lambda x: x[1])

    def bind_object_pool(self, rma_object_pool):
        """Bind the object pool, giving the field update_func query and snapshot services."""
        self._object_query_engine = ObjectQueryEngine(self.comm, rma_object_pool)

    def has_object_query_engine(self) -> bool:
        """Whether an object query engine is bound."""
        return self._object_query_engine is not None

    def build_object_snapshots(self, query_specs: Optional[Dict[str, dict]]) -> Dict[str, np.ndarray]:
        """Build the object snapshot from the queries."""
        if not query_specs:
            return {}
        if self._object_query_engine is None:
            raise RuntimeError(
                "UnifiedPhysics object query service is not bound. "
                "Engine should call physics.bind_object_pool(...) before running updates."
            )
        return self._object_query_engine.build_snapshots(query_specs)

    def set_field_update(self, name: str,
                         update_func: Callable,
                         step_interval: Optional[int] = None,
                         time_interval: Optional[float] = None,
                         enabled: bool = True,
                         queries: Optional[Dict[str, dict]] = None,
                         update_policy: Optional[dict] = None):
        """Configure the update policy of a field"""
        if name not in self._field_name_to_idx:
            raise KeyError(f"Field '{name}' not defined")

        policy = dict(update_policy or {})
        self._field_update_config[name] = {
            'enabled': enabled,
            'update_func': update_func,
            'step_interval': step_interval,
            'time_interval': time_interval,
            'queries': dict(queries or {}),
            'update_policy': policy,
            'last_update_step': 0,
            'last_update_time': 0.0,
        }

    def _call_field_update_func(self,
                                update_func: Callable,
                                snapshots: Dict[str, np.ndarray],
                                ctx: PhysicsUpdateContext,
                                step: int,
                                time: float):
        """Call the field update_func, accepting both the old and the new signature."""
        try:
            sig = inspect.signature(update_func)
            positional = [
                p for p in sig.parameters.values()
                if p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
            ]
        except (TypeError, ValueError):
            positional = []

        if len(positional) >= 3:
            second_name = positional[1].name.lower()
            if second_name in {'snapshots', 'snapshot', 'objects', 'object_snapshots'}:
                return update_func(self, snapshots, ctx)
            if len(positional) >= 4:
                return update_func(self, step, time, self.config)
            return update_func(self, step, time)

        if len(positional) == 2:
            return update_func(self, step)
        if len(positional) <= 1:
            return update_func(self)

        return update_func(self, snapshots, ctx)

    def _apply_field_update_result(self, field_name: str, result) -> bool:
        """Apply the return value of a field update_func."""
        if result is None:
            return False

        if isinstance(result, (bool, np.bool_)):
            return bool(result)

        if isinstance(result, np.ndarray):
            return self.set_field(field_name, result)

        if isinstance(result, dict):
            changed = False
            for name, value in result.items():
                if isinstance(value, np.ndarray):
                    if self.set_field(name, value):
                        changed = True
                elif np.isscalar(value):
                    if self.set_param(name, float(value)):
                        changed = True
                else:
                    raise TypeError(
                        f"Unsupported update result for '{name}': {type(value)}. "
                        "Expected numpy.ndarray or scalar."
                    )
            return changed

        raise TypeError(
            f"Field update for '{field_name}' returned unsupported type {type(result)}. "
            "Expected numpy.ndarray, dict, bool, or None."
        )

    def _run_field_update(self,
                          field_name: str,
                          cfg: dict,
                          step: int,
                          time: float,
                          trigger_source: Optional[str] = None,
                          trigger_reason: Optional[str] = None,
                          event_data: Optional[dict] = None) -> bool:
        """Run the update_func of one field."""
        snapshots = self.build_object_snapshots(cfg.get('queries', {}))
        ctx = PhysicsUpdateContext(
            self.comm,
            step=step,
            time=time,
            field_name=field_name,
            trigger_source=trigger_source,
            trigger_reason=trigger_reason,
            changed_summary=(event_data or {}).get('changed_summary', {})
        )
        result = self._call_field_update_func(
            cfg['update_func'],
            snapshots,
            ctx,
            step,
            time,
        )
        changed = self._apply_field_update_result(field_name, result)
        cfg['last_update_step'] = step
        cfg['last_update_time'] = time
        return changed

    def _should_trigger_from_object_changes(self, cfg: dict, event_data: Optional[dict]) -> bool:
        """Test whether the object changes satisfy the event trigger condition of a field."""
        if not event_data:
            return False

        policy = cfg.get('update_policy', {})
        summary = event_data.get('changed_summary', {})
        type_filter = set(policy.get('object_types', []))

        def _matches(kind: str) -> bool:
            if not policy.get(f'trigger_on_object_{kind}', False):
                return False
            key = f'{kind}d_types' if kind == 'change' else f'{kind}d_types'
            # create -> created_types, change -> changed_types, delete -> deleted_types
            if kind == 'create':
                names = set(summary.get('created_types', []))
            elif kind == 'change':
                names = set(summary.get('changed_types', []))
            else:
                names = set(summary.get('deleted_types', []))
            if not names:
                return False
            return True if not type_filter else bool(names & type_filter)

        return _matches('create') or _matches('change') or _matches('delete')

    # =========================================================================
    # Field interpolation interface
    # =========================================================================

    def _invalidate_interpolator(self, field_idx: int):
        """Invalidate the interpolator cache"""
        if field_idx in self._field_interpolators:
            del self._field_interpolators[field_idx]
        # Empty the field value cache as well
        if self._field_cache:
            self._field_cache.clear()

    def _get_interpolator(self, field_idx: int) -> RegularGridInterpolator:
        """Get or create a field interpolator"""
        if field_idx in self._field_interpolators:
            return self._field_interpolators[field_idx]

        field = self.fields[field_idx]
        x, y, z = self.field_axes[field_idx]

        interp = RegularGridInterpolator(
            (x, y, z), field,
            method='linear',
            bounds_error=False,
            fill_value=None
        )

        self._field_interpolators[field_idx] = interp
        return interp

    def interpolate_field(self, name_or_idx: Union[str, int],
                          x: float, y: float, z: float) -> float:
        """Interpolate a field at one point"""
        if isinstance(name_or_idx, str):
            idx = self._field_name_to_idx[name_or_idx]
        else:
            idx = name_or_idx

        interp = self._get_interpolator(idx)
        return float(interp([[x, y, z]])[0])

    def interpolate_field_batch(self, name_or_idx: Union[str, int],
                                positions: np.ndarray) -> np.ndarray:
        """Interpolate a field in bulk"""
        if isinstance(name_or_idx, str):
            idx = self._field_name_to_idx[name_or_idx]
        else:
            idx = name_or_idx

        interp = self._get_interpolator(idx)
        return interp(positions)

    def get_field_interpolator(self, name_or_idx: Union[str, int]) -> Callable:
        """Get a field interpolation function"""
        if isinstance(name_or_idx, str):
            idx = self._field_name_to_idx[name_or_idx]
            name = name_or_idx
        else:
            idx = name_or_idx
            name = self._field_idx_to_name.get(idx, f'field_{idx}')

        interp = self._get_interpolator(idx)

        # Return the cached version when the cache is enabled
        if self.enable_cache and self._field_cache:
            def cached_field_at_point(x, y, z):
                return self._field_cache.get(name, lambda x, y, z: float(interp([[x, y, z]])[0]), x, y, z)
            return cached_field_at_point
        else:
            def field_at_point(x, y, z):
                return float(interp([[x, y, z]])[0])
            return field_at_point

    # =========================================================================
    # History storage interface
    # =========================================================================

    def save_history(self, time: float):
        """
        Save the current state into the history (call before an update)

        Args:
            time: current simulation time

        Purpose:
            an external solver or a rate function can call get_param_rate() and
            get_field_rate() to obtain time derivatives of the physical quantities

        Usage:
            # in the main loop
            for step in range(max_steps):
                physics.save_history(current_time)  # 1. save the current state
                # ... run the physics update ...     # 2. update parameters/fields
                physics.update_time(new_time)       # 3. advance the time
                dT_dt = physics.get_param_rate('temperature')  # 4. read the rate

        Note:
            - save_history copies params and fields into _prev_* at the moment of the call
            - params/fields are modified afterwards, then update_time sets the current time
            - get_*_rate() evaluates (current - prev) / dt
        """
        if not self.enable_history:
            return

        # Save the time
        self._prev_time = self._current_time
        self._current_time = time

        # Save the parameters (deep copy)
        self._prev_params = self.params.copy()

        # Save the fields (deep copy)
        self._prev_fields = [field.copy() for field in self.fields]

        self._history_initialized = True

    def update_time(self, time: float):
        """
        Advance the current time (without saving the history)

        Args:
            time: the new simulation time

        Purpose:
            call it after modifying params/fields so the rates can be computed

        Example:
            physics.save_history(0.0)          # save the baseline
            physics.set_param('T', 600.0)      # modify a parameter
            physics.update_time(1.0)           # advance the time
            rate = physics.get_param_rate('T') # evaluates (600-500) / 1.0
        """
        if not self.enable_history:
            return
        self._current_time = time

    def get_time_delta(self) -> float:
        """
        Get the time step (current time - previous time)

        Returns:
            the increment dt, or 0 when the history is uninitialized
        """
        if not self.enable_history or not self._history_initialized:
            return 0.0
        return self._current_time - self._prev_time

    def get_prev_param(self, name_or_idx: Union[str, int]) -> Optional[float]:
        """
        Get a parameter value of the previous step

        Args:
            name_or_idx: parameter name or index

        Returns:
            the previous value, or None when the history is uninitialized
        """
        if not self.enable_history or self._prev_params is None:
            return None

        if isinstance(name_or_idx, str):
            idx = self._param_name_to_idx.get(name_or_idx)
            if idx is None:
                raise KeyError(f"Unknown parameter: {name_or_idx}")
        else:
            idx = name_or_idx

        return float(self._prev_params[idx])

    def get_param_rate(self, name_or_idx: Union[str, int]) -> Optional[float]:
        """
        Get the rate of change of a parameter (its time derivative)

        Args:
            name_or_idx: parameter name or index

        Returns:
            dP/dt, or None when the history is uninitialized or dt = 0

        Example:
            # rate of change of the temperature
            dT_dt = physics.get_param_rate('temperature')
            if dT_dt is not None:
                heating_effect = dT_dt * heat_capacity
        """
        if not self.enable_history or not self._history_initialized:
            return None

        dt = self.get_time_delta()
        if dt <= 0:
            return None

        current_val = self.get_param(name_or_idx)
        prev_val = self.get_prev_param(name_or_idx)

        if prev_val is None:
            return None

        return (current_val - prev_val) / dt

    def get_prev_field(self, name_or_idx: Union[str, int]) -> Optional[np.ndarray]:
        """
        Get the field data of the previous step

        Args:
            name_or_idx: field name or index

        Returns:
            the previous field array (a read-only view), or None when uninitialized
        """
        if not self.enable_history or not self._prev_fields:
            return None

        if isinstance(name_or_idx, str):
            idx = self._field_name_to_idx.get(name_or_idx)
            if idx is None:
                raise KeyError(f"Unknown field: {name_or_idx}")
        else:
            idx = name_or_idx

        if idx >= len(self._prev_fields) or self._prev_fields[idx] is None:
            return None

        return self._prev_fields[idx]

    def get_field_rate(self, name_or_idx: Union[str, int]) -> Optional[np.ndarray]:
        """
        Get the rate of change of a field (its time derivative)

        Args:
            name_or_idx: field name or index

        Returns:
            the field dF/dt, or None when the history is uninitialized or dt = 0

        Example:
            # time derivative of the temperature field
            dT_dt_field = physics.get_field_rate('temperature')
            if dT_dt_field is not None:
                # used to compute a thermal effect
                heat_source = dT_dt_field * heat_capacity_field
        """
        if not self.enable_history or not self._history_initialized:
            return None

        dt = self.get_time_delta()
        if dt <= 0:
            return None

        current_field = self.get_field(name_or_idx)
        prev_field = self.get_prev_field(name_or_idx)

        if prev_field is None:
            return None

        # Do the shapes match? (compression may have changed them)
        if current_field.shape != prev_field.shape:
            warnings.warn(f"Field shape changed, cannot compute rate")
            return None

        return (current_field - prev_field) / dt

    def get_prev_field_interpolator(self, name_or_idx: Union[str, int]) -> Optional[Callable]:
        """
        Get an interpolator of the field of the previous step

        Args:
            name_or_idx: field name or index

        Returns:
            the interpolator f(x, y, z), or None when the history is uninitialized

        Purpose:
            querying the previous field value at an arbitrary position
        """
        prev_field = self.get_prev_field(name_or_idx)
        if prev_field is None:
            return None

        if isinstance(name_or_idx, str):
            idx = self._field_name_to_idx[name_or_idx]
        else:
            idx = name_or_idx

        # Use the current axes (assuming they did not change)
        x, y, z = self.field_axes[idx]

        interp = RegularGridInterpolator(
            (x, y, z), prev_field,
            method='linear',
            bounds_error=False,
            fill_value=None
        )

        def field_at_point(x, y, z):
            return float(interp([[x, y, z]])[0])

        return field_at_point

    # =========================================================================
    # External solver interface
    # =========================================================================

    def register_solver(self, name: str, solver: ExternalSolver,
                        fields: Optional[List[str]] = None):
        """
        Register an external solver

        Args:
            name: solver name (e.g. 'heat_solver', 'stress_solver')
            solver: an ExternalSolver instance
            fields: fields this solver updates (None uses solver.get_supported_fields())

        Example:
            # register the Dedalus heat conduction solver
            heat_solver = DedalusHeatSolver(domain_size, resolution)
            physics.register_solver('heat', heat_solver, fields=['temperature'])

            # register the FEniCSX stress solver
            stress_solver = FEniCSXStressSolver(mesh, material)
            physics.register_solver('stress', stress_solver,
                                    fields=['stress_xx', 'stress_yy', 'stress_zz'])
        """
        self._external_solvers[name] = solver

        # Determine the fields this solver owns
        if fields is None:
            fields = solver.get_supported_fields()
        self._solver_to_fields[name] = list(fields)

        # Initialize the solver
        solver.initialize(self, self.config)

        if self.verbose and self.rank == 0:
            print(f"[UnifiedPhysics] Registered solver '{name}' for fields: {fields}")

    def unregister_solver(self, name: str):
        """Unregister an external solver"""
        if name in self._external_solvers:
            self._external_solvers[name].finalize()
            del self._external_solvers[name]
            del self._solver_to_fields[name]

    def get_solver(self, name: str) -> Optional[ExternalSolver]:
        """Get a solver"""
        return self._external_solvers.get(name)

    def call_solver(self, solver_name: str,
                    step: int,
                    time: float,
                    object_data: Optional[np.ndarray] = None,
                    trigger_source: Optional[str] = None) -> bool:
        """
        Call a solver to update its fields and parameters

        Args:
            solver_name: solver name
            step: current step
            time: current time
            object_data: object data (TWO_WAY/OUTWARD modes)
            trigger_source: what triggered the call

        Returns:
            True if a field or a parameter was updated
        """
        solver = self._external_solvers.get(solver_name)
        if solver is None:
            warnings.warn(f"Solver '{solver_name}' not registered")
            return False

        # Every rank calls the solver at the same time
        result = solver.solve(self, step, time, object_data, trigger_source)

        # Synchronize
        self.comm.Barrier()

        # Rank 0 handles the result (fields and parameters are both supported)
        changed = False
        if self.rank == 0 and result is not None:
            for name, value in result.items():
                if isinstance(value, np.ndarray):
                    # Field data
                    if self.set_field(name, value):
                        changed = True
                elif isinstance(value, (int, float)):
                    # Scalar parameter
                    if self.set_param(name, value):
                        changed = True

        # Broadcast the changed flag
        changed = self.comm.bcast(changed, root=0)

        # Synchronize the field data
        if changed:
            self._sync_all()

        return changed

    # =========================================================================
    # Event trigger mechanism
    # =========================================================================

    def register_event_trigger(self, event_type: str,
                               fields: List[str],
                               solver: Optional[str] = None,
                               condition: Optional[Callable] = None):
        """
        Register an event-triggered field update

        Executing an event of the given type updates the listed fields automatically

        Args:
            event_type: event type (e.g. 'cascade', 'precipitation')
            fields: the fields to update
            solver: name of the solver to use (None uses the default update logic)
            condition: optional condition func(event_data) -> bool;
                      the update runs only when it returns True

        Example:
            # a cascade updates the temperature and stress fields
            physics.register_event_trigger('cascade',
                                           ['temperature', 'stress_xx', 'stress_yy'],
                                           solver='fem_solver')

            # only a cascade with more than 100 defects triggers the update
            physics.register_event_trigger('cascade',
                                           ['temperature'],
                                           condition=lambda e: e.get('defect_count', 0) > 100)
        """
        self._event_triggers[event_type] = {
            'fields': list(fields),
            'solver': solver,
            'condition': condition,
        }

        if self.verbose and self.rank == 0:
            print(f"[UnifiedPhysics] Event trigger registered: '{event_type}' -> {fields}")

    def unregister_event_trigger(self, event_type: str):
        """Unregister an event trigger"""
        if event_type in self._event_triggers:
            del self._event_triggers[event_type]

    def notify_event_executed(self, event_type: str, event_data: Optional[dict] = None):
        """
        Tell the physics module that an event has executed

        Called by v16_engine after the event executes

        Args:
            event_type: event type
            event_data: data of the event (used by the condition)

        Returns:
            nothing; an internal flag is set when an update is due
        """
        event_data = event_data or {}
        triggered_fields: Set[str] = set()

        trigger_config = self._event_triggers.get(event_type)
        if trigger_config is not None:
            condition = trigger_config.get('condition')
            if condition is None or condition(event_data):
                triggered_fields.update(trigger_config['fields'])

        # The newer path: triggers declared in the update_policy of the field itself
        for field_name, cfg in self._field_update_config.items():
            if not cfg.get('enabled', True):
                continue

            policy = cfg.get('update_policy', {})
            trigger_events = set(policy.get('trigger_events', []))
            if event_type in trigger_events:
                triggered_fields.add(field_name)
                continue

            if self._should_trigger_from_object_changes(cfg, event_data):
                triggered_fields.add(field_name)

        if not triggered_fields:
            return

        self._pending_triggered_fields.update(triggered_fields)
        self._pending_trigger_source = event_type
        self._pending_event_data = event_data
        summary = event_data.get('changed_summary', {})
        if summary.get('created_count', 0) or summary.get('changed_count', 0) or summary.get('deleted_count', 0):
            self._pending_trigger_reason = 'object_change'
        else:
            self._pending_trigger_reason = 'event'

        if self.verbose and self.rank == 0:
            print(f"[UnifiedPhysics] Event '{event_type}' triggered field updates: "
                  f"{sorted(triggered_fields)}")

    def has_pending_triggers(self) -> bool:
        """Test whether an event trigger is pending"""
        return len(self._pending_triggered_fields) > 0

    def process_pending_triggers(self, step: int, time: float,
                                 object_data: Optional[np.ndarray] = None) -> bool:
        """
        Process the pending event triggers

        Called by v16_engine at a suitable moment (normally after the event executed)

        Args:
            step: current step
            time: current time
            object_data: object data

        Returns:
            True if any field was updated
        """
        if not self._pending_triggered_fields:
            return False

        changed = False
        trigger_source = self._pending_trigger_source
        trigger_reason = self._pending_trigger_reason
        event_data = self._pending_event_data or {}

        # Group the pending fields by solver
        solver_fields: Dict[Optional[str], List[str]] = {}
        for field_name in self._pending_triggered_fields:
            # Find the solver owning this field
            solver_name = None
            for sname, fields in self._solver_to_fields.items():
                if field_name in fields:
                    solver_name = sname
                    break

            if solver_name not in solver_fields:
                solver_fields[solver_name] = []
            solver_fields[solver_name].append(field_name)

        # Call each solver
        for solver_name, fields in solver_fields.items():
            if solver_name is not None:
                # Use the registered solver
                if self.call_solver(solver_name, step, time, object_data, trigger_source):
                    changed = True
            else:
                # Use the default update logic (when configured)
                for field_name in fields:
                    cfg = self._field_update_config.get(field_name)
                    if cfg and cfg.get('update_func'):
                        if self._run_field_update(
                            field_name,
                            cfg,
                            step,
                            time,
                            trigger_source=trigger_source,
                            trigger_reason=trigger_reason,
                            event_data=event_data
                        ):
                            changed = True
                            self._skip_scheduled_fields_once.add(field_name)

        # Clear the pending list
        self._pending_triggered_fields.clear()
        self._pending_trigger_source = None
        self._pending_trigger_reason = None
        self._pending_event_data = None

        if changed:
            self.stats['event_triggered_updates'] += 1

        return changed

    # =========================================================================
    # Field compression
    # =========================================================================

    def compress_field(self, name_or_idx: Union[str, int],
                       gradient_threshold: float = 0.01,
                       min_points: int = 10,
                       max_compression_ratio: float = 0.5) -> bool:
        """
        Compress a field by its gradient (fewer points in low-gradient regions)

        Principle:
        - compute the gradient of the field
        - keep the resolution high where the gradient is high
        - merge nodes where the gradient is low
        - produce a non-uniform grid

        Args:
            name_or_idx: field name or index
            gradient_threshold: gradient threshold (regions below it may be compressed)
            min_points: minimum number of points kept along each dimension
            max_compression_ratio: maximum compression ratio (points after / points before)

        Returns:
            True if the field was compressed

        Example:
            # compress the stress field, thinning the low-gradient regions
            physics.compress_field('stress_xx', gradient_threshold=0.005)
        """
        if isinstance(name_or_idx, str):
            idx = self._field_name_to_idx.get(name_or_idx)
            name = name_or_idx
            if idx is None:
                raise KeyError(f"Unknown field: {name_or_idx}")
        else:
            idx = name_or_idx
            name = self._field_idx_to_name.get(idx, f'field_{idx}')

        field = self.fields[idx]
        x_old, y_old, z_old = self.field_axes[idx]

        # Gradient along the three directions
        grad_x = np.abs(np.gradient(field, x_old, axis=0))
        grad_y = np.abs(np.gradient(field, y_old, axis=1))
        grad_z = np.abs(np.gradient(field, z_old, axis=2))

        # Compress the coordinates of each axis
        def compress_axis(coords: np.ndarray, grads: np.ndarray,
                          axis: int) -> np.ndarray:
            """
            Compress the coordinates along one axis

            grads: the gradient along that axis (already absolute)
            """
            n = len(coords)
            if n <= min_points:
                return coords.copy()

            # [fix] Guard against division by zero: no compression when gradient_threshold <= 0
            if gradient_threshold <= 0:
                return coords.copy()

            # Maximum gradient at each position (the maximum over the other axes)
            max_grad = np.max(grads, axis=tuple(i for i in range(3) if i != axis))

            # Normalize the gradient
            if max_grad.max() > 0:
                norm_grad = max_grad / max_grad.max()
            else:
                norm_grad = np.ones_like(max_grad)

            # Retention probability: the larger the gradient, the more likely a point is kept
            keep_prob = np.clip(norm_grad / gradient_threshold, 0.0, 1.0)
            keep_prob[0] = 1.0  # the end points are always kept
            keep_prob[-1] = 1.0

            # Decide which points to keep
            keep_mask = np.ones(n, dtype=bool)

            # Start in the middle and skip the low-gradient regions
            i = 1
            while i < n - 1:
                if keep_prob[i] < 0.5:
                    # Can this point be skipped?
                    skip = 1
                    while i + skip < n - 1 and keep_prob[i + skip] < 0.5:
                        skip += 1
                    # Keep only the midpoint of a skipped region
                    if skip > 1:
                        keep_mask[i:i + skip] = False
                        keep_mask[i + skip // 2] = True
                    i += skip
                else:
                    i += 1

            # Make sure at least min_points points survive
            kept_indices = np.where(keep_mask)[0]
            if len(kept_indices) < min_points:
                # Sample uniformly
                kept_indices = np.linspace(0, n - 1, min_points, dtype=int)
                keep_mask[:] = False
                keep_mask[kept_indices] = True

            # Check the compression ratio
            if np.sum(keep_mask) / n < max_compression_ratio:
                # Over-compressed: restore some points
                target_count = int(n * max_compression_ratio)
                kept_indices = np.linspace(0, n - 1, target_count, dtype=int)
                keep_mask[:] = False
                keep_mask[kept_indices] = True

            return coords[keep_mask]

        # Compress each axis
        x_new = compress_axis(x_old, grad_x, 0)
        y_new = compress_axis(y_old, grad_y, 1)
        z_new = compress_axis(z_old, grad_z, 2)

        # Did anything change?
        if (len(x_new) == len(x_old) and
            len(y_new) == len(y_old) and
            len(z_new) == len(z_old)):
            return False

        # Interpolate onto the new grid
        old_interp = RegularGridInterpolator(
            (x_old, y_old, z_old), field,
            method='linear',
            bounds_error=False,
            fill_value=None
        )

        # Build the new grid points
        X_new, Y_new, Z_new = np.meshgrid(x_new, y_new, z_new, indexing='ij')
        points_new = np.column_stack([X_new.ravel(), Y_new.ravel(), Z_new.ravel()])

        # Interpolate
        new_values = old_interp(points_new).reshape(len(x_new), len(y_new), len(z_new))

        # Update the field
        self.fields[idx] = new_values
        self.field_axes[idx] = (x_new, y_new, z_new)
        self._invalidate_interpolator(idx)
        self._fields_changed = True

        self.stats['field_compressions'] += 1

        if self.verbose and self.rank == 0:
            old_size = len(x_old) * len(y_old) * len(z_old)
            new_size = len(x_new) * len(y_new) * len(z_new)
            ratio = new_size / old_size
            print(f"[UnifiedPhysics] Field '{name}' compressed: "
                  f"{field.shape} -> {new_values.shape} "
                  f"(ratio: {ratio:.2%})")

        return True

    def decompress_field(self, name_or_idx: Union[str, int],
                         target_shape: Tuple[int, int, int]) -> bool:
        """
        Decompress a field onto the given resolution (a uniform grid)

        Args:
            name_or_idx: field name or index
            target_shape: target shape (Nx, Ny, Nz)

        Returns:
            True if the field was decompressed
        """
        if isinstance(name_or_idx, str):
            idx = self._field_name_to_idx.get(name_or_idx)
            if idx is None:
                raise KeyError(f"Unknown field: {name_or_idx}")
        else:
            idx = name_or_idx

        field = self.fields[idx]
        x_old, y_old, z_old = self.field_axes[idx]

        if field.shape == target_shape:
            return False

        # Create the new uniform grid
        lx, ly, lz = self.box_size
        nx, ny, nz = target_shape

        x_new = np.linspace(0, lx, nx)
        y_new = np.linspace(0, ly, ny)
        z_new = np.linspace(0, lz, nz)

        # Interpolate
        old_interp = RegularGridInterpolator(
            (x_old, y_old, z_old), field,
            method='linear',
            bounds_error=False,
            fill_value=None
        )

        X_new, Y_new, Z_new = np.meshgrid(x_new, y_new, z_new, indexing='ij')
        points_new = np.column_stack([X_new.ravel(), Y_new.ravel(), Z_new.ravel()])
        new_values = old_interp(points_new).reshape(target_shape)

        # Update the field
        self.fields[idx] = new_values
        self.field_axes[idx] = (x_new, y_new, z_new)
        self._invalidate_interpolator(idx)
        self._fields_changed = True

        return True

    # =========================================================================
    # External field coupling synchronization (folded in from field_coupler)
    # =========================================================================

    def sync_coupling(self, rma_object_pool, step: int, time: float) -> bool:
        """
        Run the external field coupling (folded in from field_coupler.sync_step)

        The logic follows the coupling mode:
        - TWO_WAY:  collect the objects -> call the solver -> update the fields -> affect the rates
        - OUTWARD: collect the objects -> call the solver (output only, no field update)
        - INWARD:  call the solver -> update the fields (independent of the objects)

        Args:
            rma_object_pool: RMA object pool
            step: current step
            time: current time

        Returns:
            True if a field changed (so the rates must be recomputed)
        """
        # ===== Step 0: save the history (before the update) =====
        if self.enable_history:
            self.save_history(time)

        # ===== Step 1: collect the objects =====
        object_data = None

        if self.coupling_mode in [MODE_TWO_WAY, MODE_OUTWARD]:
            local_objs = rma_object_pool.get_local_objects()

            if self.distributed_mode:
                object_data = local_objs
            else:
                # [fix] Centralized mode: Gatherv without serialization (no pickle overhead)
                object_data = self._gather_objects_zero_copy(local_objs)

        # Keep the collected object data for the external solver
        self._last_object_data = object_data

        # ===== Step 1.5: collect the parameters (computed from the objects) =====
        if self.coupling_mode in [MODE_TWO_WAY, MODE_OUTWARD] and object_data is not None:
            if self.rank == 0 and self._param_collectors:
                collected_params = self.collect_params_from_objects(object_data)
                for name, value in collected_params.items():
                    self.set_param(name, value)

        # ===== Step 2: call the external solver =====
        changed = False

        for solver_name, solver in self._external_solvers.items():
            if self.coupling_mode == MODE_OUTWARD:
                # OUTWARD mode: call the solver but do not update the fields
                solver.solve(self, step, time, object_data, None)
            else:
                # TWO_WAY and INWARD modes: update the fields
                if self.call_solver(solver_name, step, time, object_data, None):
                    changed = True

        return changed

    def _gather_objects_zero_copy(self, local_objs: np.ndarray) -> Optional[np.ndarray]:
        """
        Gather every object onto rank 0 with Gatherv (no serialization)

        [fix] Avoids the pickle overhead by transferring MPI.BYTE directly
        """
        # Collect the object count of every rank
        local_count = len(local_objs)
        all_counts = self.comm.allgather(local_count)

        total_count = sum(all_counts)
        if total_count == 0:
            if self.rank == 0:
                return np.array([], dtype=local_objs.dtype)
            return None

        # Compute the byte counts and displacements
        itemsize = local_objs.dtype.itemsize
        byte_counts = [c * itemsize for c in all_counts]
        byte_displs = [sum(byte_counts[:i]) for i in range(self.size)]

        # Rank 0 allocates the receive buffer
        if self.rank == 0:
            recv_buf = np.zeros(total_count, dtype=local_objs.dtype)
        else:
            recv_buf = None

        # Prepare the send buffer (contiguous memory)
        send_buf = np.ascontiguousarray(local_objs)

        # Gatherv transfer without serialization
        sendbuf = [send_buf.view(np.uint8), local_count * itemsize]
        if self.rank == 0:
            recvbuf = [recv_buf.view(np.uint8), byte_counts, byte_displs, MPI.BYTE]
        else:
            recvbuf = None

        self.comm.Gatherv(sendbuf, recvbuf, root=0)

        return recv_buf

    def get_all_objects(self, rma_object_pool) -> Optional[np.ndarray]:
        """
        Get the array of every object (for an external solver or post-processing)

        Args:
            rma_object_pool: RMA object pool

        Returns:
            the NumPy array of every object (in centralized mode only rank 0 has it all)

        Note:
            this method triggers MPI communication (a gather)
            for large systems prefer distributed_mode=True to avoid the memory pressure
        """
        local_objs = rma_object_pool.get_local_objects()

        if self.distributed_mode:
            return local_objs
        else:
            # [fix] Centralized mode: Gatherv without serialization
            return self._gather_objects_zero_copy(local_objs)

    def get_last_object_data(self) -> Optional[np.ndarray]:
        """
        Get the object data collected by the last sync_coupling

        Returns:
            the object array, or None when sync_coupling has not run yet

        Purpose:
            letting an external solver reach the object data outside solve()
        """
        return getattr(self, '_last_object_data', None)

    def _bcast_field_zero_copy(self, field_idx: int, root: int = 0):
        """
        Broadcast field data without serialization

        Args:
            field_idx: field index
            root: rank broadcasting the data
        """
        if self.size == 1:
            return

        # Broadcast the shape
        if self.rank == root:
            shape = self.fields[field_idx].shape
            x, y, z = self.field_axes[field_idx]
            shape_info = (shape[0], shape[1], shape[2], len(x), len(y), len(z))
        else:
            shape_info = (0, 0, 0, 0, 0, 0)

        shape_info = self.comm.bcast(shape_info, root=root)
        nx, ny, nz, lx, ly, lz = shape_info

        # Prepare the buffer
        if self.rank != root:
            self.fields[field_idx] = np.empty((nx, ny, nz), dtype=np.float64)
            x_coords = np.empty(lx, dtype=np.float64)
            y_coords = np.empty(ly, dtype=np.float64)
            z_coords = np.empty(lz, dtype=np.float64)
        else:
            x_coords, y_coords, z_coords = self.field_axes[field_idx]

        # Broadcast without serialization
        self.comm.Bcast([self.fields[field_idx], MPI.DOUBLE], root=root)
        self.comm.Bcast([x_coords, MPI.DOUBLE], root=root)
        self.comm.Bcast([y_coords, MPI.DOUBLE], root=root)
        self.comm.Bcast([z_coords, MPI.DOUBLE], root=root)

        if self.rank != root:
            self.field_axes[field_idx] = (x_coords, y_coords, z_coords)

        self._invalidate_interpolator(field_idx)

    # =========================================================================
    # Lifecycle management
    # =========================================================================

    def initialize(self):
        """Run the initialization (IC)"""
        if self.ic_func is not None:
            self.ic_func(self, self.config)

        self._initialized = True
        self._params_changed = False
        self._fields_changed = False

        # MPI synchronization
        self._sync_all(force_sync_fields=True)

    def update(self, step: int, time: float) -> bool:
        """
        Run the update

        Returns: True if the physical environment changed and a full rebuild is due
        """
        self._params_changed = False
        self._fields_changed = False

        # 1. Check the independent parameter updates
        for name, cfg in self._param_update_config.items():
            if not cfg['enabled']:
                continue
            if not self._should_update(step, time, cfg):
                continue

            new_value = cfg['update_func'](self, step, time)
            if new_value is not None:
                self.set_param(name, new_value)
                cfg['last_update_step'] = step
                cfg['last_update_time'] = time

        # 2. Check the independent field updates
        for name, cfg in self._field_update_config.items():
            if not cfg['enabled']:
                continue
            if not self._should_update(step, time, cfg):
                continue
            if self._run_field_update(name, cfg, step, time, trigger_reason='policy'):
                pass

        # 3. Call the global update function
        user_changed = False
        if self.update_func is not None:
            user_changed = self.update_func(self, step, time, self.config)

        changed = user_changed or self._params_changed or self._fields_changed

        if changed:
            self.stats['triggered_rebuilds'] += 1
            self._sync_all()

        self.stats['total_updates'] += 1
        self._last_update_step = step

        return changed

    # =========================================================================
    # Split update interface (removes the risk of an MPI deadlock)
    # =========================================================================

    def check_updates(self, step: int, time: float) -> Tuple[np.ndarray, np.ndarray, bool]:
        """
        Only decide which quantities need updating, without updating them

        Used by the winner rank after executing an event: it decides what is due and
        broadcasts the decision to every rank.

        Args:
            step: current step
            time: current time

        Returns:
            (param_flags, field_flags, global_flag)
            - param_flags: a uint64[4] array, supporting up to 256 parameters
              bit i = 1 means parameter i needs its update_func called
            - field_flags: a uint64[4] array, supporting up to 256 fields
              bit i = 1 means field i needs its update_func called
            - global_flag: bool, whether the global update_func must be called

        [fix] The flags grew from 64 to 256 bits to support large simulations
        """
        # [fix] Four uint64 words support 256 parameters/fields
        param_flags = np.zeros(4, dtype=np.uint64)
        field_flags = np.zeros(4, dtype=np.uint64)

        # Check the parameter update conditions
        for name, cfg in self._param_update_config.items():
            if not cfg['enabled']:
                continue
            if self._should_update(step, time, cfg):
                idx = self._param_name_to_idx.get(name, -1)
                if idx >= 0 and idx < 256:
                    # Find the uint64 word and the bit inside it
                    array_idx = idx // 64
                    bit_idx = idx % 64
                    param_flags[array_idx] |= np.uint64(1 << bit_idx)

        # Check the field update conditions
        for name, cfg in self._field_update_config.items():
            if not cfg['enabled']:
                continue
            if self._should_update(step, time, cfg):
                idx = self._field_name_to_idx.get(name, -1)
                if idx >= 0 and idx < 256:
                    array_idx = idx // 64
                    bit_idx = idx % 64
                    field_flags[array_idx] |= np.uint64(1 << bit_idx)

        # Is there a global update_func?
        global_flag = self.update_func is not None

        return param_flags, field_flags, global_flag

    def apply_updates(self, step: int, time: float,
                      param_flags: np.ndarray, field_flags: np.ndarray,
                      global_flag: bool) -> bool:
        """
        Run the updates from the flags (every rank uses identical flags)

        Called by every rank after the broadcast, so the flags keep them consistent.

        Args:
            step: current step
            time: current time
            param_flags: uint64[4] array of parameter update flags (256 parameters)
            field_flags: uint64[4] array of field update flags (256 fields)
            global_flag: whether to call the global update_func

        Returns:
            bool: whether the physical environment changed (forcing a full rebuild)

        [fix] The flags grew from 64 to 256 bits to support large simulations
        """
        self._params_changed = False
        self._fields_changed = False

        # 1. Run the parameter updates from param_flags
        for name, cfg in self._param_update_config.items():
            if not cfg['enabled']:
                continue
            idx = self._param_name_to_idx.get(name, -1)
            if idx < 0 or idx >= 256:
                continue

            # [fix] Test the extended flag word
            array_idx = idx // 64
            bit_idx = idx % 64
            if not (param_flags[array_idx] & np.uint64(1 << bit_idx)):
                continue

            # Run the update
            new_value = cfg['update_func'](self, step, time)
            if new_value is not None:
                self.set_param(name, new_value)
                cfg['last_update_step'] = step
                cfg['last_update_time'] = time

        # 2. Run the field updates from field_flags
        for name, cfg in self._field_update_config.items():
            if not cfg['enabled']:
                continue
            idx = self._field_name_to_idx.get(name, -1)
            if idx < 0 or idx >= 256:
                continue

            # [fix] Test the extended flag word
            array_idx = idx // 64
            bit_idx = idx % 64
            if not (field_flags[array_idx] & np.uint64(1 << bit_idx)):
                continue

            if name in self._skip_scheduled_fields_once:
                self._skip_scheduled_fields_once.remove(name)
                continue

            # Run the update
            if self._run_field_update(name, cfg, step, time, trigger_reason='policy'):
                pass

        # 3. Call the global update function (when the flag is set)
        user_changed = False
        if global_flag and self.update_func is not None:
            user_changed = self.update_func(self, step, time, self.config)

        changed = user_changed or self._params_changed or self._fields_changed

        if changed:
            self.stats['triggered_rebuilds'] += 1
            self._sync_all()

        self.stats['total_updates'] += 1
        self._last_update_step = step

        return changed

    def _should_update(self, step: int, time: float, cfg: dict) -> bool:
        """Test whether the update condition holds"""
        step_interval = cfg.get('step_interval')
        time_interval = cfg.get('time_interval')
        last_step = cfg.get('last_update_step', -1)
        last_time = cfg.get('last_update_time', -float('inf'))

        if step_interval is None and time_interval is None:
            return False

        if step_interval is not None and step - last_step >= step_interval:
            return True

        if time_interval is not None and time - last_time >= time_interval:
            return True

        return False

    def _sync_all(self, force_sync_fields: bool = False):
        """MPI synchronization"""
        if self.size == 1:
            return

        # Synchronize the parameters
        self.comm.Bcast([self.params, MPI.DOUBLE], root=0)

        # Are the fields to be synchronized?
        need_sync_fields = force_sync_fields or self._fields_changed
        need_sync_fields = self.comm.bcast(need_sync_fields, root=0)

        if not need_sync_fields:
            return

        # Synchronize the fields
        n_fields = len(self.fields)
        n_fields = self.comm.bcast(n_fields, root=0)

        while len(self.fields) < n_fields:
            self.fields.append(None)
            self.field_axes.append(None)

        for i in range(n_fields):
            if self.rank == 0:
                x, y, z = self.field_axes[i]
                shape_info = (len(x), len(y), len(z))
            else:
                shape_info = (0, 0, 0)

            shape_info = self.comm.bcast(shape_info, root=0)
            nx, ny, nz = shape_info

            if self.rank != 0:
                x = np.empty(nx, dtype=np.float64)
                y = np.empty(ny, dtype=np.float64)
                z = np.empty(nz, dtype=np.float64)
                field = np.empty((nx, ny, nz), dtype=np.float64)
            else:
                x, y, z = self.field_axes[i]
                field = self.fields[i]

            self.comm.Bcast([x, MPI.DOUBLE], root=0)
            self.comm.Bcast([y, MPI.DOUBLE], root=0)
            self.comm.Bcast([z, MPI.DOUBLE], root=0)
            self.comm.Bcast([field, MPI.DOUBLE], root=0)

            if self.rank != 0:
                self.field_axes[i] = (x, y, z)
                self.fields[i] = field

            self._invalidate_interpolator(i)

    def finalize(self):
        """Release the resources"""
        for solver in self._external_solvers.values():
            solver.finalize()
        self._external_solvers.clear()

    # =========================================================================
    # Statistics and debugging
    # =========================================================================

    def print_summary(self):
        """Print the summary"""
        if self.rank != 0:
            return

        print("\n" + "=" * 60)
        print("UnifiedPhysics Summary (v1.6.8)")
        print("=" * 60)

        print(f"\nCoupling Mode: {self.coupling_mode}")
        print(f"Distributed Mode: {self.distributed_mode}")
        print(f"Cache Enabled: {self.enable_cache}")
        print(f"History Enabled: {self.enable_history}")
        if self.enable_history and self._history_initialized:
            print(f"  Time Delta (dt): {self.get_time_delta():.6e}")

        print(f"\nParameters ({self._param_count} defined):")
        for name, idx, value in self.list_params():
            print(f"  [{idx}] {name}: {value}")

        print(f"\nFields ({len(self.fields)} defined):")
        for name, idx, shape, is_adaptive in self.list_fields():
            field = self.fields[idx]
            grid_type = "adaptive" if is_adaptive else "uniform"
            mem_bytes = field.nbytes
            if mem_bytes < 1024:
                mem_str = f"{mem_bytes} B"
            elif mem_bytes < 1024 * 1024:
                mem_str = f"{mem_bytes / 1024:.1f} KB"
            else:
                mem_str = f"{mem_bytes / 1024 / 1024:.1f} MB"
            print(f"  [{idx}] {name}: shape={shape}, {grid_type}, "
                  f"mem={mem_str}, min={field.min():.4f}, max={field.max():.4f}")

        print(f"\nExternal Solvers ({len(self._external_solvers)}):")
        for name, fields in self._solver_to_fields.items():
            print(f"  {name}: {fields}")

        print(f"\nEvent Triggers ({len(self._event_triggers)}):")
        for event_type, config in self._event_triggers.items():
            print(f"  {event_type} -> {config['fields']}")

        print(f"\nStatistics:")
        print(f"  Total updates: {self.stats['total_updates']}")
        print(f"  Triggered rebuilds: {self.stats['triggered_rebuilds']}")
        print(f"  Event-triggered updates: {self.stats['event_triggered_updates']}")
        print(f"  Field compressions: {self.stats['field_compressions']}")

        if self.enable_cache and self._field_cache:
            cache_stats = self._field_cache.get_stats()
            print(f"\nCache Statistics:")
            print(f"  Hits: {cache_stats['hits']}")
            print(f"  Misses: {cache_stats['misses']}")
            print(f"  Hit rate: {cache_stats['hit_rate']:.2%}")

        print("=" * 60)

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get the cache statistics"""
        if self._field_cache:
            return self._field_cache.get_stats()
        return {}


# ===== Factory function =====

def create_unified_physics(comm: MPI.Comm, config: dict) -> UnifiedPhysics:
    """Factory: create the unified physics module"""
    physics_config = config.get('physics', {})

    if 'box_size' not in physics_config:
        dd_config = config.get('domain_decomposition', {})
        physics_config['box_size'] = dd_config.get('box_size', [100.0, 100.0, 100.0])

    # Merge the external field coupling configuration
    if 'field_coupling' in config:
        physics_config['field_coupling'] = config['field_coupling']

    return UnifiedPhysics(comm, physics_config)


# ===== Built-in IC/update functions =====

def constant_temperature_ic(physics: UnifiedPhysics, config: dict):
    """Built-in IC: a constant temperature"""
    temp = config.get('temperature', 300.0)
    physics.set_param('temperature', temp)
    physics.set_param('k_B', 8.617333262e-5)


def linear_heating_update(physics: UnifiedPhysics, step: int, time: float,
                          config: dict) -> bool:
    """Built-in update: linear heating"""
    T0 = config.get('initial_temperature', 300.0)
    rate = config.get('heating_rate', 1.0)
    new_temp = T0 + rate * time
    return physics.set_param('temperature', new_temp)


# ===== Example solver (for the documentation and the tests) =====

class DummySolver(ExternalSolver):
    """
    Example solver (it computes nothing real)

    Used by the tests and the documentation
    """

    def __init__(self, supported_fields: List[str] = None):
        self._supported_fields = supported_fields or ['temperature']

    def initialize(self, physics: UnifiedPhysics, config: dict) -> None:
        pass

    def solve(self, physics: UnifiedPhysics,
              step: int,
              time: float,
              object_data: Optional[np.ndarray] = None,
              trigger_source: Optional[str] = None) -> Optional[Dict[str, np.ndarray]]:
        # Example: return a simple temperature field
        if physics.rank != 0:
            return None

        result = {}
        for field_name in self._supported_fields:
            if field_name in physics._field_name_to_idx:
                idx = physics._field_name_to_idx[field_name]
                shape = physics.fields[idx].shape
                # A simple example: the temperature rises linearly with time
                result[field_name] = np.full(shape, 300.0 + time * 0.1)

        return result if result else None

    def finalize(self) -> None:
        pass

    def get_supported_fields(self) -> List[str]:
        return self._supported_fields


# ===== Test code =====

if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    rank = comm.rank

    print(f"[Rank {rank}] Testing UnifiedPhysics v1.6.8...")

    # Build the configuration
    config = {
        'physics': {
            'box_size': [100.0, 100.0, 100.0],
            'verbose': True,
        },
        'field_coupling': {
            'mode': MODE_TWO_WAY,
            'enable_cache': True,
        },
    }

    # Create the physics module
    physics = create_unified_physics(comm, config)

    # Test the parameters
    physics.set_param('temperature', 500.0)
    physics.set_param('k_B', 8.617e-5)
    physics.define_param('migration_energy', 0.5)

    # Test the fields
    physics.define_field('concentration', (50, 50, 50))
    conc = physics.get_field('concentration')
    conc[:] = np.random.rand(*conc.shape)

    # Test the event triggers
    physics.register_event_trigger('cascade', ['temperature', 'concentration'])
    physics.notify_event_executed('cascade', {'defect_count': 150})
    assert physics.has_pending_triggers()

    # Test the compression
    physics.define_field('stress', (100, 100, 100), initial_value=0.0)
    stress = physics.get_field('stress')
    # Build a field with a local high gradient
    for i in range(100):
        for j in range(100):
            for k in range(100):
                if 40 <= i <= 60 and 40 <= j <= 60:
                    stress[i, j, k] = np.sin(i * 0.5) * np.cos(j * 0.5)
    physics.compress_field('stress', gradient_threshold=0.1)

    # Print the summary
    physics.print_summary()

    # Clean up
    physics.finalize()

    print(f"[Rank {rank}] UnifiedPhysics v1.6.8 test passed!")
