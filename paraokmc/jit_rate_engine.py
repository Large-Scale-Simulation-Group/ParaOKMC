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
JIT Rate Engine — Dual Launcher Architecture for OKMC v1.6.12

A production module distilled from jit_prototype.py, supporting:
- pair events: obj1 + obj2 + distance
- single-object events: obj1 only, no obj2/distance
- zero-object events: no object properties, only physics/fields

Core components:
- RateDeps: declaration of data dependencies
- PropertyResolver: property name -> column index resolution, col_table construction
- generate_dual_launchers(): codegen for the fast and general launchers
- JITRateDispatcher: runtime dispatcher

Performance: 5-10x versus NumPy vectorization (Numba + OpenMP prange)
"""

import numpy as np
import textwrap
from typing import Dict, List, Tuple, Optional, Callable, Any

try:
    from numba import njit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False

    def njit(*args, **kwargs):
        """Fallback: no-op decorator when numba is not available."""
        if args and callable(args[0]):
            return args[0]
        def wrapper(fn):
            return fn
        return wrapper

    def prange(*args):
        return range(*args)


# ============================================================================
# Layer 1: user-facing API (declarative)
# ============================================================================

class RateDeps:
    """Declaration of the data a rate function depends on

    The user declares what the rate kernel needs; the framework binds the data.

    Args:
        obj1: property names of obj1 (read from float_props)
        obj2: property names of obj2 (pair events)
        params: event parameter names (unused for now, reserved)
        fields: field names (pre-interpolated at the object position)
        physics: names of scalar physics parameters
        kind: explicit event kind ('pair', 'single', 'zero');
              None (default) means infer automatically
    """
    def __init__(self, obj1=None, obj2=None, params=None,
                 fields=None, physics=None, kind=None):
        self.obj1 = obj1 or []
        self.obj2 = obj2 or []
        self.params = params or []
        self.fields = fields or []
        self.physics = physics or []
        self._kind = kind

    @property
    def event_kind(self):
        """Infer the event kind: 'pair', 'single' or 'zero'"""
        if self._kind is not None:
            return self._kind
        if self.obj1 and self.obj2:
            return 'pair'
        elif self.obj1:
            return 'single'
        else:
            return 'zero'


# ============================================================================
# Layer 2: codegen engine (pair / single / zero)
# ============================================================================

def _build_kernel_args_and_extract(rate_deps, mode, kind):
    """
    Build the kernel argument list and the extraction code lines

    mode: 'fast' or 'general'
    kind: 'pair', 'single', 'zero'
    """
    n_o1 = len(rate_deps.obj1)
    n_o2 = len(rate_deps.obj2)
    n_fields = len(rate_deps.fields)
    n_physics = len(rate_deps.physics)

    kernel_args = []
    extract_lines = []

    # obj1 properties
    if n_o1 > 0:
        if mode == 'fast':
            for k in range(n_o1):
                var = f'v_o1_{k}'
                extract_lines.append(f'            {var} = o1_props[i, o1_cols[{k}]]')
                kernel_args.append(var)
        else:
            for k in range(n_o1):
                var = f'v_o1_{k}'
                extract_lines.append(
                    f'            {var} = o1_props[i, o1_col_table[o1_tids[i], {k}]]')
                kernel_args.append(var)

    # obj2 properties (pair events only)
    if kind == 'pair' and n_o2 > 0:
        if mode == 'fast':
            for k in range(n_o2):
                var = f'v_o2_{k}'
                extract_lines.append(f'            {var} = o2_props[i, o2_cols[{k}]]')
                kernel_args.append(var)
        else:
            for k in range(n_o2):
                var = f'v_o2_{k}'
                extract_lines.append(
                    f'            {var} = o2_props[i, o2_col_table[o2_tids[i], {k}]]')
                kernel_args.append(var)

    # distance (pair events only)
    if kind == 'pair':
        kernel_args.append('distances[i]')

    # field values
    for k in range(n_fields):
        var = f'v_field_{k}'
        extract_lines.append(f'            {var} = field_vals[i, {k}]')
        kernel_args.append(var)

    # physics parameters
    for k in range(n_physics):
        var = f'v_phys_{k}'
        extract_lines.append(f'            {var} = physics_vals[{k}]')
        kernel_args.append(var)

    return kernel_args, extract_lines


def generate_dual_launchers(event_name, rate_deps, user_kernel):
    """
    Generate the two launchers for one event class (pair / single / zero)

    Returns:
        {'fast': launcher_func, 'general': launcher_func}
    """
    kind = rate_deps.event_kind
    launchers = {}
    namespace_base = {
        'njit': njit, 'prange': prange, 'np': np,
        '_user_kernel': user_kernel,
    }

    # ---- Fast launcher ----
    kernel_args, extract_lines = _build_kernel_args_and_extract(rate_deps, 'fast', kind)
    extract_block = '\n'.join(extract_lines) if extract_lines else ''
    call_args = ', '.join(kernel_args)

    if kind == 'pair':
        fast_sig = (f"def {event_name}_fast(o1_props, o2_props, o1_cols, o2_cols,\n"
                    f"                      distances, field_vals, physics_vals, out):")
        loop_var = 'distances'
    elif kind == 'single':
        fast_sig = (f"def {event_name}_fast(o1_props, o1_cols,\n"
                    f"                      field_vals, physics_vals, out):")
        loop_var = 'out'
    else:  # zero
        fast_sig = (f"def {event_name}_fast(field_vals, physics_vals, out):")
        loop_var = 'out'

    fast_code = textwrap.dedent(f"""\
    @njit(parallel=True)
    {fast_sig}
        n = len({loop_var})
        for i in prange(n):
{extract_block}
            out[i] = _user_kernel({call_args})
    """)

    ns_fast = dict(namespace_base)
    exec(fast_code, ns_fast)
    launchers['fast'] = ns_fast[f'{event_name}_fast']
    launchers['fast']._source_code = fast_code

    # ---- General launcher ----
    kernel_args_g, extract_lines_g = _build_kernel_args_and_extract(rate_deps, 'general', kind)
    extract_block_g = '\n'.join(extract_lines_g) if extract_lines_g else ''
    call_args_g = ', '.join(kernel_args_g)

    if kind == 'pair':
        gen_sig = (f"def {event_name}_general(o1_props, o2_props, o1_tids, o2_tids,\n"
                   f"                          o1_col_table, o2_col_table,\n"
                   f"                          distances, field_vals, physics_vals, out):")
        loop_var_g = 'distances'
    elif kind == 'single':
        gen_sig = (f"def {event_name}_general(o1_props, o1_tids,\n"
                   f"                          o1_col_table,\n"
                   f"                          field_vals, physics_vals, out):")
        loop_var_g = 'out'
    else:  # zero
        gen_sig = (f"def {event_name}_general(field_vals, physics_vals, out):")
        loop_var_g = 'out'

    general_code = textwrap.dedent(f"""\
    @njit(parallel=True)
    {gen_sig}
        n = len({loop_var_g})
        for i in prange(n):
{extract_block_g}
            out[i] = _user_kernel({call_args_g})
    """)

    ns_general = dict(namespace_base)
    exec(general_code, ns_general)
    launchers['general'] = ns_general[f'{event_name}_general']
    launchers['general']._source_code = general_code

    return launchers


# ============================================================================
# Layer 2.5: property mapping resolver + col table construction
# ============================================================================

class PropertyResolver:
    """
    Resolve property names -> float_props column indices
    Build col_table: an (n_types, n_deps) int32 array
    """
    def __init__(self, props_mapping):
        self.float_map = props_mapping.get('float_props', {})

    def resolve_columns(self, obj_type, prop_names):
        """Resolve a set of property names into an array of column indices"""
        type_map = self.float_map.get(obj_type, {})
        cols = []
        for name in prop_names:
            if name not in type_map:
                raise KeyError(
                    f"Property '{name}' not found for type '{obj_type}'. "
                    f"Available: {list(type_map.keys())}")
            cols.append(type_map[name])
        return np.array(cols, dtype=np.int32)

    def build_col_table(self, type_to_int, prop_names, relevant_types=None):
        """
        Build the column lookup table: col_table[type_id, dep_idx] = col_in_float_props

        Returns:
            col_table: (n_types, n_deps) int32 array
            is_uniform: bool, whether all types share one column layout
        """
        n_types = max(type_to_int.values()) + 1 if type_to_int else 1
        n_deps = len(prop_names)

        if n_deps == 0:
            return np.zeros((n_types, 0), dtype=np.int32), True

        col_table = np.full((n_types, n_deps), -1, dtype=np.int32)

        types_to_check = relevant_types or list(type_to_int.keys())
        cols_per_type = []

        for type_name in types_to_check:
            if type_name not in type_to_int:
                continue
            tid = type_to_int[type_name]
            type_map = self.float_map.get(type_name, {})

            row = []
            for dep_idx, prop_name in enumerate(prop_names):
                if prop_name in type_map:
                    col = type_map[prop_name]
                    col_table[tid, dep_idx] = col
                    row.append(col)
                else:
                    row.append(-1)
            cols_per_type.append(tuple(row))

        is_uniform = len(set(cols_per_type)) <= 1
        return col_table, is_uniform


# ============================================================================
# Utility functions
# ============================================================================

def build_type_to_int(props_mapping):
    """Build the type_to_int mapping automatically from props_mapping

    Walks all type names in float_props and assigns integer IDs alphabetically.

    Returns:
        {'H': 0, 'SIA': 1, 'V': 2, ...}
    """
    float_map = props_mapping.get('float_props', {})
    all_types = sorted(float_map.keys())
    return {t: i for i, t in enumerate(all_types)}


def prepare_inputs_pair(events_array, count, props_mapping, type_to_int):
    """Extract the contiguous arrays the JIT path needs from events (pair events)

    Returns:
        o1_props: (N, n_float_cols) float64
        o2_props: (N, n_float_cols) float64
        o1_type_ids: (N,) int32
        o2_type_ids: (N,) int32
        distances: (N,) float64
    """
    valid = events_array[:count]
    obj1 = valid['objects'][:, 0]
    obj2 = valid['objects'][:, 1]

    o1_props = obj1['float_props'].astype(np.float64)
    o2_props = obj2['float_props'].astype(np.float64)

    # Type string -> int ID
    o1_type_strs = np.char.strip(obj1['type'].astype(str))
    o2_type_strs = np.char.strip(obj2['type'].astype(str))

    o1_type_ids = np.zeros(count, dtype=np.int32)
    o2_type_ids = np.zeros(count, dtype=np.int32)
    for tname, tid in type_to_int.items():
        o1_type_ids[o1_type_strs == tname] = tid
        o2_type_ids[o2_type_strs == tname] = tid

    # distance normally sits in params[:, 0]
    distances = valid['params'][:, 0].astype(np.float64)

    return o1_props, o2_props, o1_type_ids, o2_type_ids, distances


def prepare_inputs_single(events_array, count, props_mapping, type_to_int):
    """Extract the contiguous arrays the JIT path needs (single-object events)

    Returns:
        o1_props: (N, n_float_cols) float64
        o1_type_ids: (N,) int32
    """
    valid = events_array[:count]
    obj1 = valid['objects'][:, 0]

    o1_props = obj1['float_props'].astype(np.float64)

    o1_type_strs = np.char.strip(obj1['type'].astype(str))
    o1_type_ids = np.zeros(count, dtype=np.int32)
    for tname, tid in type_to_int.items():
        o1_type_ids[o1_type_strs == tname] = tid

    return o1_props, o1_type_ids


# ============================================================================
# Layer 3: runtime dispatcher (dual-launcher version)
# ============================================================================

class JITRateDispatcher:
    """
    Dual-launcher runtime dispatcher

    Registration phase:
      - generate the fast and general launchers
      - build col_table
      - detect automatically whether the fast path is usable

    Runtime:
      - single entry point compute_rates()
      - selects the fast or the general launcher automatically
    """

    def __init__(self, props_mapping, type_to_int):
        self.resolver = PropertyResolver(props_mapping)
        self.type_to_int = type_to_int
        self.n_types = max(type_to_int.values()) + 1 if type_to_int else 1
        self.registry = {}

    def register(self, event_name, rate_deps, user_kernel, type_pairs=None,
                 object_types=None):
        """Register an event type and generate its two launchers

        Args:
            event_name: event name
            rate_deps: RateDeps instance
            user_kernel: @njit scalar kernel function
            type_pairs: list of type pairs (required for pair events)
            object_types: applicable types for single-object events
        """
        kind = rate_deps.event_kind

        # Generate the two launchers
        launchers = generate_dual_launchers(event_name, rate_deps, user_kernel)

        # Build col_table
        o1_col_table = o2_col_table = None
        o1_cols = o2_cols = None
        is_uniform = True

        if kind in ('pair', 'single'):
            # obj1 types
            if kind == 'pair' and type_pairs:
                obj1_types = list(set(t1 for t1, t2 in type_pairs))
                obj2_types = list(set(t2 for t1, t2 in type_pairs))
            elif kind == 'single' and object_types:
                obj1_types = list(object_types)
                obj2_types = []
            elif kind == 'single':
                obj1_types = list(self.type_to_int.keys())
                obj2_types = []
            else:
                obj1_types = list(self.type_to_int.keys())
                obj2_types = list(self.type_to_int.keys())

            o1_col_table, o1_uniform = self.resolver.build_col_table(
                self.type_to_int, rate_deps.obj1, obj1_types)

            if kind == 'pair' and rate_deps.obj2:
                o2_col_table, o2_uniform = self.resolver.build_col_table(
                    self.type_to_int, rate_deps.obj2, obj2_types)
                is_uniform = o1_uniform and o2_uniform
            else:
                o2_col_table = np.zeros((self.n_types, 0), dtype=np.int32)
                is_uniform = o1_uniform

            use_fast = is_uniform

            # If uniform, extract the fixed column indices
            if use_fast and obj1_types and rate_deps.obj1:
                o1_cols = self.resolver.resolve_columns(obj1_types[0], rate_deps.obj1)
            else:
                o1_cols = np.array([], dtype=np.int32)

            if use_fast and kind == 'pair' and obj2_types and rate_deps.obj2:
                o2_cols = self.resolver.resolve_columns(obj2_types[0], rate_deps.obj2)
            else:
                o2_cols = np.array([], dtype=np.int32)
        else:
            # zero events: no object properties
            use_fast = True
            o1_cols = np.array([], dtype=np.int32)
            o2_cols = np.array([], dtype=np.int32)

        self.registry[event_name] = {
            'launchers': launchers,
            'deps': rate_deps,
            'kind': kind,
            'use_fast': use_fast,
            'o1_col_table': o1_col_table,
            'o2_col_table': o2_col_table,
            'o1_cols': o1_cols,
            'o2_cols': o2_cols,
            'type_pairs': type_pairs or [],
        }

        mode_str = "FAST" if use_fast else "GENERAL"
        return mode_str

    def compute_rates(self, event_name, **kwargs):
        """
        Compute rates - single entry point, dispatched by event kind

        pair event kwargs:
            o1_props, o2_props, o1_type_ids, o2_type_ids,
            distances, field_vals, physics_vals

        single event kwargs:
            o1_props, o1_type_ids, field_vals, physics_vals

        zero event kwargs:
            field_vals, physics_vals, n_events
        """
        entry = self.registry[event_name]
        kind = entry['kind']

        if kind == 'pair':
            return self._compute_pair(entry, **kwargs)
        elif kind == 'single':
            return self._compute_single(entry, **kwargs)
        else:
            return self._compute_zero(entry, **kwargs)

    def _compute_pair(self, entry, o1_props, o2_props,
                      o1_type_ids, o2_type_ids, distances,
                      field_vals, physics_vals):
        N = len(distances)
        out = np.empty(N, dtype=np.float64)

        if entry['use_fast']:
            entry['launchers']['fast'](
                o1_props, o2_props,
                entry['o1_cols'], entry['o2_cols'],
                distances, field_vals, physics_vals, out)
        else:
            entry['launchers']['general'](
                o1_props, o2_props,
                o1_type_ids, o2_type_ids,
                entry['o1_col_table'], entry['o2_col_table'],
                distances, field_vals, physics_vals, out)
        return out

    def _compute_single(self, entry, o1_props, o1_type_ids,
                        field_vals, physics_vals):
        N = len(o1_type_ids)
        out = np.empty(N, dtype=np.float64)

        if entry['use_fast']:
            entry['launchers']['fast'](
                o1_props, entry['o1_cols'],
                field_vals, physics_vals, out)
        else:
            entry['launchers']['general'](
                o1_props, o1_type_ids,
                entry['o1_col_table'],
                field_vals, physics_vals, out)
        return out

    def _compute_zero(self, entry, field_vals, physics_vals, n_events):
        out = np.empty(n_events, dtype=np.float64)
        # zero events share the same signature for fast and general
        entry['launchers']['fast'](field_vals, physics_vals, out)
        return out

    def has_event(self, event_name):
        return event_name in self.registry

    def get_kind(self, event_name):
        return self.registry[event_name]['kind']


# ============================================================================
# Self-test
# ============================================================================

if __name__ == '__main__':
    print("=" * 60)
    print("JIT Rate Engine - Self Test")
    print("=" * 60)

    if not NUMBA_AVAILABLE:
        print("[SKIP] Numba not available, skipping JIT tests")
        import sys
        sys.exit(0)

    N = 1000
    N_COLS = 10

    props_mapping = {
        'float_props': {
            'H':   {'diffusivity': 2, 'radius': 0},
            'V':   {'diffusivity': 2, 'radius': 0},
            'SIA': {'diffusivity': 2, 'radius': 0},
        }
    }
    type_to_int = build_type_to_int(props_mapping)

    # ---- Test 1: Pair event ----
    print("\n[Test 1] Pair event (capture)")

    @njit
    def capture_kernel(o1_diff, o1_r, o2_r, dist, temperature, k_B):
        r_cap = o1_r + o2_r
        if dist < r_cap:
            return 0.0
        return 4.0 * 3.14159265 * o1_diff * r_cap

    capture_deps = RateDeps(
        obj1=['diffusivity', 'radius'],
        obj2=['radius'],
        physics=['temperature', 'k_B'],
    )
    assert capture_deps.event_kind == 'pair'

    disp = JITRateDispatcher(props_mapping, type_to_int)
    mode = disp.register('capture', capture_deps, capture_kernel,
                         type_pairs=[('H', 'V'), ('SIA', 'V')])
    print(f"  Mode: {mode}")

    np.random.seed(42)
    o1_props = np.random.rand(N, N_COLS)
    o2_props = np.random.rand(N, N_COLS)
    o1_tids = np.zeros(N, dtype=np.int32)
    o2_tids = np.ones(N, dtype=np.int32)
    dists = np.random.rand(N) * 5.0 + 0.1
    field_vals = np.empty((N, 0), dtype=np.float64)
    phys_vals = np.array([500.0, 8.617e-5], dtype=np.float64)

    rates = disp.compute_rates('capture',
                               o1_props=o1_props, o2_props=o2_props,
                               o1_type_ids=o1_tids, o2_type_ids=o2_tids,
                               distances=dists, field_vals=field_vals,
                               physics_vals=phys_vals)
    assert len(rates) == N
    assert np.all(rates >= 0)
    print(f"  Pair rates computed: {N} events, mean={rates.mean():.4f}")
    print("  PASSED")

    # ---- Test 2: Single event ----
    print("\n[Test 2] Single event (migration)")

    @njit
    def migration_kernel(E_m, nu, temperature, k_B):
        return nu * np.exp(-E_m / (k_B * temperature))

    migration_deps = RateDeps(
        obj1=['diffusivity', 'radius'],  # reuse as E_m, nu proxies
        physics=['temperature', 'k_B'],
    )
    assert migration_deps.event_kind == 'single'

    mode = disp.register('migration', migration_deps, migration_kernel)
    print(f"  Mode: {mode}")

    rates_s = disp.compute_rates('migration',
                                 o1_props=o1_props,
                                 o1_type_ids=o1_tids,
                                 field_vals=field_vals,
                                 physics_vals=phys_vals)
    assert len(rates_s) == N
    assert np.all(rates_s > 0)
    print(f"  Single rates computed: {N} events, mean={rates_s.mean():.6e}")
    print("  PASSED")

    # ---- Test 3: Zero event ----
    print("\n[Test 3] Zero event (cascade)")

    @njit
    def cascade_kernel(cascade_rate):
        return cascade_rate

    cascade_deps = RateDeps(physics=['cascade_rate'])
    assert cascade_deps.event_kind == 'zero'

    mode = disp.register('cascade', cascade_deps, cascade_kernel)
    print(f"  Mode: {mode}")

    phys_cascade = np.array([1e6], dtype=np.float64)
    rates_z = disp.compute_rates('cascade',
                                 field_vals=np.empty((5, 0), dtype=np.float64),
                                 physics_vals=phys_cascade,
                                 n_events=5)
    assert len(rates_z) == 5
    assert np.allclose(rates_z, 1e6)
    print(f"  Zero rates: {rates_z}")
    print("  PASSED")

    # ---- Test 4: build_type_to_int ----
    print("\n[Test 4] build_type_to_int")
    t2i = build_type_to_int(props_mapping)
    assert 'H' in t2i and 'V' in t2i and 'SIA' in t2i
    assert len(set(t2i.values())) == 3
    print(f"  type_to_int: {t2i}")
    print("  PASSED")

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)
