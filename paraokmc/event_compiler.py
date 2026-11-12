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
Event compiler

Compiles user-defined event classes into the v1.6.8-compatible configuration
format so they work directly with parallel_event_builder and v16_engine.

Main features:
1. Parse the attributes and methods of an event class
2. Generate builder_func, filter_func, rate_func and execute_func
3. Generate the event parameter mapping
4. Cooperate with TypeCompiler to build the dynamic dtype
"""

import numpy as np
from typing import Dict, List, Type, Any, Callable, Optional, Tuple
from .event_class import (
    EventBase, ZeroObjectEvent, SingleObjectEvent, PairObjectEvent,
    EventRegistry, compile_event_config, get_object_type_name, get_event_type_name
)


def _is_jit_function(func: Callable) -> bool:
    """
    Detect whether a function is Numba JIT compiled

    Args:
        func: the function to test

    Returns:
        True for a JIT function, False for a plain Python function
    """
    if func is None:
        return False

    # Look for Numba's marker attributes
    # A Numba JIT function carries __wrapped__ or _dispatcher
    if hasattr(func, '__wrapped__') or hasattr(func, '_dispatcher'):
        return True

    # Inspect the module and type of the function
    func_type = type(func).__name__
    if 'Dispatcher' in func_type or 'CPUDispatcher' in func_type:
        return True

    return False


def _auto_detect_weight(event_class: Type[EventBase]) -> float:
    """
    Detect the computational weight of an event class automatically

    Rules:
    - if the user set weight explicitly to something other than 1.0, honour it
    - if rate_kernel is a JIT function, the weight is 1.0 (fast)
    - if rate_kernel is a Python function, the weight is 1000.0 (slow)

    Args:
        event_class: the event class

    Returns:
        the computational weight
    """
    # Honour an explicitly set non-default weight
    if hasattr(event_class, 'weight') and event_class.weight != 1.0:
        return event_class.weight

    # Detect the kind of rate_kernel
    if hasattr(event_class, 'rate_kernel'):
        rate_kernel = event_class.rate_kernel
        if _is_jit_function(rate_kernel):
            return 1.0  # JIT function, fast
        else:
            return 1000.0  # Python function, slow

    # Default weight
    return 1.0


class EventCompiler:
    """
    Event compiler

    Compiles event classes into v1.6.8-compatible functions and configuration.
    """

    def __init__(self, registry: EventRegistry, config: Dict):
        """
        Args:
            registry: event registry
            config: global configuration
        """
        self.registry = registry
        self.config = config

        # Cache of compilation results
        self._compiled_builders = {}
        self._compiled_filters = {}
        self._compiled_rate_funcs = {}
        self._compiled_execute_funcs = {}

    def compile_all(self) -> Dict:
        """
        Compile all events and return the v1.6.8-compatible configuration

        Returns:
            {
                'single_obj_builders': {...},
                'pair_type_filters': {...},
                'pair_param_extractors': {...},
                'zero_obj_builders': {...},
                'calculate_rate_func': callable,
                'execute_event_func': callable,
                'event_weights': {...},
                'events': {...},  # raw event definitions (consumed by TypeCompiler)
            }
        """
        result = {
            'single_obj_builders': {},
            'pair_type_filters': {},
            'pair_param_extractors': {},
            'zero_obj_builders': {},
            'event_weights': {},
            'events': {},
            'vectorized_pair_filters': {},
            'vectorized_zero_filters': {},
        }

        # Compile zero-object events
        for name in self.registry.get_zero_events():
            event_class = self.registry.get(name)
            spatial_mode = getattr(event_class, 'spatial_mode', 'global')

            if spatial_mode == 'per_cell':
                # per_cell mode: handled by VectorizedZeroEventBuilder
                # generate the vectorized_zero_filters configuration
                result['vectorized_zero_filters'][name] = {
                    'spatial_mode': 'per_cell',
                    'field_deps': list(getattr(event_class, 'field_deps', [])),
                    'params': list(event_class.params),
                }
            else:
                # global mode: generate the traditional builder_func
                result['zero_obj_builders'][name] = self._compile_zero_builder(name, event_class)

            # Detect the weight automatically
            result['event_weights'][name] = _auto_detect_weight(event_class)
            result['events'][name] = self._build_event_def(name, event_class)

        # Compile single-object events
        for name in self.registry.get_single_events():
            event_class = self.registry.get(name)
            result['single_obj_builders'][name] = self._compile_single_builder(name, event_class)
            # Detect the weight automatically
            result['event_weights'][name] = _auto_detect_weight(event_class)
            result['events'][name] = self._build_event_def(name, event_class)

        # Compile pair events
        for name in self.registry.get_pair_events():
            event_class = self.registry.get(name)
            filter_func, extractor_func = self._compile_pair_functions(name, event_class)
            result['pair_type_filters'][name] = filter_func
            result['pair_param_extractors'][name] = extractor_func
            # Detect the weight automatically
            result['event_weights'][name] = _auto_detect_weight(event_class)
            result['events'][name] = self._build_event_def(name, event_class)

            # Generate vectorized_pair_filters (used by the JIT filter)
            if hasattr(event_class, 'type_pairs') and event_class.type_pairs:
                result['vectorized_pair_filters'][name] = {
                    'type_pairs': list(event_class.type_pairs),
                    'max_distance': event_class.max_distance,
                }

        # Generate the unified rate and execute functions
        result['calculate_rate_func'] = self._compile_unified_rate_func()
        result['execute_event_func'] = self._compile_unified_execute_func()

        return result

    def _build_event_def(self, name: str, event_class: Type[EventBase]) -> Dict:
        """
        Build the event definition (consumed by TypeCompiler)

        Returns:
            a v1.6.8-compatible event definition dict

        Extracts the raw parameter values from base_config and merges them
        """
        num_obj = event_class.get_num_objects()

        event_def = {
            'objects': event_class.object_types if num_obj <= 1 else [],
            'rate_params': event_class.params,
            'weight': event_class.weight,
        }

        if num_obj == 2:
            # Pair events
            event_def['max_distance'] = event_class.max_distance
            if hasattr(event_class, 'type_pairs') and event_class.type_pairs:
                # Collect all participating types from type_pairs
                all_types = set()
                for t1, t2 in event_class.type_pairs:
                    all_types.add(t1)
                    all_types.add(t2)
                event_def['objects'] = list(all_types)

        if hasattr(event_class, 'product_type'):
            event_def['product_type'] = event_class.product_type

        # Extract the raw parameter values from base_config
        # If base_config['events'][name] exists, merge its parameter values
        if 'events' in self.config and name in self.config['events']:
            original_event = self.config['events'][name]
            for key, value in original_event.items():
                if key not in event_def:
                    event_def[key] = value

        return event_def

    def _compile_zero_builder(self, name: str,
                               event_class: Type[ZeroObjectEvent]) -> Callable:
        """
        Compile the zero-object event builder

        Returns:
            builder_func(local_objects, all_objects, config, rank) -> List[events]
        """
        def zero_builder(local_objects, all_objects, config, rank):
            comm = config.get('_comm')
            size = comm.Get_size() if comm else 1

            # Call the event class filter
            if not event_class.filter(config, rank, size):
                return []

            # Create the event
            from .rma_event_pool import create_event
            import numpy as np

            # [64-bit ID scheme] Generate the zero-object event ID (negative space)
            # Layout: [10][Rank:20bit][Type:16bit][Index:26bit]
            # Index of the event type
            zero_events = config.get('_event_registry').get_zero_events()
            type_index = zero_events.index(name) if name in zero_events else 0

            index = 0  # the first event, index 0
            val = (2 << 62) | (rank << 42) | (type_index << 26) | index
            # Convert to unsigned first, then view as signed, to avoid overflow
            event_id = np.array([val], dtype=np.uint64).view(np.int64)[0]

            event = create_event(
                event_id=event_id,
                event_type=name,
                obj_ids=[-1, -1],  # zero-object events have no object IDs
                params=list(event_class.extract_params([], config, event_name=name).values()),
                owner_rank=rank,
                dtype=config.get('_event_dtype'),
                config=config,
            )

            return [event]

        return zero_builder

    def _compile_single_builder(self, name: str,
                                  event_class: Type[SingleObjectEvent]) -> Callable:
        """
        Compile the single-object event builder

        Returns:
            builder_func(objects, config, rank) -> List[events]
        """
        def single_builder(objects, config, rank):
            if len(objects) == 0:
                return []

            # Extract the object type for filtering
            obj_types = np.array([get_object_type_name(obj, config) for obj in objects])

            # Type filter: keep only the types listed in object_types
            if event_class.object_types:
                mask = np.zeros(len(objects), dtype=bool)
                for allowed_type in event_class.object_types:
                    mask |= (obj_types == allowed_type)
            else:
                mask = np.ones(len(objects), dtype=bool)

            # Call the event class filter() (if a subclass overrides it)
            custom_filter = event_class.filter(obj_types, {}, config)
            if isinstance(custom_filter, np.ndarray):
                mask &= custom_filter
            elif isinstance(custom_filter, bool) and not custom_filter:
                return []

            # Filter the objects
            filtered_indices = np.where(mask)[0]

            if len(filtered_indices) == 0:
                return []

            # Create the events
            from .rma_event_pool import create_event

            events = []
            for idx in filtered_indices:
                obj = objects[idx]
                obj_id = int(obj['id'])

                # Extract the parameters
                single_props = {k: v[idx] for k, v in obj_props.items()}
                # Convert the NumPy structured record into a dict
                obj_dict = {name: obj[name] for name in obj.dtype.names}
                params_dict = event_class.extract_params([obj_dict], config, event_name=name)

                # Generate the event ID (simple hash scheme)
                # Layout: [rank:16][obj_id:32][type_hash:16]
                type_hash = hash(name) & 0xFFFF
                id_high = (rank << 48) | (obj_id & 0xFFFFFFFF) << 16 | type_hash
                id_low = obj_id

                event = create_event(
                    event_type=name,
                    obj_ids=[obj_id],
                    params=list(params_dict.values())[:8],  # at most 8 parameters
                    owner_rank=rank,
                    weight=event_class.weight,
                    dtype=config.get('_event_dtype'),
                    id_high=id_high,
                    id_low=id_low,
                    config=config,
                )

                events.append(event)

            return events

        return single_builder

    def _compile_pair_functions(self, name: str,
                                 event_class: Type[PairObjectEvent]) -> Tuple[Callable, Callable]:
        """
        Compile the filter and parameter extractor of pair events

        Returns:
            (filter_func, param_extractor_func)
        """
        # Type filter
        def pair_filter(obj1, obj2, distance=0.0, config=None, rank=0):
            # Scalar call (v1.6.8 compatible)
            obj1_type = get_object_type_name(obj1, config)
            obj2_type = get_object_type_name(obj2, config)

            # Distance check
            if distance > event_class.max_distance:
                return False

            # Type check
            if event_class.type_pairs:
                for t1, t2 in event_class.type_pairs:
                    if (obj1_type == t1 and obj2_type == t2) or \
                       (obj1_type == t2 and obj2_type == t1):
                        return True
                return False

            return True

        # Parameter extractor
        config = self.config

        def _record_to_dict(obj):
            """Turn a NumPy structured record or dict into a dict with named fields.

            [fix] The old implementation ran `param_name in obj` on an np.void,
            which compares field *values* rather than names and raises ValueError
            for array fields; swallowed upstream, every named parameter became 0.
            """
            if isinstance(obj, dict):
                return dict(obj)

            d = {}
            names = getattr(getattr(obj, 'dtype', None), 'names', None) or ()
            for field in names:
                if field == 'float_props':
                    continue
                value = obj[field]
                if getattr(value, 'ndim', 0) == 0:
                    d[field] = value.item() if hasattr(value, 'item') else value

            # Type name (used by extract_params / _props_mapping)
            int_to_type = config.get('_int_to_type', {})
            if 'type_id' in names and int_to_type:
                d['type'] = int_to_type.get(int(obj['type_id']), '')
            elif 'type' in names:
                d['type'] = str(obj['type']).strip()

            # Named float_props (expanded through the TypeCompiler mapping)
            if 'float_props' in names:
                fp = obj['float_props']
                d['float_props'] = list(fp)
                mapping = (config.get('_props_mapping', {})
                           .get('float_props', {})
                           .get(d.get('type', ''), {}))
                for prop_name, slot_idx in mapping.items():
                    if isinstance(slot_idx, (int, np.integer)) and slot_idx < len(fp):
                        d[prop_name] = float(fp[slot_idx])

            return d

        def param_extractor(obj1, obj2):
            # Distance (minimum image along periodic axes)
            dx = float(obj1['x']) - float(obj2['x'])
            dy = float(obj1['y']) - float(obj2['y'])
            dz = float(obj1['z']) - float(obj2['z'])
            dd = config.get('domain_decomposition', {})
            box_size = dd.get('box_size')
            periodic = dd.get('periodic', [True, True, True])
            if box_size is not None:
                deltas = [dx, dy, dz]
                for axis in range(3):
                    if periodic[axis]:
                        L = float(box_size[axis])
                        deltas[axis] -= L * np.rint(deltas[axis] / L)
                dx, dy, dz = deltas
            distance = float(np.sqrt(dx*dx + dy*dy + dz*dz))

            d1 = _record_to_dict(obj1)
            d2 = _record_to_dict(obj2)
            # obj1 fields win; obj2 fills the gaps (matching the old obj1->obj2 order)
            combined = dict(d2)
            combined.update(d1)

            values = event_class.extract_params([combined, d2], config,
                                                event_name=name)

            params = []
            for param_name in event_class.params:
                if param_name == 'distance':
                    params.append(distance)
                else:
                    try:
                        params.append(float(values.get(param_name, 0.0)))
                    except (TypeError, ValueError):
                        params.append(0.0)

            # [fix] The truncation limit follows the params slot count of the dynamic
            # dtype (it used to be hardcoded to 8, while TypeCompiler may emit more)
            event_dtype = config.get('_event_dtype')
            if event_dtype is not None and 'params' in (event_dtype.names or ()):
                max_params = event_dtype.fields['params'][0].shape[0]
            else:
                max_params = 8
            return params[:max_params]

        return pair_filter, param_extractor

    def _compile_unified_rate_func(self) -> Callable:
        """
        Compile the unified rate function

        Calls rate() directly (a scalar Python function)
        The JIT path is handled by _compute_rates_jit_batch in v16_engine

        Returns:
            calculate_rate_func(event_type, objs, params, config) -> float
        """
        registry = self.registry

        def calculate_rate(event_type: str, objs: List, params: Dict, config: Dict) -> float:
            event_class = registry.get(event_type)
            if event_class is None:
                return 0.0

            # Call the scalar rate() directly
            if event_class.has_scalar_rate():
                return event_class.rate(objs, params, config)

            return 0.0

        return calculate_rate

    def _compile_unified_execute_func(self) -> Callable:
        """
        Compile the unified event execute function

        Returns:
            execute_event_func(event, objs, config, product_type=None) -> List[Dict]
        """
        registry = self.registry

        def execute_event(event, objs: List, config: Dict, product_type: str = None) -> List[Dict]:
            event_type = None
            if 'event_type_id' in event.dtype.names:
                event_type = config.get('_event_int_to_name', {}).get(int(event['event_type_id']))
            if event_type is None:
                event_type = get_event_type_name(event, config)
            event_class = registry.get(event_type)

            if event_class is None:
                return []

            # Use product_type if the event class defines it
            if product_type is None and hasattr(event_class, 'product_type'):
                product_type = event_class.product_type

            # Call the execute method of the event class
            return event_class.execute(event, objs, config)

        return execute_event

    def _extract_object_props(self, objects: np.ndarray, config: Dict) -> Dict[str, np.ndarray]:
        """
        Extract a property dict from the object array

        Args:
            objects: NumPy structured array of objects
            config: configuration (holding the property mapping)

        Returns:
            {prop_name: np.ndarray}
        """
        props = {}
        n = len(objects)

        # Basic properties
        props['x'] = np.array([float(obj['x']) for obj in objects])
        props['y'] = np.array([float(obj['y']) for obj in objects])
        props['z'] = np.array([float(obj['z']) for obj in objects])
        props['id'] = np.array([int(obj['id']) for obj in objects])

        # Read from float_props (needs the property mapping)
        prop_mapper = config.get('_property_mapper')
        if prop_mapper and hasattr(prop_mapper, 'reverse_map'):
            for slot_idx, prop_name in prop_mapper.reverse_map.get('float', {}).items():
                try:
                    props[prop_name] = np.array([float(obj['float_props'][slot_idx])
                                                  for obj in objects])
                except (IndexError, KeyError):
                    pass

        return props


def compile_events(events_config: Dict[str, Type[EventBase]], config: Dict) -> Tuple[Dict, 'EventRegistry']:
    """
    Convenience function: compile the event configuration

    Args:
        events_config: {'event_name': EventClass, ...}
        config: global configuration

    Returns:
        (v1.6.8-compatible configuration dict, EventRegistry instance)
    """
    registry = compile_event_config(events_config)
    compiler = EventCompiler(registry, config)
    return compiler.compile_all(), registry


# ============================================================================
# Configuration merging helpers
# ============================================================================

def merge_event_config(base_config: Dict, events_config: Dict[str, Type[EventBase]]) -> Dict:
    """
    Merge an event class configuration into the base configuration

    Args:
        base_config: the user's base configuration
        events_config: event class configuration {'name': EventClass, ...}

    Returns:
        the merged full configuration
    """
    # Merge into the base configuration
    # [fix] result must be built before compiling: the filter/extractor closures
    # produced by the compiler capture the config dict passed in. At runtime the
    # engine injects _props_mapping/_int_to_type into result (engine.config is that
    # dict) - had the closures captured base_config, those keys would stay invisible
    result = base_config.copy()

    # Compile the events (closures capture result, shared with the engine)
    compiled, registry = compile_events(events_config, result)

    # Merge the event-related configuration
    result['single_obj_builders'] = {
        **result.get('single_obj_builders', {}),
        **compiled['single_obj_builders']
    }
    result['pair_type_filters'] = {
        **result.get('pair_type_filters', {}),
        **compiled['pair_type_filters']
    }
    result['pair_param_extractors'] = {
        **result.get('pair_param_extractors', {}),
        **compiled['pair_param_extractors']
    }
    result['zero_obj_builders'] = {
        **result.get('zero_obj_builders', {}),
        **compiled['zero_obj_builders']
    }
    result['event_weights'] = {
        **result.get('event_weights', {}),
        **compiled['event_weights']
    }
    result['vectorized_pair_filters'] = {
        **result.get('vectorized_pair_filters', {}),
        **compiled.get('vectorized_pair_filters', {}),
    }
    result['vectorized_zero_filters'] = {
        **result.get('vectorized_zero_filters', {}),
        **compiled.get('vectorized_zero_filters', {}),
    }

    # Install the unified rate and execute functions
    result['calculate_rate_func'] = compiled['calculate_rate_func']
    result['execute_event_func'] = compiled['execute_event_func']

    # Keep the event definitions (consumed by TypeCompiler)
    result['events'] = {
        **result.get('events', {}),
        **compiled['events']
    }

    # Reuse the compiled registry (re-registering triggers auto-wrap warnings)
    result['_event_registry'] = registry

    # Global event type mapping (int on the hot path, names for debugging)
    existing_event_name_to_int = dict(result.get('_event_name_to_int', {}))
    next_event_type_id = (max(existing_event_name_to_int.values()) + 1
                          if existing_event_name_to_int else 0)
    for event_name in registry.get_all_events().keys():
        if event_name not in existing_event_name_to_int:
            existing_event_name_to_int[event_name] = next_event_type_id
            next_event_type_id += 1
    result['_event_name_to_int'] = existing_event_name_to_int
    result['_event_int_to_name'] = {
        event_type_id: event_name
        for event_name, event_type_id in existing_event_name_to_int.items()
    }

    # Infrastructure defaults; the user may override them in base_config
    # per_cell zero events need a larger initial capacity
    has_per_cell = bool(compiled.get('vectorized_zero_filters', {}))
    default_zero_capacity = 5000 if has_per_cell else 10

    result.setdefault('rma_object_pool', {'initial_capacity': 1000, 'auto_expand': True})
    result.setdefault('rma_event_pool', {
        'capacity_zero': default_zero_capacity, 'capacity_single': 5000,
        'capacity_pair': 5000, 'auto_expand': True,
    })
    result.setdefault('incremental_update', {'enabled': True})
    result.setdefault('predictive_rebalance', {'enabled': True, 'threshold': 1.5})
    result.setdefault('runtime_load_balancer', {'enabled': True, 'check_interval': 32})
    result.setdefault('adaptive_weight_tracker', {'enabled': True})
    result.setdefault('max_steps', 1000)
    result.setdefault('max_time', float('inf'))

    return result


# ============================================================================
# Self-test
# ============================================================================

if __name__ == '__main__':
    from .event_class import SingleObjectEvent, PairObjectEvent

    print("=" * 60)
    print("v1.6.9 Event Compiler - Self Test (Unified Signature)")
    print("=" * 60)

    # Define the test event classes (using the scalar rate() interface)
    class TestMigration(SingleObjectEvent):
        """Test migration event"""
        object_types = ['H', 'V']
        params = ['E_m', 'nu']
        weight = 1.0

        @classmethod
        def rate(cls, objs, params, config):
            T = config.get('physics', {}).get('temperature', 500.0)
            k_B = 8.617e-5
            E_m = params.get('E_m', 0.5)
            nu = params.get('nu', 1e13)
            return float(nu * np.exp(-E_m / (k_B * T)))

        @classmethod
        def execute(cls, event, objs, config):
            obj = objs[0]
            return [{'id': int(obj['id']), 'x': float(obj['x']) + 0.1}]

    class TestCapture(PairObjectEvent):
        """Test capture event"""
        type_pairs = [('H', 'V')]
        max_distance = 5.0
        params = ['distance']
        weight = 2.0
        product_type = 'HV'

        @classmethod
        def rate(cls, objs, params, config):
            D = 1e-9
            r = params.get('distance', 0.0)
            return float(4 * np.pi * D * r)

        @classmethod
        def execute(cls, event, objs, config):
            obj1, obj2 = objs[0], objs[1]
            return [
                {'id': int(obj1['id']), 'deleted': True},
                {'id': int(obj2['id']), 'deleted': True},
                {'id': -1, 'type': 'HV', 'x': 0.0, 'y': 0.0, 'z': 0.0, 'new': True}
            ]

    # Test 1: compile the events
    print("\n[Test 1] Compile Events")

    base_config = {
        'physics': {'temperature': 500.0},
        'domain_decomposition': {'box_size': [100, 100, 100]},
    }

    compiled, _ = compile_events({
        'migration': TestMigration,
        'capture': TestCapture,
    }, base_config)

    print(f"  single_obj_builders: {list(compiled['single_obj_builders'].keys())}")
    print(f"  pair_type_filters: {list(compiled['pair_type_filters'].keys())}")
    print(f"  vectorized_pair_filters: {list(compiled['vectorized_pair_filters'].keys())}")
    print(f"  event_weights: {compiled['event_weights']}")
    print(f"  events: {list(compiled['events'].keys())}")

    assert 'migration' in compiled['single_obj_builders']
    assert 'capture' in compiled['pair_type_filters']
    assert 'capture' in compiled['vectorized_pair_filters']
    assert compiled['vectorized_pair_filters']['capture']['type_pairs'] == [('H', 'V')]
    assert compiled['vectorized_pair_filters']['capture']['max_distance'] == 5.0
    assert compiled['event_weights']['migration'] == 1.0
    assert compiled['event_weights']['capture'] == 2.0
    print("  PASSED")

    # Test 2: rate functions
    print("\n[Test 2] Unified Rate Function")

    rate_func = compiled['calculate_rate_func']

    # Single-object rate
    obj = {'id': 1, 'type': 'H', 'x': 0.0, 'y': 0.0, 'z': 0.0}
    params = {'E_m': 0.5, 'nu': 1e13}
    rate = rate_func('migration', [obj], params, base_config)
    print(f"  Migration rate: {rate:.2e}")
    assert rate > 0

    # Pair rate
    obj1 = {'id': 1, 'type': 'H', 'x': 0.0, 'y': 0.0, 'z': 0.0}
    obj2 = {'id': 2, 'type': 'V', 'x': 1.0, 'y': 0.0, 'z': 0.0}
    params = {'distance': 1.0}
    rate = rate_func('capture', [obj1, obj2], params, base_config)
    print(f"  Capture rate: {rate:.2e}")
    assert rate > 0
    print("  PASSED")

    # Test 3: execute functions
    print("\n[Test 3] Unified Execute Function")

    exec_func = compiled['execute_event_func']

    # Mock event
    event = {'type': 'migration', 'id': 100}
    changed = exec_func(event, [obj], base_config)
    print(f"  Migration changed: {changed}")
    assert len(changed) == 1
    assert changed[0]['x'] == 0.1

    event = {'type': 'capture', 'id': 200}
    changed = exec_func(event, [obj1, obj2], base_config)
    print(f"  Capture changed: {changed}")
    assert len(changed) == 3
    assert changed[0]['deleted'] == True
    assert changed[1]['deleted'] == True
    assert changed[2]['new'] == True
    print("  PASSED")

    # Test 4: configuration merging
    print("\n[Test 4] Merge Event Config")

    merged = merge_event_config(base_config, {
        'migration': TestMigration,
        'capture': TestCapture,
    })

    print(f"  Has calculate_rate_func: {'calculate_rate_func' in merged}")
    print(f"  Has execute_event_func: {'execute_event_func' in merged}")
    print(f"  Has _event_registry: {'_event_registry' in merged}")

    assert 'calculate_rate_func' in merged
    assert 'execute_event_func' in merged
    assert '_event_registry' in merged
    print("  PASSED")

    # Test 5: pair filter
    print("\n[Test 5] Pair Filter Function")

    pair_filter = compiled['pair_type_filters']['capture']

    # H-V pair, within range
    result = pair_filter(
        {'type': 'H', 'x': 0, 'y': 0, 'z': 0},
        {'type': 'V', 'x': 1, 'y': 0, 'z': 0},
        distance=1.0,
        config=base_config
    )
    print(f"  H-V at distance 1.0: {result}")
    assert result == True

    # H-H pair (not allowed)
    result = pair_filter(
        {'type': 'H', 'x': 0, 'y': 0, 'z': 0},
        {'type': 'H', 'x': 1, 'y': 0, 'z': 0},
        distance=1.0,
        config=base_config
    )
    print(f"  H-H at distance 1.0: {result}")
    assert result == False

    # H-V pair, out of range
    result = pair_filter(
        {'type': 'H', 'x': 0, 'y': 0, 'z': 0},
        {'type': 'V', 'x': 10, 'y': 0, 'z': 0},
        distance=10.0,
        config=base_config
    )
    print(f"  H-V at distance 10.0: {result}")
    assert result == False
    print("  PASSED")

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)
