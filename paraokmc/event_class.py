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
Event class system

Users define their own event types by subclassing EventBase, which offers:
1. event parameter declaration (params)
2. an event filter (filter)
3. the object types involved (object_types)
4. rate evaluation (rate / rate_deps+rate_kernel) - JIT or scalar Python
5. event execution (execute)

Two rate evaluation paths:
- JIT path: declare rate_deps + rate_kernel -> Numba @njit batch evaluation (fastest)
- Python path: define rate() -> one scalar Python call per event

Reading physics parameters:
    the get_physics_param() helper reads parameters from the UnifiedPhysics module:

    from .event_class import get_physics_param, get_physics_field

    class MigrationEvent(SingleObjectEvent):
        @classmethod
        def rate(cls, objs, params, config):
            T = get_physics_param(config, 'temperature', default=500.0)
            k_B = 8.617e-5
            E_m = params.get('E_m', 0.5)
            nu = params.get('nu', 1e13)
            return float(nu * np.exp(-E_m / (k_B * T)))

Usage:
    class MigrationEvent(SingleObjectEvent):
        object_types = ['H', 'V', 'SIA']  # applicable object types
        params = ['E_m', 'nu']            # names of the event parameters

        @classmethod
        def rate(cls, objs, params, config):
            T = get_physics_param(config, 'temperature', default=500.0)
            k_B = 8.617e-5
            E_m = params.get('E_m', 0.5)
            nu = params.get('nu', 1e13)
            return float(nu * np.exp(-E_m / (k_B * T)))

        @classmethod
        def execute(cls, event, objs, config):
            obj = objs[0]
            return [{'id': obj['id'], 'x': obj['x'] + dx, ...}]

    # Configuration
    config = merge_event_config(base_config, {
        'migration': MigrationEvent,
        'capture': CaptureEvent,
    })
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple, Union, Type, Callable
from dataclasses import dataclass, field

_RUNTIME_INT_TO_TYPE: Dict[int, str] = {}
_RUNTIME_EVENT_INT_TO_NAME: Dict[int, str] = {}


# ============================================================================
# Execute API - result collector
# ============================================================================

class ExecuteResult:
    """
    Collector of the results of an event execution

    Used inside execute() to gather object changes behind a clear API:
    - result.delete_obj(obj): delete an object
    - result.change_obj(obj, **updates): modify an object
    - result.new_obj(obj_type, **properties): create an object

    Example:
        @classmethod
        def execute(cls, event, objs, config):
            result = ExecuteResult()
            obj = objs[0]

            # move the object
            result.change_obj(obj, x=obj['x'] + 1.0, y=obj['y'])

            # or delete it
            # result.delete_obj(obj)

            # or create a new one
            # result.new_obj('SIA', x=1.0, y=2.0, z=3.0, float_props=[0.0, 5.0])

            return result
    """

    def __init__(self):
        self._changes = []

    def delete_obj(self, obj: np.ndarray):
        """
        Mark an object as deleted

        Args:
            obj: object record (NumPy structured array)
        """
        self._changes.append({'id': int(obj['id']), 'active': False})

    def change_obj(self, obj: np.ndarray, **updates):
        """
        Modify the properties of an object

        Args:
            obj: object record (NumPy structured array)
            **updates: properties to update (e.g. x=1.0, y=2.0, float_props=[0.0, 5.0])
        """
        change = {'id': int(obj['id'])}
        change.update(updates)
        self._changes.append(change)

    def new_obj(self, obj_type: str, **properties):
        """
        Create a new object

        Args:
            obj_type: object type (e.g. 'SIA', 'V', 'H')
            **properties: object properties (e.g. x=1.0, y=2.0, float_props=[0.0, 5.0])
        """
        # [fix] new=True is mandatory: otherwise the engine reads id=-1 as 'modify an
        # existing object', skips the ID allocation branch and may materialize an illegal object
        new_obj_dict = {'id': -1, 'type': obj_type, 'new': True}
        new_obj_dict.update(properties)
        self._changes.append(new_obj_dict)

    def to_list(self) -> List[Dict[str, Any]]:
        """Return the change list (internal use)"""
        return self._changes

    def __iter__(self):
        """Support iteration (internal use)"""
        return iter(self._changes)

    def __len__(self):
        """Return the number of changes"""
        return len(self._changes)


# ============================================================================
# Property mapping helpers
# ============================================================================

def get_props_mapping(config: Dict) -> Dict[str, Dict[str, Dict[str, int]]]:
    """
    Get the property-name-to-index mapping

    Returns:
        {
            'float_props': {obj_type: {prop_name: index}},
        }

    Example:
        mapping = get_props_mapping(config)
        # mapping['float_props']['H']['radius'] = 0
        # means the radius property of an H object lives in float_props[0]
    """
    return config.get('_props_mapping', {
        'float_props': {},
    })


def set_runtime_type_maps(int_to_type: Optional[Dict[int, str]] = None,
                          event_int_to_name: Optional[Dict[int, str]] = None):
    """Register the runtime type mapping, used as a fallback by helpers without a config."""
    global _RUNTIME_INT_TO_TYPE, _RUNTIME_EVENT_INT_TO_NAME
    _RUNTIME_INT_TO_TYPE = dict(int_to_type or {})
    _RUNTIME_EVENT_INT_TO_NAME = dict(event_int_to_name or {})


def get_object_type_name(obj, config: Optional[Dict] = None) -> str:
    """Resolve the user-visible type name from an object record or dict."""
    if isinstance(obj, dict):
        return str(obj.get('type', ''))
    dtype = getattr(obj, 'dtype', None)
    if dtype is None or dtype.names is None:
        return ''
    if 'type' in dtype.names:
        return str(obj['type']).strip()
    if 'type_id' in dtype.names:
        mapping = (config or {}).get('_int_to_type', _RUNTIME_INT_TO_TYPE)
        return mapping.get(int(obj['type_id']), '')
    return ''


def get_event_type_name(event, config: Optional[Dict] = None) -> str:
    """Resolve the user-visible event type name from an event record."""
    if isinstance(event, dict):
        return str(event.get('type', ''))
    dtype = getattr(event, 'dtype', None)
    if dtype is None or dtype.names is None:
        return ''
    if 'event_type_id' in dtype.names:
        mapping = (config or {}).get('_event_int_to_name', _RUNTIME_EVENT_INT_TO_NAME)
        return mapping.get(int(event['event_type_id']), '')
    if 'type' in dtype.names:
        return str(event['type']).strip()
    return ''


# ============================================================================
# Event object access helpers (the core API)
# ============================================================================

def get_event_object(event: np.ndarray, obj_type: str) -> Optional[np.ndarray]:
    """
    Get the object of a given type from an event

    Args:
        event: a single event record (one row of the NumPy structured array)
        obj_type: the object type wanted (e.g. 'H', 'V')

    Returns:
        the object record (NumPy structured array), or None when absent

    Example:
        # get the H object inside execute
        h_obj = get_event_object(event, 'H')
        if h_obj is not None:
            x = h_obj['x']
    """
    if 'objects' not in event.dtype.names:
        return None

    objects = event['objects']
    for i in range(len(objects)):
        obj = objects[i]
        # Is the object valid (id >= 0)?
        if obj['id'] < 0:
            continue
        if get_object_type_name(obj) == obj_type:
            return obj

    return None


def get_event_objects_by_type(events: np.ndarray, obj_type: str,
                               obj_idx: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Vectorized: get the objects of a given type from an event array

    Args:
        events: (N,) event array
        obj_type: the object type wanted (e.g. 'H')
        obj_idx: which one to take when several share the type (default 0 = the first)

    Returns:
        (objects, mask)
        - objects: the matching object records
        - mask: boolean mask marking the events that hold such an object

    Example:
        # get the H object of every event
        h_objs, mask = get_event_objects_by_type(events, 'H')
        # h_objs[mask] holds the valid H objects
    """
    n = len(events)
    objects = events['objects']  # (N, max_objs)
    max_objs = objects.shape[1]

    # Find the obj_idx-th object of the matching type in every event
    result_mask = np.zeros(n, dtype=bool)
    result_indices = np.zeros(n, dtype=np.int32)

    for i in range(n):
        count = 0
        for j in range(max_objs):
            if objects[i, j]['id'] < 0:
                continue
            if get_object_type_name(objects[i, j]) == obj_type:
                if count == obj_idx:
                    result_mask[i] = True
                    result_indices[i] = j
                    break
                count += 1

    # Extract the objects
    result_objects = np.empty(n, dtype=objects.dtype.base)
    for i in range(n):
        if result_mask[i]:
            result_objects[i] = objects[i, result_indices[i]]

    return result_objects, result_mask


def get_event_object_prop(event: np.ndarray, obj_type: str, prop_name: str,
                           config: Dict, default: Any = None) -> Any:
    """
    Get a property of the object of a given type inside an event

    Args:
        event: a single event record
        obj_type: object type (e.g. 'H', 'V')
        prop_name: property name (e.g. 'radius')
        config: configuration dict (holding the property mapping)
        default: default value

    Returns:
        the property value

    Example:
        # get the radius of the H object of an event
        radius = get_event_object_prop(event, 'H', 'radius', config, default=0.5)
    """
    obj = get_event_object(event, obj_type)
    if obj is None:
        return default

    return get_object_prop_from_record(obj, obj_type, prop_name, config, default)


def get_object_prop_from_record(obj: np.ndarray, obj_type: str, prop_name: str,
                                 config: Dict, default: Any = None) -> Any:
    """
    Get a property value from a NumPy structured record

    Args:
        obj: object record (one row of the NumPy structured array)
        obj_type: object type
        prop_name: property name
        config: configuration dict
        default: default value

    Returns:
        the property value
    """
    mapping = get_props_mapping(config)

    # Check float_props
    float_mapping = mapping.get('float_props', {}).get(obj_type, {})
    if prop_name in float_mapping:
        idx = float_mapping[prop_name]
        if 'float_props' in obj.dtype.names:
            props = obj['float_props']
            if idx < len(props):
                return float(props[idx])

    # Check the core fields
    if prop_name in obj.dtype.names:
        return obj[prop_name]

    return default


def get_event_objects_props(events: np.ndarray, obj_type: str, prop_name: str,
                             config: Dict, default: float = 0.0) -> np.ndarray:
    """
    Vectorized: get a property of the objects of a given type in an event array

    Args:
        events: (N,) event array
        obj_type: object type (e.g. 'H')
        prop_name: property name (e.g. 'radius')
        config: configuration dict
        default: default value

    Returns:
        (N,) array of property values

    Example:
        # get the radius of the H object of every event
        h_radii = get_event_objects_props(events, 'H', 'radius', config)
    """
    n = len(events)
    result = np.full(n, default, dtype=np.float64)

    objects, mask = get_event_objects_by_type(events, obj_type)
    if not np.any(mask):
        return result

    mapping = get_props_mapping(config)
    float_mapping = mapping.get('float_props', {}).get(obj_type, {})

    if prop_name in float_mapping:
        idx = float_mapping[prop_name]
        if 'float_props' in objects.dtype.names:
            result[mask] = objects[mask]['float_props'][:, idx]

    return result


def get_event_pair_props(events: np.ndarray, type1: str, type2: str,
                          prop_name: str, config: Dict,
                          default: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Vectorized: get a property of both object types of pair events

    Args:
        events: (N,) event array
        type1: first object type (e.g. 'H')
        type2: second object type (e.g. 'V')
        prop_name: property name (e.g. 'radius')
        config: configuration dict
        default: default value

    Returns:
        (prop1, prop2), two (N,) arrays

    Example:
        # get the radius of the H and V objects
        r_h, r_v = get_event_pair_props(events, 'H', 'V', 'radius', config)
        surface_dist = distance - r_h - r_v
    """
    prop1 = get_event_objects_props(events, type1, prop_name, config, default)
    prop2 = get_event_objects_props(events, type2, prop_name, config, default)
    return prop1, prop2


def get_event_param(event: np.ndarray, param_idx: int, default: float = 0.0) -> float:
    """
    Get an event parameter

    Args:
        event: a single event record
        param_idx: parameter index
        default: default value

    Returns:
        the parameter value
    """
    if 'params' not in event.dtype.names:
        return default

    params = event['params']
    if param_idx < len(params):
        return float(params[param_idx])

    return default


def get_events_params(events: np.ndarray, param_idx: int) -> np.ndarray:
    """
    Vectorized: get an event parameter

    Args:
        events: (N,) event array
        param_idx: parameter index

    Returns:
        (N,) array of parameter values
    """
    return events['params'][:, param_idx]


# ============================================================================
# Convenience wrappers (shortcuts for common operations)
# ============================================================================

def get_distance(events: np.ndarray) -> np.ndarray:
    """Distance of a pair event (assumed to sit in params[0])"""
    return events['params'][:, 0]


def get_objects_positions(events: np.ndarray, obj_idx: int = 0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Get the position of an object inside an event

    Args:
        events: (N,) event array
        obj_idx: object index (0 or 1)

    Returns:
        (x, y, z), three (N,) arrays
    """
    objects = events['objects'][:, obj_idx]
    return objects['x'], objects['y'], objects['z']


def get_pair_positions(events: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray,
                                                     np.ndarray, np.ndarray, np.ndarray]:
    """
    Get the positions of both objects of a pair event

    Returns:
        (x1, y1, z1, x2, y2, z2), six (N,) arrays
    """
    obj1 = events['objects'][:, 0]
    obj2 = events['objects'][:, 1]
    return (obj1['x'], obj1['y'], obj1['z'],
            obj2['x'], obj2['y'], obj2['z'])


def get_object_ids(events: np.ndarray) -> np.ndarray:
    """Get the object IDs of an event"""
    return events['obj_ids']


# ============================================================================
# Object property access (the original functions)
# ============================================================================


def get_prop_index(config: Dict, obj_type: str, prop_name: str) -> Tuple[str, int]:
    """
    Get the property index of a given object type

    Args:
        config: configuration dict
        obj_type: object type name (e.g. 'H', 'V')
        prop_name: property name (e.g. 'radius')

    Returns:
        (array_name, index)
        - array_name: 'float_props'
        - index: index of the property inside the array

    Raises:
        KeyError: if the property does not exist

    Example:
        array_name, idx = get_prop_index(config, 'H', 'radius')
        # array_name = 'float_props', idx = 0
        # then read it as obj['float_props'][idx]
    """
    mapping = get_props_mapping(config)

    float_mapping = mapping.get('float_props', {}).get(obj_type, {})
    if prop_name in float_mapping:
        return ('float_props', float_mapping[prop_name])

    raise KeyError(f"Property '{prop_name}' not found for object type '{obj_type}'")


def get_object_prop(obj: Dict, obj_type: str, prop_name: str, config: Dict,
                    default: Any = None) -> Any:
    """
    Get a property value of an object (through the mapping)

    Args:
        obj: object dict (holding float_props)
        obj_type: object type name
        prop_name: property name
        config: configuration dict
        default: default value (returned when the property is absent)

    Returns:
        the property value

    Example:
        radius = get_object_prop(obj, 'H', 'radius', config, default=0.5)
    """
    try:
        array_name, idx = get_prop_index(config, obj_type, prop_name)
        if array_name in obj and idx < len(obj[array_name]):
            return obj[array_name][idx]
    except KeyError:
        pass
    return default


def get_object_props_vectorized(objects: np.ndarray, obj_types: np.ndarray,
                                 prop_name: str, config: Dict,
                                 default: float = 0.0) -> np.ndarray:
    """
    Vectorized property access (mixed types supported)

    Args:
        objects: (N,) or (N, max_objs) structured array
        obj_types: (N,) array of object types
        prop_name: property name
        config: configuration dict
        default: default value

    Returns:
        (N,) array of property values

    Example:
        # get the radius of every object
        radii = get_object_props_vectorized(objects[:, 0], obj_types, 'radius', config)

    Note:
        This function copes with different types using different indices.
        When every type shares one property layout, plain NumPy indexing is faster:
        radii = objects[:, 0]['float_props'][:, radius_idx]
    """
    n = len(objects)
    result = np.full(n, default, dtype=np.float64)

    mapping = get_props_mapping(config)
    float_mapping = mapping.get('float_props', {})

    # Unique types
    unique_types = np.unique(obj_types)

    for obj_type in unique_types:
        obj_type_str = str(obj_type)
        if obj_type_str not in float_mapping:
            continue

        type_mapping = float_mapping[obj_type_str]
        if prop_name not in type_mapping:
            continue

        idx = type_mapping[prop_name]
        mask = obj_types == obj_type

        # Determine the array layout
        if objects.ndim == 1:
            # Single-object events: objects is an (N,) structured array
            if 'float_props' in objects.dtype.names:
                result[mask] = objects[mask]['float_props'][:, idx]
        else:
            # objects is assumed to be sliced correctly already
            if 'float_props' in objects.dtype.names:
                result[mask] = objects[mask]['float_props'][:, idx]

    return result


def get_pair_props_vectorized(objects: np.ndarray, obj1_idx: int, obj2_idx: int,
                               prop_name: str, config: Dict,
                               default: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Vectorized: get a property of both objects of pair events

    Args:
        objects: (N, max_objs) structured array
        obj1_idx: index of the first object (normally 0)
        obj2_idx: index of the second object (normally 1)
        prop_name: property name
        config: configuration dict
        default: default value

    Returns:
        (prop1, prop2), two (N,) arrays

    Example:
        r1, r2 = get_pair_props_vectorized(objects, 0, 1, 'radius', config)
        surface_distance = distance - r1 - r2
    """
    n = len(objects)
    prop1 = np.full(n, default, dtype=np.float64)
    prop2 = np.full(n, default, dtype=np.float64)

    mapping = get_props_mapping(config)
    float_mapping = mapping.get('float_props', {})

    if 'type_id' in objects.dtype.base.names:
        obj1_type_ids = objects[:, obj1_idx]['type_id']
        obj2_type_ids = objects[:, obj2_idx]['type_id']
        int_to_type = config.get('_int_to_type', _RUNTIME_INT_TO_TYPE)
        unique_obj1 = np.unique(obj1_type_ids)
        unique_obj2 = np.unique(obj2_type_ids)
    else:
        obj1_type_ids = objects[:, obj1_idx]['type']
        obj2_type_ids = objects[:, obj2_idx]['type']
        int_to_type = {}
        unique_obj1 = np.unique(obj1_type_ids)
        unique_obj2 = np.unique(obj2_type_ids)

    # Handle the first object
    for obj_type in unique_obj1:
        obj_type_str = int_to_type.get(int(obj_type), '') if int_to_type else str(obj_type)
        if obj_type_str not in float_mapping:
            continue
        type_mapping = float_mapping[obj_type_str]
        if prop_name not in type_mapping:
            continue
        idx = type_mapping[prop_name]
        mask = obj1_type_ids == obj_type
        if 'float_props' in objects[:, obj1_idx].dtype.names:
            prop1[mask] = objects[mask, obj1_idx]['float_props'][:, idx]

    # Handle the second object
    for obj_type in unique_obj2:
        obj_type_str = int_to_type.get(int(obj_type), '') if int_to_type else str(obj_type)
        if obj_type_str not in float_mapping:
            continue
        type_mapping = float_mapping[obj_type_str]
        if prop_name not in type_mapping:
            continue
        idx = type_mapping[prop_name]
        mask = obj2_type_ids == obj_type
        if 'float_props' in objects[:, obj2_idx].dtype.names:
            prop2[mask] = objects[mask, obj2_idx]['float_props'][:, idx]

    return prop1, prop2


# ============================================================================
# Physics parameter and field access helpers
# ============================================================================

def get_physics_param(config: Dict, name: str, default: Any = None) -> Any:
    """
    Get a physics parameter from the UnifiedPhysics module or from config

    Priority:
    1. config['physics'].get_param(name) (the UnifiedPhysics module)
    2. config['physics'][name] (dict mode)
    3. config[name] (top-level config)
    4. default

    Args:
        config: configuration dict
        name: parameter name
        default: default value

    Returns:
        the parameter value

    Example:
        T = get_physics_param(config, 'temperature', default=500.0)
        k_B = get_physics_param(config, 'k_B', default=8.617e-5)
    """
    physics = config.get('physics')

    if physics is not None:
        # Try the get_param method of UnifiedPhysics
        if hasattr(physics, 'get_param'):
            try:
                value = physics.get_param(name)
                if value is not None:
                    # A callable is supported (e.g. a temperature schedule)
                    if callable(value):
                        current_time = config.get('_current_time', 0.0)
                        return value(current_time)
                    return value
            except (KeyError, AttributeError):
                pass

        # Try dict access
        if isinstance(physics, dict) and name in physics:
            value = physics[name]
            if callable(value):
                current_time = config.get('_current_time', 0.0)
                return value(current_time)
            return value

    # Fall back to the top level of config
    if name in config:
        value = config[name]
        if callable(value):
            current_time = config.get('_current_time', 0.0)
            return value(current_time)
        return value

    return default


def get_physics_field(config: Dict, name: str) -> Optional[Callable]:
    """
    Get a field interpolator from the UnifiedPhysics module

    Args:
        config: configuration dict
        name: field name

    Returns:
        the interpolator f(x, y, z) -> value, or None

    Example:
        stress_field = get_physics_field(config, 'stress_xx')
        if stress_field is not None:
            stress = stress_field(x, y, z)  # single-point query
            stress_batch = stress_field(x_arr, y_arr, z_arr)  # batch query
    """
    physics = config.get('physics')

    if physics is not None and hasattr(physics, 'get_field_interpolator'):
        try:
            return physics.get_field_interpolator(name)
        except (KeyError, AttributeError):
            pass

    return None


def get_physics_field_cached(config: Dict, name: str) -> Optional[Callable]:
    """
    Get a cached field interpolator from the UnifiedPhysics module

    Same as get_physics_field but returns the cached version when available

    Args:
        config: configuration dict
        name: field name

    Returns:
        the cached interpolator, or None
    """
    physics = config.get('physics')

    if physics is not None:
        # Prefer the cached version
        if hasattr(physics, 'get_field_cached'):
            try:
                return physics.get_field_cached(name)
            except (KeyError, AttributeError):
                pass

        # Fall back to the plain version
        if hasattr(physics, 'get_field_interpolator'):
            try:
                return physics.get_field_interpolator(name)
            except (KeyError, AttributeError):
                pass

    return None


def get_physics_params(config: Dict, *param_names: str) -> tuple:
    """
    Get several physics parameters at once

    Args:
        config: configuration dict
        *param_names: parameter names

    Returns:
        a tuple of parameter values

    Example:
        T, k_B, D_eff = get_physics_params(config, 'temperature', 'k_B', 'D_eff')
    """
    # Common defaults
    defaults = {
        'temperature': 300.0,
        'k_B': 8.617333262e-5,
        'attempt_frequency': 1e13,
        'migration_energy': 0.5,
        'D_eff': 1e-9,
    }

    values = []
    for name in param_names:
        value = get_physics_param(config, name, default=defaults.get(name, 0.0))
        values.append(value)

    return tuple(values) if len(values) > 1 else (values[0],)


# ============================================================================
# Vectorized physics parameter and field access
# ============================================================================

def get_params_array(config: Dict) -> Optional[np.ndarray]:
    """
    Get the parameter array of UnifiedPhysics (a direct reference, no copy)

    Args:
        config: configuration dict

    Returns:
        a reference to the NumPy parameter array, or None

    Purpose:
        letting a rate evaluation index the parameters with NumPy directly,
        avoiding repeated dict lookups and function calls

    Example:
        params_arr = get_params_array(config)
        param_indices = get_param_indices(config, ['temperature', 'k_B'])
        if params_arr is not None and param_indices:
            T = params_arr[param_indices['temperature']]
            k_B = params_arr[param_indices['k_B']]
    """
    physics = config.get('physics')

    if physics is not None and hasattr(physics, 'params'):
        return physics.params  # return the reference directly, no copy

    return None


def get_param_indices(config: Dict, param_names: List[str]) -> Dict[str, int]:
    """
    Get the mapping from parameter name to index

    Args:
        config: configuration dict
        param_names: names of the parameters needed

    Returns:
        a {parameter name: index} dict

    Example:
        indices = get_param_indices(config, ['temperature', 'k_B'])
        # indices = {'temperature': 0, 'k_B': 1}
    """
    physics = config.get('physics')
    result = {}

    if physics is not None and hasattr(physics, 'get_param_index'):
        for name in param_names:
            try:
                result[name] = physics.get_param_index(name)
            except (KeyError, AttributeError):
                pass

    return result


def interpolate_field_batch(config: Dict, field_name: str,
                            x: np.ndarray, y: np.ndarray, z: np.ndarray) -> Optional[np.ndarray]:
    """
    Batch field interpolation (NumPy vectorized)

    Args:
        config: configuration dict
        field_name: field name
        x, y, z: coordinate arrays (of equal shape)

    Returns:
        the interpolated array (same shape as the input), or None

    Note:
        Calls the batch interpolation method of UnifiedPhysics directly, which
        uses the vectorized scipy.interpolate.RegularGridInterpolator

    Example:
        # inside a rate evaluation
        if objects is not None:
            x = objects[:, 0]['x']
            y = objects[:, 0]['y']
            z = objects[:, 0]['z']
            stress = interpolate_field_batch(config, 'stress_xx', x, y, z)
            if stress is not None:
                E_m = params[:, 0] - stress * 0.01
    """
    physics = config.get('physics')

    if physics is None:
        return None

    # Route 1: the batch interpolation method of UnifiedPhysics
    if hasattr(physics, 'interpolate_field_batch'):
        try:
            # Build the positions array: (N, 3)
            positions = np.column_stack([x.ravel(), y.ravel(), z.ravel()])
            result = physics.interpolate_field_batch(field_name, positions)
            return result.reshape(x.shape) if result is not None else None
        except (KeyError, AttributeError):
            pass

    # Route 2: the interpolator (which already accepts batches)
    if hasattr(physics, 'get_field_interpolator'):
        try:
            interp = physics.get_field_interpolator(field_name)
            if interp is not None:
                # RegularGridInterpolator accepts batched input
                positions = np.column_stack([x.ravel(), y.ravel(), z.ravel()])
                result = interp(positions)
                return result.reshape(x.shape)
        except (KeyError, AttributeError):
            pass

    return None


def get_field_at_objects(config: Dict, field_name: str,
                         objects: np.ndarray, obj_idx: int = 0) -> Optional[np.ndarray]:
    """
    Get the field value at the object positions (NumPy vectorized)

    Args:
        config: configuration dict
        field_name: field name
        objects: (N, max_objects) structured array of objects
        obj_idx: object index (0 = first object, 1 = second object)

    Returns:
        an (N,) array of field values, or None

    Example:
        class StressAssistedMigration(SingleObjectEvent):
            @classmethod
            def rate(cls, objs, params, config):
                # read the stress at the object position
                stress = get_field_at_objects(config, 'stress_xx', objects, obj_idx=0)
                if stress is not None:
                    E_m = params[:, 0] - stress * 0.01
                else:
                    E_m = params[:, 0]
                # ...
    """
    if objects is None:
        return None

    # Handle the shape of the object array
    if objects.ndim == 1:
        # Single-object events: objects is an (N,) structured array
        x = objects['x']
        y = objects['y']
        z = objects['z']
    elif objects.ndim == 2:
        # Pair events: objects is an (N, 2) structured array
        x = objects[:, obj_idx]['x']
        y = objects[:, obj_idx]['y']
        z = objects[:, obj_idx]['z']
    else:
        return None

    return interpolate_field_batch(config, field_name, x, y, z)


# ============================================================================
# Event base class
# ============================================================================

class EventBase(ABC):
    """
    Event base class

    Every event type must subclass it and provide:
    - object_types: the object types taking part in the event
    - params: names of the event parameters (read from objects or globals)
    - filter(): the event filter
    - rate(): scalar rate evaluation (or rate_deps + rate_kernel for the JIT path)
    - execute(): event execution
    """

    # Subclasses must define these
    object_types: List[str] = []  # types of the participating objects
    params: List[str] = []        # names of the event parameters

    # Optional configuration
    max_distance: float = 0.0     # maximum interaction distance of a pair event
    weight: float = 1.0           # computational weight (used by load balancing)
                                  # detected automatically: JIT = 1.0, Python = 1000.0
                                  # an explicit weight overrides the detection

    # [feat-2] Class-level rate increment switch (off by default):
    # True  = on the reuse path (single incremental reuse / pair structural retention)
    # this class may keep the DONE rate computed in the previous step (provided the
    # objects involved did not change and no field or parameter forces a rebuild)
    # False = the rate is recomputed every step (the structure is reused, the rate is not)
    # Use: safe for classes whose rate depends only on object properties and constants;
    #      keep it False when the rate follows a time-varying field or temperature schedule
    incremental_rate: bool = False

    # JIT acceleration
    rate_deps = None       # Optional[RateDeps] - declaration of the data dependencies
    rate_kernel = None     # Optional[Callable] - @njit scalar kernel (preferred) or a Python function

    @classmethod
    def get_num_objects(cls) -> int:
        """Number of objects the event needs"""
        return len(cls.object_types)

    @classmethod
    def filter(cls, *args, **kwargs) -> Union[bool, np.ndarray]:
        """
        Event filter

        Subclasses may override this to filter events.
        The default returns True (no filtering).

        Two modes are supported:
        - scalar: filter(obj, config) -> bool
        - vectorized: filter(obj_types, obj_props, config) -> np.ndarray[bool]
        """
        return True

    @classmethod
    def rate(cls, objs: List[Dict], params: Dict, config: Dict) -> float:
        """
        Scalar rate evaluation (a Python function, suited to complex logic)

        Use when:
        - the logic branches in complicated ways
        - non-numeric properties must be read
        - debugging or prototyping

        Args:
            objs: the participating objects (as dicts)
            params: dict of event parameters {'param_name': value, ...}
            config: the global configuration

        Returns:
            the rate (Hz)
        """
        raise NotImplementedError("Subclass must implement rate() or rate_deps+rate_kernel")

    @classmethod
    def has_scalar_rate(cls) -> bool:
        """Test whether scalar rate evaluation is implemented (ignoring the base default)"""
        # Look for an override of rate in this class or its direct parents
        # The default implementation of EventBase does not count
        for klass in cls.__mro__:
            if klass is EventBase:
                break
            if 'rate' in klass.__dict__:
                return True
        return False

    @classmethod
    def check_rate_implementation(cls, event_name: str = None) -> bool:
        """
        Report which rate implementations exist

        Returns:
            has_scalar: whether a scalar rate() is implemented
        """
        return cls.has_scalar_rate()

    @classmethod
    @abstractmethod
    def execute(cls, event: Dict, objs: List[Dict], config: Dict) -> List[Dict]:
        """
        Execute the event

        Args:
            event: the event record
            objs: the participating objects
            config: the global configuration

        Returns:
            the list of changed objects, each holding:
            - 'id': object ID
            - 'deleted': whether it is deleted (optional)
            - 'new': whether it is new (optional)
            - any other property updates
        """
        pass

    @classmethod
    def extract_params(cls, objs: List[Dict], config: Dict, event_name: str = None) -> Dict[str, Any]:
        """
        Extract the event parameters from the objects and the configuration

        Default behaviour: look the names declared in params up in the object
        properties and in config. Subclasses may override it.

        The event_name argument reads parameters from config['events'][event_name]

        Lookup order (highest priority first):
        1. object properties (objs[0][param_name])
        2. event configuration (config['events'][event_name][param_name])
        3. physics module (config['physics'].get_param(param_name) or config['physics'][param_name])
        4. top-level configuration (config[param_name])
        5. the default (0.0)
        """
        result = {}
        for param_name in cls.params:
            # 1. Try the first object
            if objs and param_name in objs[0]:
                result[param_name] = objs[0][param_name]
                continue

            # 2. Try config['events'][event_name]
            if event_name and 'events' in config:
                event_config = config['events'].get(event_name, {})
                if param_name in event_config:
                    result[param_name] = event_config[param_name]
                    continue

            # 3. Try config['physics']
            # physics may be a UnifiedPhysics object or a dict
            if 'physics' in config:
                physics = config['physics']
                if hasattr(physics, 'get_param'):
                    # A UnifiedPhysics object
                    try:
                        value = physics.get_param(param_name)
                        if value is not None:
                            result[param_name] = value
                            continue
                    except (KeyError, IndexError):
                        pass  # the parameter is absent, keep looking
                elif isinstance(physics, dict) and param_name in physics:
                    # A plain dict
                    result[param_name] = physics[param_name]
                    continue

            # 4. Try the top level of config
            if param_name in config:
                result[param_name] = config[param_name]
            elif param_name not in result:
                # 5. The default
                result[param_name] = 0.0
        return result


# ============================================================================
# Zero-object events
# ============================================================================

class ZeroObjectEvent(EventBase):
    """
    Base class of zero-object events

    For cascades, global sources and other events that need no particular object.

    Spatial modes:
    - spatial_mode='global': one event per rank at most (e.g. a cascade)
    - spatial_mode='per_cell': one event per local cell (e.g. dislocation nucleation),
      which also needs field_deps declaring the fields the rate depends on

    Example (global mode):
        class CascadeEvent(ZeroObjectEvent):
            params = ['cascade_rate', 'defect_count']

            @classmethod
            def rate(cls, objs, params, config):
                return float(params.get('cascade_rate', 1e5))

            @classmethod
            def execute(cls, event, objs, config):
                # create many defect objects
                return [{'id': -1, 'type': 'SIA', 'new': True, ...}, ...]

    Example (per_cell mode):
        class DislocationNucleation(ZeroObjectEvent):
            spatial_mode = 'per_cell'
            field_deps = ['stress_xx']
            params = ['stress', 'cell_x', 'cell_y', 'cell_z']

            @classmethod
            def rate(cls, objs, params, config):
                stress = params.get('stress', 0.0)
                threshold = 1e9
                if stress > threshold:
                    return 1e10 * np.exp(-(threshold / stress))
                return 0.0

            @classmethod
            def execute(cls, event, objs, config):
                cell_x = event['params'][1]
                cell_y = event['params'][2]
                cell_z = event['params'][3]
                return [{'id': -1, 'type': 'dislocation', 'new': True,
                         'x': cell_x, 'y': cell_y, 'z': cell_z}]
    """
    object_types = []

    # Spatial mode: 'global' (one event overall) or 'per_cell' (one event per cell)
    spatial_mode: str = 'global'

    # Fields the per_cell rate depends on (interpolated into params automatically)
    field_deps: List[str] = []

    # Owner rank policy
    owner_rank_mode: str = 'rank0'  # 'rank0', 'round_robin', 'random'

    @classmethod
    def get_num_objects(cls) -> int:
        return 0

    @classmethod
    def filter(cls, config: Dict, rank: int, size: int) -> bool:
        """
        Zero-object event filter

        Args:
            config: the global configuration
            rank: current rank
            size: total number of ranks

        Returns:
            whether to create the event on this rank
        """
        if cls.owner_rank_mode == 'rank0':
            return rank == 0
        elif cls.owner_rank_mode == 'round_robin':
            return True  # every rank creates one; the ID keeps them apart
        elif cls.owner_rank_mode == 'random':
            return True
        return rank == 0


# ============================================================================
# Single-object events
# ============================================================================

class SingleObjectEvent(EventBase):
    """
    Base class of single-object events

    For migration, decay and other events involving one object.

    Example:
        class MigrationEvent(SingleObjectEvent):
            object_types = ['H', 'V', 'SIA', 'HV']
            params = ['E_m', 'nu']

            @classmethod
            def rate(cls, objs, params, config):
                T = get_physics_param(config, 'temperature', default=500.0)
                k_B = 8.617e-5
                E_m = params.get('E_m', 0.5)
                nu = params.get('nu', 1e13)
                return float(nu * np.exp(-E_m / (k_B * T)))

            @classmethod
            def execute(cls, event, objs, config):
                obj = objs[0]
                step_size = config.get('migration_step', 1.0)
                rng = config['rng']  # the unified deterministic generator injected by the engine
                dx = rng.uniform(-step_size, step_size)
                dy = rng.uniform(-step_size, step_size)
                dz = rng.uniform(-step_size, step_size)
                return [{'id': obj['id'], 'x': obj['x'] + dx,
                         'y': obj['y'] + dy, 'z': obj['z'] + dz}]
    """

    @classmethod
    def get_num_objects(cls) -> int:
        return 1


# ============================================================================
# Pair events
# ============================================================================

class PairObjectEvent(EventBase):
    """
    Base class of pair events

    For capture, recombination, annihilation and other two-object events.

    Example:
        class CaptureEvent(PairObjectEvent):
            type_pairs = [('H', 'V')]
            max_distance = 5.0
            params = ['distance']

            @classmethod
            def rate(cls, objs, params, config):
                D_eff = get_physics_param(config, 'D_eff', default=1e-9)
                distance = params.get('distance', 1.0)
                return float(4 * np.pi * D_eff * distance)

            @classmethod
            def execute(cls, event, objs, config):
                obj1, obj2 = objs[0], objs[1]
                return [
                    {'id': obj1['id'], 'deleted': True},
                    {'id': obj2['id'], 'deleted': True},
                    {'id': -1, 'type': 'HV', 'new': True, ...}
                ]
    """

    # Allowed type pairs: [('TypeA', 'TypeB'), ...]
    # An empty list allows every combination
    type_pairs: List[Tuple[str, str]] = []

    # Product type (optional)
    product_type: Optional[str] = None

    @classmethod
    def get_num_objects(cls) -> int:
        return 2


# ============================================================================
# Event registry
# ============================================================================

class EventRegistry:
    """
    Event registry

    Holds every registered event type and offers queries and validation.

    Two rate evaluation paths:
    - JIT events: rate_deps + rate_kernel are defined -> batch @njit evaluation
    - Python events: rate() is defined -> one scalar Python call per event
    """

    def __init__(self):
        self._events: Dict[str, Type[EventBase]] = {}
        self._zero_events: List[str] = []
        self._single_events: List[str] = []
        self._pair_events: List[str] = []

        # Scalar Python events (everything that is not JIT)
        self._python_events: List[str] = []

        # JIT accelerated events
        self._jit_events: List[str] = []

    def register(self, name: str, event_class: Type[EventBase]):
        """Register an event type"""
        if not issubclass(event_class, EventBase):
            raise TypeError(f"Event class must inherit from EventBase: {event_class}")

        self._events[name] = event_class

        # Classify by the number of objects
        num_obj = event_class.get_num_objects()
        if num_obj == 0:
            self._zero_events.append(name)
        elif num_obj == 1:
            self._single_events.append(name)
        else:
            self._pair_events.append(name)

        # JIT detection: the event class defines both rate_deps and rate_kernel
        has_jit = (getattr(event_class, 'rate_deps', None) is not None and
                   getattr(event_class, 'rate_kernel', None) is not None)
        if has_jit:
            self._jit_events.append(name)

        # Everything else counts as a scalar Python event
        if not has_jit:
            self._python_events.append(name)

    def get(self, name: str) -> Optional[Type[EventBase]]:
        """Get an event class"""
        return self._events.get(name)

    def get_zero_events(self) -> List[str]:
        """Get the names of the zero-object events"""
        return self._zero_events.copy()

    def get_single_events(self) -> List[str]:
        """Get the names of the single-object events"""
        return self._single_events.copy()

    def get_pair_events(self) -> List[str]:
        """Get the names of the pair events"""
        return self._pair_events.copy()

    def get_python_events(self) -> List[str]:
        """Get the names of the scalar Python rate events (everything not JIT)"""
        return self._python_events.copy()

    # Backward-compatible alias
    def get_scalar_events(self) -> List[str]:
        """Get the names of the scalar Python rate events (backward compatible)"""
        return self._python_events.copy()

    def get_all_events(self) -> Dict[str, Type[EventBase]]:
        """Get every event"""
        return self._events.copy()

    def is_python(self, name: str) -> bool:
        """Test whether an event uses a scalar Python rate"""
        return name in self._python_events

    def get_jit_events(self) -> List[str]:
        """Get the names of the events with a JIT kernel"""
        return self._jit_events.copy()

    def is_jit(self, name: str) -> bool:
        """Test whether an event has a JIT kernel"""
        return name in self._jit_events

    def validate(self) -> List[str]:
        """
        Validate every registered event class

        Returns:
            the list of errors; an empty list means validation passed
        """
        errors = []

        for name, event_class in self._events.items():
            # Check the required attributes
            if not hasattr(event_class, 'object_types'):
                errors.append(f"{name}: missing 'object_types' attribute")

            # Is a rate method implemented? (a JIT kernel counts)
            has_rate = (event_class.has_scalar_rate() or
                        (getattr(event_class, 'rate_deps', None) is not None and
                         getattr(event_class, 'rate_kernel', None) is not None))
            if not has_rate:
                errors.append(f"{name}: must implement rate() or rate_deps+rate_kernel")

            # Check the execute method
            try:
                if event_class.execute is EventBase.execute:
                    errors.append(f"{name}: must implement execute()")
            except AttributeError:
                errors.append(f"{name}: must implement execute()")

        return errors


# ============================================================================
# Compile the event configuration
# ============================================================================

def compile_event_config(events_config: Dict[str, Type[EventBase]]) -> EventRegistry:
    """
    Compile the event configuration

    Registers the user-defined event classes into an EventRegistry.

    Args:
        events_config: {'event_name': EventClass, ...}

    Returns:
        the EventRegistry instance
    """
    registry = EventRegistry()

    for name, event_class in events_config.items():
        registry.register(name, event_class)

    errors = registry.validate()
    if errors:
        raise ValueError(f"Event configuration errors:\n" + "\n".join(errors))

    return registry


# ============================================================================
# Built-in event classes (with the unified signature)
# ============================================================================

class ArrheniusMigration(SingleObjectEvent):
    """
    Arrhenius migration event

    Rate: nu * exp(-E_m / (k_B * T))

    Parameter mapping:
    - params[:, 0]: E_m (migration energy, eV)
    - params[:, 1]: nu (attempt frequency, Hz)
    """
    object_types = []  # to be set by the user
    params = ['E_m', 'nu']

    @classmethod
    def rate(cls, objs: List[Dict], params: Dict, config: Dict) -> float:
        T = get_physics_param(config, 'temperature', default=300.0)
        k_B = 8.617e-5  # eV/K
        E_m = params.get('E_m', 0.5)
        nu = params.get('nu', 1e13)
        return float(nu * np.exp(-E_m / (k_B * T)))

    @classmethod
    def execute(cls, event: Dict, objs: List[Dict], config: Dict) -> List[Dict]:
        obj = objs[0]
        step_size = config.get('migration_step', 1.0)
        box_size = config.get('domain_decomposition', {}).get('box_size', [100, 100, 100])
        periodic = config.get('domain_decomposition', {}).get('periodic', [True, True, True])

        # Random displacement
        # [fix] Use the unified deterministic generator injected by the engine
        # (config['rng']) so (seed, step, state) reproduces; fall back to np.random
        rng = config.get('rng')
        if rng is not None:
            dx = rng.uniform(-step_size, step_size)
            dy = rng.uniform(-step_size, step_size)
            dz = rng.uniform(-step_size, step_size)
        else:
            dx = np.random.uniform(-step_size, step_size)
            dy = np.random.uniform(-step_size, step_size)
            dz = np.random.uniform(-step_size, step_size)

        new_x = float(obj['x']) + dx
        new_y = float(obj['y']) + dy
        new_z = float(obj['z']) + dz

        # Periodic boundary conditions
        if periodic[0]:
            new_x = new_x % box_size[0]
        if periodic[1]:
            new_y = new_y % box_size[1]
        if periodic[2]:
            new_z = new_z % box_size[2]

        return [{'id': int(obj['id']), 'x': new_x, 'y': new_y, 'z': new_z}]


class DiffusionCapture(PairObjectEvent):
    """
    Diffusion-limited capture event

    Rate: 4 * pi * D_eff * r_capture

    Parameter mapping:
    - params[0]: distance
    - params[1]: D_eff (optional, read from config by default)
    - params[2]: r_capture (optional, default 1.0)
    """
    type_pairs = []  # to be set by the user
    max_distance = 3.0
    params = ['distance', 'D_eff', 'r_capture']
    product_type: str = 'Complex'

    @classmethod
    def rate(cls, objs: List[Dict], params: Dict, config: Dict) -> float:
        D_eff = params.get('D_eff', 0.0)
        if D_eff == 0.0:
            D_eff = get_physics_param(config, 'D_eff', default=1e-9)
        r_capture = params.get('r_capture', 1.0)
        return float(4 * np.pi * D_eff * r_capture)

    @classmethod
    def execute(cls, event: Dict, objs: List[Dict], config: Dict) -> List[Dict]:
        obj1, obj2 = objs[0], objs[1]

        # Position of the product (the midpoint)
        new_x = (float(obj1['x']) + float(obj2['x'])) / 2
        new_y = (float(obj1['y']) + float(obj2['y'])) / 2
        new_z = (float(obj1['z']) + float(obj2['z'])) / 2

        # Type of the product
        product = getattr(cls, 'product_type', 'Complex')

        return [
            {'id': int(obj1['id']), 'deleted': True},
            {'id': int(obj2['id']), 'deleted': True},
            {'id': -1, 'type': product, 'x': new_x, 'y': new_y, 'z': new_z, 'new': True}
        ]


class Annihilation(PairObjectEvent):
    """
    Annihilation event (the two objects annihilate each other)

    Rate: 4 * pi * D_eff * distance

    Parameter mapping:
    - params[0]: distance
    - params[1]: D_eff (optional)
    """
    type_pairs = []  # to be set by the user, e.g. [('H', 'V')]
    max_distance = 3.0
    params = ['distance', 'D_eff']

    @classmethod
    def rate(cls, objs: List[Dict], params: Dict, config: Dict) -> float:
        distance = params.get('distance', 0.0)
        D_eff = params.get('D_eff', 0.0)
        if D_eff == 0.0:
            D_eff = get_physics_param(config, 'D_eff', default=1e-9)
        return float(4 * np.pi * D_eff * distance)

    @classmethod
    def execute(cls, event: Dict, objs: List[Dict], config: Dict) -> List[Dict]:
        obj1, obj2 = objs[0], objs[1]
        return [
            {'id': int(obj1['id']), 'deleted': True},
            {'id': int(obj2['id']), 'deleted': True}
        ]


# ============================================================================
# Self-test
# ============================================================================

if __name__ == '__main__':
    print("=" * 60)
    print("v1.6.9 Event Class System - Self Test")
    print("=" * 60)

    # Test 1: scalar Python rate
    print("\n[Test 1] Python Scalar Rate")

    class MyMigration(SingleObjectEvent):
        object_types = ['H', 'V']
        params = ['E_m', 'nu']

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

    # Test the scalar rate
    config = {'physics': {'temperature': 500.0}}
    rate = MyMigration.rate([], {'E_m': 0.5, 'nu': 1e13}, config)
    print(f"  Scalar rate: {rate}")
    assert rate > 0
    print("  PASSED")

    # Test 2: classification of scalar events
    print("\n[Test 2] Event Classification")

    class ScalarOnlyEvent(SingleObjectEvent):
        object_types = ['H']
        params = []

        @classmethod
        def rate(cls, objs, params, config):
            return 1.0

        @classmethod
        def execute(cls, event, objs, config):
            return []

    registry2 = compile_event_config({'scalar_only': ScalarOnlyEvent})
    print(f"  Python events: {registry2.get_python_events()}")
    assert 'scalar_only' in registry2.get_python_events()
    print("  PASSED")

    # Test 3: the built-in event classes (scalar rate)
    print("\n[Test 3] Built-in Event Classes (Scalar Rate)")

    # Arrhenius migration
    rate = ArrheniusMigration.rate([], {'E_m': 0.5, 'nu': 1e13},
                                    {'physics': {'temperature': 600.0}})
    print(f"  ArrheniusMigration rate: {rate}")
    assert rate > 0

    # Diffusion capture
    rate = DiffusionCapture.rate([], {'distance': 1.0, 'D_eff': 1e-9, 'r_capture': 1.0}, config)
    print(f"  DiffusionCapture rate: {rate}")
    assert rate > 0

    # Annihilation
    rate = Annihilation.rate([], {'distance': 2.0, 'D_eff': 1e-9}, config)
    print(f"  Annihilation rate: {rate}")
    assert rate > 0
    print("  PASSED")

    # Test 4: the property mapping helpers
    print("\n[Test 4] Property Mapping Helper Functions")

    # A mock configuration
    props_mapping = {
        'float_props': {
            'H': {'radius': 0, 'energy': 1},
            'V': {'radius': 0, 'formation_energy': 1},
        },
    }
    config_with_mapping = {'_props_mapping': props_mapping}

    # Test get_props_mapping
    m = get_props_mapping(config_with_mapping)
    assert 'float_props' in m
    assert 'H' in m['float_props']
    assert m['float_props']['H']['radius'] == 0
    print("  get_props_mapping: PASSED")

    # Test get_prop_index
    arr_name, idx = get_prop_index(config_with_mapping, 'H', 'radius')
    assert arr_name == 'float_props'
    assert idx == 0
    print("  get_prop_index: PASSED")

    # Test get_object_prop
    obj = {'float_props': [0.5, 1.2]}
    radius = get_object_prop(obj, 'H', 'radius', config_with_mapping)
    assert radius == 0.5
    print("  get_object_prop: PASSED")

    # Test a property that does not exist
    missing = get_object_prop(obj, 'H', 'nonexistent', config_with_mapping, default=999)
    assert missing == 999
    print("  get_object_prop (default): PASSED")

    # Test 5: the event object access helpers
    print("\n[Test 5] Event Object Access Helper Functions")

    # Build a mock event dtype
    obj_dtype = np.dtype([
        ('id', np.int64),
        ('type', 'U16'),
        ('x', np.float64),
        ('y', np.float64),
        ('z', np.float64),
        ('float_props', np.float64, (2,)),
    ])

    event_dtype = np.dtype([
        ('id', np.int64),
        ('type', 'U32'),
        ('params', np.float64, (4,)),
        ('obj_ids', np.int64, (2,)),
        ('objects', obj_dtype, (2,)),
    ])

    # Create the test event
    test_event = np.zeros(1, dtype=event_dtype)[0]
    test_event['id'] = 1
    test_event['type'] = 'capture'
    test_event['params'] = [2.5, 1e-9, 0.0, 0.0]  # distance, D_eff
    test_event['obj_ids'] = [100, 200]

    # Fill the first object (H)
    test_event['objects'][0]['id'] = 100
    test_event['objects'][0]['type'] = 'H'
    test_event['objects'][0]['x'] = 10.0
    test_event['objects'][0]['y'] = 20.0
    test_event['objects'][0]['z'] = 30.0
    test_event['objects'][0]['float_props'] = [0.5, 1.2]  # radius, energy

    # Fill the second object (V)
    test_event['objects'][1]['id'] = 200
    test_event['objects'][1]['type'] = 'V'
    test_event['objects'][1]['x'] = 12.0
    test_event['objects'][1]['y'] = 22.0
    test_event['objects'][1]['z'] = 32.0
    test_event['objects'][1]['float_props'] = [0.8, 2.0]  # radius, formation_energy

    # Test get_event_object
    h_obj = get_event_object(test_event, 'H')
    assert h_obj is not None
    assert h_obj['id'] == 100
    assert str(h_obj['type']).strip() == 'H'
    print("  get_event_object('H'): PASSED")

    v_obj = get_event_object(test_event, 'V')
    assert v_obj is not None
    assert v_obj['id'] == 200
    print("  get_event_object('V'): PASSED")

    # Test get_event_object_prop
    h_radius = get_event_object_prop(test_event, 'H', 'radius', config_with_mapping)
    assert h_radius == 0.5
    print("  get_event_object_prop('H', 'radius'): PASSED")

    v_radius = get_event_object_prop(test_event, 'V', 'radius', config_with_mapping)
    assert v_radius == 0.8
    print("  get_event_object_prop('V', 'radius'): PASSED")

    # Test get_event_param
    distance = get_event_param(test_event, 0)
    assert distance == 2.5
    print("  get_event_param(0): PASSED")

    # Test get_object_prop_from_record (core fields)
    x = get_object_prop_from_record(h_obj, 'H', 'x', config_with_mapping)
    assert x == 10.0
    print("  get_object_prop_from_record('x'): PASSED")

    # Test 6: vectorized event access (build an event array)
    print("\n[Test 6] Vectorized Event Access")

    events = np.zeros(3, dtype=event_dtype)
    for i in range(3):
        events[i]['id'] = i
        events[i]['type'] = 'capture'
        events[i]['params'] = [1.0 + i, 1e-9, 0.0, 0.0]
        events[i]['obj_ids'] = [100 + i, 200 + i]

        events[i]['objects'][0]['id'] = 100 + i
        events[i]['objects'][0]['type'] = 'H'
        events[i]['objects'][0]['float_props'] = [0.5 + i * 0.1, 1.0]

        events[i]['objects'][1]['id'] = 200 + i
        events[i]['objects'][1]['type'] = 'V'
        events[i]['objects'][1]['float_props'] = [0.8 + i * 0.1, 2.0]

    # Test get_distance
    distances = get_distance(events)
    assert len(distances) == 3
    assert distances[0] == 1.0
    assert distances[2] == 3.0
    print("  get_distance: PASSED")

    # Test get_events_params
    params_col = get_events_params(events, 1)  # D_eff
    assert len(params_col) == 3
    assert all(p == 1e-9 for p in params_col)
    print("  get_events_params: PASSED")

    # Test get_objects_positions
    x, y, z = get_objects_positions(events, obj_idx=0)
    assert len(x) == 3
    print("  get_objects_positions: PASSED")

    # Test get_object_ids
    ids = get_object_ids(events)
    assert ids.shape == (3, 2)
    assert ids[0, 0] == 100
    print("  get_object_ids: PASSED")

    print("\n" + "=" * 60)
    print("All tests passed!")


# ============================================================================
# Execute API - helper functions
# ============================================================================

def change_obj(obj, **kwargs):
    """
    Mark an object as modified and update its properties

    Args:
        obj: an element of the object array
        **kwargs: properties to update (e.g. x=1.0, y=2.0)

    Returns:
        dict: the id together with the updated properties

    Example:
        return [change_obj(obj, x=new_x, y=new_y)]
    """
    result = {'id': int(obj['id']), 'changed': True}
    for key, value in kwargs.items():
        result[key] = value
    return result


def new_obj(**kwargs):
    """
    Create a new object (id=-1 marks it as new)

    Args:
        **kwargs: object properties (type is required, e.g. type='V', x=1.0, y=2.0)

    Returns:
        dict: id=-1, new=True and every property

    Example:
        return [new_obj(type='V', x=x, y=y, z=z, float_props=[0.0, 1.0])]
    """
    result = {'id': -1, 'new': True}
    result.update(kwargs)
    return result


def delete_obj(obj):
    """
    Mark an object as deleted

    Args:
        obj: an element of the object array

    Returns:
        dict: the id together with active=False

    Example:
        return [delete_obj(obj)]
    """
    return {'id': int(obj['id']), 'active': False}
