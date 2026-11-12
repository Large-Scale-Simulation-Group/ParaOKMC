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
User configuration system - the seven-block model

A unified user configuration interface compiling the 7 user blocks into a flat config.

The seven blocks:
- domain: domain decomposition and spatial geometry
- simulation: run control
- objects: object type definitions
- ic: initial conditions
- events: event registration
- physics: physical fields and parameters
- outputs: output and post-processing

Main entry points:
    OKMCengine(config) -> OKMCEngineV16
"""

import numpy as np
from typing import Dict, List, Any, Callable, Type, Optional, Set, Tuple
from .event_class import (
    EventBase, ZeroObjectEvent, SingleObjectEvent, PairObjectEvent
)
from .event_compiler import merge_event_config
from .v16_engine import create_okmc_engine_v16
from .config_system import create_initial_objects_func


class ConfigValidationError(Exception):
    """Configuration validation error"""
    pass


def _get_physics_field_names(physics: dict) -> List[str]:
    """Return the field names declared in physics, accepting both the old field_names and the new fields form."""
    field_names: List[str] = []
    if physics.get('field_names'):
        field_names.extend(list(physics['field_names']))
    if physics.get('fields'):
        for name in physics['fields'].keys():
            if name not in field_names:
                field_names.append(name)
    return field_names


def normalize_user_config(user_config: dict) -> dict:
    """
    Normalize the user configuration

    Args:
        user_config: configuration dict supplied by the user

    Returns:
        the normalized configuration dict
    """
    # Deep copy so the caller's configuration is untouched
    import copy
    config = copy.deepcopy(user_config)

    # Make sure every required block exists
    required_blocks = ['domain', 'simulation', 'objects', 'ic', 'events', 'physics', 'outputs']
    for block in required_blocks:
        if block not in config:
            config[block] = {}

    return config


def validate_user_config(user_config: dict) -> None:
    """
    Validate the user configuration

    Args:
        user_config: the normalized user configuration

    Raises:
        ConfigValidationError: if the configuration is invalid
    """
    # 1. Top-level structure
    if not isinstance(user_config, dict):
        raise ConfigValidationError("Config must be a dictionary")

    allowed_blocks = {'domain', 'simulation', 'objects', 'ic', 'events', 'physics', 'outputs'}
    user_blocks = set(user_config.keys())
    extra_blocks = user_blocks - allowed_blocks

    if extra_blocks:
        raise ConfigValidationError(
            f"Invalid top-level blocks: {extra_blocks}. "
            f"ParaOKMC now requires top-level blocks: domain, simulation, objects, ic, events, physics, outputs. "
            f"Legacy flat config is no longer accepted by OKMCengine(...)."
        )

    # 2. Validate each block
    _validate_domain(user_config.get('domain', {}))
    _validate_simulation(user_config.get('simulation', {}))
    _validate_objects(user_config.get('objects', {}))
    _validate_ic(user_config.get('ic', {}), user_config.get('objects', {}))
    _validate_events(user_config.get('events', {}), user_config.get('objects', {}),
                     user_config.get('physics', {}))
    _validate_physics(user_config.get('physics', {}))
    _validate_outputs(user_config.get('outputs', {}))

    # 3. Cross-block: event names referenced by physics.triggers must exist
    physics = user_config.get('physics', {})
    events = user_config.get('events', {})
    if 'triggers' in physics and physics['triggers']:
        for i, trigger_def in enumerate(physics['triggers']):
            event_name = trigger_def['event']
            if event_name not in events:
                raise ConfigValidationError(
                    f"physics.triggers[{i}]: event '{event_name}' is not defined in config['events']. "
                    f"Available events: {list(events.keys())}"
                )

    # 4. Cross-block: physics.fields[*].update_policy.trigger_events must be registered
    if 'fields' in physics and physics['fields']:
        for field_name, field_def in physics['fields'].items():
            policy = field_def.get('update_policy', {})
            for event_name in policy.get('trigger_events', []):
                if event_name not in events:
                    raise ConfigValidationError(
                        f"physics.fields['{field_name}'].update_policy.trigger_events "
                        f"contains unknown event '{event_name}'. Available events: {list(events.keys())}"
                    )

            for query_name, query_def in field_def.get('queries', {}).items():
                for obj_type in query_def.get('types', []):
                    if obj_type not in user_config.get('objects', {}):
                        raise ConfigValidationError(
                            f"physics.fields['{field_name}'].queries['{query_name}'] "
                            f"references unknown object type '{obj_type}'. "
                            f"Available objects: {list(user_config.get('objects', {}).keys())}"
                        )

            for obj_type in policy.get('object_types', []):
                if obj_type not in user_config.get('objects', {}):
                    raise ConfigValidationError(
                        f"physics.fields['{field_name}'].update_policy.object_types "
                        f"contains unknown object type '{obj_type}'. "
                        f"Available objects: {list(user_config.get('objects', {}).keys())}"
                    )


def _validate_domain(domain: dict) -> None:
    """Validate the domain block"""
    if not domain:
        raise ConfigValidationError("domain block is required")

    # lattice_constant
    if 'lattice_constant' not in domain:
        raise ConfigValidationError("domain.lattice_constant is required")
    lc = domain['lattice_constant']
    if not isinstance(lc, (int, float)) or lc <= 0:
        raise ConfigValidationError(f"domain.lattice_constant must be positive number, got {lc}")

    # lattice_grid
    if 'lattice_grid' not in domain:
        raise ConfigValidationError("domain.lattice_grid is required")
    grid = domain['lattice_grid']
    if not isinstance(grid, (list, tuple)) or len(grid) != 3:
        raise ConfigValidationError(f"domain.lattice_grid must be [nx, ny, nz], got {grid}")
    if not all(isinstance(x, int) and x > 0 for x in grid):
        raise ConfigValidationError(f"domain.lattice_grid must be positive integers, got {grid}")

    # periodic
    if 'periodic' not in domain:
        raise ConfigValidationError("domain.periodic is required")
    periodic = domain['periodic']
    if not isinstance(periodic, (list, tuple)) or len(periodic) != 3:
        raise ConfigValidationError(f"domain.periodic must be [bool, bool, bool], got {periodic}")
    if not all(isinstance(x, bool) for x in periodic):
        raise ConfigValidationError(f"domain.periodic must be booleans, got {periodic}")

    # max_interaction_distance
    if 'max_interaction_distance' not in domain:
        raise ConfigValidationError("domain.max_interaction_distance is required")
    mid = domain['max_interaction_distance']
    if not isinstance(mid, (int, float)) or mid <= 0:
        raise ConfigValidationError(f"domain.max_interaction_distance must be positive, got {mid}")

    # h_factor (optional)
    if 'h_factor' in domain:
        hf = domain['h_factor']
        if not isinstance(hf, int) or hf <= 0:
            raise ConfigValidationError(f"domain.h_factor must be positive integer, got {hf}")


def _validate_simulation(simulation: dict) -> None:
    """Validate the simulation block"""
    if not simulation:
        raise ConfigValidationError("simulation block is required")

    # max_steps
    if 'max_steps' not in simulation:
        raise ConfigValidationError("simulation.max_steps is required")
    ms = simulation['max_steps']
    if not isinstance(ms, int) or ms <= 0:
        raise ConfigValidationError(f"simulation.max_steps must be positive integer, got {ms}")

    # max_time (optional)
    if 'max_time' in simulation:
        mt = simulation['max_time']
        if not isinstance(mt, (int, float)) or (mt <= 0 and mt != float('inf')):
            raise ConfigValidationError(f"simulation.max_time must be positive or inf, got {mt}")

    # random_seed (optional)
    if 'random_seed' in simulation:
        rs = simulation['random_seed']
        if not isinstance(rs, int):
            raise ConfigValidationError(f"simulation.random_seed must be integer, got {rs}")


def _validate_objects(objects: dict) -> None:
    """Validate the objects block"""
    if not objects:
        raise ConfigValidationError("objects block must be non-empty")

    if not isinstance(objects, dict):
        raise ConfigValidationError("objects must be a dictionary")

    for obj_type, obj_def in objects.items():
        if not isinstance(obj_def, dict):
            raise ConfigValidationError(f"objects['{obj_type}'] must be a dictionary")

        # size (optional)
        if 'size' in obj_def:
            size = obj_def['size']
            if not isinstance(size, (int, float)) or size < 0:
                raise ConfigValidationError(f"objects['{obj_type}'].size must be non-negative, got {size}")

        # properties (optional)
        if 'properties' in obj_def:
            props = obj_def['properties']
            if not isinstance(props, dict):
                raise ConfigValidationError(f"objects['{obj_type}'].properties must be a dictionary")
            for prop_name, prop_value in props.items():
                if not isinstance(prop_value, (int, float, str)):
                    raise ConfigValidationError(
                        f"objects['{obj_type}'].properties['{prop_name}'] must be scalar, got {type(prop_value)}"
                    )


def _validate_ic(ic: dict, objects: dict) -> None:
    """Validate the ic block"""
    if not ic:
        raise ConfigValidationError("ic block is required")

    # mode
    if 'mode' not in ic:
        raise ConfigValidationError("ic.mode is required (must be 'builtin' or 'function')")

    mode = ic['mode']
    if mode not in ('builtin', 'function', 'dump'):
        raise ConfigValidationError(f"ic.mode must be 'builtin', 'function' or 'dump', got '{mode}'")

    if mode == 'builtin':
        if 'initial_conditions' not in ic:
            raise ConfigValidationError("ic.initial_conditions is required when mode='builtin'")

        initial_conditions = ic['initial_conditions']
        if not isinstance(initial_conditions, dict):
            raise ConfigValidationError("ic.initial_conditions must be a dictionary")

        # The object type must be registered
        for obj_type in initial_conditions.keys():
            if obj_type not in objects:
                raise ConfigValidationError(
                    f"ic.initial_conditions references unknown object type '{obj_type}'. "
                    f"Define it in config['objects'] first."
                )

    elif mode == 'function':
        if 'func' not in ic:
            raise ConfigValidationError("ic.func is required when mode='function'")
        if not callable(ic['func']):
            raise ConfigValidationError("ic.func must be callable")

    elif mode == 'dump':
        if 'file' not in ic or not isinstance(ic['file'], str):
            raise ConfigValidationError("ic.file (a dump file or a dump directory) is required when mode='dump'")
        if 'frame' in ic and not isinstance(ic['frame'], int):
            raise ConfigValidationError(f"ic.frame must be an integer step (or -1 for the last), got {ic['frame']}")
        if ic.get('id_mode', 'keep') not in ('keep', 'renumber'):
            raise ConfigValidationError("ic.id_mode must be 'keep' or 'renumber'")


def _validate_events(events: dict, objects: dict, physics: dict) -> None:
    """Validate the events block"""
    if not events:
        # An empty event set is allowed (useful in tests)
        return

    if not isinstance(events, dict):
        raise ConfigValidationError("events must be a dictionary")

    for event_name, event_class in events.items():
        # Must be a subclass of EventBase
        if not isinstance(event_class, type) or not issubclass(event_class, EventBase):
            raise ConfigValidationError(
                f"Event '{event_name}': value must be an EventBase subclass, got {type(event_class)}"
            )

        # Validate according to the base class
        if issubclass(event_class, ZeroObjectEvent):
            _validate_zero_object_event(event_name, event_class, physics)
        elif issubclass(event_class, SingleObjectEvent):
            _validate_single_object_event(event_name, event_class, objects, physics)
        elif issubclass(event_class, PairObjectEvent):
            _validate_pair_object_event(event_name, event_class, objects, physics)
        else:
            raise ConfigValidationError(
                f"Event '{event_name}': must inherit from ZeroObjectEvent, SingleObjectEvent, or PairObjectEvent"
            )


def _validate_zero_object_event(event_name: str, event_class: Type[EventBase], physics: dict) -> None:
    """Validate a ZeroObjectEvent"""
    # object_types must not be defined
    if hasattr(event_class, 'object_types') and event_class.object_types:
        raise ConfigValidationError(
            f"Event '{event_name}': ZeroObjectEvent must not define object_types"
        )

    # type_pairs must not be defined
    if hasattr(event_class, 'type_pairs') and event_class.type_pairs:
        raise ConfigValidationError(
            f"Event '{event_name}': ZeroObjectEvent must not define type_pairs"
        )

    # Validate rate_deps
    if hasattr(event_class, 'rate_deps') and event_class.rate_deps:
        _validate_rate_deps(event_name, event_class, {}, physics)


def _validate_single_object_event(event_name: str, event_class: Type[EventBase],
                                   objects: dict, physics: dict) -> None:
    """Validate a SingleObjectEvent"""
    # object_types is required
    if not hasattr(event_class, 'object_types') or not event_class.object_types:
        raise ConfigValidationError(
            f"Event '{event_name}': SingleObjectEvent must define non-empty object_types"
        )

    # The object types must be registered
    for obj_type in event_class.object_types:
        if obj_type not in objects:
            raise ConfigValidationError(
                f"Event '{event_name}': object type '{obj_type}' in object_types is not defined in config['objects']"
            )

    # Validate rate_deps
    if hasattr(event_class, 'rate_deps') and event_class.rate_deps:
        _validate_rate_deps(event_name, event_class, objects, physics)


def _validate_pair_object_event(event_name: str, event_class: Type[EventBase],
                                 objects: dict, physics: dict) -> None:
    """Validate a PairObjectEvent"""
    # type_pairs is required
    if not hasattr(event_class, 'type_pairs') or not event_class.type_pairs:
        raise ConfigValidationError(
            f"Event '{event_name}': PairObjectEvent must define non-empty type_pairs"
        )

    # The object types must be registered
    for pair in event_class.type_pairs:
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            raise ConfigValidationError(
                f"Event '{event_name}': type_pairs must be list of (type1, type2) tuples, got {pair}"
            )
        type1, type2 = pair
        if type1 not in objects:
            raise ConfigValidationError(
                f"Event '{event_name}': object type '{type1}' in type_pairs is not defined in config['objects']"
            )
        if type2 not in objects:
            raise ConfigValidationError(
                f"Event '{event_name}': object type '{type2}' in type_pairs is not defined in config['objects']"
            )

    # max_distance is required
    if not hasattr(event_class, 'max_distance') or event_class.max_distance is None:
        raise ConfigValidationError(
            f"Event '{event_name}': PairObjectEvent must define max_distance"
        )
    if not isinstance(event_class.max_distance, (int, float)) or event_class.max_distance <= 0:
        raise ConfigValidationError(
            f"Event '{event_name}': max_distance must be positive number, got {event_class.max_distance}"
        )

    # Validate rate_deps
    if hasattr(event_class, 'rate_deps') and event_class.rate_deps:
        _validate_rate_deps(event_name, event_class, objects, physics)


def _validate_rate_deps(event_name: str, event_class: Type[EventBase],
                        objects: dict, physics: dict) -> None:
    """Validate the rate_deps dependencies"""
    rate_deps = event_class.rate_deps

    # Validate the obj1 dependencies
    if hasattr(rate_deps, 'obj1') and rate_deps.obj1:
        obj_types = []
        if hasattr(event_class, 'object_types'):
            obj_types = event_class.object_types
        elif hasattr(event_class, 'type_pairs'):
            obj_types = list(set(pair[0] for pair in event_class.type_pairs))

        for obj_type in obj_types:
            if obj_type not in objects:
                raise ConfigValidationError(
                    f"Event '{event_name}': object type '{obj_type}' not defined in config['objects']"
                )

            # Strict: report an error even when properties is missing or empty
            obj_def = objects[obj_type]
            props = obj_def.get('properties', {})

            for dep in rate_deps.obj1:
                if dep not in props:
                    raise ConfigValidationError(
                        f"Event '{event_name}': missing object property '{dep}' "
                        f"in objects['{obj_type}']['properties'] for rate_deps.obj1. "
                        f"Available properties: {list(props.keys())}"
                    )

    # Validate the obj2 dependencies
    if hasattr(rate_deps, 'obj2') and rate_deps.obj2:
        if hasattr(event_class, 'type_pairs'):
            obj_types = list(set(pair[1] for pair in event_class.type_pairs))
            for obj_type in obj_types:
                if obj_type not in objects:
                    raise ConfigValidationError(
                        f"Event '{event_name}': object type '{obj_type}' not defined in config['objects']"
                    )

                obj_def = objects[obj_type]
                props = obj_def.get('properties', {})

                for dep in rate_deps.obj2:
                    if dep not in props:
                        raise ConfigValidationError(
                            f"Event '{event_name}': missing object property '{dep}' "
                            f"in objects['{obj_type}']['properties'] for rate_deps.obj2. "
                            f"Available properties: {list(props.keys())}"
                        )

    # Validate the physics dependencies - strict
    if hasattr(rate_deps, 'physics') and rate_deps.physics:
        # A physics block with params is required
        if not physics or 'params' not in physics:
            raise ConfigValidationError(
                f"Event '{event_name}': rate_deps.physics requires config['physics']['params'], "
                f"but physics block is empty or missing 'params'"
            )

        physics_params = physics['params']
        for dep in rate_deps.physics:
            if dep not in physics_params:
                raise ConfigValidationError(
                    f"Event '{event_name}': physics quantity '{dep}' requested by rate_deps.physics "
                    f"is not defined in config['physics']['params']. "
                    f"Available params: {list(physics_params.keys())}"
                )

    # Validate the fields dependencies - strict
    if hasattr(rate_deps, 'fields') and rate_deps.fields:
        field_names = _get_physics_field_names(physics)
        if not field_names:
            raise ConfigValidationError(
                f"Event '{event_name}': rate_deps.fields requires physics fields to be defined, "
                f"but physics block is empty or missing both 'field_names' and 'fields'"
            )
        for dep in rate_deps.fields:
            if dep not in field_names:
                raise ConfigValidationError(
                    f"Event '{event_name}': field '{dep}' requested by rate_deps.fields "
                    f"is not defined in config['physics']. "
                    f"Available fields: {field_names}"
                )

    # Validate the params dependencies - strict
    if hasattr(rate_deps, 'params') and rate_deps.params:
        # params refers to event parameters, so check the event class params
        event_params = getattr(event_class, 'params', [])
        if event_params is None:
            event_params = []

        for dep in rate_deps.params:
            # Special case: distance is always available for pair events
            if dep == 'distance':
                if not (isinstance(event_class, type) and issubclass(event_class, PairObjectEvent)):
                    raise ConfigValidationError(
                        f"Event '{event_name}': param 'distance' in rate_deps.params "
                        f"is only available for PairObjectEvent"
                    )
                continue

            # Every other parameter must be declared in the event params
            if dep not in event_params:
                raise ConfigValidationError(
                    f"Event '{event_name}': param '{dep}' requested by rate_deps.params "
                    f"is not defined in event class params. "
                    f"Available params: {event_params}"
                )


def _validate_physics(physics: dict) -> None:
    """Validate the physics block"""
    # physics may be empty
    if not physics:
        return

    # params (optional)
    if 'params' in physics:
        params = physics['params']
        if not isinstance(params, dict):
            raise ConfigValidationError("physics.params must be a dictionary")
        for param_name, param_value in params.items():
            if not isinstance(param_value, (int, float, bool)):
                raise ConfigValidationError(
                    f"physics.params['{param_name}'] must be scalar, got {type(param_value)}"
                )

    # field_names and field_shapes (legacy)
    if 'field_names' in physics:
        field_names = physics['field_names']
        if not isinstance(field_names, list):
            raise ConfigValidationError("physics.field_names must be a list")

        if 'field_shapes' in physics:
            field_shapes = physics['field_shapes']
            if not isinstance(field_shapes, list):
                raise ConfigValidationError("physics.field_shapes must be a list")
            if len(field_shapes) != len(field_names):
                raise ConfigValidationError(
                    f"physics.field_shapes length ({len(field_shapes)}) must match "
                    f"field_names length ({len(field_names)})"
                )

    # fields (new style)
    if 'fields' in physics:
        fields = physics['fields']
        if not isinstance(fields, dict):
            raise ConfigValidationError("physics.fields must be a dictionary")

        for field_name, field_def in fields.items():
            if not isinstance(field_def, dict):
                raise ConfigValidationError(
                    f"physics.fields['{field_name}'] must be a dictionary"
                )

            if 'shape' not in field_def:
                raise ConfigValidationError(
                    f"physics.fields['{field_name}'].shape is required"
                )

            shape = field_def['shape']
            if not isinstance(shape, (list, tuple)) or len(shape) != 3:
                raise ConfigValidationError(
                    f"physics.fields['{field_name}'].shape must be [nx, ny, nz], got {shape}"
                )
            if not all(isinstance(x, int) and x > 0 for x in shape):
                raise ConfigValidationError(
                    f"physics.fields['{field_name}'].shape must contain positive integers, got {shape}"
                )

            if 'update_func' in field_def and field_def['update_func'] is not None:
                if not callable(field_def['update_func']):
                    raise ConfigValidationError(
                        f"physics.fields['{field_name}'].update_func must be callable"
                    )

            if 'queries' in field_def:
                queries = field_def['queries']
                if not isinstance(queries, dict):
                    raise ConfigValidationError(
                        f"physics.fields['{field_name}'].queries must be a dictionary"
                    )
                for query_name, query_def in queries.items():
                    if not isinstance(query_def, dict):
                        raise ConfigValidationError(
                            f"physics.fields['{field_name}'].queries['{query_name}'] must be a dictionary"
                        )
                    if not query_def:
                        raise ConfigValidationError(
                            f"physics.fields['{field_name}'].queries['{query_name}'] cannot be empty"
                        )
                    if query_def.get('all', False) and 'types' in query_def:
                        raise ConfigValidationError(
                            f"physics.fields['{field_name}'].queries['{query_name}'] "
                            "cannot define both 'all' and 'types'"
                        )
                    if 'types' in query_def:
                        if not isinstance(query_def['types'], list) or not all(
                            isinstance(t, str) for t in query_def['types']
                        ):
                            raise ConfigValidationError(
                                f"physics.fields['{field_name}'].queries['{query_name}'].types "
                                "must be a list of strings"
                            )
                    elif not query_def.get('all', False):
                        raise ConfigValidationError(
                            f"physics.fields['{field_name}'].queries['{query_name}'] must define "
                            "'all: True' or 'types: [...]'"
                        )

            if 'update_policy' in field_def:
                policy = field_def['update_policy']
                if not isinstance(policy, dict):
                    raise ConfigValidationError(
                        f"physics.fields['{field_name}'].update_policy must be a dictionary"
                    )

                if 'step_interval' in policy:
                    step_interval = policy['step_interval']
                    if not isinstance(step_interval, int) or step_interval <= 0:
                        raise ConfigValidationError(
                            f"physics.fields['{field_name}'].update_policy.step_interval must be positive integer"
                        )

                if 'time_interval' in policy:
                    time_interval = policy['time_interval']
                    if not isinstance(time_interval, (int, float)) or time_interval <= 0:
                        raise ConfigValidationError(
                            f"physics.fields['{field_name}'].update_policy.time_interval must be positive number"
                        )

                if 'trigger_events' in policy:
                    trigger_events = policy['trigger_events']
                    if not isinstance(trigger_events, list) or not all(
                        isinstance(e, str) for e in trigger_events
                    ):
                        raise ConfigValidationError(
                            f"physics.fields['{field_name}'].update_policy.trigger_events "
                            "must be a list of event names"
                        )

                for key in (
                    'trigger_on_object_change',
                    'trigger_on_object_create',
                    'trigger_on_object_delete',
                ):
                    if key in policy and not isinstance(policy[key], bool):
                        raise ConfigValidationError(
                            f"physics.fields['{field_name}'].update_policy.{key} must be bool"
                        )

                if 'object_types' in policy:
                    object_types = policy['object_types']
                    if not isinstance(object_types, list) or not all(
                        isinstance(t, str) for t in object_types
                    ):
                        raise ConfigValidationError(
                            f"physics.fields['{field_name}'].update_policy.object_types "
                            "must be a list of strings"
                        )

    # solvers (optional)
    if 'solvers' in physics:
        solvers = physics['solvers']
        if not isinstance(solvers, list):
            raise ConfigValidationError("physics.solvers must be a list")

        field_names = _get_physics_field_names(physics)
        for i, solver_def in enumerate(solvers):
            if not isinstance(solver_def, dict):
                raise ConfigValidationError(f"physics.solvers[{i}] must be a dictionary")

            # name is required
            if 'name' not in solver_def:
                raise ConfigValidationError(f"physics.solvers[{i}] must have 'name' field")

            # solver is required and must satisfy the ExternalSolver contract (initialize + solve)
            if 'solver' not in solver_def:
                raise ConfigValidationError(f"physics.solvers[{i}] must have 'solver' field")
            solver_obj = solver_def['solver']
            if not hasattr(solver_obj, 'initialize'):
                raise ConfigValidationError(
                    f"physics.solvers[{i}]['solver'] must have 'initialize' method "
                    f"(compatible with ExternalSolver interface)"
                )
            if not hasattr(solver_obj, 'solve'):
                raise ConfigValidationError(
                    f"physics.solvers[{i}]['solver'] must have 'solve' method "
                    f"(compatible with ExternalSolver interface)"
                )

            # The fields must be declared in field_names
            if 'fields' in solver_def:
                solver_fields = solver_def['fields']
                if not isinstance(solver_fields, list):
                    raise ConfigValidationError(f"physics.solvers[{i}]['fields'] must be a list")
                for field in solver_fields:
                    if field not in field_names:
                        raise ConfigValidationError(
                            f"physics.solvers[{i}]: field '{field}' is not defined in physics.field_names. "
                            f"Available fields: {field_names}"
                        )

    # triggers (optional) - revalidated in validate_user_config (it needs events)
    if 'triggers' in physics:
        triggers = physics['triggers']
        if not isinstance(triggers, list):
            raise ConfigValidationError("physics.triggers must be a list")

        field_names = _get_physics_field_names(physics)
        for i, trigger_def in enumerate(triggers):
            if not isinstance(trigger_def, dict):
                raise ConfigValidationError(f"physics.triggers[{i}] must be a dictionary")

            # event is required
            if 'event' not in trigger_def:
                raise ConfigValidationError(f"physics.triggers[{i}] must have 'event' field")

            # fields is required
            if 'fields' not in trigger_def:
                raise ConfigValidationError(f"physics.triggers[{i}] must have 'fields' field")

            trigger_fields = trigger_def['fields']
            if not isinstance(trigger_fields, list):
                raise ConfigValidationError(f"physics.triggers[{i}]['fields'] must be a list")

            # The fields must be declared in field_names
            for field in trigger_fields:
                if field not in field_names:
                    raise ConfigValidationError(
                        f"physics.triggers[{i}]: field '{field}' is not defined in physics.field_names. "
                        f"Available fields: {field_names}"
                    )

            # condition, if present, must be callable
            if 'condition' in trigger_def and trigger_def['condition'] is not None:
                if not callable(trigger_def['condition']):
                    raise ConfigValidationError(
                        f"physics.triggers[{i}]['condition'] must be callable, got {type(trigger_def['condition'])}"
                    )


def _validate_outputs(outputs: dict) -> None:
    """Validate the outputs block"""
    # outputs may be empty
    if not outputs:
        return

    # interval (optional)
    if 'interval' in outputs:
        interval = outputs['interval']
        if not isinstance(interval, int) or interval <= 0:
            raise ConfigValidationError(f"outputs.interval must be positive integer, got {interval}")

    # ovito_dump (optional)
    if 'ovito_dump' in outputs:
        dump = outputs['ovito_dump']
        if not isinstance(dump, dict):
            raise ConfigValidationError(
                f"outputs.ovito_dump must be a dictionary, got {type(dump).__name__}"
            )

        if 'enabled' in dump:
            if not isinstance(dump['enabled'], bool):
                raise ConfigValidationError(
                    f"outputs.ovito_dump.enabled must be bool, got {type(dump['enabled']).__name__}"
                )

        if 'output_dir' in dump:
            if not isinstance(dump['output_dir'], str):
                raise ConfigValidationError(
                    f"outputs.ovito_dump.output_dir must be string, got {type(dump['output_dir']).__name__}"
                )

        if 'prefix' in dump:
            if not isinstance(dump['prefix'], str):
                raise ConfigValidationError(
                    f"outputs.ovito_dump.prefix must be string, got {type(dump['prefix']).__name__}"
                )

        if 'mode' in dump and dump['mode'] not in ('distributed', 'collect'):
            raise ConfigValidationError(
                f"outputs.ovito_dump.mode must be 'distributed' (one file per rank, default) "
                f"or 'collect' (one file written by rank 0), got {dump['mode']!r}"
            )

        if 'properties' in dump:
            props = dump['properties']
            if not isinstance(props, list):
                raise ConfigValidationError(
                    f"outputs.ovito_dump.properties must be a list, got {type(props).__name__}"
                )
            for i, p in enumerate(props):
                if not isinstance(p, str):
                    raise ConfigValidationError(
                        f"outputs.ovito_dump.properties[{i}] must be string, got {type(p).__name__}"
                    )

    # postprocessors (optional)
    if 'postprocessors' in outputs:
        postprocessors = outputs['postprocessors']
        if not isinstance(postprocessors, list):
            raise ConfigValidationError("outputs.postprocessors must be a list")
        for i, pp in enumerate(postprocessors):
            if not callable(pp):
                raise ConfigValidationError(f"outputs.postprocessors[{i}] must be callable")

    # event_hooks (optional): run on every rank after each applied state-change message
    if 'event_hooks' in outputs:
        hooks = outputs['event_hooks']
        if not isinstance(hooks, list):
            raise ConfigValidationError("outputs.event_hooks must be a list")
        for i, hook in enumerate(hooks):
            if not callable(hook):
                raise ConfigValidationError(f"outputs.event_hooks[{i}] must be callable")

    # csv (not yet supported)
    if 'csv' in outputs:
        raise ConfigValidationError("outputs.csv is not yet supported")


def compile_user_config(user_config: dict) -> dict:
    """
    Compile the user's 7 blocks into the internal flat config

    Args:
        user_config: the validated user configuration

    Returns:
        the internal flat config, ready for merge_event_config and create_okmc_engine_v16
    """
    domain = user_config.get('domain', {})
    simulation = user_config.get('simulation', {})
    objects = user_config.get('objects', {})
    ic = user_config.get('ic', {})
    events = user_config.get('events', {})
    physics = user_config.get('physics', {})
    outputs = user_config.get('outputs', {})

    # Compile domain
    lattice_constant = domain['lattice_constant']
    lattice_grid = domain['lattice_grid']
    box_size = [
        lattice_grid[0] * lattice_constant,
        lattice_grid[1] * lattice_constant,
        lattice_grid[2] * lattice_constant,
    ]

    domain_decomposition = {
        'lattice_constant': lattice_constant,
        'lattice_grid': lattice_grid,
        'box_size': box_size,
        'periodic': domain['periodic'],
        'max_interaction_distance': domain['max_interaction_distance'],
    }
    if 'h_factor' in domain:
        domain_decomposition['h_factor'] = domain['h_factor']

    # Compile simulation
    flat_config = {
        'domain_decomposition': domain_decomposition,
        'lattice_constant': lattice_constant,
        'grid_size': lattice_grid,
        'max_steps': simulation['max_steps'],
        'max_time': simulation.get('max_time', float('inf')),
        'random_seed': simulation.get('random_seed', 42),
        'verbose': simulation.get('verbose', False),
        'progress_interval': simulation.get('progress_interval', 500),
    }

    # Compile the simulation sub-configuration
    if 'incremental_update' in simulation:
        flat_config['incremental_update'] = simulation['incremental_update']
    if 'predictive_rebalance' in simulation:
        flat_config['predictive_rebalance'] = simulation['predictive_rebalance']
    if 'runtime_load_balancer' in simulation:
        flat_config['runtime_load_balancer'] = simulation['runtime_load_balancer']
    if 'adaptive_weight' in simulation:
        flat_config['adaptive_weight'] = simulation['adaptive_weight']
    if 'rma_object_pool' in simulation:
        flat_config['rma_object_pool'] = simulation['rma_object_pool']
    if 'rma_event_pool' in simulation:
        flat_config['rma_event_pool'] = simulation['rma_event_pool']
    if 'object_load_balancer' in simulation:
        flat_config['object_load_balancer'] = simulation['object_load_balancer']

    # Compile objects
    flat_config['objects'] = objects

    # Compile ic
    if ic.get('mode') == 'builtin':
        initial_conditions = ic['initial_conditions']
        flat_config['initial_objects_func'] = create_initial_objects_func(
            initial_conditions, objects, lattice_grid, lattice_constant,
            seed=simulation.get('random_seed', 42)
        )
    elif ic.get('mode') == 'function':
        flat_config['initial_objects_func'] = ic['func']
    elif ic.get('mode') == 'dump':
        from .post_processor import create_dump_reader
        flat_config['initial_objects_func'] = create_dump_reader(
            ic['file'], frame=ic.get('frame', -1), id_mode=ic.get('id_mode', 'keep'))
    else:
        # Default: empty IC
        flat_config['initial_objects_func'] = lambda rank, size, config: []

    # Compile events (build the minimal events dict first)
    flat_config['events'] = {name: {} for name in events.keys()}

    # Compile physics
    if physics:
        physics = dict(physics)

    # The new physics.fields is compiled into internal field_names/field_shapes
        if 'fields' in physics and physics['fields']:
            field_names = list(physics.get('field_names', []))
            field_shapes = [tuple(shape) for shape in physics.get('field_shapes', [])]
            existing = {name: shape for name, shape in zip(field_names, field_shapes)}

            for field_name, field_def in physics['fields'].items():
                field_shape = tuple(field_def['shape'])
                if field_name not in existing:
                    field_names.append(field_name)
                    field_shapes.append(field_shape)
                    existing[field_name] = field_shape

            physics['field_names'] = field_names
            physics['field_shapes'] = field_shapes

        # params, if present, must be registered through ic_func
        if 'params' in physics and physics['params']:
            original_ic_func = physics.get('ic_func', None)

            def physics_ic_with_params(physics_obj, config,
                                       original_ic_func=original_ic_func,
                                       params_dict=physics['params']):
                for param_name, param_value in params_dict.items():
                    physics_obj.set_param(param_name, param_value)

                if original_ic_func is not None:
                    original_ic_func(physics_obj, config)

            physics['ic_func'] = physics_ic_with_params

        # Register the new fields configuration: initial value, queries, update_policy, update_func
        if 'fields' in physics and physics['fields']:
            original_ic_func = physics.get('ic_func', None)

            def physics_ic_with_fields(physics_obj, config,
                                       original_ic_func=original_ic_func,
                                       fields_dict=physics['fields']):
                if original_ic_func is not None:
                    original_ic_func(physics_obj, config)

                for field_name, field_def in fields_dict.items():
                    if 'initial_value' in field_def:
                        field = np.full(tuple(field_def['shape']), float(field_def['initial_value']), dtype=np.float64)
                        physics_obj.set_field(field_name, field)

                    if 'update_func' in field_def and field_def['update_func'] is not None:
                        policy = dict(field_def.get('update_policy', {}))
                        physics_obj.set_field_update(
                            field_name,
                            field_def['update_func'],
                            step_interval=policy.get('step_interval', None),
                            time_interval=policy.get('time_interval', None),
                            enabled=policy.get('enabled', True),
                            queries=field_def.get('queries', {}),
                            update_policy=policy,
                        )

            physics['ic_func'] = physics_ic_with_fields

        # Register the solvers
        if 'solvers' in physics and physics['solvers']:
            original_ic_func = physics.get('ic_func', None)

            def physics_ic_with_solvers(physics_obj, config,
                                        original_ic_func=original_ic_func,
                                        solver_defs=physics['solvers']):
                if original_ic_func is not None:
                    original_ic_func(physics_obj, config)

                for solver_def in solver_defs:
                    solver_name = solver_def['name']
                    solver_obj = solver_def['solver']
                    fields = solver_def.get('fields', [])
                    physics_obj.register_solver(solver_name, solver_obj, fields)

            physics['ic_func'] = physics_ic_with_solvers

        # Register the triggers
        if 'triggers' in physics and physics['triggers']:
            original_ic_func = physics.get('ic_func', None)

            def physics_ic_with_triggers(physics_obj, config,
                                         original_ic_func=original_ic_func,
                                         trigger_defs=physics['triggers']):
                if original_ic_func is not None:
                    original_ic_func(physics_obj, config)

                for trigger_def in trigger_defs:
                    event_name = trigger_def['event']
                    fields = trigger_def['fields']
                    solver_name = trigger_def.get('solver', None)
                    condition = trigger_def.get('condition', None)
                    physics_obj.register_event_trigger(event_name, fields, solver_name, condition)

            physics['ic_func'] = physics_ic_with_triggers

        flat_config['physics'] = physics

    # Compile outputs
    if outputs:
        post_processing = {}
        if 'interval' in outputs:
            post_processing['interval'] = outputs['interval']

        # Assemble analysis_funcs
        analysis_funcs = []

        # ovito_dump
        if outputs.get('ovito_dump', {}).get('enabled', False):
            from .post_processor import create_dump_writer, create_distributed_dump_writer
            dump_config = outputs['ovito_dump']
            # 'distributed' (default): every rank writes prefix.<step>.rank_<r>.dump with its own
            # objects and no gather; OVITO loads the set through the wildcard
            # prefix.<step>.rank_*.dump, and ic.mode='dump' reads the directory back.
            # 'collect': rank 0 gathers everything and writes one prefix.<step>.dump.
            factory = (create_distributed_dump_writer
                       if dump_config.get('mode', 'distributed') == 'distributed'
                       else create_dump_writer)
            dump_writer = factory(
                output_dir=dump_config.get('output_dir', 'dump'),
                prefix=dump_config.get('prefix', 'state'),
                properties=dump_config.get('properties', ['id', 'type', 'x', 'y', 'z']),
                box_size=box_size,  # box_size derived from domain
                periodic=domain['periodic'],  # periodic taken from domain
                type_map={name: i + 1 for i, name in enumerate(objects.keys())},
            )
            analysis_funcs.append(dump_writer)

        # postprocessors
        if 'postprocessors' in outputs:
            analysis_funcs.extend(outputs['postprocessors'])

        if analysis_funcs:
            post_processing['analysis_funcs'] = analysis_funcs

        # event hooks (per-step observers fed by the broadcast state-change message)
        if outputs.get('event_hooks'):
            post_processing['event_hooks'] = list(outputs['event_hooks'])

        if post_processing:
            flat_config['post_processing'] = post_processing

    return flat_config


def OKMCengine(user_config: dict):
    """
    Unified ParaOKMC user entry point

    Takes the 7 user blocks, compiles them and creates an OKMCEngineV16 instance.

    Args:
        user_config: user configuration dict; the following top-level blocks are required:
            - domain: domain decomposition and spatial geometry
            - simulation: run control
            - objects: object type definitions
            - ic: initial conditions
            - events: event registration
            - physics: physical fields and parameters
            - outputs: output and post-processing

    Returns:
        an OKMCEngineV16 instance

    Example:
        ```python
        from paraokmc import OKMCengine

        config = {
            "domain": {...},
            "simulation": {...},
            "objects": {...},
            "ic": {...},
            "events": {
                "H_migration": HMigration,
                "capture": HV1Capture,
            },
            "physics": {...},
            "outputs": {...},
        }

        engine = OKMCengine(config)
        stats = engine.run()
        engine.destroy()
        ```
    """
    # 1. Normalize
    normalized = normalize_user_config(user_config)

    # 2. Validate
    validate_user_config(normalized)

    # 3. Compile into the flat config
    flat_config = compile_user_config(normalized)

    # 4. Merge the event configuration
    events = normalized.get('events', {})
    if events:
        flat_config = merge_event_config(flat_config, events)

    # 5. Create the engine
    return create_okmc_engine_v16(flat_config)
