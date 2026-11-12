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
Built-in migration library (event builders and execute functions)

Common event builders, filters and execute functions

- The execute functions can read physics parameters from the UnifiedPhysics module
- Adds the get_physics_box_size() helper
"""

import numpy as np
from typing import List, Callable, Tuple, Optional
from .rma_event_pool import create_event
from .config_system import deterministic_direction_rands


# ===== Physics parameter helpers =====

def get_physics_box_size(config: dict) -> List[float]:
    """
    Get box_size from the UnifiedPhysics module or from config

    Priority:
    1. config['physics'].box_size
    2. config['domain_decomposition']['box_size']
    3. config['box_size']
    4. the default [100.0, 100.0, 100.0]

    Args:
        config: configuration dict

    Returns:
        box_size [Lx, Ly, Lz]
    """
    physics = config.get('physics')

    # Prefer the physics module
    if physics is not None:
        try:
            return list(physics.box_size)
        except AttributeError:
            pass

    # Then domain_decomposition
    dd_config = config.get('domain_decomposition', {})
    if 'box_size' in dd_config:
        return list(dd_config['box_size'])

    # Then the top level of config
    if 'box_size' in config:
        return list(config['box_size'])

    # Default
    return [100.0, 100.0, 100.0]


# ===== Single-object event builders =====

def create_migration_builder(migration_distance: float = 1.0,
                             migration_mode: str = 'isotropic',
                             allowed_types: Optional[List[str]] = None) -> Callable:
    """
    Create a migration event builder

    Args:
        migration_distance: migration distance
        migration_mode: migration mode ('isotropic', 'lattice')
        allowed_types: object types allowed to migrate (None means all)

    Returns:
        builder_func(objects, config, rank) -> List[np.ndarray]
    """

    def builder_func(objects: np.ndarray, config: dict, rank: int) -> List[np.ndarray]:
        """
        Build the migration events

        Args:
            objects: object array (local or target objects)
            config: configuration dict
            rank: current rank

        Returns:
            the event list
        """
        events = []
        event_id_base = rank * 10000000
        int_to_type = config.get('_int_to_type', {})

        for i, obj in enumerate(objects):
            obj_type = int_to_type.get(int(obj['type_id']), '')

            # Type filter
            if allowed_types is not None and obj_type not in allowed_types:
                continue

            # Pass config so the dtype is resolved automatically
            # create_event reads config['_event_dtype'] itself
            event = create_event(
                event_id=event_id_base + i,
                event_type='migration',
                obj_ids=[int(obj['id'])],
                params=[migration_distance, 0.0],  # [distance, direction_flag]
                owner_rank=rank,
                config=config  # pass config, resolving the dtype automatically
            )
            events.append(event)

        return events

    return builder_func


def create_rotation_builder(allowed_types: Optional[List[str]] = None) -> Callable:
    """
    Create a rotation event builder (for extended defects such as dislocation loops)

    Args:
        allowed_types: object types allowed to rotate

    Returns:
        builder_func(objects, config, rank) -> List[np.ndarray]
    """

    def builder_func(objects: np.ndarray, config: dict, rank: int) -> List[np.ndarray]:
        """Build the rotation events"""
        events = []
        event_id_base = rank * 10000000 + 1000000  # a different ID range
        int_to_type = config.get('_int_to_type', {})

        for i, obj in enumerate(objects):
            obj_type = int_to_type.get(int(obj['type_id']), '')

            if allowed_types is not None and obj_type not in allowed_types:
                continue

            # Pass config so the dtype is resolved automatically
            event = create_event(
                event_id=event_id_base + i,
                event_type='rotation',
                obj_ids=[int(obj['id'])],
                params=[0.0, 0.0],
                owner_rank=rank,
                config=config  # pass config, resolving the dtype automatically
            )
            events.append(event)

        return events

    return builder_func


# ===== Pair event filters =====

def create_capture_filter(type_pairs: List[Tuple[str, str]]) -> Callable:
    """
    Create a capture type filter

    Args:
        type_pairs: type pairs allowed to capture, e.g. [('H', 'V'), ('I', 'V')]

    Returns:
        filter_func(obj1, obj2) -> bool
    """

    # Normalize the type pairs (both directions)
    allowed_pairs = set()
    for t1, t2 in type_pairs:
        allowed_pairs.add((t1, t2))
        allowed_pairs.add((t2, t1))

    def filter_func(obj1: np.ndarray, obj2: np.ndarray) -> bool:
        """
        Decide whether two objects may undergo capture

        Args:
            obj1, obj2: object records (NumPy structured arrays)

        Returns:
            True/False
        """
        type1 = str(obj1['type'])
        type2 = str(obj2['type'])
        return (type1, type2) in allowed_pairs

    return filter_func


def create_clustering_filter(same_type: bool = True,
                             allowed_types: Optional[List[str]] = None) -> Callable:
    """
    Create a clustering type filter (aggregation of like objects)

    Args:
        same_type: whether both objects must share a type
        allowed_types: types allowed to aggregate

    Returns:
        filter_func(obj1, obj2) -> bool
    """

    def filter_func(obj1: np.ndarray, obj2: np.ndarray) -> bool:
        """Decide whether two objects may aggregate"""
        type1 = str(obj1['type'])
        type2 = str(obj2['type'])

        # Type filter
        if allowed_types is not None:
            if type1 not in allowed_types or type2 not in allowed_types:
                return False

        # Same-type check
        if same_type:
            return type1 == type2
        else:
            return True

    return filter_func


def create_recombination_filter(type_pairs: List[Tuple[str, str]]) -> Callable:
    """
    Create a recombination type filter (e.g. I+V -> lattice)

    Args:
        type_pairs: type pairs allowed to recombine

    Returns:
        filter_func(obj1, obj2) -> bool
    """
    return create_capture_filter(type_pairs)  # the logic matches capture


# ===== Pair event parameter extractors =====

def create_distance_extractor(box_size: Optional[List[float]] = None,
                              periodic: bool = False) -> Callable:
    """
    Create a distance parameter extractor

    Args:
        box_size: box dimensions (for the periodic boundary)
        periodic: whether to apply periodic boundary conditions

    Returns:
        extractor_func(obj1, obj2) -> List[float]
    """

    def extractor_func(obj1: np.ndarray, obj2: np.ndarray) -> List[float]:
        """
        Extract the distance between two objects

        Args:
            obj1, obj2: object records

        Returns:
            [distance, 0.0]
        """
        pos1 = np.array([obj1['x'], obj1['y'], obj1['z']])
        pos2 = np.array([obj2['x'], obj2['y'], obj2['z']])

        if periodic and box_size is not None:
            # Periodic boundary: the minimum image distance
            dr = pos1 - pos2
            box = np.array(box_size)
            dr = dr - box * np.round(dr / box)
            distance = np.linalg.norm(dr)
        else:
            # Direct distance
            distance = np.linalg.norm(pos1 - pos2)

        return [distance, 0.0]

    return extractor_func


def create_binding_energy_extractor(binding_energies: dict) -> Callable:
    """
    Create a binding energy parameter extractor

    Args:
        binding_energies: dict of binding energies, e.g.
            {
                ('H', 'V'): 0.5,  # eV
                ('I', 'V'): 1.0,
            }

    Returns:
        extractor_func(obj1, obj2) -> List[float]
    """

    def extractor_func(obj1: np.ndarray, obj2: np.ndarray) -> List[float]:
        """
        Extract the binding energy and the distance

        Returns:
            [distance, binding_energy]
        """
        pos1 = np.array([obj1['x'], obj1['y'], obj1['z']])
        pos2 = np.array([obj2['x'], obj2['y'], obj2['z']])
        distance = np.linalg.norm(pos1 - pos2)

        type1 = str(obj1['type'])
        type2 = str(obj2['type'])

        # Look the binding energy up (both directions)
        Eb = binding_energies.get((type1, type2), 0.0)
        if Eb == 0.0:
            Eb = binding_energies.get((type2, type1), 0.0)

        return [distance, Eb]

    return extractor_func


# ===== Predefined event configurations =====

def get_hydrogen_tungsten_events(box_size: List[float]) -> dict:
    """
    Get the standard event configuration for H in W

    Args:
        box_size: box dimensions

    Returns:
        {
            'single_obj_builders': {...},
            'pair_type_filters': {...},
            'pair_param_extractors': {...},
        }
    """
    return {
        'single_obj_builders': {
            'migration': create_migration_builder(
                migration_distance=1.0,
                allowed_types=['H', 'V', 'I']
            ),
        },
        'pair_type_filters': {
            'capture': create_capture_filter([('H', 'V')]),
            'recombination': create_recombination_filter([('I', 'V')]),
        },
        'pair_param_extractors': {
            'capture': create_distance_extractor(box_size, periodic=True),
            'recombination': create_distance_extractor(box_size, periodic=True),
        },
    }


def get_iron_radiation_events(box_size: List[float]) -> dict:
    """
    Get the standard event configuration for irradiation damage in Fe

    Args:
        box_size: box dimensions

    Returns:
        the event configuration dict
    """
    return {
        'single_obj_builders': {
            'migration': create_migration_builder(
                migration_distance=0.287,  # half the Fe lattice constant
                allowed_types=['I', 'V', 'I2', 'V2']
            ),
        },
        'pair_type_filters': {
            'IV_recombination': create_recombination_filter([('I', 'V')]),
            'I_clustering': create_clustering_filter(same_type=True, allowed_types=['I']),
            'V_clustering': create_clustering_filter(same_type=True, allowed_types=['V']),
        },
        'pair_param_extractors': {
            'IV_recombination': create_distance_extractor(box_size, periodic=True),
            'I_clustering': create_binding_energy_extractor({('I', 'I'): 0.8}),
            'V_clustering': create_binding_energy_extractor({('V', 'V'): 0.3}),
        },
    }


def get_frenkel_pair_events(box_size: List[float]) -> dict:
    """
    Get the event configuration of a simple Frenkel pair evolution

    Args:
        box_size: box dimensions

    Returns:
        the event configuration dict
    """
    return {
        'single_obj_builders': {
            'migration': create_migration_builder(
                migration_distance=1.0,
                allowed_types=['I', 'V']
            ),
        },
        'pair_type_filters': {
            'recombination': create_recombination_filter([('I', 'V')]),
        },
        'pair_param_extractors': {
            'recombination': create_distance_extractor(box_size, periodic=True),
        },
    }


# ===== Event execution helpers =====

def default_execute_migration(event: np.ndarray,
                              objs: List[np.ndarray],
                              config: dict,
                              jump_distance: float = 1.0) -> List[dict]:
    """
    Default migration execute function (deterministic random number version)

    box_size can be read from the UnifiedPhysics module

    Args:
        event: the event record
        objs: the participating objects
        config: configuration dict
        jump_distance: jump length

    Returns:
        the list of changed objects
    """
    if len(objs) == 0:
        return []

    obj = objs[0]

    # [fix] Direction sampling goes through deterministic_direction_rands: config['rng']
    # is preferred and resampled per call; without an engine context the direction comes
    # from the 128-bit event ID. The old code read the compatibility id field, which is
    theta_rand, phi_rand = deterministic_direction_rands(event, config)

    theta = 2 * np.pi * theta_rand
    phi = np.arccos(2 * phi_rand - 1)
    dx = jump_distance * np.sin(phi) * np.cos(theta)
    dy = jump_distance * np.sin(phi) * np.sin(theta)
    dz = jump_distance * np.cos(phi)

    # Get box_size from the physics module or from config
    box_size = get_physics_box_size(config)
    new_x = (obj['x'] + dx) % box_size[0]
    new_y = (obj['y'] + dy) % box_size[1]
    new_z = (obj['z'] + dz) % box_size[2]

    changed_obj = {
        'id': int(obj['id']),
        'type': config.get('_int_to_type', {}).get(int(obj['type_id']), ''),
        'x': float(new_x),
        'y': float(new_y),
        'z': float(new_z),
        'deleted': False,
        'new': False,
    }

    return [changed_obj]


def default_execute_capture(event: np.ndarray,
                            objs: List[np.ndarray],
                            config: dict,
                            product_type: str = 'HV') -> List[dict]:
    """
    Default capture execute function (H + V -> HV)

    Args:
        event: the event record
        objs: the participating objects
        config: configuration dict
        product_type: type of the product

    Returns:
        the list of changed objects
    """
    if len(objs) < 2:
        return []

    obj1, obj2 = objs[0], objs[1]

    # Delete the original objects
    changed = [
        {'id': int(obj1['id']), 'deleted': True},
        {'id': int(obj2['id']), 'deleted': True},
    ]

    # Create the product (halfway between the two)
    new_x = (obj1['x'] + obj2['x']) / 2
    new_y = (obj1['y'] + obj2['y']) / 2
    new_z = (obj1['z'] + obj2['z']) / 2

    new_obj = {
        'id': int(obj1['id']) * 1000 + int(obj2['id']) % 1000,
        'type': product_type,
        'x': float(new_x),
        'y': float(new_y),
        'z': float(new_z),
        'deleted': False,
        'new': True,
    }
    changed.append(new_obj)

    return changed


def default_execute_recombination(event: np.ndarray,
                                  objs: List[np.ndarray],
                                  config: dict) -> List[dict]:
    """
    Default recombination execute function (I + V -> lattice, i.e. annihilation)

    Args:
        event: the event record
        objs: the participating objects
        config: configuration dict

    Returns:
        the list of changed objects
    """
    if len(objs) < 2:
        return []

    obj1, obj2 = objs[0], objs[1]

    # Delete both objects (they annihilate on recombination)
    changed = [
        {'id': int(obj1['id']), 'deleted': True},
        {'id': int(obj2['id']), 'deleted': True},
    ]

    return changed


if __name__ == "__main__":
    print("===== Builtin Migrations Test =====")

    # Create the migration event builder
    migration_builder = create_migration_builder(
        migration_distance=1.0,
        allowed_types=['H', 'V']
    )

    # Create the test objects
    from .rma_object_pool import OBJECT_DTYPE
    objects = np.array([
        (1, 'H', 0.0, 0.0, 0.0, 0, True, [0]*20, [0]*10, ['']*5),
        (2, 'V', 1.0, 1.0, 1.0, 0, True, [0]*20, [0]*10, ['']*5),
    ], dtype=OBJECT_DTYPE)

    # Build the events
    events = migration_builder(objects, {}, rank=0)
    print(f"Built {len(events)} migration events")

    # Test the capture filter
    capture_filter = create_capture_filter([('H', 'V')])
    can_capture = capture_filter(objects[0], objects[1])
    print(f"H-V can capture: {can_capture}")

    # Test the distance extractor
    distance_extractor = create_distance_extractor(box_size=[10.0, 10.0, 10.0], periodic=True)
    params = distance_extractor(objects[0], objects[1])
    print(f"Distance params: {params}")

    # Test the predefined configurations
    events_config = get_hydrogen_tungsten_events(box_size=[100.0, 100.0, 100.0])
    print(f"\nHydrogen-Tungsten events config:")
    print(f"  Single builders: {list(events_config['single_obj_builders'].keys())}")
    print(f"  Pair filters: {list(events_config['pair_type_filters'].keys())}")

    print("\nBuiltin Migrations test completed!")
