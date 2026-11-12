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
Configuration helper library

Provides built-in rate functions, execute functions, event filters and IC helpers.
ConfigCompiler and compile_config have been removed; event classes
(event_class.py + event_compiler.py) are the only event configuration path.
"""

import numpy as np
from typing import Dict, List, Callable, Any, Tuple


# ===== Built-in rate functions =====

def arrhenius_rate(objs, params, config):
    """
    Arrhenius rate: nu0 * exp(-Em / kBT)

    Args:
        objs: object list (available for richer models; unused here)
        params: parameter dict {'migration_energy': float, 'attempt_frequency': float}
        config: configuration dict

    Returns:
        rate (Hz)
    """
    migration_energy = params.get('migration_energy', 1.0)
    attempt_frequency = params.get('attempt_frequency', 1e13)

    T = config.get('temperature', 300.0)
    # Handle temperature scheduler (callable)
    if callable(T):
        current_time = config.get('_current_time', 0.0)
        T = T(current_time)
    kB = 8.617e-5  # eV/K
    return attempt_frequency * np.exp(-migration_energy / (kB * T))


def distance_rate(objs, params, config):
    """
    Distance-dependent rate: k / (distance^2 + epsilon)

    Args:
        objs: object list (gives access to object properties)
        params: parameter dict {'distance': float, 'k_prefactor': float}
        config: configuration dict

    Returns:
        rate (Hz)
    """
    distance = params.get('distance', 1.0)
    k_prefactor = params.get('k_prefactor', 1e12)
    return k_prefactor / (distance**2 + 0.01)


# ===== Built-in execute functions =====

def deterministic_direction_rands(event, config):
    """Single entry point for migration direction sampling -> (theta_rand, phi_rand)

    Prefers the deterministic generator injected by the engine, config['rng']
    (resampled on every execution). Without an engine context the direction is
    derived from the event ID: first the 128-bit ID (id_high/id_low), looked up
    by dtype field for structured arrays and by key for dicts - missing either
    form would let an event carrying id=0 degrade to a fixed direction; only
    then the compatibility id field (0 if absent).
    """
    rng = config.get('rng')
    if rng is not None:
        return rng.random(), rng.random()

    names = getattr(getattr(event, 'dtype', None), 'names', None)
    if names is None and isinstance(event, dict):
        names = event.keys()
    names = names or ()
    if 'id_high' in names and 'id_low' in names:
        event_id = (int(event['id_high'])
                    ^ (int(event['id_low']) * 0x9e3779b97f4a7c15)) \
            & 0xFFFFFFFFFFFFFFFF
    elif 'id' in names:
        event_id = int(event['id'])
    else:
        event_id = 0
    user_seed = config.get('_user_seed', 0)

    MASK_64 = 0xFFFFFFFFFFFFFFFF
    combined_theta = ((user_seed << 32) & MASK_64) | (event_id & 0xFFFFFFFF)
    theta_seed = combined_theta ^ ((user_seed + event_id) * 0x9e3779b97f4a7c15)
    theta_seed = (theta_seed ^ (theta_seed >> 30)) * 0xbf58476d1ce4e5b9
    theta_seed = theta_seed & MASK_64
    theta_seed = (theta_seed ^ (theta_seed >> 27)) * 0x94d049bb133111eb
    theta_seed = theta_seed & MASK_64
    theta_seed = theta_seed ^ (theta_seed >> 31)
    theta_rand = (theta_seed & 0xFFFFFFFF) / 4294967296.0

    combined_phi = ((event_id << 32) & MASK_64) | (user_seed & 0xFFFFFFFF)
    phi_seed = combined_phi ^ ((event_id + user_seed) * 0x9e3779b97f4a7c15)
    phi_seed = (phi_seed ^ (phi_seed >> 30)) * 0xbf58476d1ce4e5b9
    phi_seed = phi_seed & MASK_64
    phi_seed = (phi_seed ^ (phi_seed >> 27)) * 0x94d049bb133111eb
    phi_seed = phi_seed & MASK_64
    phi_seed = phi_seed ^ (phi_seed >> 31)
    phi_rand = (phi_seed & 0xFFFFFFFF) / 4294967296.0

    return theta_rand, phi_rand


def random_migration(event, objs, config):
    """Random migration (deterministic random number version)"""
    if len(objs) == 0:
        return []

    obj = objs[0]
    lc = config.get('lattice_constant', 1.0)

    theta_rand, phi_rand = deterministic_direction_rands(event, config)

    theta = 2 * np.pi * theta_rand
    phi = np.arccos(2 * phi_rand - 1)
    dx = lc * np.sin(phi) * np.cos(theta)
    dy = lc * np.sin(phi) * np.sin(theta)
    dz = lc * np.cos(phi)

    box_size = config['domain_decomposition']['box_size']
    new_x = (obj['x'] + dx) % box_size[0]
    new_y = (obj['y'] + dy) % box_size[1]
    new_z = (obj['z'] + dz) % box_size[2]

    return [{
        'id': int(obj['id']),
        'type': str(obj['type']),
        'x': float(new_x),
        'y': float(new_y),
        'z': float(new_z),
        'deleted': False,
        'new': False,
    }]


def capture_and_annihilate(event, objs, config):
    """Capture and annihilate (e.g. I+V -> lattice)"""
    if len(objs) < 2:
        return []

    return [
        {'id': int(objs[0]['id']), 'deleted': True},
        {'id': int(objs[1]['id']), 'deleted': True},
    ]


def capture_and_create(event, objs, config, product_type):
    """Capture and create a new object (e.g. H+V -> HV)"""
    if len(objs) < 2:
        return []

    obj1, obj2 = objs[0], objs[1]

    return [
        {'id': int(obj1['id']), 'deleted': True},
        {'id': int(obj2['id']), 'deleted': True},
        {
            # New objects get a placeholder ID; the engine assigns the real one
            # as [rank:16][step:32][counter:16] so helpers cannot clash.
            'id': -1,
            'type': product_type,
            'x': (obj1['x'] + obj2['x']) / 2,
            'y': (obj1['y'] + obj2['y']) / 2,
            'z': (obj1['z'] + obj2['z']) / 2,
            'new': True,
        }
    ]


# ===== Built-in event filters =====

def filter_rank0_only(config, rank, size):
    """Zero-object event filter: build on rank 0 only"""
    return rank == 0


def filter_per_rank(config, rank, size):
    """Zero-object event filter: build on every rank"""
    return True


def filter_always(obj, config, rank):
    """Single-object event filter: always build"""
    return True


def filter_pair_always(obj1, obj2, distance, config, rank):
    """Pair event filter: always build"""
    return True


# ===== IC helpers =====

def create_initial_objects_func(initial_conditions, objects_def, grid_size,
                                lattice_constant, seed=42):
    """
    Create the initial object generator (extracted from the former ConfigCompiler)

    Args:
        initial_conditions: {'H': {'count': 30, 'distribution': 'random'}, ...}
        objects_def: {'H': {'properties': {...}}, ...}
        grid_size: (nx, ny, nz)
        lattice_constant: lattice constant
        seed: random seed (default 42; overridden at runtime by config['_user_seed'])

    Returns:
        initial_objects_func(rank, size, config) -> List[dict]
    """
    ic = initial_conditions
    lc = lattice_constant

    def initial_objects_func(rank, size, config):
        """
        New initialization logic:
        1. every rank generates all objects from the same random seed
        2. decide whether each object lies in this rank's centre or boundary region
        3. keep only the centre and boundary objects
        4. use the new ID format: (rank << 48) | (step << 16) | counter
           - top 16 bits: birth rank
           - middle 32 bits: birth step (0 for initial objects)
           - low 16 bits: local counter
        """
        saved_state = np.random.get_state()
        ic_seed = config.get('_user_seed', seed)
        # np.random.seed accepts only [0, 2^32): mask large/negative seeds to 32 bits
        np.random.seed(ic_seed & 0xFFFFFFFF)

        # Fetch the domain decomposition
        domain_decomp = config.get('_domain_decomp', None)
        if domain_decomp is None:
            # Without a domain decomposition return an empty list (serial mode)
            if rank == 0:
                print(f"[DEBUG] Rank {rank}: _domain_decomp is None, returning empty list")
            np.random.set_state(saved_state)
            return []

        # Bounds of this rank's subdomain
        bounds = domain_decomp.subdomain_bounds
        box_size = domain_decomp.box_size
        periodic = domain_decomp.periodic
        r_cut = config.get('r_cut', 15.0)  # default cutoff radius

        # Generate all objects (every rank uses the same random seed)
        all_objects = []
        global_counter = 0

        for obj_type, ic_def in ic.items():
            count = ic_def.get('count', 0)
            distribution = ic_def.get('distribution', 'random')

            for i in range(count):
                if distribution == 'random':
                    x = np.random.random() * grid_size[0] * lc
                    y = np.random.random() * grid_size[1] * lc
                    z = np.random.random() * grid_size[2] * lc
                else:
                    nx, ny, nz = grid_size
                    idx = i
                    iz = idx // (nx * ny)
                    iy = (idx % (nx * ny)) // nx
                    ix = idx % nx
                    x = (ix + 0.5) * lc
                    y = (iy + 0.5) * lc
                    z = (iz + 0.5) * lc

                # Decide whether the object is in this rank's centre or boundary region
                # Distance to each boundary
                dx_min = abs(x - bounds['x_min'])
                dx_max = abs(x - bounds['x_max'])
                dy_min = abs(y - bounds['y_min'])
                dy_max = abs(y - bounds['y_max'])
                dz_min = abs(z - bounds['z_min'])
                dz_max = abs(z - bounds['z_max'])

                # Periodic boundary handling
                if periodic[0]:
                    dx_wrap = box_size[0] - max(dx_min, dx_max)
                    dx_min = min(dx_min, dx_wrap)
                    dx_max = min(dx_max, dx_wrap)
                if periodic[1]:
                    dy_wrap = box_size[1] - max(dy_min, dy_max)
                    dy_min = min(dy_min, dy_wrap)
                    dy_max = min(dy_max, dy_wrap)
                if periodic[2]:
                    dz_wrap = box_size[2] - max(dz_min, dz_max)
                    dz_min = min(dz_min, dz_wrap)
                    dz_max = min(dz_max, dz_wrap)

                # Minimum distance to a boundary
                dist_to_boundary = min(dx_min, dx_max, dy_min, dy_max, dz_min, dz_max)

                # Object radius
                obj_radius = 0.0
                if obj_type in objects_def:
                    obj_def = objects_def[obj_type]
                    if 'properties' in obj_def:
                        props = obj_def['properties']
                        for k in ('size', 'radius', 'r'):
                            if k in props:
                                obj_radius = float(props[k]) / 2.0
                                break

                # Centre or boundary region?
                # If the distance < (r_cut + obj_radius) it is a boundary object,
                # otherwise test whether it lies inside the subdomain
                is_in_subdomain = (
                    bounds['x_min'] <= x <= bounds['x_max'] and
                    bounds['y_min'] <= y <= bounds['y_max'] and
                    bounds['z_min'] <= z <= bounds['z_max']
                )

                if not is_in_subdomain:
                    continue  # outside this rank's subdomain, skip

                # Inside the subdomain: keep the object
                global_counter += 1

                obj = {
                    'id': 0,  # temporary ID, reassigned below
                    'type': obj_type,
                    'x': x,
                    'y': y,
                    'z': z,
                }

                if obj_type in objects_def:
                    obj_def = objects_def[obj_type]
                    if 'properties' in obj_def:
                        # Build float_props following the TypeCompiler mapping
                        props = obj_def['properties']
                        size_key = None
                        for k in props:
                            if k.lower() in ('size', 'radius', 'r'):
                                size_key = k
                                break
                        n_non_size = sum(1 for k in props if k != size_key)
                        total = 1 + n_non_size
                        float_props = [0.0] * total
                        if size_key is not None:
                            float_props[0] = float(props[size_key])
                        f_idx = 1
                        for k, v in props.items():
                            if k == size_key:
                                continue
                            float_props[f_idx] = float(v) if not isinstance(v, str) else 0.0
                            f_idx += 1
                        if float_props:
                            obj['float_props'] = float_props
                    for prop_name, prop_value in obj_def.items():
                        if prop_name != 'properties':
                            obj[prop_name] = prop_value

                all_objects.append(obj)

        # Reassign IDs in the new format: (rank << 48) | (step << 16) | counter
        local_counter = 1
        for obj in all_objects:
            obj['id'] = (rank << 48) | (0 << 16) | local_counter
            local_counter += 1

        np.random.set_state(saved_state)
        return all_objects

    return initial_objects_func


if __name__ == "__main__":
    # Test the helper functions
    print("=== config_system.py self-test ===")

    # Test arrhenius_rate
    rate = arrhenius_rate([], {'migration_energy': 0.3, 'attempt_frequency': 1e13},
                          {'temperature': 500.0})
    print(f"arrhenius_rate(Em=0.3, T=500): {rate:.2e}")
    assert rate > 0

    # Test create_initial_objects_func
    ic = {'H': {'count': 5, 'distribution': 'random'},
          'V': {'count': 3, 'distribution': 'random'}}
    objects_def = {'H': {'size': 1}, 'V': {'size': 1}}
    ic_func = create_initial_objects_func(ic, objects_def, (30, 30, 30), 1.0)
    objs = ic_func(0, 1, {'_user_seed': 42})
    print(f"create_initial_objects_func: {len(objs)} objects")
    assert len(objs) == 8
    types = [o['type'] for o in objs]
    assert types.count('H') == 5
    assert types.count('V') == 3

    print("All tests passed!")
