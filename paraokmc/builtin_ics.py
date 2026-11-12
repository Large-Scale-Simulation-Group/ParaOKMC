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
Built-in initial condition library

Common generators for the initial object distribution
"""

import numpy as np
from typing import List, Dict, Callable, Optional


# ===== Uniform grid distribution =====

def create_uniform_grid_ic(object_types: List[str],
                           densities: Dict[str, float],
                           box_size: List[float],
                           lattice_constant: float = 1.0,
                           seed: Optional[int] = None) -> Callable:
    """
    Create an initial condition function with a uniform grid distribution

    Args:
        object_types: list of object types (e.g. ['H', 'V', 'I'])
        densities: density per type (objects per volume), e.g. {'H': 0.01, 'V': 0.01}
        box_size: box dimensions [Lx, Ly, Lz]
        lattice_constant: lattice constant (sets the grid spacing)
        seed: random seed (optional)

    Returns:
        initial_objects_func(rank, size, config) -> List[dict]
    """

    def initial_objects_func(rank, size, config):
        """Generate the initial objects"""
        rng = np.random.default_rng(seed)

        # Total volume
        volume = box_size[0] * box_size[1] * box_size[2]

        objects = []
        base_id = (rank << 48) + 1  # [fix] orthogonal to the runtime/config_system ID space, never 0

        for obj_type in object_types:
            density = densities.get(obj_type, 0.0)
            if density <= 0:
                continue

            # Number of objects of this type
            n_objects = int(density * volume)

            # Draw uniformly inside the whole box
            for i in range(n_objects):
                # Random position
                x = rng.random() * box_size[0]
                y = rng.random() * box_size[1]
                z = rng.random() * box_size[2]

                obj = {
                    'id': base_id + len(objects),
                    'type': obj_type,
                    'x': x,
                    'y': y,
                    'z': z,
                }
                objects.append(obj)

        return objects

    return initial_objects_func


# ===== Random distribution =====

def create_random_ic(object_types: List[str],
                     counts: Dict[str, int],
                     box_size: List[float],
                     min_separation: float = 0.5,
                     seed: Optional[int] = None) -> Callable:
    """
    Create an initial condition function with a random distribution (minimum separation)

    Args:
        object_types: list of object types
        counts: number of objects per type, e.g. {'H': 100, 'V': 50}
        box_size: box dimensions [Lx, Ly, Lz]
        min_separation: minimum separation (avoids overlap)
        seed: random seed (optional)

    Returns:
        initial_objects_func(rank, size, config) -> List[dict]
    """

    def initial_objects_func(rank, size, config):
        """Generate the initial objects"""
        rng = np.random.default_rng(seed)

        objects = []
        positions = []  # positions accepted so far
        base_id = (rank << 48) + 1  # [fix] orthogonal to the runtime/config_system ID space, never 0

        for obj_type in object_types:
            count = counts.get(obj_type, 0)
            if count <= 0:
                continue

            for i in range(count):
                # Try to place without overlap (at most 1000 attempts)
                for attempt in range(1000):
                    x = rng.random() * box_size[0]
                    y = rng.random() * box_size[1]
                    z = rng.random() * box_size[2]

                    pos = np.array([x, y, z])

                    # Check the distance to the objects placed already
                    too_close = False
                    for existing_pos in positions:
                        dist = np.linalg.norm(pos - existing_pos)
                        if dist < min_separation:
                            too_close = True
                            break

                    if not too_close:
                        positions.append(pos)
                        obj = {
                            'id': base_id + len(objects),
                            'type': obj_type,
                            'x': x,
                            'y': y,
                            'z': z,
                        }
                        objects.append(obj)
                        break

        return objects

    return initial_objects_func


# ===== Layered distribution =====

def create_layered_ic(layers: List[Dict],
                      box_size: List[float],
                      axis: str = 'z',
                      seed: Optional[int] = None) -> Callable:
    """
    Create an initial condition function with a layered distribution

    Args:
        layers: list of layer configurations, each holding:
            {
                'z_range': [z_min, z_max],  # extent of the layer
                'object_types': ['H', 'V'],  # object types in this layer
                'densities': {'H': 0.01, 'V': 0.005}  # densities
            }
        box_size: box dimensions [Lx, Ly, Lz]
        axis: layer normal ('x', 'y', 'z')
        seed: random seed (optional)

    Returns:
        initial_objects_func(rank, size, config) -> List[dict]
    """

    def initial_objects_func(rank, size, config):
        """Generate the initial objects"""
        rng = np.random.default_rng(seed)

        objects = []
        base_id = (rank << 48) + 1  # [fix] orthogonal to the runtime/config_system ID space, never 0

        axis_idx = {'x': 0, 'y': 1, 'z': 2}[axis]

        for layer in layers:
            coord_range = layer['z_range']  # in fact a generic coordinate range
            obj_types = layer['object_types']
            densities = layer['densities']

            # Layer volume
            layer_thickness = coord_range[1] - coord_range[0]
            if axis == 'x':
                layer_volume = layer_thickness * box_size[1] * box_size[2]
            elif axis == 'y':
                layer_volume = box_size[0] * layer_thickness * box_size[2]
            else:  # z
                layer_volume = box_size[0] * box_size[1] * layer_thickness

            for obj_type in obj_types:
                density = densities.get(obj_type, 0.0)
                if density <= 0:
                    continue

                n_objects = int(density * layer_volume)

                for i in range(n_objects):
                    # Draw inside the layer
                    pos = [
                        rng.random() * box_size[0],
                        rng.random() * box_size[1],
                        rng.random() * box_size[2]
                    ]
                    # Clamp to the layer extent
                    pos[axis_idx] = coord_range[0] + rng.random() * layer_thickness

                    obj = {
                        'id': base_id + len(objects),
                        'type': obj_type,
                        'x': pos[0],
                        'y': pos[1],
                        'z': pos[2],
                    }
                    objects.append(obj)

        return objects

    return initial_objects_func


# ===== Frenkel pair distribution =====

def create_frenkel_pairs_ic(n_pairs: int,
                             box_size: List[float],
                             separation: float = 1.0,
                             interstitial_type: str = 'I',
                             vacancy_type: str = 'V',
                             seed: Optional[int] = None) -> Callable:
    """
    Create an initial condition function with a distribution of Frenkel pairs

    A Frenkel pair: a neighbouring interstitial and vacancy

    Args:
        n_pairs: number of Frenkel pairs
        box_size: box dimensions [Lx, Ly, Lz]
        separation: interstitial-vacancy separation
        interstitial_type: interstitial type
        vacancy_type: vacancy type
        seed: random seed (optional)

    Returns:
        initial_objects_func(rank, size, config) -> List[dict]
    """

    def initial_objects_func(rank, size, config):
        """Generate the initial objects"""
        rng = np.random.default_rng(seed)

        objects = []
        base_id = (rank << 48) + 1  # [fix] orthogonal to the runtime/config_system ID space, never 0

        for i in range(n_pairs):
            # Random centre position
            center = np.array([
                rng.random() * box_size[0],
                rng.random() * box_size[1],
                rng.random() * box_size[2]
            ])

            # Random orientation
            theta = 2 * np.pi * rng.random()
            phi = np.arccos(2 * rng.random() - 1)
            direction = np.array([
                np.sin(phi) * np.cos(theta),
                np.sin(phi) * np.sin(theta),
                np.cos(phi)
            ])

            # Interstitial position
            pos_I = center + 0.5 * separation * direction
            pos_I = pos_I % box_size  # periodic boundary

            # Vacancy position
            pos_V = center - 0.5 * separation * direction
            pos_V = pos_V % box_size  # periodic boundary

            # Create the interstitial
            obj_I = {
                'id': base_id + 2 * i,
                'type': interstitial_type,
                'x': float(pos_I[0]),
                'y': float(pos_I[1]),
                'z': float(pos_I[2]),
            }
            objects.append(obj_I)

            # Create the vacancy
            obj_V = {
                'id': base_id + 2 * i + 1,
                'type': vacancy_type,
                'x': float(pos_V[0]),
                'y': float(pos_V[1]),
                'z': float(pos_V[2]),
            }
            objects.append(obj_V)

        return objects

    return initial_objects_func


# ===== Irradiation cascade initial condition =====

def create_cascade_ic(cascade_center: List[float],
                      cascade_radius: float,
                      n_interstitials: int,
                      n_vacancies: int,
                      box_size: List[float],
                      interstitial_type: str = 'I',
                      vacancy_type: str = 'V',
                      seed: Optional[int] = None) -> Callable:
    """
    Create an initial condition function for an irradiation cascade

    Args:
        cascade_center: cascade centre [x, y, z]
        cascade_radius: cascade radius
        n_interstitials: number of interstitials
        n_vacancies: number of vacancies
        box_size: box dimensions [Lx, Ly, Lz]
        interstitial_type: interstitial type
        vacancy_type: vacancy type
        seed: random seed (optional)

    Returns:
        initial_objects_func(rank, size, config) -> List[dict]
    """

    def initial_objects_func(rank, size, config):
        """Generate the initial objects"""
        rng = np.random.default_rng(seed)

        objects = []
        base_id = (rank << 48) + 1  # [fix] orthogonal to the runtime/config_system ID space, never 0
        center = np.array(cascade_center)

        # Generate the interstitials (concentrated near the cascade centre)
        for i in range(n_interstitials):
            # Sample inside the sphere (biased towards the centre)
            r = cascade_radius * rng.random()**(1/3)
            theta = 2 * np.pi * rng.random()
            phi = np.arccos(2 * rng.random() - 1)

            pos = center + r * np.array([
                np.sin(phi) * np.cos(theta),
                np.sin(phi) * np.sin(theta),
                np.cos(phi)
            ])

            pos = pos % box_size  # periodic boundary

            obj = {
                'id': base_id + i,
                'type': interstitial_type,
                'x': float(pos[0]),
                'y': float(pos[1]),
                'z': float(pos[2]),
            }
            objects.append(obj)

        # Generate the vacancies (also inside the cascade region)
        for i in range(n_vacancies):
            r = cascade_radius * rng.random()**(1/3)
            theta = 2 * np.pi * rng.random()
            phi = np.arccos(2 * rng.random() - 1)

            pos = center + r * np.array([
                np.sin(phi) * np.cos(theta),
                np.sin(phi) * np.sin(theta),
                np.cos(phi)
            ])

            pos = pos % box_size  # periodic boundary

            obj = {
                'id': base_id + n_interstitials + i,
                'type': vacancy_type,
                'x': float(pos[0]),
                'y': float(pos[1]),
                'z': float(pos[2]),
            }
            objects.append(obj)

        return objects

    return initial_objects_func


if __name__ == "__main__":
    print("===== Builtin ICs Test =====")

    # Test the uniform distribution
    ic_uniform = create_uniform_grid_ic(
        object_types=['H', 'V'],
        densities={'H': 0.01, 'V': 0.005},
        box_size=[10.0, 10.0, 10.0],
        seed=42
    )

    objects = ic_uniform(rank=0, size=1, config={})
    print(f"Uniform grid IC: {len(objects)} objects")
    print(f"  First object: {objects[0]}")

    # Test the random distribution
    ic_random = create_random_ic(
        object_types=['H', 'V'],
        counts={'H': 10, 'V': 5},
        box_size=[10.0, 10.0, 10.0],
        min_separation=1.0,
        seed=42
    )

    objects = ic_random(rank=0, size=1, config={})
    print(f"\nRandom IC: {len(objects)} objects")

    # Test the Frenkel pairs
    ic_frenkel = create_frenkel_pairs_ic(
        n_pairs=5,
        box_size=[10.0, 10.0, 10.0],
        separation=1.0,
        seed=42
    )

    objects = ic_frenkel(rank=0, size=1, config={})
    print(f"\nFrenkel pairs IC: {len(objects)} objects (should be 10)")

    print("\nBuiltin ICs test completed!")
