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
Configuration analyzer

Core features:
1. Analyse the user configuration and count the slots object properties and event parameters need
2. Generate OBJECT_DTYPE and EVENT_DTYPE dynamically
3. Implement the 'adaptive memory layout' - allocate only what is needed

Design idea:
- static hardcoding -> dynamic adaptation
- scan the config during initialization and compute the exact slot counts
- clear the way for the v1.7.0 JIT parallelization
"""

import numpy as np
from typing import Dict, Any, List, Tuple


class TypeCompiler:
    """
    Type compiler: analyses the configuration and generates the dynamic dtype

    Features:
    1. Scan the object property demand (number of float/int/str slots)
    2. Scan the event parameter demand (number of params, number of objects)
    3. Generate OBJECT_DTYPE and EVENT_DTYPE dynamically
    4. Generate the property-name-to-index mapping
    """

    def __init__(self, config: dict, verbose: bool = False):
        """
        Initialize the type compiler

        Args:
            config: user configuration dict
            verbose: whether to print details
        """
        self.config = config
        self.verbose = verbose

        # Object property demand
        self.obj_specs = {
            'float': 0,      # number of float properties
            'int': 0,        # number of integer properties
            'str': 0,        # number of string properties
            'str_len': 0,    # maximum string length
            'type_len': 16,  # maximum length of an object type name (default 16)
        }

        # Event parameter demand
        self.evt_specs = {
            'params': 0,     # number of parameter slots
            'objs': 0,       # maximum number of participating objects
            'type_len': 32,  # maximum length of an event type name (default 32)
        }

        # Property name to index mapping (everything lives in float_props)
        self.float_props_mapping: Dict[str, Dict[str, int]] = {}  # {obj_type: {prop_name: index}}

        # Type name <-> integer ID mapping (from the key order of config['objects'])
        self.type_to_int: Dict[str, int] = {}
        self.int_to_type: Dict[int, str] = {}

    def analyze(self):
        """
        Analyse the configuration and count the slot demand

        Note:
        - this is the heart of the adaptive memory layout
        - only the slots actually needed are counted, so nothing is wasted
        """
        # 1. Analyse the object property demand
        self._analyze_object_properties()

        # 2. Analyse the event parameter demand
        self._analyze_event_parameters()

        if self.verbose:
            print("=" * 60)
            print("[TypeCompiler] Configuration Analysis Results:")
            print("-" * 60)
            print("Object Specs:")
            print(f"  - Float properties: {self.obj_specs['float']}")
            print("Event Specs:")
            print(f"  - Parameters:       {self.evt_specs['params']}")
            print(f"  - Participating objects: {self.evt_specs['objs']}")
            print("=" * 60)

    def _analyze_object_properties(self):
        """
        Analyse the object property demand

        Strategy:
        1. walk all object type definitions
        2. count the property slots each type needs
        3. take the maximum as the global slot count
        4. scan the maximum type name length (to prevent truncation)
        5. build the property-name-to-index mapping
        """
        # Route 1: read config['objects'] (when the user supplied object definitions)
        if 'objects' in self.config:
            for idx, (obj_name, obj_def) in enumerate(self.config['objects'].items()):
                # Build type_to_int following the key order
                self.type_to_int[obj_name] = idx
                self.int_to_type[idx] = obj_name

                # Scan the type name length
                self.obj_specs['type_len'] = max(self.obj_specs['type_len'], len(obj_name))

                # Scan the properties and build the mapping
                properties = obj_def.get('properties', {})
                self._count_properties_with_mapping(obj_name, properties)

        # Route 2: infer from initial_objects_func (more general)
        # That function only runs later, so the user must declare the property spec in
        # the config, or the dtype has to be adjusted after the first call (harder)

        # The conservative v1.6.2 defaults (20/20/10) were removed
        # Without user-declared properties the counts stay 0 and get_object_dtype()
        # omits those fields, saving memory until they are genuinely needed
        if self.verbose and self.obj_specs['float'] == 0 and self.obj_specs['int'] == 0:
            print("[TypeCompiler] No object properties defined, using minimal dtype (core fields only)")

    def _count_properties_with_mapping(self, obj_type: str, properties: dict):
        """
        Count the properties and build the name-to-index mapping

        [fix] The size property is forced into float_props[0]
        Reason: v16_engine._update_interaction_distance() assumes float_props[0] is the radius
        Fix: however the user declares properties, size always ends up at index 0

        Every property lives in float_props (int/str have been removed)

        Args:
            obj_type: object type name
            properties: property dict {'radius': 1.0, 'charge': -1, 'label': 'H'}
        """
        # [fix] size/radius is forced into float_props[0]
        # Indexing starts at 1, reserving slot 0 for size
        f_idx = 1  # start at 1; 0 is reserved for size

        float_mapping = {}

        # [fix] Look for a size or radius property
        has_size_prop = False
        size_prop_name = None
        for key in properties.keys():
            if key.lower() in ('size', 'radius', 'r'):
                has_size_prop = True
                size_prop_name = key
                break

        # If the user declared size/radius, map it to index 0
        if has_size_prop:
            float_mapping[size_prop_name] = 0

        for key, value in properties.items():
            # Skip the size/radius property already handled
            if has_size_prop and key == size_prop_name:
                continue

            # Every property goes to float_props
            float_mapping[key] = f_idx
            f_idx += 1

        # Store the mapping
        self.float_props_mapping[obj_type] = float_mapping

        # Update the maxima
        # [fix] float is at least 1 (so the size slot always exists)
        self.obj_specs['float'] = max(self.obj_specs['float'], f_idx, 1)

    def _analyze_event_parameters(self):
        """
        Analyse the event parameter demand

        Strategy:
        1. walk all event type definitions
        2. count the parameters and participating objects
        3. take the maxima
        4. scan the maximum event type name length
        """
        # Route 1: read config['events']
        if 'events' in self.config:
            for evt_name, evt_def in self.config['events'].items():
                # Scan the event type name length
                self.evt_specs['type_len'] = max(self.evt_specs['type_len'], len(evt_name))

                # Number of parameters
                rate_params = evt_def.get('rate_params', [])
                n_params = len(rate_params)
                self.evt_specs['params'] = max(self.evt_specs['params'], n_params)

                # Number of participating objects
                objects = evt_def.get('objects', [])
                n_objs = len(objects)
                self.evt_specs['objs'] = max(self.evt_specs['objs'], n_objs)

        # Route 2: infer from the builder configuration
        # Single-object events: 1 object
        if 'single_obj_builders' in self.config and len(self.config['single_obj_builders']) > 0:
            for evt_name in self.config['single_obj_builders'].keys():
                self.evt_specs['type_len'] = max(self.evt_specs['type_len'], len(evt_name))
            self.evt_specs['objs'] = max(self.evt_specs['objs'], 1)

        # Pair events: 2 objects
        if 'pair_type_filters' in self.config and len(self.config['pair_type_filters']) > 0:
            for evt_name in self.config['pair_type_filters'].keys():
                self.evt_specs['type_len'] = max(self.evt_specs['type_len'], len(evt_name))
            self.evt_specs['objs'] = max(self.evt_specs['objs'], 2)

        # Zero-object events: 0 objects
        if 'zero_obj_builders' in self.config and len(self.config['zero_obj_builders']) > 0:
            for evt_name in self.config['zero_obj_builders'].keys():
                self.evt_specs['type_len'] = max(self.evt_specs['type_len'], len(evt_name))
            self.evt_specs['objs'] = max(self.evt_specs['objs'], 0)

        # The conservative v1.6.2 default (params=8) was removed
        # Without user-declared parameters the count stays 0;
        # get_event_dtype() applies max(params, 1) for at least one slot (a NumPy constraint)
        if self.verbose and self.evt_specs['params'] == 0:
            print("[TypeCompiler] No event params defined, using minimal params array (size=1)")

    def get_object_dtype(self) -> np.dtype:
        """
        Generate the dynamic OBJECT_DTYPE

        Returns:
            a NumPy dtype object

        Properties:
        - only the necessary fields are present
        - if a property class has zero entries, no memory is allocated for it
        """
        # Base fields (required)
        fields = [
            ('id', np.int64),
            ('type_id', np.int32),                        # integer type ID (hot path)
            ('x', np.float64),
            ('y', np.float64),
            ('z', np.float64),
            ('size', np.float64),        # physical size (radius/scale, used for the ghost region)
            ('owner_rank', np.int32),
            ('active', np.bool_),
            ('new', np.bool_),           # new-object flag (for incremental event building)
            ('changed', np.bool_),       # changed-object flag (for incremental event building)
        ]

        # Dynamic fields (added on demand)
        if self.obj_specs['float'] > 0:
            fields.append(('float_props', np.float64, (self.obj_specs['float'],)))

        dtype = np.dtype(fields, align=True)

        if self.verbose:
            print(f"[TypeCompiler] Generated OBJECT_DTYPE (size={dtype.itemsize} bytes)")
            for name in dtype.names:
                print(f"  - {name}: {dtype.fields[name][0]}")

        return dtype

    def get_event_dtype(self, object_dtype: np.dtype) -> np.dtype:
        """
        Generate the dynamic EVENT_DTYPE (the core v1.6.3 change)

        Args:
            object_dtype: object dtype (embedded into the event)

        Returns:
            a NumPy dtype object

        Core change:
        - events no longer store obj_ids (references) alone
        - an event embeds a copy of the OBJECT_DTYPE structure (self-contained)
        - contiguous in memory and JIT friendly

        Key design:
        - obj_ids is kept as a fast index (for incremental updates and communication)
        - an objects field stores the full object data (the heart of v1.6.3)
        - the two fields together serve both performance and functionality
        """
        max_objs = max(self.evt_specs['objs'], 6)  # at least 6 objects (multi-object reactions)
        max_params = max(self.evt_specs['params'], 1)  # at least one parameter

        fields = [
            ('id', np.int64),
            ('id_high', np.int64),  # 128-bit ID support
            ('id_low', np.int64),   # 128-bit ID support
            ('event_type_id', np.int32),  # integer event type ID (hot path)
            ('status', np.int32),  # 0=TODO, 1=BUSY, 2=DONE, 3=SENT
            ('owner_rank', np.int32),  # [Fix] the missing owner_rank field
            ('rate', np.float64),
            ('weight', np.float64),

            # [key] obj_ids is retained (fast index and incremental updates)
            # even in self-contained mode obj_ids remains important metadata
            ('obj_ids', np.int64, (max_objs,)),

            # [core v1.6.3 change] Embed the object data directly (self-contained events)
            # This is the essence of v1.6.3: an event becomes an independent unit of work
            ('objects', object_dtype, (max_objs,)),

            # Event parameters
            ('params', np.float64, (max_params,)),
        ]

        dtype = np.dtype(fields, align=True)

        if self.verbose:
            print(f"[TypeCompiler] Generated EVENT_DTYPE (size={dtype.itemsize} bytes)")
            print(f"  - Embedded objects: {max_objs} × {object_dtype.itemsize} = {max_objs * object_dtype.itemsize} bytes")
            print(f"  - obj_ids metadata: {max_objs} × 8 = {max_objs * 8} bytes")
            print(f"  - Total size: {dtype.itemsize} bytes")

        return dtype

    def get_specs(self) -> Dict[str, Any]:
        """
        Get a summary of the analysis

        Returns:
            a dict holding obj_specs and evt_specs
        """
        return {
            'obj_specs': self.obj_specs.copy(),
            'evt_specs': self.evt_specs.copy(),
        }

    def get_props_mapping(self) -> Dict[str, Any]:
        """
        Get the property-name-to-index mapping

        Returns:
            {
                'float_props': {obj_type: {prop_name: index}},
            }
        """
        return {
            'float_props': self.float_props_mapping.copy(),
            'type_to_int': self.type_to_int.copy(),       # type name -> integer ID
            'int_to_type': self.int_to_type.copy(),       # integer ID -> type name
        }

    def get_prop_index(self, obj_type: str, prop_name: str) -> Tuple[str, int]:
        """
        Get the array kind and index of a property

        Args:
            obj_type: object type name (e.g. 'H', 'V')
            prop_name: property name (e.g. 'radius')

        Returns:
            (array_name, index)
            - array_name: 'float_props'
            - index: index of the property inside the array

        Raises:
            KeyError: if the property does not exist
        """
        if obj_type in self.float_props_mapping:
            if prop_name in self.float_props_mapping[obj_type]:
                return ('float_props', self.float_props_mapping[obj_type][prop_name])

        raise KeyError(f"Property '{prop_name}' not found for object type '{obj_type}'")


# ===== Factory function =====

def create_type_compiler(config: dict, verbose: bool = False) -> TypeCompiler:
    """
    Factory: create a type compiler

    Args:
        config: user configuration dict
        verbose: whether to print details

    Returns:
        a TypeCompiler instance
    """
    compiler = TypeCompiler(config, verbose)
    compiler.analyze()
    return compiler


# ===== Test code =====

if __name__ == "__main__":
    print("=" * 70)
    print("TypeCompiler Test")
    print("=" * 70)

    # Test 1: a simple configuration
    print("\n[Test 1] Simple configuration")
    print("-" * 70)
    test_config1 = {
        'objects': {
            'H': {
                'properties': {
                    'radius': 1.0,
                    'charge': 1,
                }
            },
            'V': {
                'properties': {
                    'radius': 1.5,
                    'charge': -1,
                    'label': 'vacancy'
                }
            }
        },
        'events': {
            'migration': {
                'rate_params': ['Em', 'nu0'],
                'objects': ['H'],
            },
            'capture': {
                'rate_params': ['Ec', 'rc'],
                'objects': ['H', 'V'],
            }
        }
    }

    compiler1 = create_type_compiler(test_config1, verbose=True)
    obj_dtype1 = compiler1.get_object_dtype()
    evt_dtype1 = compiler1.get_event_dtype(obj_dtype1)

    # Test 2: compatibility mode (no object/event definitions)
    print("\n[Test 2] Compatibility mode (v1.6.2 defaults)")
    print("-" * 70)
    test_config2 = {
        'single_obj_builders': {'migration': None},
        'pair_type_filters': {'capture': None},
    }

    compiler2 = create_type_compiler(test_config2, verbose=True)
    obj_dtype2 = compiler2.get_object_dtype()
    evt_dtype2 = compiler2.get_event_dtype(obj_dtype2)

    print("\n" + "=" * 70)
    print("TypeCompiler Test Completed")
    print("=" * 70)
