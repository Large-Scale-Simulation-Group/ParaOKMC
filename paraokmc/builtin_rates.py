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
Built-in rate library

Common rate functions
- Supports reading physics parameters from the UnifiedPhysics module
- Prefers the physics module, remaining backward compatible with the old config
"""

import numpy as np
from typing import Callable, Dict, Optional, List
from .helpers import arrhenius, distance_3d, KB_EV_K


# ===== Physics parameter helpers =====

def get_physics_temperature(config: dict, current_time: float = 0.0) -> float:
    """
    Get the temperature from the UnifiedPhysics module or from config

    Priority:
    1. config['physics'].get_param('temperature')
    2. config['temperature'] (a callable temperature schedule is supported)
    3. default value 300.0 K

    Args:
        config: configuration dict
        current_time: current simulation time (used by the temperature schedule)

    Returns:
        temperature (K)
    """
    physics = config.get('physics')

    # Prefer the physics module
    if physics is not None:
        try:
            return physics.get_param('temperature')
        except (KeyError, AttributeError):
            pass

    # Fall back to the top level of config
    T = config.get('temperature', 300.0)

    # Support a callable temperature schedule
    if callable(T):
        T = T(current_time)

    return float(T)


# ===== Rate calculator factory =====

class RateCalculatorFactory:
    """
    Rate calculator factory

    Creates configurable rate functions

    Can read the temperature from the UnifiedPhysics module
    """

    def __init__(self, params: dict):
        """
        Initialize the factory

        Args:
            params: dict of physics parameters, e.g.
                {
                    'temperature': 300.0 or a TemperatureScheduler object,
                    'nu0_migration': 1e13,
                    'Em_H': 0.3,
                    'Em_V': 1.0,
                    ...
                }
        """
        self.params = params

        # The temperature may be a constant or a schedule
        self.temperature = params.get('temperature', 300.0)

    def get_temperature(self, current_time: float = 0.0, config: dict = None) -> float:
        """
        Get the current temperature

        Prefers config['physics']

        Args:
            current_time: current simulation time
            config: configuration dict (optional, to reach the physics module)

        Returns:
            temperature (K)
        """
        # If a config is given, prefer the physics module
        if config is not None:
            physics = config.get('physics')
            if physics is not None:
                try:
                    return physics.get_param('temperature')
                except (KeyError, AttributeError):
                    pass

        # Fall back to the temperature given when the factory was built
        if callable(self.temperature):
            return self.temperature(current_time)
        else:
            return float(self.temperature)

    def create_rate_calculator(self,
                               migration_rates: Dict[str, dict],
                               reaction_rates: Dict[str, Callable],
                               default_rate: float = 0.0) -> Callable:
        """
        Create a rate function

        Args:
            migration_rates: migration rate configuration, e.g.
                {
                    'H': {'Em': 0.3, 'nu0': 1e13},
                    'V': {'Em': 1.0, 'nu0': 1e13},
                }
            reaction_rates: reaction rate functions, e.g.
                {
                    'capture': capture_rate_func(objs, params, T),
                    'recombination': recomb_rate_func(objs, params, T),
                }
            default_rate: default rate for unknown event types

        Returns:
            calculate_rate_func(event_type, objs, params, config) -> float
        """

        def calculate_rate_func(event_type: str, objs: List[np.ndarray],
                               params: List[float], config: dict) -> float:
            """
            Compute the rate of an event

            Prefers the temperature from the UnifiedPhysics module

            Args:
                event_type: event type string
                objs: participating objects (NumPy structured array)
                params: list of event parameters
                config: configuration dict

            Returns:
                rate (1/s)
            """
            # Temperature: physics module first, then config, then factory default
            current_time = config.get('current_time', config.get('_current_time', 0.0))
            T = self.get_temperature(current_time, config)

            # Migration events
            if event_type == 'migration':
                if len(objs) > 0:
                    obj_type = str(objs[0]['type'])
                    if obj_type in migration_rates:
                        Em = migration_rates[obj_type]['Em']
                        nu0 = migration_rates[obj_type]['nu0']
                        return arrhenius(Em, nu0, T)

            # Reaction events
            if event_type in reaction_rates:
                rate_func = reaction_rates[event_type]
                return rate_func(objs, params, T)

            # Default rate
            return default_rate

        return calculate_rate_func


# ===== Common reaction rate functions =====

def create_capture_rate(capture_radius: float,
                        nu0: float = 1e13,
                        distance_dependent: bool = True) -> Callable:
    """
    Create a capture reaction rate function

    Args:
        capture_radius: capture radius
        nu0: attempt frequency
        distance_dependent: whether the rate depends on distance

    Returns:
        rate_func(objs, params, T) -> float
    """

    def rate_func(objs: List[np.ndarray], params: List[float], T: float) -> float:
        """
        Compute the capture rate

        params[0] = distance (supplied by parallel_event_builder)
        """
        if len(objs) < 2:
            return 0.0

        distance = params[0] if len(params) > 0 else 0.0

        if distance_dependent:
            # Distance dependent: rate ~ 1/r^2
            if distance < 0.1:
                distance = 0.1  # avoid division by zero
            return nu0 / (distance**2 + 0.01)
        else:
            # Constant rate as long as the pair is within capture_radius
            if distance <= capture_radius:
                return nu0
            else:
                return 0.0

    return rate_func


def create_recombination_rate(recomb_radius: float,
                              Em_barrier: float = 0.0,
                              nu0: float = 1e13) -> Callable:
    """
    Create a recombination reaction rate function (e.g. I+V -> lattice)

    Args:
        recomb_radius: recombination radius
        Em_barrier: recombination barrier
        nu0: attempt frequency

    Returns:
        rate_func(objs, params, T) -> float
    """

    def rate_func(objs: List[np.ndarray], params: List[float], T: float) -> float:
        """
        Compute the recombination rate

        params[0] = distance
        """
        if len(objs) < 2:
            return 0.0

        distance = params[0] if len(params) > 0 else 0.0

        if distance <= recomb_radius:
            return arrhenius(Em_barrier, nu0, T)
        else:
            return 0.0

    return rate_func


def create_dissociation_rate(Em_dissociation: float,
                             nu0: float = 1e13) -> Callable:
    """
    Create a dissociation reaction rate function (e.g. H2 -> 2H)

    Args:
        Em_dissociation: dissociation energy
        nu0: attempt frequency

    Returns:
        rate_func(objs, params, T) -> float
    """

    def rate_func(objs: List[np.ndarray], params: List[float], T: float) -> float:
        """
        Compute the dissociation rate
        """
        return arrhenius(Em_dissociation, nu0, T)

    return rate_func


def create_emission_rate(Em_emission: float,
                        Eb_binding: float,
                        nu0: float = 1e13) -> Callable:
    """
    Create an emission reaction rate function (e.g. release of a defect from a cluster)

    Args:
        Em_emission: emission barrier
        Eb_binding: binding energy
        nu0: attempt frequency

    Returns:
        rate_func(objs, params, T) -> float
    """

    def rate_func(objs: List[np.ndarray], params: List[float], T: float) -> float:
        """
        Compute the emission rate

        Effective barrier = Em_emission + Eb_binding
        """
        Eff_barrier = Em_emission + Eb_binding
        return arrhenius(Eff_barrier, nu0, T)

    return rate_func


# ===== Predefined rate calculators =====

def create_hydrogen_tungsten_rates(temperature: float = 300.0) -> Callable:
    """
    Create the standard rate calculator for H in W

    Args:
        temperature: temperature (K)

    Returns:
        calculate_rate_func
    """
    params = {'temperature': temperature}

    factory = RateCalculatorFactory(params)

    migration_rates = {
        'H': {'Em': 0.206, 'nu0': 1e13},  # H migration
        'V': {'Em': 1.66, 'nu0': 1e13},   # V migration
        'I': {'Em': 0.013, 'nu0': 1e13},  # I migration
    }

    reaction_rates = {
        'capture': create_capture_rate(capture_radius=2.0, nu0=1e12),
        'recombination': create_recombination_rate(recomb_radius=1.5, Em_barrier=0.1),
    }

    return factory.create_rate_calculator(migration_rates, reaction_rates)


def create_iron_radiation_rates(temperature: float = 300.0) -> Callable:
    """
    Create the standard rate calculator for irradiation damage in Fe

    Args:
        temperature: temperature (K)

    Returns:
        calculate_rate_func
    """
    params = {'temperature': temperature}

    factory = RateCalculatorFactory(params)

    migration_rates = {
        'I': {'Em': 0.34, 'nu0': 6e12},   # Fe interstitial
        'V': {'Em': 0.67, 'nu0': 6e12},   # Fe vacancy
        'I2': {'Em': 0.42, 'nu0': 6e12},  # di-interstitial
        'V2': {'Em': 0.62, 'nu0': 6e12},  # di-vacancy
    }

    reaction_rates = {
        'IV_recombination': create_recombination_rate(recomb_radius=1.0, Em_barrier=0.0),
        'clustering': create_capture_rate(capture_radius=2.0, nu0=1e12),
    }

    return factory.create_rate_calculator(migration_rates, reaction_rates)


# ===== Decorators for custom rate calculators =====

def with_temperature_dependence(base_rate_func: Callable,
                               temperature_scheduler) -> Callable:
    """
    Add temperature dependence to a rate function

    Args:
        base_rate_func: base rate function
        temperature_scheduler: temperature schedule

    Returns:
        the wrapped rate function
    """

    def wrapped_rate_func(event_type: str, objs: List[np.ndarray],
                         params: List[float], config: dict) -> float:
        """Wrapped rate function"""
        current_time = config.get('current_time', 0.0)
        T = temperature_scheduler(current_time)

        # Update the temperature inside config
        config_copy = config.copy()
        config_copy['temperature'] = T

        return base_rate_func(event_type, objs, params, config_copy)

    return wrapped_rate_func


def with_rate_scaling(base_rate_func: Callable,
                     scale_factor: float) -> Callable:
    """
    Add a scale factor to a rate function

    Args:
        base_rate_func: base rate function
        scale_factor: scale factor

    Returns:
        the wrapped rate function
    """

    def wrapped_rate_func(event_type: str, objs: List[np.ndarray],
                         params: List[float], config: dict) -> float:
        """Wrapped rate function"""
        base_rate = base_rate_func(event_type, objs, params, config)
        return base_rate * scale_factor

    return wrapped_rate_func


if __name__ == "__main__":
    print("===== Builtin Rates Test =====")

    # Test the rate calculator factory
    params = {'temperature': 500.0}
    factory = RateCalculatorFactory(params)

    migration_rates = {
        'H': {'Em': 0.3, 'nu0': 1e13},
        'V': {'Em': 1.0, 'nu0': 1e13},
    }

    reaction_rates = {
        'capture': create_capture_rate(capture_radius=2.0, nu0=1e12),
    }

    calc_rate = factory.create_rate_calculator(migration_rates, reaction_rates)

    # Test the migration rate
    obj_H = np.zeros(1, dtype=[('id', np.int64), ('type', 'U16')])[0]
    obj_H['id'] = 1
    obj_H['type'] = 'H'

    rate_H = calc_rate('migration', [obj_H], [], {'current_time': 0.0})
    print(f"H migration rate at 500K: {rate_H:.3e} Hz")

    # Test the capture rate
    obj_V = np.zeros(1, dtype=[('id', np.int64), ('type', 'U16')])[0]
    obj_V['id'] = 2
    obj_V['type'] = 'V'

    distance = 1.5
    rate_capture = calc_rate('capture', [obj_H, obj_V], [distance], {'current_time': 0.0})
    print(f"H-V capture rate (d={distance}): {rate_capture:.3e} Hz")

    # Test the predefined rate calculators
    calc_HW = create_hydrogen_tungsten_rates(temperature=500.0)
    rate_H_W = calc_HW('migration', [obj_H], [], {'current_time': 0.0})
    print(f"\nH in W migration rate at 500K: {rate_H_W:.3e} Hz")

    print("\nBuiltin Rates test completed!")
