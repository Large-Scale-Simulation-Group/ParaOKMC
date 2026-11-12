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
Helper function library

Common physics, mathematics and utility helpers
"""

import numpy as np
from typing import Callable, Optional


# ===== Physical constants =====

KB_EV_K = 8.617e-5  # Boltzmann constant (eV/K)
KB_J_K = 1.380649e-23  # Boltzmann constant (J/K)
AVOGADRO = 6.02214076e23  # Avogadro constant
EV_TO_J = 1.602176634e-19  # eV to J


# ===== Deterministic random number generation =====

def deterministic_random(user_seed, rank, local_counter):
    """
    Deterministic random number generator (from user_seed + rank + local_counter)

    Design goals:
    - independent streams per rank (through rank)
    - different numbers on repeated calls in one context (through local_counter)
    - fully deterministic and reproducible (through user_seed)
    - no global synchronization; every rank generates independently

    Args:
        user_seed (int): user-defined seed (taken from config['random_seed'])
        rank (int): MPI rank
        local_counter (int): local call counter (0, 1, 2, ...)

    Returns:
        float: a random number in [0, 1)

    Use cases:
        1. Inside event execute functions:
           >>> event_id_high = int(event['id_high'])
           >>> user_seed = config.get('random_seed', 0)
           >>> rank = config.get('rank', 0)
           >>> # use event_id_high as an extra seed
           >>> rand = deterministic_random(user_seed ^ event_id_high, rank, 0)

        2. Inside initialization functions:
           >>> user_seed = config.get('random_seed', 0)
           >>> rank = comm.rank
           >>> for i in range(n_objects):
           >>>     x = deterministic_random(user_seed, rank, i * 3 + 0) * box_size
           >>>     y = deterministic_random(user_seed, rank, i * 3 + 1) * box_size
           >>>     z = deterministic_random(user_seed, rank, i * 3 + 2) * box_size

        3. Inside post-processing functions:
           >>> user_seed = config.get('random_seed', 0)
           >>> rank = comm.rank
           >>> rand = deterministic_random(user_seed, rank, step)

    Notes:
        - the same (user_seed, rank, local_counter) always gives the same number
        - different ranks give different numbers (even for equal seed and counter)
        - the SplitMix64 hash provides good pseudo-random quality
    """
    MASK_64 = 0xFFFFFFFFFFFFFFFF

    # Combine the three inputs: user_seed, rank, local_counter
    combined = ((user_seed & 0xFFFFFFFF) << 32) | ((rank << 16) | (local_counter & 0xFFFF))

    # SplitMix64 hash (good-quality pseudo-random generation)
    seed = combined ^ ((user_seed + rank + local_counter) * 0x9e3779b97f4a7c15)
    seed = (seed ^ (seed >> 30)) * 0xbf58476d1ce4e5b9 & MASK_64
    seed = (seed ^ (seed >> 27)) * 0x94d049bb133111eb & MASK_64
    seed = (seed ^ (seed >> 31)) & MASK_64

    return (seed & 0xFFFFFFFF) / 4294967296.0


def deterministic_randint(user_seed, rank, local_counter, low, high):
    """
    Generate a deterministic random integer in [low, high)

    Args:
        user_seed (int): user-defined seed
        rank (int): MPI rank
        local_counter (int): local call counter
        low (int): lower bound (inclusive)
        high (int): upper bound (exclusive)

    Returns:
        int: a random integer in [low, high)

    Example:
        >>> # pick a direction at random (0-3)
        >>> direction = deterministic_randint(user_seed, rank, 0, 0, 4)
    """
    rand = deterministic_random(user_seed, rank, local_counter)
    return low + int(rand * (high - low))


def deterministic_choice(user_seed, rank, local_counter, choices):
    """
    Pick one element from a list deterministically at random

    Args:
        user_seed (int): user-defined seed
        rank (int): MPI rank
        local_counter (int): local call counter
        choices (list): candidate list

    Returns:
        the chosen element

    Example:
        >>> # pick a sign at random
        >>> sign = deterministic_choice(user_seed, rank, 0, [-1, 1])
    """
    idx = deterministic_randint(user_seed, rank, local_counter, 0, len(choices))
    return choices[idx]


# ===== Temperature schedule =====

class TemperatureScheduler:
    """
    Temperature schedule: supports a time-dependent temperature

    Supported modes:
    - constant: fixed temperature
    - linear: linear heating/cooling
    - exponential: exponential decay
    - custom: user-supplied function
    """

    def __init__(self, mode: str = 'constant', **kwargs):
        """
        Initialize the temperature schedule

        Args:
            mode: schedule mode
            kwargs: mode-specific parameters
                constant: T0
                linear: T0, T1, t0, t1
                exponential: T0, T_inf, tau
                custom: func (callable)
        """
        self.mode = mode
        self.kwargs = kwargs

        if mode == 'constant':
            self.T0 = kwargs.get('T0', 300.0)
        elif mode == 'linear':
            self.T0 = kwargs['T0']
            self.T1 = kwargs['T1']
            self.t0 = kwargs.get('t0', 0.0)
            self.t1 = kwargs['t1']
        elif mode == 'exponential':
            self.T0 = kwargs['T0']
            self.T_inf = kwargs.get('T_inf', 300.0)
            self.tau = kwargs['tau']
        elif mode == 'custom':
            self.func = kwargs['func']
        else:
            raise ValueError(f"Unknown temperature mode: {mode}")

    def get_temperature(self, time: float) -> float:
        """
        Get the temperature at a given time

        Args:
            time: simulation time

        Returns:
            temperature (K)
        """
        if self.mode == 'constant':
            return self.T0
        elif self.mode == 'linear':
            if time <= self.t0:
                return self.T0
            elif time >= self.t1:
                return self.T1
            else:
                alpha = (time - self.t0) / (self.t1 - self.t0)
                return self.T0 + alpha * (self.T1 - self.T0)
        elif self.mode == 'exponential':
            return self.T_inf + (self.T0 - self.T_inf) * np.exp(-time / self.tau)
        elif self.mode == 'custom':
            return self.func(time)
        else:
            return self.T0

    def __call__(self, time: float) -> float:
        """Allow the object to be called directly"""
        return self.get_temperature(time)


# ===== Mathematical helpers =====

def arrhenius(Em: float, nu0: float, temperature: float) -> float:
    """
    Arrhenius rate

    rate = nu0 * exp(-Em / kB / T)

    Args:
        Em: activation energy (eV)
        nu0: attempt frequency (1/s)
        temperature: temperature (K)

    Returns:
        rate (1/s)
    """
    if temperature <= 0:
        return 0.0
    return nu0 * np.exp(-Em / (KB_EV_K * temperature))


def distance_3d(pos1: np.ndarray, pos2: np.ndarray) -> float:
    """
    Distance between two points in 3D

    Args:
        pos1: position 1 [x, y, z]
        pos2: position 2 [x, y, z]

    Returns:
        distance
    """
    dx = pos1[0] - pos2[0]
    dy = pos1[1] - pos2[1]
    dz = pos1[2] - pos2[2]
    return np.sqrt(dx**2 + dy**2 + dz**2)


def periodic_distance_3d(pos1: np.ndarray, pos2: np.ndarray, box_size: np.ndarray) -> float:
    """
    3D distance under periodic boundary conditions

    Args:
        pos1: position 1 [x, y, z]
        pos2: position 2 [x, y, z]
        box_size: box dimensions [Lx, Ly, Lz]

    Returns:
        minimum image distance
    """
    dr = pos1 - pos2
    dr = dr - box_size * np.round(dr / box_size)
    return np.linalg.norm(dr)


def apply_periodic_bc(position: np.ndarray, box_size: np.ndarray) -> np.ndarray:
    """
    Apply periodic boundary conditions

    Args:
        position: position [x, y, z]
        box_size: box dimensions [Lx, Ly, Lz]

    Returns:
        the wrapped position
    """
    return position - box_size * np.floor(position / box_size)


# ===== Random jump helpers =====

class RandomJump:
    """
    Random jump generator

    Produces isotropic or directed random jump vectors
    """

    def __init__(self, jump_distance: float, mode: str = 'isotropic', direction: Optional[np.ndarray] = None):
        """
        Initialize the random jump generator

        Args:
            jump_distance: jump length
            mode: mode ('isotropic', 'directional', 'lattice')
            direction: direction vector for the directed mode (normalized automatically)
        """
        self.jump_distance = jump_distance
        self.mode = mode

        if mode == 'directional' and direction is not None:
            self.direction = np.array(direction) / np.linalg.norm(direction)
        else:
            self.direction = None

    def generate(self, rng: Optional[np.random.Generator] = None) -> np.ndarray:
        """
        Generate a random jump vector

        Args:
            rng: NumPy random generator (optional)

        Returns:
            jump vector [dx, dy, dz]
        """
        if rng is None:
            rng = np.random.default_rng()

        if self.mode == 'isotropic':
            # Isotropic: sample uniformly on the unit sphere
            theta = 2 * np.pi * rng.random()
            phi = np.arccos(2 * rng.random() - 1)
            dx = self.jump_distance * np.sin(phi) * np.cos(theta)
            dy = self.jump_distance * np.sin(phi) * np.sin(theta)
            dz = self.jump_distance * np.cos(phi)
            return np.array([dx, dy, dz])

        elif self.mode == 'directional':
            # Directed: along the given direction with a small perturbation
            perturbation = 0.1 * self.jump_distance * (rng.random(3) - 0.5)
            return self.jump_distance * self.direction + perturbation

        elif self.mode == 'lattice':
            # Lattice: one of the six principal directions
            directions = [
                [1, 0, 0], [-1, 0, 0],
                [0, 1, 0], [0, -1, 0],
                [0, 0, 1], [0, 0, -1]
            ]
            choice = rng.integers(0, 6)
            return self.jump_distance * np.array(directions[choice])

        else:
            raise ValueError(f"Unknown jump mode: {self.mode}")


# ===== Statistics helpers =====

class RateStatistics:
    """
    Rate statistics helper

    Collects and analyses the distribution of event rates
    """

    def __init__(self):
        self.rates = {}

    def record(self, event_type: str, rate: float):
        """Record a rate"""
        if event_type not in self.rates:
            self.rates[event_type] = []
        self.rates[event_type].append(rate)

    def get_statistics(self) -> dict:
        """
        Get the statistics

        Returns:
            {
                event_type: {
                    'count': number of samples,
                    'mean': mean,
                    'std': standard deviation,
                    'min': minimum,
                    'max': maximum,
                }
            }
        """
        stats = {}
        for event_type, rate_list in self.rates.items():
            if len(rate_list) > 0:
                stats[event_type] = {
                    'count': len(rate_list),
                    'mean': np.mean(rate_list),
                    'std': np.std(rate_list),
                    'min': np.min(rate_list),
                    'max': np.max(rate_list),
                }
        return stats

    def reset(self):
        """Reset the statistics"""
        self.rates = {}


# ===== Unit conversion =====

def eV_to_J(energy_eV: float) -> float:
    """eV to J"""
    return energy_eV * EV_TO_J


def J_to_eV(energy_J: float) -> float:
    """J to eV"""
    return energy_J / EV_TO_J


def nm_to_m(length_nm: float) -> float:
    """nanometres to metres"""
    return length_nm * 1e-9


def m_to_nm(length_m: float) -> float:
    """metres to nanometres"""
    return length_m * 1e9


if __name__ == "__main__":
    print("===== Temperature Scheduler Test =====")

    # Constant temperature
    T_const = TemperatureScheduler('constant', T0=500.0)
    print(f"Constant T(0) = {T_const(0):.2f} K")
    print(f"Constant T(100) = {T_const(100):.2f} K")

    # Linear heating
    T_linear = TemperatureScheduler('linear', T0=300.0, T1=800.0, t0=0.0, t1=1000.0)
    print(f"Linear T(0) = {T_linear(0):.2f} K")
    print(f"Linear T(500) = {T_linear(500):.2f} K")
    print(f"Linear T(1000) = {T_linear(1000):.2f} K")

    # Exponential decay
    T_exp = TemperatureScheduler('exponential', T0=800.0, T_inf=300.0, tau=500.0)
    print(f"Exponential T(0) = {T_exp(0):.2f} K")
    print(f"Exponential T(500) = {T_exp(500):.2f} K")
    print(f"Exponential T(1500) = {T_exp(1500):.2f} K")

    print("\n===== Arrhenius Test =====")
    Em = 0.5  # eV
    nu0 = 1e13  # 1/s
    T = 500.0  # K
    rate = arrhenius(Em, nu0, T)
    print(f"Rate(Em={Em} eV, nu0={nu0:.1e} Hz, T={T} K) = {rate:.3e} Hz")

    print("\n===== Distance Test =====")
    pos1 = np.array([0.0, 0.0, 0.0])
    pos2 = np.array([3.0, 4.0, 0.0])
    d = distance_3d(pos1, pos2)
    print(f"Distance = {d:.2f}")

    print("\n===== Random Jump Test =====")
    jump_gen = RandomJump(jump_distance=1.0, mode='isotropic')
    for i in range(3):
        jump = jump_gen.generate()
        print(f"Jump {i}: {jump}, length={np.linalg.norm(jump):.3f}")

    print("\nHelpers module test completed!")
