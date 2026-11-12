"""
Tutorial 1 - a first model: Frenkel pairs in a periodic box.

Two object types, I (interstitial) and V (vacancy), both migrate with an
Arrhenius rate; an I and a V closer than 2 A recombine. This is the smallest
complete ParaOKMC model: two event classes and the seven configuration blocks.

Run:
    python 01_first_model.py
    mpirun -np 2 python 01_first_model.py        # identical model, two ranks
"""
import numpy as np
from mpi4py import MPI
from numba import njit

from paraokmc import OKMCengine, SingleObjectEvent, PairObjectEvent, RateDeps, change_obj, delete_obj


# ----------------------------------------------------------------- events

class Migration(SingleObjectEvent):
    """One random jump of length a0 for an I or a V."""

    object_types = ['I', 'V']                      # the event applies to both types

    # rate = nu * exp(-E_m / (k_B T)); E_m and nu are object properties,
    # temperature and k_B are physics parameters. The engine evaluates the
    # kernel for every candidate event at once.
    rate_deps = RateDeps(obj1=['E_m', 'nu'], physics=['temperature', 'k_B'])

    @staticmethod
    @njit
    def rate_kernel(E_m, nu, temperature, k_B):
        return nu * np.exp(-E_m / (k_B * temperature))

    @classmethod
    def execute(cls, event, objs, config):
        obj, rng = objs[0], config['rng']           # rng: the engine's reproducible generator
        a0 = config['lattice_constant']
        box = np.asarray(config['grid_size'], dtype=float) * a0
        theta, phi = 2.0 * np.pi * rng.random(), np.arccos(2.0 * rng.random() - 1.0)
        step = a0 * np.array([np.sin(phi) * np.cos(theta), np.sin(phi) * np.sin(theta), np.cos(phi)])
        new = (np.array([obj['x'], obj['y'], obj['z']]) + step) % box     # periodic wrap
        return [change_obj(obj, x=new[0], y=new[1], z=new[2])]


class Recombination(PairObjectEvent):
    """I + V -> nothing, as soon as they are within max_distance."""

    type_pairs = [('I', 'V')]
    max_distance = 2.0                             # A; candidate pairs exist only within this range

    @classmethod
    def rate(cls, objs, params, config):           # plain-Python rate: a constant here
        return 1.0e12

    @classmethod
    def execute(cls, event, objs, config):
        return [delete_obj(objs[0]), delete_obj(objs[1])]


# ----------------------------------------------------------------- configuration

config = {
    "domain": {                                    # geometry: box = lattice_grid * lattice_constant
        "lattice_constant": 2.87,                  # A (bcc Fe)
        "lattice_grid": [20, 20, 20],
        "periodic": [True, True, True],
        "max_interaction_distance": 2.0,           # >= the largest max_distance of any pair event
    },
    "simulation": {                                # how long to run and how
        "max_steps": 2000,
        "max_time": 1.0,                           # s
        "random_seed": 7,
        "progress_interval": 500,                  # progress line every 500 steps
    },
    "objects": {                                   # types, their size and scalar properties
        "I": {"size": 1.0, "properties": {"E_m": 0.34, "nu": 6.0e12}},
        "V": {"size": 1.0, "properties": {"E_m": 0.67, "nu": 6.0e12}},
    },
    "ic": {                                        # initial condition: random placement
        "mode": "builtin",
        "initial_conditions": {"I": {"count": 40, "distribution": "random"},
                               "V": {"count": 40, "distribution": "random"}},
    },
    "events": {"migration": Migration, "recombination": Recombination},
    "physics": {"params": {"temperature": 600.0, "k_B": 8.617333262e-5}},
    "outputs": {"interval": 500},                  # post-processing cadence (nothing registered yet)
}


if __name__ == '__main__':
    comm = MPI.COMM_WORLD
    engine = OKMCengine(config)                    # validates the config, decomposes the box, builds the pools
    stats = engine.run()                           # the whole simulation loop
    engine.destroy()                               # frees the MPI windows
    if comm.rank == 0:
        print(f"steps = {stats['total_steps']}, simulated time = {stats['total_time']:.3e} s, "
              f"objects left = {stats['object_count']}")
