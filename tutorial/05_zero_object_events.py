"""
Tutorial 5 - zero-object events: a source of Frenkel pairs.

A ZeroObjectEvent is attached to no object. It models a global process such
as irradiation that creates defects at a given rate. By default one such
event exists on rank 0 (owner_rank_mode = 'rank0'); its rate is counted once
in the global sum like any other event, and when it is selected its execute()
creates new objects anywhere in the box.

Here a source creates I + V pairs at 'dpa_rate' per second while I and V
migrate and recombine, so the population reaches a steady state.

Run:
    python 05_zero_object_events.py
    mpirun -np 2 python 05_zero_object_events.py
"""
import numpy as np
from mpi4py import MPI
from numba import njit

from paraokmc import OKMCengine, ZeroObjectEvent, SingleObjectEvent, PairObjectEvent, RateDeps, change_obj, delete_obj, new_obj


def box_of(config):
    return np.asarray(config['grid_size'], dtype=float) * config['lattice_constant']


def default_props(config, obj_type):
    """float_props of a new object filled with the defaults declared in objects[type]."""
    props = [0.0] * config['_object_dtype'].fields['float_props'][0].shape[0]
    slots = config['_props_mapping']['float_props'][obj_type]
    for name, value in config['objects'][obj_type].get('properties', {}).items():
        props[slots[name]] = value
    return props


class FrenkelSource(ZeroObjectEvent):
    rate_deps = RateDeps(physics=['pair_production_rate'])      # no objects: a zero-object kernel

    @staticmethod
    @njit
    def rate_kernel(pair_production_rate):
        return pair_production_rate

    @classmethod
    def execute(cls, event, objs, config):                      # objs is empty
        rng, box = config['rng'], box_of(config)
        pos = np.array([rng.random(), rng.random(), rng.random()]) * box
        d = 4.0 * config['lattice_constant']                    # separate the pair so it does not recombine at once
        theta, phi = 2.0 * np.pi * rng.random(), np.arccos(2.0 * rng.random() - 1.0)
        shift = d * np.array([np.sin(phi) * np.cos(theta), np.sin(phi) * np.sin(theta), np.cos(phi)])
        v_pos, i_pos = pos, (pos + shift) % box
        # a new object needs its full property vector: copy the declared defaults
        return [new_obj(type='V', x=v_pos[0], y=v_pos[1], z=v_pos[2], size=1.0, float_props=default_props(config, 'V')),
                new_obj(type='I', x=i_pos[0], y=i_pos[1], z=i_pos[2], size=1.0, float_props=default_props(config, 'I'))]


class Migration(SingleObjectEvent):
    object_types = ['I', 'V']
    rate_deps = RateDeps(obj1=['E_m'], physics=['temperature'])

    @staticmethod
    @njit
    def rate_kernel(E_m, temperature):
        return 6.0e12 * np.exp(-E_m / (8.617333262e-5 * temperature))

    @classmethod
    def execute(cls, event, objs, config):
        o, rng, a0 = objs[0], config['rng'], config['lattice_constant']
        theta, phi = 2.0 * np.pi * rng.random(), np.arccos(2.0 * rng.random() - 1.0)
        step = a0 * np.array([np.sin(phi) * np.cos(theta), np.sin(phi) * np.sin(theta), np.cos(phi)])
        new = (np.array([o['x'], o['y'], o['z']]) + step) % box_of(config)
        return [change_obj(o, x=new[0], y=new[1], z=new[2])]


class Recombination(PairObjectEvent):
    type_pairs = [('I', 'V')]
    max_distance = 2.5
    rate_deps = RateDeps(kind='pair')

    @staticmethod
    @njit
    def rate_kernel(distance):
        return 1.0e13

    @classmethod
    def execute(cls, event, objs, config):
        return [delete_obj(objs[0]), delete_obj(objs[1])]


def population(step, time, pool, comm):
    n = comm.reduce(len(pool.get_local_objects()), op=MPI.SUM, root=0)
    if comm.rank == 0:
        print(f"  step {step:>6d}  t = {time:.3e} s  objects = {n}")


config = {
    "domain": {"lattice_constant": 2.87, "lattice_grid": [20, 20, 20],
               "periodic": [True, True, True], "max_interaction_distance": 2.5},
    "simulation": {"max_steps": 6000, "random_seed": 1, "progress_interval": 0,
                   "rma_object_pool": {"initial_capacity": 2000, "auto_expand": True}},
    "objects": {"I": {"size": 1.0, "properties": {"E_m": 0.34}},
                "V": {"size": 1.0, "properties": {"E_m": 0.67}}},
    "ic": {"mode": "builtin", "initial_conditions": {}},        # start empty: the source fills the box
    "events": {"source": FrenkelSource, "migration": Migration, "recombination": Recombination},
    "physics": {"params": {"temperature": 500.0, "pair_production_rate": 2.0e10}},
    "outputs": {"interval": 1000, "postprocessors": [population]},
}

if __name__ == '__main__':
    engine = OKMCengine(config)
    stats = engine.run()
    engine.destroy()
    if MPI.COMM_WORLD.rank == 0:
        print(f"done: {stats['total_steps']} steps, {stats['object_count']} objects at t = {stats['total_time']:.3e} s")
