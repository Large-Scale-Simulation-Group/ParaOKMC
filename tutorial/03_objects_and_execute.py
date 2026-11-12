"""
Tutorial 3 - objects, properties and the consequence of an event.

Objects carry x, y, z, a size and the scalar properties declared under
objects[type]['properties']. execute() returns a list of changes built with
change_obj / delete_obj / new_obj. This script forms and dissolves clusters:

    H + V  -> HV        (capture; a new object with its own properties)
    HV     -> H + V     (dissociation; the cluster disappears, two objects appear)
    H                   migrates; V and HV do not

A new object must be given its full property vector (float_props) in the
slot order the engine assigned; make_props() below builds it from the
declared defaults plus any override.

Run:
    python 03_objects_and_execute.py
    mpirun -np 2 python 03_objects_and_execute.py
"""
import numpy as np
from mpi4py import MPI
from numba import njit

from paraokmc import OKMCengine, SingleObjectEvent, PairObjectEvent, RateDeps, change_obj, delete_obj, new_obj

K_B = 8.617333262e-5


def box_of(config):
    return np.asarray(config['grid_size'], dtype=float) * config['lattice_constant']


def make_props(config, obj_type, **overrides):
    """Full float_props vector of a new object: declared defaults, then overrides."""
    n = config['_object_dtype'].fields['float_props'][0].shape[0]
    slots = config['_props_mapping']['float_props'][obj_type]
    props = np.zeros(n)
    for name, value in config['objects'][obj_type].get('properties', {}).items():
        props[slots[name]] = value
    for name, value in overrides.items():
        props[slots[name]] = value
    return props.tolist()


class HMigration(SingleObjectEvent):
    object_types = ['H']
    rate_deps = RateDeps(obj1=['E_m', 'nu'], physics=['temperature'])

    @staticmethod
    @njit
    def rate_kernel(E_m, nu, temperature):
        return nu * np.exp(-E_m / (8.617333262e-5 * temperature))

    @classmethod
    def execute(cls, event, objs, config):
        o, rng, a0 = objs[0], config['rng'], config['lattice_constant']
        theta, phi = 2.0 * np.pi * rng.random(), np.arccos(2.0 * rng.random() - 1.0)
        step = a0 * np.array([np.sin(phi) * np.cos(theta), np.sin(phi) * np.sin(theta), np.cos(phi)])
        new = (np.array([o['x'], o['y'], o['z']]) + step) % box_of(config)
        # a property can change together with the position
        return [change_obj(o, x=new[0], y=new[1], z=new[2], jumps=o['jumps'] + 1)]


class Capture(PairObjectEvent):
    type_pairs = [('H', 'V')]
    max_distance = 3.0
    rate_deps = RateDeps(kind='pair')

    @staticmethod
    @njit
    def rate_kernel(distance):
        return 1.0e13

    @classmethod
    def execute(cls, event, objs, config):
        h, v = (objs[0], objs[1]) if objs[0]['type'] == 'H' else (objs[1], objs[0])
        # the cluster inherits the vacancy position and remembers how far the H travelled
        return [delete_obj(h), delete_obj(v),
                new_obj(type='HV', x=v['x'], y=v['y'], z=v['z'],
                        size=config['objects']['HV']['size'],
                        float_props=make_props(config, 'HV', h_jumps=h['jumps']))]


class Dissociation(SingleObjectEvent):
    object_types = ['HV']
    rate_deps = RateDeps(obj1=['E_b'], physics=['temperature'])

    @staticmethod
    @njit
    def rate_kernel(E_b, temperature):
        return 6.0e12 * np.exp(-(0.39 + E_b) / (8.617333262e-5 * temperature))

    @classmethod
    def execute(cls, event, objs, config):
        c, rng, a0 = objs[0], config['rng'], config['lattice_constant']
        theta, phi = 2.0 * np.pi * rng.random(), np.arccos(2.0 * rng.random() - 1.0)
        r = 4.0 * a0                                # release the H outside the capture radius
        h_pos = (np.array([c['x'], c['y'], c['z']]) +
                 r * np.array([np.sin(phi) * np.cos(theta), np.sin(phi) * np.sin(theta), np.cos(phi)])) % box_of(config)
        return [delete_obj(c),
                new_obj(type='V', x=c['x'], y=c['y'], z=c['z'], size=config['objects']['V']['size'],
                        float_props=make_props(config, 'V')),
                new_obj(type='H', x=h_pos[0], y=h_pos[1], z=h_pos[2], size=config['objects']['H']['size'],
                        float_props=make_props(config, 'H', jumps=c['h_jumps']))]


def count_types(step, time, pool, comm):
    """Post-processor: how many objects of each type exist (summed over ranks)."""
    local = {}
    for o in pool.get_local_objects():
        t = pool.get_type_name(o)
        local[t] = local.get(t, 0) + 1
    totals = comm.gather(local, root=0)
    if comm.rank == 0:
        merged = {}
        for d in totals:
            for t, n in d.items():
                merged[t] = merged.get(t, 0) + n
        print(f"  step {step:>6d}  t = {time:.3e} s  " + "  ".join(f"{t}={n}" for t, n in sorted(merged.items())))


config = {
    "domain": {"lattice_constant": 3.17, "lattice_grid": [16, 16, 16],
               "periodic": [True, True, True], "max_interaction_distance": 3.0},
    "simulation": {"max_steps": 4000, "random_seed": 3, "progress_interval": 0},
    "objects": {
        "H":  {"size": 0.5, "properties": {"E_m": 0.39, "nu": 6.0e12, "jumps": 0.0}},
        "V":  {"size": 1.0, "properties": {}},
        "HV": {"size": 1.2, "properties": {"E_b": 1.2, "h_jumps": 0.0}},
    },
    "ic": {"mode": "builtin", "initial_conditions": {"H": {"count": 50, "distribution": "random"},
                                                     "V": {"count": 25, "distribution": "random"}}},
    "events": {"h_migration": HMigration, "capture": Capture, "dissociation": Dissociation},
    "physics": {"params": {"temperature": 900.0}},
    "outputs": {"interval": 1000, "postprocessors": [count_types]},
}

if __name__ == '__main__':
    engine = OKMCengine(config)
    stats = engine.run()
    engine.destroy()
    if MPI.COMM_WORLD.rank == 0:
        print(f"done: {stats['total_steps']} steps, {stats['object_count']} objects")
