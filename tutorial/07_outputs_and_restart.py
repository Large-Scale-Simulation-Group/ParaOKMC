"""
Tutorial 7 - output, observers and restarting from a saved frame.

Three ways to look at a running simulation:
  * outputs.ovito_dump writes a LAMMPS-style dump every `interval` steps and
    once more after the last step. By default each rank writes its own file
    (prefix.<step>.rank_<r>.dump); OVITO loads them all with the wildcard
    prefix.<step>.rank_*.dump. Declared object properties become columns.
  * outputs.postprocessors are callables f(step, time, pool, comm) run every
    `interval` steps; they see the local objects and may reduce over comm.
  * outputs.event_hooks are callables f(event_type, changed, step, time, pool,
    comm, config) run on every rank after every applied event; `changed` is
    the broadcast record of what the event did (deleted records included), so
    an observable like the number of recombinations needs no MPI of its own.

A saved frame is an initial condition: ic.mode = "dump" restores it, with
the same or a different rank count.

Run:
    python 07_outputs_and_restart.py                       # 1500 steps -> out/07_dump/
    mpirun -np 2 python 07_outputs_and_restart.py --restart   # 1000 more steps from the last frame
"""
import argparse
import os
import numpy as np
from mpi4py import MPI
from numba import njit

from paraokmc import OKMCengine, SingleObjectEvent, PairObjectEvent, RateDeps, change_obj, delete_obj

HERE = os.path.dirname(os.path.abspath(__file__))
DUMP_DIR = os.path.join(HERE, 'out', '07_dump')
STATS = {'recombinations': 0}


def box_of(config):
    return np.asarray(config['grid_size'], dtype=float) * config['lattice_constant']


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
        return [change_obj(o, x=new[0], y=new[1], z=new[2], jumps=o['jumps'] + 1)]


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


def count_recombinations(event_type, changed, step, time, pool, comm, config):
    """Event hook: every rank sees the same broadcast record, so every rank holds the same count."""
    if event_type == 'recombination':
        STATS['recombinations'] += 1


def summary(step, time, pool, comm):
    """Post-processor: population so far (summed over ranks) and the recombination count."""
    local = pool.get_local_objects()
    n = comm.reduce(len(local), op=MPI.SUM, root=0)
    if comm.rank == 0:
        print(f"  step {step:>6d}  t = {time:.3e} s  objects = {n}  recombinations so far = {STATS['recombinations']}")


def make_config(restart):
    if restart:
        ic = {"mode": "dump", "file": DUMP_DIR, "frame": -1, "id_mode": "keep"}
        steps = 1000
    else:
        ic = {"mode": "builtin", "initial_conditions": {"I": {"count": 150, "distribution": "random"},
                                                        "V": {"count": 150, "distribution": "random"}}}
        steps = 1500
    return {
        "domain": {"lattice_constant": 2.87, "lattice_grid": [24, 24, 24],
                   "periodic": [True, True, True], "max_interaction_distance": 2.5},
        "simulation": {"max_steps": steps, "random_seed": 21 if not restart else 22, "progress_interval": 0},
        "objects": {"I": {"size": 1.0, "properties": {"E_m": 0.34, "jumps": 0.0}},
                    "V": {"size": 1.0, "properties": {"E_m": 0.67, "jumps": 0.0}}},
        "ic": ic,
        "events": {"migration": Migration, "recombination": Recombination},
        "physics": {"params": {"temperature": 700.0}},
        "outputs": {
            "interval": 500,
            "ovito_dump": {"enabled": True, "output_dir": DUMP_DIR if not restart else DUMP_DIR + '_restart',
                           "prefix": "frenkel", "mode": "distributed",
                           "properties": ["id", "type", "x", "y", "z", "size", "jumps"]},
            "postprocessors": [summary],
            "event_hooks": [count_recombinations],
        },
    }


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--restart', action='store_true', help='continue from the last frame in out/07_dump')
    args = p.parse_args()
    comm = MPI.COMM_WORLD
    engine = OKMCengine(make_config(args.restart))
    stats = engine.run()
    engine.destroy()
    if comm.rank == 0:
        where = DUMP_DIR + ('_restart' if args.restart else '')
        frames = sorted({f.split('.')[1] for f in os.listdir(where) if f.endswith('.dump')}, key=int)
        print(f"done: {stats['total_steps']} steps, {stats['object_count']} objects; frames {frames} in {where}")
