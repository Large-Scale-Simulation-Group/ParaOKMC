"""
Tutorial 4 - your own initial condition, and how random numbers work.

ic.mode = "function" hands the engine a function

    func(rank, size, config) -> list[dict]

that every rank calls once. Each rank must return the objects that lie in
its own subdomain and nothing else; config['_domain_decomp'].get_owner_rank()
answers that question, so the simplest pattern is: generate the whole
configuration identically on every rank from a seeded generator, keep the
objects you own. Ids must be unique across ranks.

Random numbers inside events come from config['rng']. Every draw is a pure
function of (random_seed, step, draw index), so a run is reproducible and
every rank would obtain the same numbers if it asked. Do not use np.random
inside an event: ranks would drift apart.

Run twice with the same seed and compare the fingerprint lines:
    python 04_initial_condition_and_random_numbers.py --seed 5
    python 04_initial_condition_and_random_numbers.py --seed 5
    python 04_initial_condition_and_random_numbers.py --seed 6     # different trajectory
"""
import argparse
import numpy as np
from mpi4py import MPI
from numba import njit

from paraokmc import OKMCengine, SingleObjectEvent, RateDeps, change_obj


class Hop(SingleObjectEvent):
    """Jump along one of the six <100> directions; the direction is drawn from config['rng']."""
    object_types = ['A']
    rate_deps = RateDeps(physics=['hop_rate'])

    @staticmethod
    @njit
    def rate_kernel(hop_rate):
        return hop_rate

    @classmethod
    def execute(cls, event, objs, config):
        o, rng, a0 = objs[0], config['rng'], config['lattice_constant']
        box = np.asarray(config['grid_size'], dtype=float) * a0
        axis = rng.randint(0, 3)                   # 0, 1 or 2
        sign = 1.0 if rng.random() < 0.5 else -1.0
        pos = np.array([o['x'], o['y'], o['z']])
        pos[axis] = (pos[axis] + sign * a0) % box[axis]
        return [change_obj(o, x=pos[0], y=pos[1], z=pos[2])]


def two_slabs_ic(n_per_slab):
    """All particles start in two thin slabs at x = L/4 and x = 3L/4."""
    def initial_objects(rank, size, config):
        dd = config['_domain_decomp']
        a0 = config['lattice_constant']
        box = np.asarray(config['grid_size'], dtype=float) * a0
        gen = np.random.default_rng(config['_user_seed'])      # same stream on every rank
        mine = []
        for k in range(2 * n_per_slab):
            x0 = box[0] * (0.25 if k < n_per_slab else 0.75)
            pos = [x0 + a0 * gen.uniform(-0.5, 0.5), gen.uniform(0, box[1]), gen.uniform(0, box[2])]
            if dd.get_owner_rank(pos) == rank:                  # keep what this rank owns
                mine.append({'id': k + 1, 'type': 'A', 'x': pos[0], 'y': pos[1], 'z': pos[2],
                             'size': 1.0, 'float_props': []})
        return mine
    return initial_objects


def fingerprint(step, time, pool, comm):
    """Post-processor: a number that changes if any coordinate changes."""
    local = pool.get_local_objects()
    s = float(np.sum(local['x'] * 1.0 + local['y'] * 2.0 + local['z'] * 3.0)) if len(local) else 0.0
    total = comm.reduce(s, op=MPI.SUM, root=0)
    if comm.rank == 0:
        print(f"  step {step:>5d}  t = {time:.6e}  fingerprint = {total:.6f}")


def make_config(seed):
    return {
        "domain": {"lattice_constant": 1.0, "lattice_grid": [40, 20, 20],
                   "periodic": [True, True, True], "max_interaction_distance": 1.0},
        "simulation": {"max_steps": 2000, "random_seed": seed, "progress_interval": 0},
        "objects": {"A": {"size": 1.0}},
        "ic": {"mode": "function", "func": two_slabs_ic(200)},
        "events": {"hop": Hop},
        "physics": {"params": {"hop_rate": 1.0e12}},
        "outputs": {"interval": 1000, "postprocessors": [fingerprint]},
    }


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--seed', type=int, default=5)
    args = p.parse_args()
    engine = OKMCengine(make_config(args.seed))
    stats = engine.run()
    engine.destroy()
