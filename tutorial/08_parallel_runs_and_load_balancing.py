"""
Tutorial 8 - running in parallel and the load-balancing switches.

Nothing in a model refers to ranks: the same script runs on 1 or 64 processes.
What the rank count changes is the spatial decomposition (a cuboid grid of
subdomains) and therefore how the work is shared. The residence-time
statistics are the same; the trajectory is not bitwise identical because
the catalogue of events is enumerated in a different order.

This script puts every particle in a small cube near one corner of the box (a
hotspot that one rank owns whatever the rank count) so that a plain
decomposition leaves the other ranks idle, and lets you switch the three
balancing levels on or off:

    level 1  object_load_balancer   re-cuts the box by object density (RCB)
    level 2  predictive_rebalance   moves event records to under-loaded ranks before rate evaluation
    level 3  runtime_load_balancer  idle ranks steal pending events during rate evaluation

--levels selects which of them run ("" = none, "23" = the library defaults,
"123" = all). Compare the wall time per step printed at the end:
    mpirun -np 4 python 08_parallel_runs_and_load_balancing.py --levels ""
    mpirun -np 4 python 08_parallel_runs_and_load_balancing.py --levels 1
    mpirun -np 4 python 08_parallel_runs_and_load_balancing.py --levels 123
    mpirun -np 4 python 08_parallel_runs_and_load_balancing.py --levels 123 --uniform   # no hotspot
"""
import argparse
import time
import numpy as np
from mpi4py import MPI
from numba import njit

from paraokmc import OKMCengine, SingleObjectEvent, RateDeps, change_obj


class Hop(SingleObjectEvent):
    object_types = ['A']
    rate_deps = RateDeps(physics=['hop_rate'])
    incremental_rate = True        # the rate is constant: let the engine reuse it for untouched objects

    @staticmethod
    @njit
    def rate_kernel(hop_rate):
        return hop_rate

    @classmethod
    def execute(cls, event, objs, config):
        o, rng, a0 = objs[0], config['rng'], config['lattice_constant']
        box = np.asarray(config['grid_size'], dtype=float) * a0
        theta, phi = 2.0 * np.pi * rng.random(), np.arccos(2.0 * rng.random() - 1.0)
        step = a0 * np.array([np.sin(phi) * np.cos(theta), np.sin(phi) * np.sin(theta), np.cos(phi)])
        new = (np.array([o['x'], o['y'], o['z']]) + step) % box
        return [change_obj(o, x=new[0], y=new[1], z=new[2])]


class HopPython(Hop):
    """The same hop with a plain-Python rate that is recomputed every step.

    Level 3 (work stealing) acts on the Python-rate loop only: events whose class has a
    compiled kernel are evaluated in one batch and are never stolen.
    """
    rate_deps = None
    rate_kernel = None
    incremental_rate = False

    @classmethod
    def rate(cls, objs, params, config):
        return config['physics'].get_param('hop_rate')


def ic(n, uniform):
    def initial_objects(rank, size, config):
        dd = config['_domain_decomp']
        box = np.asarray(config['grid_size'], dtype=float) * config['lattice_constant']
        gen = np.random.default_rng(config['_user_seed'])
        lo, hi = (0.0, 1.0) if uniform else (0.05, 0.15)        # hotspot: a 10 % cube near a corner
        pos = gen.uniform(lo, hi, size=(n, 3)) * box
        return [{'id': k + 1, 'type': 'A', 'x': float(p[0]), 'y': float(p[1]), 'z': float(p[2]),
                 'size': 1.0, 'float_props': []}
                for k, p in enumerate(pos) if dd.get_owner_rank(list(p)) == rank]
    return initial_objects


def make_config(args):
    lv = set(args.levels)
    return {
        "domain": {"lattice_constant": 1.0, "lattice_grid": [60, 60, 60],
                   "periodic": [True, True, True], "max_interaction_distance": 1.0},
        "simulation": {
            "max_steps": args.steps, "random_seed": 1, "progress_interval": 0,
            "object_load_balancer": {"enabled": '1' in lv, "balance_interval": 200, "imbalance_threshold": 0.2},
            "predictive_rebalance": {"enabled": '2' in lv, "threshold": 1.5},
            "runtime_load_balancer": {"enabled": '3' in lv, "check_interval": args.check_interval},
            "rma_object_pool": {"initial_capacity": args.n, "auto_expand": True},
            "rma_event_pool": {"capacity_single": args.n, "capacity_pair": 64, "capacity_zero": 1, "auto_expand": True},
        },
        "objects": {"A": {"size": 1.0}},
        "ic": {"mode": "function", "func": ic(args.n, args.uniform)},
        "events": {"hop": Hop if args.rate == 'kernel' else HopPython},
        "physics": {"params": {"hop_rate": 1.0e12}},
        "outputs": {},
    }


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--levels', default='23', help='balancing levels to enable, e.g. "", "1", "23", "123"')
    p.add_argument('--uniform', action='store_true', help='spread the particles over the whole box')
    p.add_argument('--n', type=int, default=20000)
    p.add_argument('--steps', type=int, default=2000)
    p.add_argument('--rate', choices=['kernel', 'python'], default='kernel',
                   help='compiled kernel with rate reuse (Hop) or Python rate recomputed every step (HopPython)')
    p.add_argument('--check-interval', type=int, default=32, help='level-3 check interval (steps)')
    args = p.parse_args()
    comm = MPI.COMM_WORLD
    t0 = time.perf_counter()
    engine = OKMCengine(make_config(args))
    stats = engine.run()
    engine.destroy()
    wall = time.perf_counter() - t0
    if comm.rank == 0:
        print(f"ranks={comm.size} levels={args.levels or 'none'} rate={args.rate} {'uniform' if args.uniform else 'hotspot'}: "
              f"{stats['total_steps']} steps in {wall:.1f} s = {1e3 * wall / stats['total_steps']:.2f} ms/step; "
              f"per-step averages: rate {1e3 * stats['avg_rate_time']:.2f} ms, build {1e3 * stats['avg_build_time']:.2f} ms, "
              f"select {1e3 * stats['avg_select_time']:.2f} ms, execute {1e3 * stats['avg_execute_time']:.2f} ms")
