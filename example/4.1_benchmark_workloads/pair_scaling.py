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

"""Pair workload of the scaling benchmarks (Section 4.1).

H and V objects uniformly distributed; migration dominates (one object makes a small
step per event); H-V capture pair events are built but never selected (rate 1e-6, so
the pair structure stays in place). Constant density: the box grows with N, about 2.6
H-V neighbours per object.

--pair-structure on   structural increment for pair events (the library default)
--pair-structure off  pair events rebuilt in full every step (the escape hatch
                      simulation.incremental_update.pair_structure = False)

The initial condition generates every object on every rank from one seed and keeps
those inside the rank's subdomain.

Run:
    mpirun -np 8 python pair_scaling.py --n 20000 --steps 300 --pair-structure on
"""
import argparse
import sys

import numpy as np
from mpi4py import MPI

from paraokmc import OKMCengine, SingleObjectEvent, PairObjectEvent

COMM = MPI.COMM_WORLD
LC = 3.17
BOX = [0.0, 0.0, 0.0]   # set from N in main()


class Mig(SingleObjectEvent):
    object_types = ['H', 'V']
    params = []

    @classmethod
    def rate(cls, objs, params, config):
        return 1.0e6

    @classmethod
    def execute(cls, event, objs, config):
        obj = objs[0]
        rng = config['rng']
        out = {'id': int(obj['id'])}
        for axis, box_len in zip(('x', 'y', 'z'), BOX):
            out[axis] = (float(obj[axis])
                         + (rng.random() * 3.0 - 1.5)) % box_len
        return [out]


class Cap(PairObjectEvent):
    type_pairs = [('H', 'V')]
    max_distance = 6.0
    params = ['distance']

    @classmethod
    def rate(cls, objs, params, config):
        return 1.0e-6   # built but never selected: the pair structure stays in place

    @classmethod
    def execute(cls, event, objs, config):
        return []


def make_ic(n):
    def initial_objects_func(rank, size, config):
        # generate all coordinates as arrays (weak scaling reaches millions of objects)
        # and build dictionaries only for the objects of this subdomain
        r = np.random.default_rng(5)
        xs = r.uniform(0, BOX[0], n)
        ys = r.uniform(0, BOX[1], n)
        zs = r.uniform(0, BOX[2], n)
        dd = config.get('_domain_decomp')
        if dd is None:
            keep = np.arange(n) if rank == 0 else np.arange(0)
        else:
            b = dd.subdomain_bounds
            keep = np.flatnonzero(
                (xs >= b['x_min']) & (xs < b['x_max'])
                & (ys >= b['y_min']) & (ys < b['y_max'])
                & (zs >= b['z_min']) & (zs < b['z_max']))
        return [{'id': int(i) + 1,
                 'type': 'H' if (int(i) % 2 == 0) else 'V',
                 'x': float(xs[i]), 'y': float(ys[i]), 'z': float(zs[i])}
                for i in keep]
    return initial_objects_func


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, default=20000)
    parser.add_argument('--steps', type=int, default=300)
    parser.add_argument('--seed', type=int, default=9)
    parser.add_argument('--pair-structure', choices=['on', 'off'],
                        default='on')
    args = parser.parse_args()

    # constant density: N = 20000 <-> grid 60^3 (about 2.6 H-V neighbours per object)
    edge = max(20, int(round(60 * (args.n / 20000.0) ** (1.0 / 3.0))))
    grid = (edge, edge, edge)
    global BOX
    BOX = [grid[i] * LC for i in range(3)]

    size = COMM.size
    # pool capacities per rank (local objects ~ n/size plus a boundary margin); sizing
    # them for the global N would waste hundreds of MB per rank in weak scaling
    cap_single = max(4096, args.n * 3 // size)
    cap_pair = max(20000, args.n * 8 // size)

    config = {
        "domain": {
            "lattice_constant": LC,
            "lattice_grid": list(grid),
            "h_factor": 1,
            "periodic": [True, True, True],
            "max_interaction_distance": 6.0,
        },
        "simulation": {
            "max_steps": args.steps,
            "max_time": 1.0e30,
            "verbose": False,
            "random_seed": args.seed,
            "progress_interval": 1000,   # > 0 prints the [ParaOKMC] Done/Perf lines
            "rma_object_pool": {"initial_capacity": cap_single, "auto_expand": True},
            "rma_event_pool": {"capacity_zero": 4, "capacity_single": cap_single,
                               "capacity_pair": cap_pair, "auto_expand": True},
            "incremental_update": {"pair_structure": args.pair_structure == "on"},
            "predictive_rebalance": {"enabled": False},
            "adaptive_weight": {"enabled": False},
            "runtime_load_balancer": {"enabled": False},
            "object_load_balancer": {"enabled": False},
        },
        "objects": {"H": {}, "V": {}},
        "ic": {"mode": "function", "func": make_ic(args.n)},
        "events": {"mig": Mig, "cap": Cap},
        "physics": {"params": {"temperature": 500.0}},
        "outputs": {},
    }
    engine = OKMCengine(config)
    engine.run()   # prints the [ParaOKMC] Done/Perf lines

    n_pair = COMM.allreduce(int(engine.rma_event_pool.count_pair),
                            op=MPI.SUM)
    objs = engine.rma_object_pool.get_local_objects()
    owned = objs[(objs['owner_rank'] == COMM.rank) & objs['active']]
    n_obj = COMM.allreduce(len(owned), op=MPI.SUM)
    if COMM.rank == 0:
        print(f"[pair_scaling] n={args.n} steps={args.steps} np={size} "
              f"pair_structure={args.pair_structure} "
              f"pairs={n_pair} objs={n_obj}", flush=True)
    engine.destroy()


if __name__ == '__main__':
    main()
