"""
Regression: events of a deleted object must not survive the next build.

Background
----------
A pure deletion (absorption, annihilation) leaves changed_objects empty, so the
single-event in-place filter never runs; with the pool-level ghost cleanup gated
on an event-moving LB (commit 4396358) the DONE event of the vanished object
stayed in the pool, was selected, and apply_changed_objects re-created the
object from the stale snapshot: the object was resurrected and absorbed again, the
absorption count exceeded the number of objects, and the run never reached
total rate zero. Fixed in _build_events step 0 (deleted objects are always
cleaned).

What is verified (LB off, np = 1 or more)
-----------------------------------------
1. after the step that deletes an object, no single event in the pool refers to it
2. no later state-change message refers to the deleted id (no resurrection)
3. every object is deleted at most once
4. the run stops by itself when the last mobile object is absorbed (total rate 0)

Run:
    python tests/test_ghost_event_after_delete.py
    mpirun -np 4 python tests/test_ghost_event_after_delete.py
"""

from __future__ import annotations

import numpy as np
from mpi4py import MPI

from paraokmc import OKMCengine
from paraokmc import SingleObjectEvent, PairObjectEvent
from paraokmc import change_obj, delete_obj

COMM = MPI.COMM_WORLD
BOX = 20.0
N_A, N_S = 12, 16
MAX_STEPS = 200_000


class Hop(SingleObjectEvent):
    object_types = ['A']
    params = []

    @classmethod
    def rate(cls, objs, params, config):
        return 1.0

    @classmethod
    def execute(cls, event, objs, config):
        obj = objs[0]
        rng = config['rng']
        return [change_obj(
            obj,
            x=(float(obj['x']) + rng.random() * 2.0 - 1.0) % BOX,
            y=(float(obj['y']) + rng.random() * 2.0 - 1.0) % BOX,
            z=(float(obj['z']) + rng.random() * 2.0 - 1.0) % BOX)]


class Absorb(PairObjectEvent):
    type_pairs = [('A', 'S')]
    max_distance = 2.5
    params = []

    @classmethod
    def rate(cls, objs, params, config):
        return 1.0e6

    @classmethod
    def execute(cls, event, objs, config):
        a_type = config['_type_to_int']['A']
        t0 = objs[0]['type']
        is_a0 = (int(t0) == a_type) if isinstance(t0, (int, np.integer)) else (str(t0) == 'A')
        return [delete_obj(objs[0] if is_a0 else objs[1])]


def make_ic(seed):
    def ic(rank, size, config):
        rng = np.random.RandomState(seed)
        dd = config['_domain_decomp']
        objs, counters = [], {}

        def add(t, pos):
            owner = int(dd.get_owner_rank(list(pos)))
            counters[owner] = counters.get(owner, 0) + 1
            if owner == rank:
                objs.append({'id': (owner << 48) | counters[owner], 'type': t,
                             'x': float(pos[0]), 'y': float(pos[1]), 'z': float(pos[2]),
                             'size': 1.0, 'float_props': []})
        for _ in range(N_S):
            add('S', rng.random_sample(3) * BOX)
        for _ in range(N_A):
            add('A', rng.random_sample(3) * BOX)
        return objs
    return ic


STATE = {'deleted': {}, 'violations': [], 'absorbed': 0}
ENGINE = {}


def watch(event_type, changed, step, time_value, pool, comm, config):
    eng = ENGINE['e']
    ep = eng.rma_event_pool
    n1 = ep.count_single
    ev_ids = ep.single_obj_array[:n1]['obj_ids'][:, 0]
    # (1) the step after a deletion: the build has run, nothing may refer to the id
    for oid, s in STATE['deleted'].items():
        if step == s + 1 and int(np.sum(ev_ids == oid)) > 0:
            STATE['violations'].append(f"step {step}: single event of deleted id {oid} survived the build")
    for rec in changed:
        oid = int(rec['id'])
        if not rec['active']:
            if oid in STATE['deleted']:
                STATE['violations'].append(f"step {step}: id {oid} deleted twice (first at {STATE['deleted'][oid]})")
            STATE['deleted'][oid] = step
            STATE['absorbed'] += 1
        elif oid in STATE['deleted']:
            # (2) a change message for an object that no longer exists = resurrection
            STATE['violations'].append(f"step {step}: message for deleted id {oid} (deleted at {STATE['deleted'][oid]})")


def main():
    config = {
        'domain': {'lattice_constant': 2.0, 'lattice_grid': [10, 10, 10], 'h_factor': 1,
                   'periodic': [True, True, True], 'max_interaction_distance': 2.5},
        'simulation': {
            'max_steps': MAX_STEPS, 'max_time': 1.0e30, 'progress_interval': 0,
            'verbose': False, 'random_seed': 11,
            'rma_object_pool': {'initial_capacity': 512, 'auto_expand': True},
            'rma_event_pool': {'capacity_zero': 1, 'capacity_single': 512,
                               'capacity_pair': 2048, 'auto_expand': True},
            'incremental_update': {'enabled': True},
            'predictive_rebalance': {'enabled': False},
            'adaptive_weight': {'enabled': False},
            'runtime_load_balancer': {'enabled': False},
            'object_load_balancer': {'enabled': False},
        },
        'objects': {'A': {'size': 1.0, 'properties': {}}, 'S': {'size': 1.0, 'properties': {}}},
        'ic': {'mode': 'function', 'func': make_ic(3)},
        'events': {'hop': Hop, 'absorb': Absorb},
        'physics': {'params': {'temperature': 300.0}},
        'outputs': {'event_hooks': [watch]},
    }
    engine = OKMCengine(config)
    ENGINE['e'] = engine
    try:
        stats = engine.run()
    finally:
        engine.destroy()

    n_steps = int(stats.get('total_steps', -1))
    violations = STATE['violations']
    # every rank saw the same broadcast, so the checks are replicated; gather for reporting
    all_viol = COMM.allgather(violations)
    all_abs = COMM.allgather(STATE['absorbed'])
    assert all(a == all_abs[0] for a in all_abs), all_abs
    assert not any(all_viol), all_viol[0][:5] if all_viol[0] else [v for v in all_viol if v][0][:5]
    assert all_abs[0] == N_A, f"absorbed {all_abs[0]} != {N_A}"
    # (4) the run must have stopped by itself (total rate 0), long before max_steps
    assert n_steps < MAX_STEPS, f"run did not terminate: {n_steps} steps"
    if COMM.rank == 0:
        print(f"[test_ghost_event_after_delete] np={COMM.size} absorbed={all_abs[0]}/{N_A} "
              f"steps={n_steps} (max {MAX_STEPS}) no ghost events, no resurrection -> PASS",
              flush=True)


if __name__ == '__main__':
    main()
