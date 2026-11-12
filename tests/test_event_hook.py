"""
Post-processor event hooks (outputs.event_hooks).

What is verified
----------------
1. The hook runs on every rank once per step with the broadcast state-change
   message, and a deleted object record still carries its type and properties.
2. Statistics accumulated in the hook are identical on every rank (replicated
   state, no reduction needed) and equal to the reference obtained the old way:
   per-rank accumulation inside execute() on the winner rank + MPI allreduce.
3. A hook never touches the physics / rebuild machinery: the run is
   bit-identical with and without the hook.

Run:
    python tests/test_event_hook.py
    mpirun -np 4 python tests/test_event_hook.py
"""

from __future__ import annotations

import numpy as np
from mpi4py import MPI

from paraokmc import OKMCengine
from paraokmc import SingleObjectEvent, PairObjectEvent
from paraokmc import change_obj, delete_obj

COMM = MPI.COMM_WORLD
BOX = 20.0

# reference path: accumulated on the executing rank only (needs allreduce)
_EXEC_STATS = {'absorbed': 0, 'weight_sum': 0.0}
# hook path: accumulated on every rank from the broadcast (replicated)
_HOOK_STATS = {'absorbed': 0, 'weight_sum': 0.0, 'calls': 0, 'events': {}}


def _prop_idx(config, obj_type, name):
    return config['_props_mapping']['float_props'][obj_type][name]


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
        idx = _prop_idx(config, 'A', 'weight')
        props = np.array(obj['float_props'], copy=True)
        props[idx] += 1.0                      # one unit of "weight" per hop
        return [change_obj(
            obj,
            x=(float(obj['x']) + rng.random() * 2.0 - 1.0) % BOX,
            y=(float(obj['y']) + rng.random() * 2.0 - 1.0) % BOX,
            z=(float(obj['z']) + rng.random() * 2.0 - 1.0) % BOX,
            float_props=props.tolist())]


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
        a = objs[0] if is_a0 else objs[1]
        idx = _prop_idx(config, 'A', 'weight')
        _EXEC_STATS['absorbed'] += 1
        _EXEC_STATS['weight_sum'] += float(a['float_props'][idx])
        return [delete_obj(a)]


def hook(event_type, changed, step, time_value, pool, comm, config):
    _HOOK_STATS['calls'] += 1
    _HOOK_STATS['events'][event_type] = _HOOK_STATS['events'].get(event_type, 0) + 1
    if len(changed) == 0:
        return
    deleted = changed[~changed['active']]
    a_type = config['_type_to_int']['A']
    idx = _prop_idx(config, 'A', 'weight')
    for rec in deleted:
        assert int(rec['type_id']) == a_type, "deleted record lost its type"
        _HOOK_STATS['absorbed'] += 1
        _HOOK_STATS['weight_sum'] += float(rec['float_props'][idx])


def make_ic(n_a, n_s, seed):
    def ic(rank, size, config):
        rng = np.random.RandomState(seed)
        dd = config['_domain_decomp']
        objs = []
        counters = {}

        def add(t, pos, props):
            owner = int(dd.get_owner_rank(list(pos)))
            counters[owner] = counters.get(owner, 0) + 1
            if owner == rank:
                objs.append({'id': (owner << 48) | counters[owner], 'type': t,
                             'x': float(pos[0]), 'y': float(pos[1]), 'z': float(pos[2]),
                             'size': 1.0, 'float_props': props})

        n_float = config['_object_dtype'].fields['float_props'][0].shape[0]
        for _ in range(n_s):
            add('S', rng.random_sample(3) * BOX, [0.0] * n_float)
        for _ in range(n_a):
            add('A', rng.random_sample(3) * BOX, [0.0] * n_float)
        return objs
    return ic


def make_config(with_hook: bool, max_steps: int, seed: int):
    outputs = {'event_hooks': [hook]} if with_hook else {}
    return {
        'domain': {
            'lattice_constant': 2.0,
            'lattice_grid': [10, 10, 10],
            'h_factor': 1,
            'periodic': [True, True, True],
            'max_interaction_distance': 2.5,
        },
        'simulation': {
            'max_steps': max_steps,
            'max_time': 1.0e30,
            'progress_interval': 0,
            'verbose': False,
            'random_seed': seed,
            'rma_object_pool': {'initial_capacity': 512, 'auto_expand': True},
            'rma_event_pool': {'capacity_zero': 1, 'capacity_single': 512,
                               'capacity_pair': 2048, 'auto_expand': True},
            'incremental_update': {'enabled': True},
            'predictive_rebalance': {'enabled': False},
            'adaptive_weight': {'enabled': False},
            'runtime_load_balancer': {'enabled': False},
            'object_load_balancer': {'enabled': False},
        },
        'objects': {
            'A': {'size': 1.0, 'properties': {'weight': 0.0}},
            'S': {'size': 1.0, 'properties': {}},
        },
        'ic': {'mode': 'function', 'func': make_ic(60, 12, seed)},
        'events': {'hop': Hop, 'absorb': Absorb},
        'physics': {'params': {'temperature': 300.0}},
        'outputs': outputs,
    }


def run(with_hook: bool, max_steps: int = 3000, seed: int = 7):
    for d in (_EXEC_STATS, _HOOK_STATS):
        for k in d:
            d[k] = {} if isinstance(d[k], dict) else 0 if isinstance(d[k], int) else 0.0
    engine = OKMCengine(make_config(with_hook, max_steps, seed))
    try:
        stats = engine.run()
        sim_t = float(engine.current_time)
    finally:
        engine.destroy()
    return stats, sim_t


def main():
    rank, size = COMM.rank, COMM.size

    # --- run with the hook -------------------------------------------------
    stats, sim_t = run(with_hook=True)
    ref_abs = COMM.allreduce(_EXEC_STATS['absorbed'], op=MPI.SUM)
    ref_w = COMM.allreduce(_EXEC_STATS['weight_sum'], op=MPI.SUM)
    hook_abs_all = COMM.allgather(_HOOK_STATS['absorbed'])
    hook_w_all = COMM.allgather(_HOOK_STATS['weight_sum'])
    calls_all = COMM.allgather(_HOOK_STATS['calls'])
    n_steps = int(stats.get('total_steps', -1))

    assert ref_abs > 0, "the test must absorb at least one object"
    assert all(v == hook_abs_all[0] for v in hook_abs_all), hook_abs_all
    assert all(v == hook_w_all[0] for v in hook_w_all), hook_w_all
    assert hook_abs_all[0] == ref_abs, (hook_abs_all[0], ref_abs)
    assert hook_w_all[0] == ref_w, (hook_w_all[0], ref_w)
    assert all(c == n_steps for c in calls_all), (calls_all, n_steps)
    assert sum(_HOOK_STATS['events'].values()) == n_steps
    assert _HOOK_STATS['events'].get('absorb', 0) == ref_abs

    # --- run without the hook: trajectory must be identical -----------------
    stats2, sim_t2 = run(with_hook=False)
    assert int(stats2.get('total_steps', -2)) == n_steps
    assert sim_t2 == sim_t, (sim_t2, sim_t)
    ref_abs2 = COMM.allreduce(_EXEC_STATS['absorbed'], op=MPI.SUM)
    assert ref_abs2 == ref_abs

    if rank == 0:
        print(f"[test_event_hook] np={size} steps={n_steps} absorbed={ref_abs} "
              f"weight_sum={ref_w:.1f} hook==allreduce on every rank, "
              f"trajectory unchanged (sim_t={sim_t:.6e}) -> PASS", flush=True)


if __name__ == '__main__':
    main()
