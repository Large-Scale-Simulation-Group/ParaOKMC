"""
Load balancing must never change the global total rate.

N particles start in a corner hotspot; each hops at rate K_HOP and decays (is
deleted) at rate K_DECAY. With n alive particles the exact global total rate is
n * (K_HOP + K_DECAY) at every step. A lost event lowers it, a double-counted
one (a stolen event counted by both thief and victim) raises it. The test runs
the model with each balancing configuration, with and without rate reuse, and
checks the identity at every step.

Run:
    mpirun -np 4 python tests/test_lb_rate_conservation.py
"""
import os
import os
import sys
import numpy as np
from mpi4py import MPI
from numba import njit

from paraokmc import OKMCengine, SingleObjectEvent, RateDeps, change_obj, delete_obj

K_HOP, K_DECAY = 1.0e12, 2.0e11


class Violation(Exception):
    """Raised on every rank at once (the check follows an allreduce) to stop a run early."""
N0 = 3000
CASE = {'n0': N0}
ALIVE = {'n': N0}


class Hop(SingleObjectEvent):
    object_types = ['A']
    rate_deps = RateDeps(physics=['k_hop'])

    @staticmethod
    @njit
    def rate_kernel(k_hop):
        return k_hop

    @classmethod
    def execute(cls, event, objs, config):
        o, rng = objs[0], config['rng']
        box = np.asarray(config['grid_size'], dtype=float) * config['lattice_constant']
        axis = rng.randint(0, 3)
        pos = np.array([o['x'], o['y'], o['z']])
        pos[axis] = (pos[axis] + (1.0 if rng.random() < 0.5 else -1.0)) % box[axis]
        return [change_obj(o, x=pos[0], y=pos[1], z=pos[2])]


class HopPy(Hop):
    """Same hop with a plain-Python rate: level 3 only steals on the Python rate loop."""
    rate_deps = None
    rate_kernel = None

    @classmethod
    def rate(cls, objs, params, config):
        return K_HOP


class Decay(SingleObjectEvent):
    object_types = ['A']
    rate_deps = RateDeps(physics=['k_decay'])

    @staticmethod
    @njit
    def rate_kernel(k_decay):
        return k_decay

    @classmethod
    def execute(cls, event, objs, config):
        return [delete_obj(objs[0])]


def count_deletions(event_type, changed, step, time, pool, comm, config):
    ALIVE['n'] -= int(np.count_nonzero(~changed['active'])) if len(changed) else 0


def ic(rank, size, config):
    dd = config['_domain_decomp']
    box = np.asarray(config['grid_size'], dtype=float) * config['lattice_constant']
    pos = np.random.default_rng(3).uniform(0.02, 0.12, size=(CASE['n0'], 3)) * box
    return [{'id': k + 1, 'type': 'A', 'x': float(p[0]), 'y': float(p[1]), 'z': float(p[2]),
             'size': 1.0, 'float_props': []}
            for k, p in enumerate(pos) if dd.get_owner_rank(list(p)) == rank]


def _diagnose(inner, comm):
    """Which (object, event type) is missing or duplicated, and where do the copies live."""
    ep, pool = inner.rma_event_pool, inner.rma_object_pool
    names = inner.config.get('_event_int_to_name', {})
    ev = ep.single_obj_array[:ep.count_single]
    live = [(int(e['obj_ids'][0]), names.get(int(e['event_type_id']), '?'), int(e['status']),
             float(e['rate']), comm.rank) for e in ev]
    objs = pool.get_local_objects()
    mine = [(int(o['id']), comm.rank) for o in objs]
    all_live = comm.gather(live, root=0)
    all_objs = comm.gather(mine, root=0)
    if comm.rank != 0:
        return
    events = {}
    for part in all_live:
        for oid, et, st, r, rk in part:
            events.setdefault((oid, et), []).append((rk, st, r))
    owners = {oid: rk for part in all_objs for oid, rk in part}
    for oid, rk in sorted(owners.items()):
        for et in ('hop', 'decay'):
            copies = events.get((oid, et), [])
            counted = [c for c in copies if c[1] == 2 and c[2] > 0]
            if len(counted) != 1:
                print(f"    object {oid} (owner rank {rk}) event {et}: copies (rank, status, rate) = {copies}; "
                      f"other event: {events.get((oid, 'decay' if et == 'hop' else 'hop'))}", flush=True)
    stray = [(k, v) for k, v in events.items() if k[0] not in owners]
    if stray:
        print(f"    events of objects that do not exist: {stray[:5]}", flush=True)


def run(levels, incremental, python_rate=False, steps=400, n0=N0):
    CASE['n0'] = n0
    hop = HopPy if python_rate else Hop
    hop.incremental_rate = incremental
    # LB_MIXED=1: only the hop reuses its rate, so level 2 stays on
    Decay.incremental_rate = (not incremental) if os.environ.get('LB_MIXED') else incremental
    ALIVE['n'] = n0
    cfg = {
        "domain": {"lattice_constant": 1.0, "lattice_grid": [40, 40, 40],
                   "periodic": [True, True, True], "max_interaction_distance": 1.0},
        "simulation": {
            "max_steps": steps, "random_seed": 5, "progress_interval": 0,
            "object_load_balancer": {"enabled": '1' in levels, "balance_interval": 50},
            "predictive_rebalance": {"enabled": '2' in levels, "threshold": 0.5, "check_interval": 7},
            "runtime_load_balancer": {"enabled": '3' in levels, "check_interval": 8},
        },
        "objects": {"A": {"size": 1.0}},
        "ic": {"mode": "function", "func": ic},
        "events": {"hop": hop, "decay": Decay},
        "physics": {"params": {"k_hop": K_HOP, "k_decay": K_DECAY}},
        "outputs": {"event_hooks": [count_deletions]},
    }
    eng = OKMCengine(cfg)
    inner = getattr(eng, 'engine', eng)
    comm = inner.comm
    worst = [0.0]
    original = inner._get_total_rate

    def checked():
        local = original()
        total = comm.allreduce(local, op=MPI.SUM)
        expect = ALIVE['n'] * (K_HOP + K_DECAY)
        err = abs(total - expect) / expect
        worst[0] = max(worst[0], err)
        if err > 1e-9:
            if comm.rank == 0:
                print(f"  first violation at step {inner.current_step}: total={total:.6e} "
                      f"expected={expect:.6e} ratio={total / expect:.4f}", flush=True)
            if os.environ.get('LB_DIAG'):
                _diagnose(inner, comm)
            raise Violation()
        return local
    inner._get_total_rate = checked
    try:
        eng.run()
    except Violation:
        pass
    steals = comm.allreduce(inner.runtime_load_balancer.stats.get('total_tasks_sent', 0), op=MPI.SUM)
    moved2 = inner.predictive_rebalancer.stats.get('total_events_moved', 0)
    eng.destroy()
    return worst[0], ALIVE['n'], steals, moved2


if __name__ == '__main__':
    comm = MPI.COMM_WORLD
    failed = False
    cases = [(lv, inc, False) for lv in ['', '1', '2', '3', '23', '123'] for inc in (False, True)]
    cases += [(lv, inc, True) for lv in ['', '3', '23'] for inc in (False, True)]
    import os
    only = os.environ.get('LB_CASE')          # e.g. "3,python,False" runs a single case
    if only:
        lv, rk, ic_ = only.split(',')
        cases = [(lv, ic_ == 'True', rk == 'python')]
    only = os.environ.get('LB_CASE')          # e.g. "2,python,False" runs a single case
    if only:
        lv, rk, ic_ = only.split(',')
        cases = [(lv, ic_ == 'True', rk == 'python')]
    for levels, inc, py in cases:
            err, alive, steals, moved2 = run(levels, inc, python_rate=py,
                                              steps=int(os.environ.get('LB_STEPS', 40)) if py else 400,
                                              n0=int(os.environ.get('LB_N', 200)) if py else N0)
            ok = err < 1e-9
            failed |= not ok
            if comm.rank == 0:
                print(f"levels={levels or 'none':4s} rate={'python' if py else 'kernel'} incremental_rate={inc!s:5s} "
                      f"max rel. error of total rate = {err:.2e}  alive={alive}  stolen={steals}  moved_by_level2={moved2}  "
                      f"{'PASSED' if ok else 'FAILED'}", flush=True)
    sys.exit(1 if failed else 0)
