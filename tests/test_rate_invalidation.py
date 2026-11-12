"""
Physics changes invalidate RATES, never event STRUCTURE.

Design rule: the structure of the event pool (which objects carry which events)
depends only on the objects and the geometry; a physics parameter or field only
enters the rates. So when an event changes a physics parameter through
physics.set_param_in_event(), the engine must

  - do nothing at all when no event class's rate depends on that parameter
    (a counter, say): no flag, no rebuild, no rate reset;
  - otherwise drop the rates of the dependent classes to TODO (recomputed in
    the next rate phase) and rewrite the params column cached in the events of
    scalar-rate classes, without touching the event structure.

What is verified (np = 1 or more)
---------------------------------
1. a counter parameter bumped on every execution of one event causes zero rate
   invalidations, and the trajectory is bit-identical to a run without it;
2. a temperature change invalidates exactly the dependent classes: in the step
   after the change every DONE rate of the JIT class (rate_deps.physics) equals
   the Arrhenius value at the new temperature, and every event of the scalar
   class (params=['temperature']) carries the new temperature in its cached
   params column and the matching rate;
3. the single-event pool is built in full exactly once (the first step); every
   later step, temperature changes included, goes through the in-place path.

Run:
    python tests/test_rate_invalidation.py
    mpirun -np 4 python tests/test_rate_invalidation.py
"""

from __future__ import annotations

import math

import numpy as np
from mpi4py import MPI
from numba import njit

from paraokmc import OKMCengine
from paraokmc import SingleObjectEvent
from paraokmc import change_obj
from paraokmc import RateDeps

COMM = MPI.COMM_WORLD
BOX = 20.0
KB = 8.617333262e-5
NU, EM = 1.0e13, 0.45
T0 = 500.0
BUMP_EVERY = 40          # every BUMP_EVERY-th execution of 'C' raises the temperature
DT_BUMP = 25.0
SCALAR_COEF = 1.0e-3


@njit
def arrhenius_kernel(temperature):
    return NU * np.exp(-EM / (KB * temperature))


def arrhenius(T):
    return NU * math.exp(-EM / (KB * T))


def _hop(obj, config):
    rng = config['rng']
    return [change_obj(obj,
                       x=(float(obj['x']) + rng.random() * 2.0 - 1.0) % BOX,
                       y=(float(obj['y']) + rng.random() * 2.0 - 1.0) % BOX,
                       z=(float(obj['z']) + rng.random() * 2.0 - 1.0) % BOX)]


class HopJIT(SingleObjectEvent):
    """JIT rate depending on the temperature; rates are reused across steps."""
    object_types = ['A']
    params = []
    rate_deps = RateDeps(physics=['temperature'])
    rate_kernel = arrhenius_kernel
    incremental_rate = True

    @classmethod
    def execute(cls, event, objs, config):
        return _hop(objs[0], config)


class HopScalar(SingleObjectEvent):
    """Scalar rate reading the temperature from the cached params column."""
    object_types = ['B']
    params = ['temperature']
    incremental_rate = True

    @classmethod
    def rate(cls, objs, params, config):
        return SCALAR_COEF * float(params['temperature'])

    @classmethod
    def execute(cls, event, objs, config):
        return _hop(objs[0], config)


class Driver(SingleObjectEvent):
    """Bumps a counter on every execution; every BUMP_EVERY-th execution also
    raises the temperature (when bumping is enabled)."""
    object_types = ['C']
    params = []
    bump_temperature = True

    @classmethod
    def rate(cls, objs, params, config):
        return 1.0e12

    @classmethod
    def execute(cls, event, objs, config):
        physics = config['physics']
        n = physics.get_param('counter') + 1.0
        physics.set_param_in_event('counter', n)
        if cls.bump_temperature and int(n) % BUMP_EVERY == 0:
            physics.set_param_in_event('temperature', physics.get_param('temperature') + DT_BUMP)
        return _hop(objs[0], config)


def make_ic(seed, n_a=12, n_b=12, n_c=1):
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
        for t, n in (('A', n_a), ('B', n_b), ('C', n_c)):
            for _ in range(n):
                add(t, rng.random_sample(3) * BOX)
        return objs
    return ic


CHECK = {'violations': [], 'checked': 0}
ENGINE = {}


def check_rates(event_type, changed, step, time_value, pool, comm, config):
    """Runs after the rate phase of this step: every DONE rate must match the
    CURRENT temperature (the change of the previous step has been applied)."""
    eng = ENGINE['e']
    physics = config['physics']
    T = physics.get_param('temperature')
    ep = eng.rma_event_pool
    n = ep.count_single
    if n == 0:
        return
    seg = ep.single_obj_array[:n]
    gid_jit = config['_event_name_to_int']['hop_jit']
    gid_sc = config['_event_name_to_int']['hop_scalar']
    done = seg['status'] == 2
    jit = done & (seg['event_type_id'] == gid_jit)
    sc = done & (seg['event_type_id'] == gid_sc)
    k = arrhenius(T)
    if np.any(jit):
        bad = np.abs(seg['rate'][jit] - k) > 1e-9 * k
        if np.any(bad):
            CHECK['violations'].append(f"step {step}: {int(bad.sum())} JIT rates stale (T={T})")
    if np.any(sc):
        bad_p = np.abs(seg['params'][sc, 0] - T) > 1e-12
        bad_r = np.abs(seg['rate'][sc] - SCALAR_COEF * T) > 1e-12
        if np.any(bad_p) or np.any(bad_r):
            CHECK['violations'].append(
                f"step {step}: scalar cached params stale ({int(bad_p.sum())}) / rates stale ({int(bad_r.sum())}) (T={T})")
    CHECK['checked'] += 1


def make_config(seed, with_hook, bump):
    Driver.bump_temperature = bump
    return {
        'domain': {'lattice_constant': 2.0, 'lattice_grid': [10, 10, 10], 'h_factor': 1,
                   'periodic': [True, True, True], 'max_interaction_distance': 2.5},
        'simulation': {
            'max_steps': 3000, 'max_time': 1.0e30, 'progress_interval': 0,
            'verbose': False, 'random_seed': seed,
            'rma_object_pool': {'initial_capacity': 512, 'auto_expand': True},
            'rma_event_pool': {'capacity_zero': 1, 'capacity_single': 512,
                               'capacity_pair': 64, 'auto_expand': True},
            'incremental_update': {'enabled': True},
            'predictive_rebalance': {'enabled': False},
            'adaptive_weight': {'enabled': False},
            'runtime_load_balancer': {'enabled': False},
            'object_load_balancer': {'enabled': False},
        },
        'objects': {'A': {'size': 1.0, 'properties': {}},
                    'B': {'size': 1.0, 'properties': {}},
                    'C': {'size': 1.0, 'properties': {}}},
        'ic': {'mode': 'function', 'func': make_ic(5)},
        'events': {'hop_jit': HopJIT, 'hop_scalar': HopScalar, 'driver': Driver},
        'physics': {'params': {'temperature': T0, 'counter': 0.0}},
        'outputs': {'event_hooks': [check_rates]} if with_hook else {},
    }


def run(seed, with_hook, bump):
    CHECK['violations'] = []
    CHECK['checked'] = 0
    engine = OKMCengine(make_config(seed, with_hook, bump))
    ENGINE['e'] = engine
    # A structural rebuild can only be forced through set_force_rebuild (the object LB
    # is off here, so the physics path was its only caller): count the calls.
    forced = {'n': 0}
    orig = engine.incremental_manager.set_force_rebuild

    def counted(*a, **k):
        forced['n'] += 1
        return orig(*a, **k)
    engine.incremental_manager.set_force_rebuild = counted
    try:
        stats = engine.run()
        out = {
            'steps': int(stats.get('total_steps', -1)),
            'sim_t': float(engine.current_time),
            'invalidations': int(engine._n_rate_invalidations),
            'counter': float(engine.physics.get_param('counter')),
            'T': float(engine.physics.get_param('temperature')),
            'forced_rebuilds': forced['n'],
            'violations': list(CHECK['violations']),
            'checked': CHECK['checked'],
            'deps_param': {k: sorted(v) for k, v in engine._rate_deps_by_param.items()},
            'deps_all': sorted(engine._rate_deps_all),
            'scalar_cols': {k: sorted(v) for k, v in engine._scalar_param_cols.items()},
        }
    finally:
        engine.destroy()
    return out


def main():
    rank, size = COMM.rank, COMM.size

    # Three runs; every rank runs all of them, and the checks are made on the gathered
    # results so that no rank can leave early and strand the others in a collective.
    a = run(seed=3, with_hook=False, bump=False)          # counter only
    orig_execute = Driver.execute

    @classmethod
    def silent_execute(cls, event, objs, config):
        return _hop(objs[0], config)
    Driver.execute = silent_execute
    ref = run(seed=3, with_hook=False, bump=False)        # same, no physics writes at all
    Driver.execute = orig_execute
    b = run(seed=3, with_hook=True, bump=True)            # temperature bumps

    gathered = COMM.allgather({
        'deps_param': a['deps_param'], 'deps_all': a['deps_all'], 'scalar_cols': a['scalar_cols'],
        'a_counter': a['counter'], 'a_inv': a['invalidations'], 'a_forced': a['forced_rebuilds'],
        'a_steps': a['steps'], 'a_sim_t': a['sim_t'],
        'ref_steps': ref['steps'], 'ref_sim_t': ref['sim_t'],
        'b_counter': b['counter'], 'b_T': b['T'], 'b_inv': b['invalidations'],
        'b_forced': b['forced_rebuilds'], 'b_checked': b['checked'], 'b_steps': b['steps'],
        'b_viol': b['violations'][:5],
    })
    r0 = gathered[0]

    # dependency tables: temperature -> {hop_jit, hop_scalar}; counter -> nothing
    assert len(r0['deps_param'].get('temperature', [])) == 2, r0['deps_param']
    assert 'counter' not in r0['deps_param'], r0['deps_param']
    assert r0['deps_all'] == [], r0['deps_all']
    assert list(r0['scalar_cols'].keys()) == ['temperature'], r0['scalar_cols']

    # (1) counter only: no invalidation, nothing forced, trajectory unchanged
    assert r0['a_counter'] > 0, "the driver never ran"
    for g in gathered:
        assert g['a_inv'] == 0, g['a_inv']
        assert g['a_forced'] == 0, g['a_forced']
        assert g['ref_steps'] == g['a_steps'] and g['ref_sim_t'] == g['a_sim_t'], \
            (g['ref_steps'], g['a_steps'], g['ref_sim_t'], g['a_sim_t'])

    # (2)+(3) temperature bumps: exact invalidation count, fresh rates, nothing forced
    n_bumps = int(r0['b_counter']) // BUMP_EVERY
    assert n_bumps >= 3, f"only {n_bumps} temperature changes; raise max_steps"
    for g in gathered:
        assert abs(g['b_T'] - (T0 + n_bumps * DT_BUMP)) < 1e-9, (g['b_T'], n_bumps)
        assert g['b_inv'] == n_bumps, (g['b_inv'], n_bumps)
        assert g['b_forced'] == 0, g['b_forced']
        assert g['b_checked'] == g['b_steps'], (g['b_checked'], g['b_steps'])
        assert not g['b_viol'], g['b_viol']

    if rank == 0:
        print(f"[test_rate_invalidation] np={size} counter-only: 0 invalidations, trajectory unchanged; "
              f"{n_bumps} temperature changes -> {r0['b_inv']} invalidations, all rates fresh "
              f"in {r0['b_checked']} checked steps, forced structural rebuilds=0 -> PASS",
              flush=True)


if __name__ == '__main__':
    main()
