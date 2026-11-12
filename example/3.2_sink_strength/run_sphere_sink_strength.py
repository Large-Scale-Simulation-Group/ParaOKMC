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

"""
e1: Sink strength of spherical absorbers.

Reference (all parameters taken from the paper itself):
    L. Malerba, C.S. Becquart, C. Domain,
    "Object kinetic Monte Carlo study of sink strengths",
    Journal of Nuclear Materials 360 (2007) 159-169.

What is validated
-----------------
A single mobile defect species performs jump-by-jump migration in a bcc
lattice box containing randomly placed, non-overlapping, unsaturable
spherical absorbers. The average number of jumps before absorption <nj>
gives the sink strength (paper Eq. (1), with the n = 3 convention):

    k^2 = 2*3 / (dj^2 * <nj>)

which is compared against the theoretical expressions:

    3D migration (paper Eq. (2), 2nd order):
        k^2 = 4*pi*N*R * (1 + R*sqrt(4*pi*N*R))
    1D migration (paper Eq. (3)):
        k^2 = 6 * (pi*R^2*N)^2
    mixed 1D/3D (paper Eqs. (4a-c) master curve), with lch = dj*sqrt(n'_j)

R is the capture distance (the paper's R values 0.75...10.25 nm already
include the defect capture radius), used directly as the pair-event
max_distance.

How the model is written
------------------------
Two event classes carry the whole physics; everything an event needs is a
class attribute or a method of that class:

- SIAMigration (single-object): a <111> walk whose axis is redrawn every
  `jumps_before_turn` jumps (1 = fully 3D). The Arrhenius rate is a Numba
  kernel defined inside the class; execute() makes one jump, tracks the
  current axis and the jumps left on it in two object properties, and
  increments `jump_count`.
- SphereAbsorption (pair-object): absorbers are inert objects of type 'S';
  `max_distance` is the capture distance R and the rate is huge, so a defect
  is absorbed by the first jump that brings it within R. The rate is a plain
  Python rate() on purpose; the same class in the compiled rate_deps +
  rate_kernel form follows it, commented out (see the note there).

The box follows from config['grid_size'] * config['lattice_constant'].
Defects do not interact with each other (no SIA-SIA event registered).

The jump statistics are collected by a post-processor event hook
(outputs.event_hooks): the engine calls it on every rank right after the
broadcast state-change message of a step has been applied, and a deleted
defect record still carries its jump_count, so every rank holds the same
totals and no MPI reduction is needed. The run ends when the last defect is
absorbed (total rate zero).

Run:
    python run_sphere_sink_strength.py --regime 3d --R-nm 6.25 --N-cm3 1.5e17
    python run_sphere_sink_strength.py --regime 1d --R-nm 10.25 --N-cm3 6e16
    mpirun -np 4 python run_sphere_sink_strength.py --regime mixed --jumps-before-turn 100
"""

from __future__ import annotations

import argparse
import math
from typing import Dict, Tuple

import numpy as np
from mpi4py import MPI
from numba import njit

from paraokmc import OKMCengine
from paraokmc import SingleObjectEvent, PairObjectEvent
from paraokmc import change_obj, delete_obj
from paraokmc import RateDeps


A0_FE_A = 2.87                                    # bcc Fe lattice parameter [A]
JUMP_DISTANCE_A = math.sqrt(3.0) * A0_FE_A / 2.0  # dj = sqrt(3)/2 a0 [A]
DOMAIN_COARSEN = 25                               # decomposition cell = 25 a0 (granularity only)

_STATS = {'absorbed': 0, 'jump_sum': 0.0, 'jump_sq_sum': 0.0}


def _box_size(config: Dict) -> np.ndarray:
    """Simulation box [A] from the domain block the engine passes along."""
    return np.asarray(config['grid_size'], dtype=np.float64) * float(config['lattice_constant'])


# ============================================================ events

class SIAMigration(SingleObjectEvent):
    """<111> walk; the axis is redrawn every `jumps_before_turn` jumps (1 = fully 3D)."""

    object_types = ['SIA']
    # rate reuse: the rate depends only on a constant parameter, so an event whose
    # object did not move keeps its rate instead of being recomputed every step
    incremental_rate = True

    # Arrhenius parameters of the paper
    ATTEMPT_HZ = 8.071e13
    MIGRATION_EV = 0.31
    KB_EV_K = 8.617333262145e-5

    # the four <111> directions; a jump is +/- axis * a0/2
    AXES_111 = np.array([[1, 1, 1], [1, 1, -1], [1, -1, 1], [-1, 1, 1]], dtype=np.float64)

    jumps_before_turn = 1              # set by create_config(): 1 = 3D, large = 1D

    # --- rate: compiled kernel, evaluated for the whole event array at once ---
    rate_deps = RateDeps(physics=['temperature'])

    @staticmethod
    @njit
    def rate_kernel(temperature):
        return 8.071e13 * np.exp(-0.31 / (8.617333262145e-5 * temperature))

    # --- axis bookkeeping in the object properties axis_idx / segment_left ---
    @classmethod
    def next_axis(cls, props, mapping, rng) -> Tuple[int, int]:
        n_turn = int(cls.jumps_before_turn)
        if n_turn <= 1:
            return int(rng.randint(0, 4)), 0
        axis_idx = int(round(props[mapping['axis_idx']]))
        segment_left = int(round(props[mapping['segment_left']]))
        if axis_idx < 0 or axis_idx >= 4 or segment_left <= 0:
            axis_idx, segment_left = int(rng.randint(0, 4)), n_turn
        return axis_idx, segment_left - 1

    # --- consequence ---
    @classmethod
    def execute(cls, event, objs, config):
        obj, rng = objs[0], config['rng']
        mapping = config['_props_mapping']['float_props']['SIA']
        props = np.array(obj['float_props'], copy=True)
        axis_idx, segment_left = cls.next_axis(props, mapping, rng)
        sign = 1.0 if rng.random() >= 0.5 else -1.0
        step = sign * 0.5 * A0_FE_A * cls.AXES_111[axis_idx]
        new_pos = (np.array([obj['x'], obj['y'], obj['z']], dtype=np.float64) + step) % _box_size(config)

        props[mapping['jump_count']] += 1.0
        props[mapping['axis_idx']] = float(axis_idx)
        props[mapping['segment_left']] = float(segment_left)
        return [change_obj(obj, x=new_pos[0], y=new_pos[1], z=new_pos[2], float_props=props.tolist())]


class SphereAbsorption(PairObjectEvent):
    """Instantaneous absorption of a defect by an inert spherical absorber ('S')."""

    type_pairs = [('SIA', 'S')]
    # rate reuse: the rate depends only on the pair geometry, which changes only when
    # one of the two objects moves, so untouched pairs keep their rate
    incremental_rate = True
    ABSORB_RATE_HZ = 1.0e18            # >> jump rate (~1.5e11 Hz at 573 K)
    max_distance = 1.0                 # = capture distance R, set by create_config()

    @staticmethod
    def sia_of(objs, config):
        """The SIA member of the pair (the other one is the absorber)."""
        t = objs[0]['type']
        is_sia = (int(t) == config['_type_to_int']['SIA']) if isinstance(t, (int, np.integer)) else (str(t) == 'SIA')
        return objs[0] if is_sia else objs[1]

    @classmethod
    def rate(cls, objs, params, config):
        return cls.ABSORB_RATE_HZ

    @classmethod
    def execute(cls, event, objs, config):
        return [delete_obj(cls.sia_of(objs, config))]


# ---------------------------------------------------------------------------
# The same event with a compiled rate (teaching variant, off by default)
#
# ParaOKMC offers two ways to write a rate:
#   * rate(objs, params, config): plain Python, called once per candidate
#     event; it may read anything (object coordinates, config, class state).
#   * rate_deps + rate_kernel: a Numba kernel the engine applies to the whole
#     event array at once. It sees only what rate_deps declares: float
#     properties of obj1 and obj2, the pair distance, field values and
#     physics scalars, in that order. obj1 is always the first type in
#     `type_pairs`.
# The example keeps the Python form above so that both forms appear side by
# side (SIAMigration uses the compiled one). A pair kernel that needs no
# object property must say kind='pair' explicitly, because RateDeps infers the
# event kind from the declared properties. To try it, comment out the class
# above and uncomment this one; the trajectory is identical.
# ---------------------------------------------------------------------------
# class SphereAbsorption(PairObjectEvent):
#     """Instantaneous absorption, rate as a compiled kernel (obj1 = SIA, obj2 = S)."""
#
#     type_pairs = [('SIA', 'S')]
#     # rate reuse: the rate depends only on the pair geometry, which changes only when
#     # one of the two objects moves, so untouched pairs keep their rate
#     incremental_rate = True
#     ABSORB_RATE_HZ = 1.0e18
#     max_distance = 1.0                 # = capture distance R, set by create_config()
#
#     rate_deps = RateDeps(kind='pair')  # nothing but the pair distance is needed
#
#     @staticmethod
#     @njit
#     def rate_kernel(distance):
#         return 1.0e18                  # the pair exists only within max_distance
#
#     @staticmethod
#     def sia_of(objs, config):
#         t = objs[0]['type']
#         is_sia = (int(t) == config['_type_to_int']['SIA']) if isinstance(t, (int, np.integer)) else (str(t) == 'SIA')
#         return objs[0] if is_sia else objs[1]
#
#     @classmethod
#     def execute(cls, event, objs, config):
#         return [delete_obj(cls.sia_of(objs, config))]


# ============================================================ observables

def _absorption_hook(event_type, changed, step, time_value, pool, comm, config):
    """Event hook: accumulate the jump count of every absorbed SIA (same totals on every rank)."""
    if len(changed) == 0:
        return
    deleted = changed[~changed['active']]
    if len(deleted) == 0:
        return
    sia_type = config['_type_to_int']['SIA']
    idx = config['_props_mapping']['float_props']['SIA']['jump_count']
    for rec in deleted:
        if int(rec['type_id']) != sia_type:
            continue
        nj = float(rec['float_props'][idx])
        _STATS['absorbed'] += 1
        _STATS['jump_sum'] += nj
        _STATS['jump_sq_sum'] += nj * nj


def _make_reporter(num_defects: int):
    def reporter(step, time_value, pool, comm):
        total_abs = int(_STATS['absorbed'])
        if comm.rank == 0:
            print(f"[e1] step={step:>9d} active={num_defects - total_abs} "
                  f"absorbed={total_abs}/{num_defects}", flush=True)
    return reporter


# ============================================================ initial condition

def _create_ic_func(n_sinks: int, n_defects: int, capture_a: float, seed: int):
    """Absorbers: random, capture spheres non-overlapping. Defects: random,
    outside every capture sphere (the paper excludes defects created inside absorbers)."""

    def initial_objects_func(rank, size, config):
        saved_state = np.random.get_state()
        np.random.seed(config.get('_user_seed', seed))
        domain_decomp = config['_domain_decomp']
        box = _box_size(config)

        def min_image(delta):
            return delta - box * np.round(delta / box)

        # absorbers: rejection sampling with centre distance >= 2R
        centers = np.empty((0, 3), dtype=np.float64)
        attempts = 0
        while len(centers) < n_sinks:
            attempts += 1
            if attempts > 1000 * n_sinks:
                raise RuntimeError("Could not place non-overlapping absorbers; volume fraction too high for this box.")
            pos = np.random.random(3) * box
            if len(centers) > 0 and np.min(np.linalg.norm(min_image(pos[None, :] - centers), axis=1)) < 2.0 * capture_a:
                continue
            centers = np.vstack([centers, pos])

        # defects: uniform, outside all capture spheres
        defect_pos = []
        while len(defect_pos) < n_defects:
            pos = np.random.random(3) * box
            if np.min(np.linalg.norm(min_image(pos[None, :] - centers), axis=1)) > capture_a:
                defect_pos.append(pos)

        n_float = config['_object_dtype'].fields['float_props'][0].shape[0]
        props_of = config['_props_mapping']['float_props']
        sia_props = np.zeros(n_float)
        sia_props[props_of['SIA']['axis_idx']] = -1.0
        sink_props = np.zeros(n_float)
        if 'radius' in props_of.get('S', {}):
            sink_props[props_of['S']['radius']] = capture_a

        local_objects, owner_counters = [], {}

        def add(obj_type, pos, size_a, props):
            owner_rank = int(domain_decomp.get_owner_rank(list(pos)))
            owner_counters[owner_rank] = owner_counters.get(owner_rank, 0) + 1
            if owner_rank == rank:
                local_objects.append({'id': (owner_rank << 48) | owner_counters[owner_rank],
                                      'type': obj_type, 'x': float(pos[0]), 'y': float(pos[1]),
                                      'z': float(pos[2]), 'size': size_a, 'float_props': props.tolist()})

        for pos in centers:
            add('S', pos, 2.0 * capture_a, sink_props)
        for pos in defect_pos:
            add('SIA', pos, A0_FE_A, sia_props)

        np.random.set_state(saved_state)
        return local_objects

    return initial_objects_func


# ============================================================ configuration

def _regime_to_turns(regime: str, jumps_before_turn: int | None) -> int:
    if regime == '3d':
        return 1
    if regime == '1d':
        return int(10**12)
    if jumps_before_turn is None:
        raise ValueError("--jumps-before-turn is required for regime='mixed'")
    return int(jumps_before_turn)


def create_config(regime: str, capture_nm: float, density_cm3: float, target_absorbed: int,
                  grid: Tuple[int, int, int], random_seed: int, progress_interval: int,
                  report_interval: int, dump_dir: str, dump_interval: int, max_steps: int,
                  jumps_before_turn: int | None) -> Dict:
    global _STATS
    _STATS = {'absorbed': 0, 'jump_sum': 0.0, 'jump_sq_sum': 0.0}

    n_turn = _regime_to_turns(regime, jumps_before_turn)
    capture_a = 10.0 * capture_nm

    # Decomposition grid decoupled from the physical lattice: one cell per lattice site
    # (300x350x400 = 4.2e7 cells) would make the ghost-cell setup take minutes; the physics
    # only needs the box size, which stays exactly grid * a0.
    if any(g % DOMAIN_COARSEN != 0 for g in grid):
        raise ValueError(f"grid {grid} must be divisible by {DOMAIN_COARSEN}")
    box_size = np.array(grid, dtype=np.float64) * A0_FE_A
    volume_m3 = float(np.prod(box_size)) * 1.0e-30
    n_sinks = max(1, int(round(density_cm3 * 1.0e6 * volume_m3)))
    actual_density_m3 = n_sinks / volume_m3

    # the case parameters live on the event classes
    SIAMigration.jumps_before_turn = n_turn
    SphereAbsorption.max_distance = capture_a

    outputs = {'event_hooks': [_absorption_hook]}
    if report_interval > 0:
        outputs['interval'] = max(1, int(report_interval))
        outputs['postprocessors'] = [_make_reporter(int(target_absorbed))]
    if dump_dir:
        # per-rank OVITO dump every `dump_interval` steps plus a final frame;
        # it shares the post-processing interval
        outputs['interval'] = max(1, int(dump_interval or report_interval or 1))
        outputs['ovito_dump'] = {'enabled': True, 'output_dir': dump_dir, 'prefix': 'e1',
                                 'mode': 'distributed',
                                 'properties': ['id', 'type', 'x', 'y', 'z', 'size', 'jump_count']}

    config = {
        'domain': {
            'lattice_constant': DOMAIN_COARSEN * A0_FE_A,
            'lattice_grid': [g // DOMAIN_COARSEN for g in grid],
            'h_factor': 1,
            'periodic': [True, True, True],
            'max_interaction_distance': capture_a + 1.0e-6,
        },
        'simulation': {
            'max_steps': int(max_steps),
            'max_time': 1.0e30,
            'progress_interval': int(progress_interval),
            'verbose': False,
            'random_seed': int(random_seed),
            'rma_object_pool': {'initial_capacity': max(64, n_sinks + target_absorbed + 8), 'auto_expand': True},
            'rma_event_pool': {'capacity_zero': 1, 'capacity_single': max(64, int(target_absorbed) + 8),
                               'capacity_pair': max(256, int(target_absorbed)), 'auto_expand': True},
            'predictive_rebalance': {'enabled': False},
            'adaptive_weight': {'enabled': False},
            'runtime_load_balancer': {'enabled': False},
            'object_load_balancer': {'enabled': False},
        },
        'objects': {
            'SIA': {'size': A0_FE_A,
                    'properties': {'jump_count': 0.0, 'axis_idx': -1.0, 'segment_left': 0.0}},
            'S': {'size': 2.0 * capture_a, 'properties': {'radius': capture_a}},   # inert absorber
        },
        'ic': {'mode': 'function', 'func': _create_ic_func(n_sinks, int(target_absorbed), capture_a, random_seed)},
        'events': {'sia_migration': SIAMigration, 'sphere_absorption': SphereAbsorption},
        'physics': {'params': {'temperature': 573.0}},
        'outputs': outputs,
    }
    # what the header and the summary report
    config['_case'] = dict(box_size=box_size, num_defects=int(target_absorbed), jumps_before_turn=n_turn,
                           regime=regime, capture_a=capture_a, n_sinks=n_sinks, density_m3=actual_density_m3)
    return config


# ============================================================ reporting

def _theory_k2_m2(regime: str, capture_m: float, density_m3: float, jumps_before_turn: int) -> Dict[str, float]:
    """Theoretical sink strengths (paper Eqs. (2), (3), (4))."""
    k2_3_1st = 4.0 * math.pi * density_m3 * capture_m
    k2_3 = k2_3_1st * (1.0 + capture_m * math.sqrt(k2_3_1st))
    k2_1 = 6.0 * (math.pi * capture_m**2 * density_m3) ** 2
    out = {'k2_3d_1st': k2_3_1st, 'k2_3d_2nd': k2_3, 'k2_1d': k2_1}
    if regime == 'mixed':
        lch_m = JUMP_DISTANCE_A * 1.0e-10 * math.sqrt(float(jumps_before_turn))
        x2 = lch_m**2 * k2_1 / 12.0 + k2_1**2 / k2_3**2
        out['k2_master'] = 0.5 * (1.0 + math.sqrt(1.0 + 4.0 / x2)) * k2_1
    return out


def _print_header(case: Dict, comm: MPI.Comm):
    if comm.rank != 0:
        return
    fv = (4.0 / 3.0) * math.pi * (case['capture_a'] * 1e-10) ** 3 * case['density_m3']
    box = case['box_size']
    print("=" * 72)
    print("e1: spherical absorber sink strength (Malerba et al., JNM 360 (2007))")
    print("=" * 72)
    print(f"regime={case['regime']}  jumps_before_turn={case['jumps_before_turn']}  "
          f"R(capture)={case['capture_a'] / 10.0:.3f} nm")
    print(f"N={case['density_m3'] * 1e-6:.4e} cm^-3  n_sinks={case['n_sinks']}  "
          f"fV={fv:.4e}  defects={case['num_defects']}")
    print(f"box={box[0]:.1f} x {box[1]:.1f} x {box[2]:.1f} A^3  dj={JUMP_DISTANCE_A:.4f} A")
    print("=" * 72)


def _print_summary(case: Dict, stats: Dict, comm: MPI.Comm):
    total_abs = int(_STATS['absorbed'])          # replicated on every rank by the event hook
    jump_sum = float(_STATS['jump_sum'])
    jump_sq = float(_STATS['jump_sq_sum'])
    if comm.rank != 0:
        return
    print("")
    print("Results")
    print("-" * 72)
    print(f"absorbed = {total_abs} / {case['num_defects']}")
    if total_abs == 0:
        print("no absorption events: k^2 unavailable")
        return
    avg = jump_sum / total_abs
    var = max(0.0, jump_sq / total_abs - avg * avg)
    sem = math.sqrt(var / total_abs)
    dj_m = JUMP_DISTANCE_A * 1.0e-10
    k2 = 6.0 / (dj_m * dj_m * avg)
    k2_err = k2 * sem / avg
    print(f"<nj> = {avg:.6e} +/- {sem:.2e}")
    print(f"k^2(sim) = {k2:.6e} m^-2  (+/- {k2_err:.2e})")

    th = _theory_k2_m2(case['regime'], case['capture_a'] * 1e-10, case['density_m3'], case['jumps_before_turn'])
    if case['regime'] == '3d':
        ref = th['k2_3d_2nd']
        print(f"k^2(theory, Eq.2 1st order) = {th['k2_3d_1st']:.6e} m^-2")
        print(f"k^2(theory, Eq.2 2nd order) = {ref:.6e} m^-2")
    elif case['regime'] == '1d':
        ref = th['k2_1d']
        print(f"k^2(theory, Eq.3) = {ref:.6e} m^-2")
    else:
        ref = th['k2_master']
        print(f"k^2(theory 3D 2nd) = {th['k2_3d_2nd']:.6e} m^-2")
        print(f"k^2(theory 1D)     = {th['k2_1d']:.6e} m^-2")
        print(f"k^2(master curve)  = {ref:.6e} m^-2")
    print(f"relative deviation vs reference = {100.0 * (k2 - ref) / ref:+.2f} %")
    print(f"engine_steps = {stats.get('total_steps', 'n/a')}")
    print("-" * 72)


def parse_args():
    p = argparse.ArgumentParser(description="e1: spherical absorber sink strength")
    p.add_argument('--regime', choices=['3d', '1d', 'mixed'], default='3d')
    p.add_argument('--jumps-before-turn', type=int, default=None)
    p.add_argument('--R-nm', type=float, default=6.25, help="capture distance in nm (paper values 0.75...10.25)")
    p.add_argument('--N-cm3', type=float, default=1.5e17, help="absorber density in cm^-3 (paper: 1e16...1.5e17)")
    p.add_argument('--target-absorbed', type=int, default=1000)
    p.add_argument('--grid', type=int, nargs=3, default=[300, 350, 400], metavar=('NX', 'NY', 'NZ'))
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--progress-interval', type=int, default=0)
    p.add_argument('--report-interval', type=int, default=100000)
    p.add_argument('--dump-dir', type=str, default=None,
                   help='write OVITO/LAMMPS dump frames (one file per rank) into this directory')
    p.add_argument('--dump-interval', type=int, default=0,
                   help='steps between dump frames (0: use --report-interval); a final frame is always written')
    p.add_argument('--max-steps', type=int, default=2 * 10**8)
    return p.parse_args()


def main():
    args = parse_args()
    comm = MPI.COMM_WORLD

    config = create_config(regime=args.regime, capture_nm=args.R_nm, density_cm3=args.N_cm3,
                           target_absorbed=args.target_absorbed, grid=tuple(args.grid),
                           random_seed=args.seed, progress_interval=args.progress_interval,
                           report_interval=args.report_interval, dump_dir=args.dump_dir,
                           dump_interval=args.dump_interval, max_steps=args.max_steps,
                           jumps_before_turn=args.jumps_before_turn)
    case = config.pop('_case')
    _print_header(case, comm)

    engine = OKMCengine(config)
    try:
        stats = engine.run()
    finally:
        engine.destroy()
    _print_summary(case, stats, comm)


if __name__ == '__main__':
    main()
