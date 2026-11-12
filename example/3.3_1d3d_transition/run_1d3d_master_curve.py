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
e3: 1D-to-3D diffusion-reaction master curve.

Reference (all parameters taken from the paper itself):
    H. Trinkaus, H.L. Heinisch, A.V. Barashev, S.I. Golubov, B.N. Singh,
    "1D to 3D diffusion-reaction kinetics of defects in crystals",
    Physical Review B 66, 060105(R) (2002).

What is validated
-----------------
A point defect hops site to site along a randomly chosen close-packed
<110> direction of an fcc lattice (lattice parameter of Cu), performing a
1D random walk and switching to another randomly chosen direction after a
fixed number of hops n_ch (n_ch = 1 is the 3D limit). It is absorbed by
randomly placed spherical sinks of capture radius R and number density N
(paper: N = 1e22 m^-3, box 1 um^3, periodic boundaries). The average
number of jumps before absorption <j> gives the sink strength

    k^2 = 6 / (a^2 * <j>)        (D = D_lo/3 convention, as in the paper)

One invocation runs one n_ch; sweeping n_ch = 1...1e6 and plotting
    y = k^2 / k^2_(1)   vs   x, x^2 = l_ch^2 k^2_(1)/12 + k^4_(1)/k^4_(3)
reproduces the paper's Fig. 2 master curve y = (1 + sqrt(1 + 4/x^2))/2,
with l_ch = a*sqrt(n_ch), k^2_(1) = 6*(pi*R^2*N)^2 (paper Eq. (8a)) and
k^2_(3) taken from the measured 3D limit (n_ch = 1), exactly as the paper
prescribes to account for lattice discreteness.

How the model is written
------------------------
Two event classes carry the whole physics; everything an event needs is a
class attribute or a method of that class:

- DefectHop (single-object): a <110> hop of length a = a0/sqrt(2); the axis
  is redrawn every n_ch hops. The (constant) hop rate is a Numba kernel
  defined inside the class; execute() tracks the current axis and the hops
  left on it in two object properties and increments `jump_count`.
- SphereAbsorption (pair-object): sinks are inert objects of type 'S';
  `max_distance` is the capture radius R and the rate is huge, so a defect
  is absorbed by the first hop that brings it within R. The rate is a plain
  Python rate() on purpose; the same class in the compiled rate_deps +
  rate_kernel form follows it, commented out (see the note there).

The box follows from config['grid_size'] * config['lattice_constant']; the
decomposition cell width (100 A) is decoupled from the hop length.

The jump statistics are collected by a post-processor event hook
(outputs.event_hooks): the engine calls it on every rank right after the
broadcast state-change message of a step has been applied, and a deleted
defect record still carries its jump_count, so every rank holds the same
totals and no MPI reduction is needed. The run ends when the last defect is
absorbed (total rate zero).

Run:
    python run_1d3d_master_curve.py --n-ch 1                      # 3D limit
    python run_1d3d_master_curve.py --n-ch 1000 --k3-sim-m2 <k2 from n_ch=1>
    mpirun -np 4 python run_1d3d_master_curve.py --n-ch 100000
"""

from __future__ import annotations

import argparse
import math
import os
from typing import Dict, Tuple

import numpy as np
from mpi4py import MPI
from numba import njit

from paraokmc import OKMCengine
from paraokmc import SingleObjectEvent, PairObjectEvent
from paraokmc import change_obj, delete_obj
from paraokmc import RateDeps


A0_CU_A = 3.615                              # fcc Cu lattice parameter [A]
JUMP_DISTANCE_A = A0_CU_A / math.sqrt(2.0)   # nearest-neighbour hop along <110> [A]
DOMAIN_CELL_A = 100.0                        # decomposition cell width [A]

_STATS = {'absorbed': 0, 'jump_sum': 0.0, 'jump_sq_sum': 0.0}


def _box_size(config: Dict) -> np.ndarray:
    """Simulation box [A] from the domain block the engine passes along."""
    return np.asarray(config['grid_size'], dtype=np.float64) * float(config['lattice_constant'])


# ============================================================ events

class DefectHop(SingleObjectEvent):
    """1D random walk along a <110> axis, redrawn every n_ch hops (1 = 3D)."""

    object_types = ['DEF']
    # rate reuse: the rate depends only on a constant parameter, so an event whose
    # object did not move keeps its rate instead of being recomputed every step
    incremental_rate = True

    ATTEMPT_HZ = 1.0e13                # arbitrary: k^2 depends on the jump statistics only

    # the six <110> axes; a hop is +/- axis * a0/2 (12 nearest-neighbour directions)
    AXES_110 = np.array([[1, 1, 0], [1, -1, 0], [1, 0, 1], [1, 0, -1], [0, 1, 1], [0, 1, -1]], dtype=np.float64)

    n_ch = 1                           # set by create_config()

    # --- rate: compiled kernel, evaluated for the whole event array at once ---
    rate_deps = RateDeps(physics=['temperature'])

    @staticmethod
    @njit
    def rate_kernel(temperature):
        return 1.0e13 + 0.0 * temperature

    # --- axis bookkeeping in the object properties axis_idx / segment_left ---
    @classmethod
    def next_axis(cls, props, mapping, rng) -> Tuple[int, int]:
        n_ch = int(cls.n_ch)
        if n_ch <= 1:
            return int(rng.randint(0, 6)), 0
        axis_idx = int(round(props[mapping['axis_idx']]))
        segment_left = int(round(props[mapping['segment_left']]))
        if axis_idx < 0 or axis_idx >= 6 or segment_left <= 0:
            axis_idx, segment_left = int(rng.randint(0, 6)), n_ch
        return axis_idx, segment_left - 1

    # --- consequence ---
    @classmethod
    def execute(cls, event, objs, config):
        obj, rng = objs[0], config['rng']
        mapping = config['_props_mapping']['float_props']['DEF']
        props = np.array(obj['float_props'], copy=True)
        axis_idx, segment_left = cls.next_axis(props, mapping, rng)
        sign = 1.0 if rng.random() >= 0.5 else -1.0
        step = sign * 0.5 * A0_CU_A * cls.AXES_110[axis_idx]
        new_pos = (np.array([obj['x'], obj['y'], obj['z']], dtype=np.float64) + step) % _box_size(config)

        props[mapping['jump_count']] += 1.0
        props[mapping['axis_idx']] = float(axis_idx)
        props[mapping['segment_left']] = float(segment_left)
        return [change_obj(obj, x=new_pos[0], y=new_pos[1], z=new_pos[2], float_props=props.tolist())]


class SphereAbsorption(PairObjectEvent):
    """Instantaneous absorption by an inert spherical sink ('S')."""

    type_pairs = [('DEF', 'S')]
    # rate reuse: the rate depends only on the pair geometry, which changes only when
    # one of the two objects moves, so untouched pairs keep their rate
    incremental_rate = True
    ABSORB_RATE_HZ = 1.0e18
    max_distance = 1.0                 # = capture radius R, set by create_config()

    @staticmethod
    def defect_of(objs, config):
        """The defect member of the pair (the other one is the sink)."""
        t = objs[0]['type']
        is_def = (int(t) == config['_type_to_int']['DEF']) if isinstance(t, (int, np.integer)) else (str(t) == 'DEF')
        return objs[0] if is_def else objs[1]

    @classmethod
    def rate(cls, objs, params, config):
        return cls.ABSORB_RATE_HZ

    @classmethod
    def execute(cls, event, objs, config):
        return [delete_obj(cls.defect_of(objs, config))]


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
# side (DefectHop uses the compiled one). A pair kernel that needs no object
# property must say kind='pair' explicitly, because RateDeps infers the event
# kind from the declared properties. To try it, comment out the class above
# and uncomment this one; the trajectory is identical.
# ---------------------------------------------------------------------------
# class SphereAbsorption(PairObjectEvent):
#     """Instantaneous absorption, rate as a compiled kernel (obj1 = DEF, obj2 = S)."""
#
#     type_pairs = [('DEF', 'S')]
#     # rate reuse: the rate depends only on the pair geometry, which changes only when
#     # one of the two objects moves, so untouched pairs keep their rate
#     incremental_rate = True
#     ABSORB_RATE_HZ = 1.0e18
#     max_distance = 1.0                 # = capture radius R, set by create_config()
#
#     rate_deps = RateDeps(kind='pair')  # nothing but the pair distance is needed
#
#     @staticmethod
#     @njit
#     def rate_kernel(distance):
#         return 1.0e18                  # the pair exists only within max_distance
#
#     @staticmethod
#     def defect_of(objs, config):
#         t = objs[0]['type']
#         is_def = (int(t) == config['_type_to_int']['DEF']) if isinstance(t, (int, np.integer)) else (str(t) == 'DEF')
#         return objs[0] if is_def else objs[1]
#
#     @classmethod
#     def execute(cls, event, objs, config):
#         return [delete_obj(cls.defect_of(objs, config))]


# ============================================================ observables

def _absorption_hook(event_type, changed, step, time_value, pool, comm, config):
    """Event hook: accumulate the jump count of every absorbed defect (same totals on every rank)."""
    if len(changed) == 0:
        return
    deleted = changed[~changed['active']]
    if len(deleted) == 0:
        return
    def_type = config['_type_to_int']['DEF']
    idx = config['_props_mapping']['float_props']['DEF']['jump_count']
    for rec in deleted:
        if int(rec['type_id']) != def_type:
            continue
        nj = float(rec['float_props'][idx])
        _STATS['absorbed'] += 1
        _STATS['jump_sum'] += nj
        _STATS['jump_sq_sum'] += nj * nj


def _make_reporter(num_defects: int):
    def reporter(step, time_value, pool, comm):
        total_abs = int(_STATS['absorbed'])
        if comm.rank == 0:
            print(f"[e3] step={step:>9d} active={num_defects - total_abs} "
                  f"absorbed={total_abs}/{num_defects}", flush=True)
    return reporter


# ============================================================ initial condition

def _create_ic_func(n_sinks: int, n_defects: int, capture_a: float, seed: int):
    """Sinks: purely random positions, as the paper states ("randomly arranged
    spherical sinks", no non-overlap requirement). Defects: random, outside every
    capture sphere."""

    def initial_objects_func(rank, size, config):
        saved_state = np.random.get_state()
        np.random.seed(config.get('_user_seed', seed))
        domain_decomp = config['_domain_decomp']
        box = _box_size(config)

        def min_image(delta):
            return delta - box * np.round(delta / box)

        centers = np.random.random((n_sinks, 3)) * box
        defect_pos = []
        while len(defect_pos) < n_defects:
            pos = np.random.random(3) * box
            if np.min(np.linalg.norm(min_image(pos[None, :] - centers), axis=1)) > capture_a:
                defect_pos.append(pos)

        n_float = config['_object_dtype'].fields['float_props'][0].shape[0]
        props_of = config['_props_mapping']['float_props']
        def_props = np.zeros(n_float)
        def_props[props_of['DEF']['axis_idx']] = -1.0
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
            add('DEF', pos, JUMP_DISTANCE_A, def_props)

        np.random.set_state(saved_state)
        return local_objects

    return initial_objects_func


# ============================================================ configuration

def create_config(n_ch: int, capture_nm: float, density_m3: float, box_nm: Tuple[float, float, float],
                  target_absorbed: int, random_seed: int, progress_interval: int, report_interval: int,
                  dump_dir: str, dump_interval: int, max_steps: int) -> Dict:
    global _STATS
    _STATS = {'absorbed': 0, 'jump_sum': 0.0, 'jump_sq_sum': 0.0}

    capture_a = 10.0 * capture_nm
    grid = tuple(max(4, int(round(b * 10.0 / DOMAIN_CELL_A))) for b in box_nm)
    box_size = np.array(grid, dtype=np.float64) * DOMAIN_CELL_A
    volume_m3 = float(np.prod(box_size)) * 1.0e-30
    n_sinks = max(1, int(round(density_m3 * volume_m3)))
    actual_density = n_sinks / volume_m3

    # the case parameters live on the event classes
    DefectHop.n_ch = int(n_ch)
    SphereAbsorption.max_distance = capture_a

    outputs = {'event_hooks': [_absorption_hook]}
    if report_interval > 0:
        outputs['interval'] = max(1, int(report_interval))
        outputs['postprocessors'] = [_make_reporter(int(target_absorbed))]
    if dump_dir:
        # per-rank OVITO dump every `dump_interval` steps plus a final frame;
        # it shares the post-processing interval
        outputs['interval'] = max(1, int(dump_interval or report_interval or 1))
        outputs['ovito_dump'] = {'enabled': True, 'output_dir': dump_dir, 'prefix': 'e3',
                                 'mode': 'distributed',
                                 'properties': ['id', 'type', 'x', 'y', 'z', 'size', 'jump_count']}

    config = {
        'domain': {
            'lattice_constant': DOMAIN_CELL_A,
            'lattice_grid': list(grid),
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
            'DEF': {'size': JUMP_DISTANCE_A,
                    'properties': {'jump_count': 0.0, 'axis_idx': -1.0, 'segment_left': 0.0}},
            'S': {'size': 2.0 * capture_a, 'properties': {'radius': capture_a}},   # inert sink
        },
        'ic': {'mode': 'function', 'func': _create_ic_func(n_sinks, int(target_absorbed), capture_a, random_seed)},
        'events': {'defect_hop': DefectHop, 'sphere_absorption': SphereAbsorption},
        'physics': {'params': {'temperature': 300.0}},
        'outputs': outputs,
    }
    # what the header and the summary report
    config['_case'] = dict(box_size=box_size, num_defects=int(target_absorbed), n_ch=int(n_ch),
                           capture_a=capture_a, n_sinks=n_sinks, density_m3=actual_density)
    return config


# ============================================================ reporting

def _print_header(case: Dict, comm: MPI.Comm):
    if comm.rank != 0:
        return
    box = case['box_size']
    print("=" * 72)
    print("e3: 1D-to-3D master curve (Trinkaus et al., PRB 66, 060105(R) (2002))")
    print("=" * 72)
    print(f"n_ch={case['n_ch']}  R={case['capture_a'] / 10.0:.3f} nm  "
          f"N={case['density_m3']:.4e} m^-3  n_sinks={case['n_sinks']}")
    print(f"box={box[0] / 10.0:.0f} x {box[1] / 10.0:.0f} x {box[2] / 10.0:.0f} nm^3  "
          f"a(hop)={JUMP_DISTANCE_A:.4f} A  defects={case['num_defects']}")
    print("=" * 72)


def _print_summary(case: Dict, stats: Dict, comm: MPI.Comm, k3_sim_m2: float | None, csv_path: str | None):
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
    a_m = JUMP_DISTANCE_A * 1.0e-10
    k2 = 6.0 / (a_m * a_m * avg)
    k2_err = k2 * sem / avg

    R_m = case['capture_a'] * 1.0e-10
    N_m3 = case['density_m3']
    k2_1 = 6.0 * (math.pi * R_m**2 * N_m3) ** 2            # paper Eq. (8a)
    k2_3_1st = 4.0 * math.pi * R_m * N_m3
    k2_3_analytic = k2_3_1st * (1.0 + R_m * math.sqrt(k2_3_1st))
    k2_3 = k3_sim_m2 if k3_sim_m2 is not None else k2_3_analytic

    lch_m = a_m * math.sqrt(float(case['n_ch']))
    x2 = lch_m**2 * k2_1 / 12.0 + k2_1**2 / k2_3**2
    x = math.sqrt(x2)
    y_sim = k2 / k2_1
    y_theory = 0.5 * (1.0 + math.sqrt(1.0 + 4.0 / x2))

    print(f"<j> = {avg:.6e} +/- {sem:.2e}")
    print(f"k^2(sim) = {k2:.6e} m^-2  (+/- {k2_err:.2e})")
    print(f"k^2_1D(theory, Eq.8a) = {k2_1:.6e} m^-2")
    print(f"k^2_3D(analytic)      = {k2_3_analytic:.6e} m^-2")
    if k3_sim_m2 is not None:
        print(f"k^2_3D(simulated)     = {k3_sim_m2:.6e} m^-2   [used for x]")
    print(f"master curve: x = {x:.6e}  y(sim) = {y_sim:.6e}  y(theory) = {y_theory:.6e}")
    if case['n_ch'] <= 1:
        print(f"relative deviation vs 3D reference = {100.0 * (k2 - k2_3) / k2_3:+.2f} %")
    else:
        print(f"relative deviation y(sim)/y(theory) - 1 = {100.0 * (y_sim / y_theory - 1.0):+.2f} %")
    print(f"engine_steps = {stats.get('total_steps', 'n/a')}")
    print("-" * 72)

    if csv_path:
        header = ("n_ch,R_nm,N_m3,absorbed,avg_jumps,sem_jumps,"
                  "k2_sim_m2,k2_err_m2,k2_1d_m2,k2_3d_used_m2,x,y_sim,y_theory\n")
        line = (f"{case['n_ch']},{case['capture_a'] / 10.0},{N_m3:.6e},{total_abs},"
                f"{avg:.6e},{sem:.6e},{k2:.6e},{k2_err:.6e},{k2_1:.6e},{k2_3:.6e},"
                f"{x:.6e},{y_sim:.6e},{y_theory:.6e}\n")
        write_header = not os.path.exists(csv_path)
        with open(csv_path, 'a', encoding='utf-8') as f:
            if write_header:
                f.write(header)
            f.write(line)
        print(f"appended to {csv_path}")


def parse_args():
    p = argparse.ArgumentParser(description="e3: Trinkaus 1D-to-3D master curve")
    p.add_argument('--n-ch', type=int, default=1, help="hops before direction change (paper: 1...1e6; 1 = 3D limit)")
    p.add_argument('--R-nm', type=float, default=10.0, help="sink capture radius in nm (fV = 4.2%% at N=1e22)")
    p.add_argument('--N-m3', type=float, default=1.0e22, help="sink number density in m^-3 (paper: 1e22)")
    p.add_argument('--box-nm', type=float, nargs=3, default=[1000.0, 1000.0, 1000.0], metavar=('LX', 'LY', 'LZ'),
                   help="box size in nm (paper: 1 um cube)")
    p.add_argument('--target-absorbed', type=int, default=1000)
    p.add_argument('--k3-sim-m2', type=float, default=None,
                   help="measured 3D-limit k^2 from an n_ch=1 run, used to normalize x as prescribed by the paper")
    p.add_argument('--csv', type=str, default=None, help="append one result line to this CSV (for the Fig. 2 sweep)")
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

    config = create_config(n_ch=args.n_ch, capture_nm=args.R_nm, density_m3=args.N_m3, box_nm=tuple(args.box_nm),
                           target_absorbed=args.target_absorbed, random_seed=args.seed,
                           progress_interval=args.progress_interval, report_interval=args.report_interval,
                           dump_dir=args.dump_dir, dump_interval=args.dump_interval, max_steps=args.max_steps)
    case = config.pop('_case')
    _print_header(case, comm)

    engine = OKMCengine(config)
    try:
        stats = engine.run()
    finally:
        engine.destroy()
    _print_summary(case, stats, comm, args.k3_sim_m2, args.csv)


if __name__ == '__main__':
    main()
