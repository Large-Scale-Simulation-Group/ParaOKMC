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
e2: Sink strength of a straight dislocation (3D-migrating defects).

Reference (all parameters taken from the paper itself):
    V. Jansson, L. Malerba, A. De Backer, C.S. Becquart, C. Domain,
    "Sink strength calculations of dislocations and loops using OKMC",
    Journal of Nuclear Materials 442 (2013) 218-226.

What is validated
-----------------
A single SIA species performs 3D jump-by-jump migration in a bcc Fe box
(lx = 100 a0 fixed; ly, lz set the dislocation density rho_d = 1/(ly*lz)).
Each jump is an independent, uniformly random <111> step (fully 3D motion).
The straight dislocation runs along x through the box centre: with periodic
boundary conditions this is a regular array of infinite parallel
dislocations, exactly as in the paper. The measured sink strength

    k^2 = 2*3 / (dj^2 * <nj>)          (n = 3 convention)

is compared against the 3D Wiedersich expression (Jansson et al. 2013, Table 1):

    k^2 = 2*pi*rho_d*(1-q^2) / (ln(1/q) - 3/4 + q^2*(1 - q^2/4)),
    q   = rd*sqrt(pi*rho_d)

rd is the capture radius of the dislocation (the defect capture radius is
already folded in, as in the paper's rd values ending in .25 nm).

How the model is written
------------------------
Two event classes carry the whole physics; everything an event needs is a
class attribute or a method of that class:

- SIAMigration (single-object): the Arrhenius rate is a Numba kernel defined
  inside the class (`rate_kernel`, fed by `rate_deps`); execute() makes one
  random <111> jump and increments the object's `jump_count` property.
- DislocationAbsorption (pair-object): the dislocation is ONE inert object of
  type 'DISL' at the box centre. `max_distance` is a sphere that contains the
  whole capture cylinder (periodic along x); the cylindrical condition itself
  is the rate: huge inside the capture radius, zero outside. It is written as
  a plain Python rate() on purpose; the same class in the compiled
  rate_deps + rate_kernel form follows it, commented out (see the note there).

The box follows from config['grid_size'] * config['lattice_constant'];
the capture radius and the dislocation axis are set on the absorption class
by create_config(). Physical constants from the paper: a0 = 2.87 A,
dj = sqrt(3)/2*a0, nu = 8.071e13 s^-1, Em = 0.31 eV, T = 573 K.

The jump statistics are collected by a post-processor event hook
(outputs.event_hooks): the engine calls it on every rank right after the
broadcast state-change message of a step has been applied, and a deleted
defect record still carries its jump_count, so every rank holds the same
totals and no MPI reduction is needed. The run ends when the last defect is
absorbed (total rate zero).

Run:
    python run_dislocation_sink_strength.py --rho-m2 1e15 --rd-nm 6.25
    mpirun -np 2 python run_dislocation_sink_strength.py --rho-m2 5e14 --rd-nm 4.25
"""

from __future__ import annotations

import argparse
import math
from typing import Dict

import numpy as np
from mpi4py import MPI
from numba import njit

from paraokmc import OKMCengine
from paraokmc import SingleObjectEvent, PairObjectEvent
from paraokmc import change_obj, delete_obj
from paraokmc import RateDeps


A0_FE_A = 2.87                                    # bcc Fe lattice parameter [A]
JUMP_DISTANCE_A = math.sqrt(3.0) * A0_FE_A / 2.0  # dj = sqrt(3)/2 a0 [A]
DOMAIN_COARSEN = 5                                # decomposition cell = 5 a0 (granularity only)

_STATS = {'absorbed': 0, 'jump_sum': 0.0, 'jump_sq_sum': 0.0}


def _box_size(config: Dict) -> np.ndarray:
    """Simulation box [A] from the domain block the engine passes along."""
    return np.asarray(config['grid_size'], dtype=np.float64) * float(config['lattice_constant'])


# ============================================================ events

class SIAMigration(SingleObjectEvent):
    """3D migration: every jump is an independent random <111> step."""

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

    # --- rate: compiled kernel, evaluated for the whole event array at once ---
    rate_deps = RateDeps(physics=['temperature'])

    @staticmethod
    @njit
    def rate_kernel(temperature):
        return 8.071e13 * np.exp(-0.31 / (8.617333262145e-5 * temperature))

    # --- consequence ---
    @classmethod
    def execute(cls, event, objs, config):
        obj, rng = objs[0], config['rng']
        axis = cls.AXES_111[rng.randint(0, 4)]
        sign = 1.0 if rng.random() >= 0.5 else -1.0
        step = sign * 0.5 * A0_FE_A * axis
        box = _box_size(config)
        new_pos = (np.array([obj['x'], obj['y'], obj['z']], dtype=np.float64) + step) % box

        props = np.array(obj['float_props'], copy=True)
        mapping = config['_props_mapping']['float_props']['SIA']
        props[mapping['jump_count']] += 1.0
        if 'pos_y' in mapping:             # only present with the compiled DislocationAbsorption below
            props[mapping['pos_y']] = new_pos[1]
            props[mapping['pos_z']] = new_pos[2]
        return [change_obj(obj, x=new_pos[0], y=new_pos[1], z=new_pos[2], float_props=props.tolist())]


class DislocationAbsorption(PairObjectEvent):
    """Absorption of an SIA by the straight dislocation (inert 'DISL' object).

    A candidate pair exists for every SIA within `max_distance` of the
    dislocation object; the rate applies the actual cylindrical condition.
    """

    type_pairs = [('SIA', 'DISL')]
    # rate reuse: the rate depends only on the pair geometry, which changes only when
    # one of the two objects moves, so untouched pairs keep their rate
    incremental_rate = True
    ABSORB_RATE_HZ = 1.0e18            # >> jump rate: absorption on contact

    # case geometry, set once by create_config()
    max_distance = 1.0                 # radius of a sphere containing the capture cylinder
    capture_a = 1.0                    # capture radius rd [A]
    axis_yz = (0.0, 0.0)               # the dislocation line runs along x through this (y, z)

    @classmethod
    def inside_capture_cylinder(cls, x: float, y: float, z: float, box) -> bool:
        """Minimum-image planar distance to the dislocation axis <= rd."""
        dy = y - cls.axis_yz[0]
        dz = z - cls.axis_yz[1]
        dy -= box[1] * round(dy / box[1])
        dz -= box[2] * round(dz / box[2])
        return math.hypot(dy, dz) <= cls.capture_a

    @staticmethod
    def sia_of(objs, config):
        """The SIA member of the pair (the other one is the dislocation)."""
        t = objs[0]['type']
        is_sia = (int(t) == config['_type_to_int']['SIA']) if isinstance(t, (int, np.integer)) else (str(t) == 'SIA')
        return objs[0] if is_sia else objs[1]

    @classmethod
    def rate(cls, objs, params, config):
        sia = cls.sia_of(objs, config)
        if cls.inside_capture_cylinder(float(sia['x']), float(sia['y']), float(sia['z']), _box_size(config)):
            return cls.ABSORB_RATE_HZ
        return 0.0

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
#     physics scalars. obj1 is always the first type in `type_pairs`.
# The example keeps the Python form above so that both forms appear side by
# side (SIAMigration uses the compiled one). Here the compiled form buys no
# time, because every SIA has at most one candidate pair; with large pair
# catalogues (e.g. clustering) it is the faster path.
#
# A kernel does not receive x, y, z. The transverse coordinates it needs are
# therefore mirrored into object properties: pos_y/pos_z of the SIA (kept
# current by SIAMigration.execute), axis_y/axis_z of the dislocation, and the
# box cross-section as two physics scalars. create_config() and the initial
# condition add these automatically when this class is active. To try it,
# comment out the class above and uncomment this one; the trajectory and the
# measured k^2 are identical (checked on the published cases).
# ---------------------------------------------------------------------------
# class DislocationAbsorption(PairObjectEvent):
#     """Cylindrical capture test as a compiled kernel (obj1 = SIA, obj2 = DISL)."""
#
#     type_pairs = [('SIA', 'DISL')]
#     # rate reuse: the rate depends only on the pair geometry, which changes only when
#     # one of the two objects moves, so untouched pairs keep their rate
#     incremental_rate = True
#     ABSORB_RATE_HZ = 1.0e18
#
#     # case geometry, set once by create_config() (also used by the initial condition)
#     max_distance = 1.0
#     capture_a = 1.0
#     axis_yz = (0.0, 0.0)
#
#     # the data the kernel needs, in the order the kernel receives them:
#     # obj1 properties, obj2 properties, distance, fields, physics
#     rate_deps = RateDeps(obj1=['pos_y', 'pos_z'],
#                          obj2=['axis_y', 'axis_z', 'radius'],
#                          physics=['box_y', 'box_z'])
#
#     @staticmethod
#     @njit
#     def rate_kernel(pos_y, pos_z, axis_y, axis_z, radius, distance, box_y, box_z):
#         dy = pos_y - axis_y
#         dz = pos_z - axis_z
#         dy -= box_y * np.round(dy / box_y)      # minimum image in the cross-section
#         dz -= box_z * np.round(dz / box_z)
#         if dy * dy + dz * dz <= radius * radius:
#             return 1.0e18
#         return 0.0
#
#     @classmethod
#     def inside_capture_cylinder(cls, x: float, y: float, z: float, box) -> bool:
#         """Same test in Python, used by the initial condition."""
#         dy = y - cls.axis_yz[0]
#         dz = z - cls.axis_yz[1]
#         dy -= box[1] * round(dy / box[1])
#         dz -= box[2] * round(dz / box[2])
#         return math.hypot(dy, dz) <= cls.capture_a
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


def _uses_compiled_absorption() -> bool:
    """True when the commented-out compiled DislocationAbsorption is active."""
    return getattr(DislocationAbsorption, 'rate_kernel', None) is not None


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
            print(f"[e2] step={step:>9d} active={num_defects - total_abs} "
                  f"absorbed={total_abs}/{num_defects}", flush=True)
    return reporter


# ============================================================ initial condition

def _create_ic_func(n_defects: int, seed: int):
    """One inert dislocation at the box centre + defects placed at random
    outside the capture cylinder (the paper excludes defects created inside the sink)."""

    def initial_objects_func(rank, size, config):
        saved_state = np.random.get_state()
        np.random.seed(config.get('_user_seed', seed))
        domain_decomp = config['_domain_decomp']
        box = _box_size(config)
        n_float = config['_object_dtype'].fields['float_props'][0].shape[0]
        props_of = config['_props_mapping']['float_props']

        sia_props = np.zeros(n_float)
        disl_props = np.zeros(n_float)
        if 'radius' in props_of.get('DISL', {}):
            disl_props[props_of['DISL']['radius']] = DislocationAbsorption.capture_a
        mirrored = 'pos_y' in props_of.get('SIA', {})     # compiled-absorption variant only
        if mirrored:
            disl_props[props_of['DISL']['axis_y']] = DislocationAbsorption.axis_yz[0]
            disl_props[props_of['DISL']['axis_z']] = DislocationAbsorption.axis_yz[1]

        local_objects, owner_counters = [], {}

        def add(obj_type, pos, size_a, props):
            owner_rank = int(domain_decomp.get_owner_rank(list(pos)))
            owner_counters[owner_rank] = owner_counters.get(owner_rank, 0) + 1
            if owner_rank == rank:
                local_objects.append({'id': (owner_rank << 48) | owner_counters[owner_rank],
                                      'type': obj_type, 'x': float(pos[0]), 'y': float(pos[1]),
                                      'z': float(pos[2]), 'size': size_a, 'float_props': props.tolist()})

        add('DISL', 0.5 * box, 2.0 * DislocationAbsorption.capture_a, disl_props)
        placed = 0
        while placed < n_defects:
            pos = np.random.random(3) * box
            if DislocationAbsorption.inside_capture_cylinder(pos[0], pos[1], pos[2], box):
                continue
            if mirrored:
                sia_props[props_of['SIA']['pos_y']] = pos[1]
                sia_props[props_of['SIA']['pos_z']] = pos[2]
            add('SIA', pos, A0_FE_A, sia_props)
            placed += 1

        np.random.set_state(saved_state)
        return local_objects

    return initial_objects_func


# ============================================================ configuration

def create_config(rho_m2: float, rd_nm: float, target_absorbed: int, random_seed: int,
                  progress_interval: int, report_interval: int, dump_dir: str,
                  dump_interval: int, max_steps: int) -> Dict:
    global _STATS
    _STATS = {'absorbed': 0, 'jump_sum': 0.0, 'jump_sq_sum': 0.0}

    capture_a = 10.0 * rd_nm

    # box: lx = 100 a0 (paper); square cross-section ly = lz from the target density
    # (fine for 3D migration, which conserves no transverse coordinate)
    a0_m = A0_FE_A * 1.0e-10
    cross_cells = max(DOMAIN_COARSEN,
                      int(round(math.sqrt(1.0 / rho_m2) / a0_m / DOMAIN_COARSEN)) * DOMAIN_COARSEN)
    grid = (100, cross_cells, cross_cells)
    box_size = np.array(grid, dtype=np.float64) * A0_FE_A
    actual_rho = 1.0 / ((grid[1] * a0_m) * (grid[2] * a0_m))

    # pair search radius: a sphere that contains the whole capture cylinder (PBC in x)
    max_distance_a = math.hypot(0.5 * box_size[0], capture_a) + 1.0
    if max_distance_a >= 0.5 * min(box_size[1], box_size[2]):
        raise ValueError(f"pair search radius {max_distance_a:.1f} A exceeds half the box "
                         f"cross-section; decrease rd or rho (minimum image would be ambiguous)")

    # the case geometry lives on the event class
    DislocationAbsorption.max_distance = max_distance_a
    DislocationAbsorption.capture_a = capture_a
    DislocationAbsorption.axis_yz = (0.5 * box_size[1], 0.5 * box_size[2])

    outputs = {'event_hooks': [_absorption_hook]}
    if report_interval > 0:
        outputs['interval'] = max(1, int(report_interval))
        outputs['postprocessors'] = [_make_reporter(int(target_absorbed))]
    if dump_dir:
        # per-rank OVITO dump every `dump_interval` steps plus a final frame;
        # it shares the post-processing interval
        outputs['interval'] = max(1, int(dump_interval or report_interval or 1))
        outputs['ovito_dump'] = {'enabled': True, 'output_dir': dump_dir, 'prefix': 'e2',
                                 'mode': 'distributed',
                                 'properties': ['id', 'type', 'x', 'y', 'z', 'size', 'jump_count']}

    config = {
        'domain': {
            'lattice_constant': DOMAIN_COARSEN * A0_FE_A,
            'lattice_grid': [g // DOMAIN_COARSEN for g in grid],
            'h_factor': 1,
            'periodic': [True, True, True],
            'max_interaction_distance': max_distance_a + 1.0e-6,
        },
        'simulation': {
            'max_steps': int(max_steps),
            'max_time': 1.0e30,
            'progress_interval': int(progress_interval),
            'verbose': False,
            'random_seed': int(random_seed),
            'rma_object_pool': {'initial_capacity': max(64, target_absorbed + 8), 'auto_expand': True},
            'rma_event_pool': {'capacity_zero': 1, 'capacity_single': max(64, int(target_absorbed) + 8),
                               'capacity_pair': max(256, int(target_absorbed)), 'auto_expand': True},
            'predictive_rebalance': {'enabled': False},
            'adaptive_weight': {'enabled': False},
            'runtime_load_balancer': {'enabled': False},
            'object_load_balancer': {'enabled': False},
        },
        'objects': {
            'SIA': {'size': A0_FE_A, 'properties': {'jump_count': 0.0}},
            'DISL': {'size': 2.0 * capture_a, 'properties': {'radius': capture_a}},   # inert
        },
        'ic': {'mode': 'function', 'func': _create_ic_func(int(target_absorbed), random_seed)},
        'events': {'sia_migration': SIAMigration, 'dislocation_absorption': DislocationAbsorption},
        'physics': {'params': {'temperature': 573.0}},
        'outputs': outputs,
    }
    if _uses_compiled_absorption():
        # the kernel cannot read x/y/z: mirror the transverse coordinates into properties
        config['objects']['SIA']['properties'].update({'pos_y': 0.0, 'pos_z': 0.0})
        config['objects']['DISL']['properties'].update({'axis_y': 0.5 * box_size[1], 'axis_z': 0.5 * box_size[2]})
        config['physics']['params'].update({'box_y': float(box_size[1]), 'box_z': float(box_size[2])})
    # what the header and the summary report
    config['_case'] = dict(box_size=box_size, num_defects=int(target_absorbed), capture_a=capture_a,
                           rho_target_m2=rho_m2, rho_actual_m2=actual_rho, max_distance_a=max_distance_a)
    return config


# ============================================================ reporting

def _theory_k2_m2(rd_m: float, rho_m2: float) -> Dict[str, float]:
    """Wiedersich 3D expression (Jansson et al. 2013, Table 1)."""
    q = rd_m * math.sqrt(math.pi * rho_m2)
    wiedersich = 2.0 * math.pi * rho_m2 * (1.0 - q * q) / (math.log(1.0 / q) - 0.75 + q * q * (1.0 - 0.25 * q * q))
    return {'q': q, 'k2_3d_wiedersich': wiedersich}


def _print_header(case: Dict, comm: MPI.Comm):
    if comm.rank != 0:
        return
    box = case['box_size']
    print("=" * 72)
    print("e2: straight dislocation sink strength, 3D migration")
    print("    (Jansson et al., JNM 442 (2013))")
    print("=" * 72)
    print(f"rd(capture)={case['capture_a'] / 10.0:.3f} nm  defects={case['num_defects']}")
    print(f"rho_d(target)={case['rho_target_m2']:.3e} m^-2  rho_d(actual)={case['rho_actual_m2']:.3e} m^-2")
    print(f"box={box[0]:.1f} x {box[1]:.1f} x {box[2]:.1f} A^3  pair_search_radius={case['max_distance_a']:.1f} A")
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
    th = _theory_k2_m2(case['capture_a'] * 1e-10, case['rho_actual_m2'])
    ref = th['k2_3d_wiedersich']
    print(f"k^2(theory, Wiedersich) = {ref:.6e} m^-2  (q={th['q']:.4f})")
    print(f"relative deviation vs reference = {100.0 * (k2 - ref) / ref:+.2f} %")
    print(f"engine_steps = {stats.get('total_steps', 'n/a')}")
    print("-" * 72)


def parse_args():
    p = argparse.ArgumentParser(description="e2: straight dislocation sink strength (3D migration)")
    p.add_argument('--rd-nm', type=float, default=6.25, help="dislocation capture radius in nm (paper: 2.25...9.25)")
    p.add_argument('--rho-m2', type=float, default=1.0e15, help="dislocation density in m^-2 (paper: 5e13...1e15)")
    p.add_argument('--target-absorbed', type=int, default=500)
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

    config = create_config(rho_m2=args.rho_m2, rd_nm=args.rd_nm, target_absorbed=args.target_absorbed,
                           random_seed=args.seed, progress_interval=args.progress_interval,
                           report_interval=args.report_interval, dump_dir=args.dump_dir,
                           dump_interval=args.dump_interval, max_steps=args.max_steps)
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
