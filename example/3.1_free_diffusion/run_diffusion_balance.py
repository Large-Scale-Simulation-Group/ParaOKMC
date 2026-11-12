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
Pure diffusion benchmark with an exact analytical solution.

Reference:
    Martin-Bragado et al., "MMonCa: An Object Kinetic Monte Carlo simulator
    for damage irradiation evolution and defect diffusion"
    Computer Physics Communications 184 (2013) 2703-2710
    DOI: 10.1016/j.cpc.2013.07.011

Its Section 4.1 compares KMC diffusion against an exact theoretical solution in a
100 x 300 x 300 nm^3 periodic box with D = 100 nm^2/s. This e0 example keeps
the same physical setting but uses a narrow periodic slab initial condition
along x, which admits a clean Fourier-series analytical solution and is also
useful for object-load-balance checks.
"""

from __future__ import annotations

import argparse
import os
import math
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
from mpi4py import MPI
from numba import njit

from paraokmc import OKMCengine
from paraokmc.event_class import SingleObjectEvent, change_obj
from paraokmc.jit_rate_engine import RateDeps


ARTICLE_BOX_NM = (100.0, 300.0, 300.0)
ARTICLE_NUM_PARTICLES = 448_820
ARTICLE_DIFFUSIVITY_NM2_S = 100.0
ARTICLE_SIM_TIME_S = 10.0

DEFAULT_NUM_PARTICLES = 4096
DEFAULT_SEED_SLAB_WIDTH_NM = 10.0
DEFAULT_JUMP_LENGTH_NM = 5.0
DEFAULT_NUM_X_BINS = 32
DEFAULT_OUTPUT_DIR = "e0_out"

LATTICE_UNIT_NM = 1.0
LATTICE_CONSTANT_A = 10.0 * LATTICE_UNIT_NM

_JUMP_DIRS = np.array(
    [
        [1.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, -1.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, -1.0],
    ],
    dtype=np.float64,
)
_RUNTIME = {"jump_length_a": 0.0}


def jump_rate_from_diffusivity(diffusivity_nm2_s: float, jump_length_nm: float) -> float:
    """3D random walk relation: D = lambda^2 * Gamma / 6."""
    return 6.0 * float(diffusivity_nm2_s) / max(float(jump_length_nm) ** 2, 1.0e-12)


class TracerMigration(SingleObjectEvent):
    """Lattice jump of a non-interacting tracer: one of six <100> directions,
    at the constant rate 6D/l^2 held in the physics parameter jump_rate_hz."""

    object_types = ["Tracer"]
    params = []
    rate_deps = RateDeps(physics=["jump_rate_hz"])

    @staticmethod
    @njit
    def rate_kernel(jump_rate_hz):
        return jump_rate_hz

    @classmethod
    def execute(cls, event, objs, config):
        obj = objs[0]
        rng = config["rng"]
        direction = _JUMP_DIRS[rng.randint(0, 6)]
        jump = float(_RUNTIME["jump_length_a"])
        lc = float(config["lattice_constant"])
        gs = config["grid_size"]
        bx, by, bz = gs[0] * lc, gs[1] * lc, gs[2] * lc

        return [
            change_obj(
                obj,
                x=(float(obj["x"]) + direction[0] * jump) % bx,
                y=(float(obj["y"]) + direction[1] * jump) % by,
                z=(float(obj["z"]) + direction[2] * jump) % bz,
            )
        ]


def _periodic_slab_density(
    x_nm: np.ndarray,
    time_s: float,
    total_particles: int,
    box_length_nm: float,
    slab_width_nm: float,
    diffusivity_nm2_s: float,
    n_terms: int = 256,
) -> np.ndarray:
    shifted = x_nm - 0.5 * box_length_nm
    density = np.full_like(shifted, float(total_particles) / box_length_nm, dtype=np.float64)

    if slab_width_nm <= 0.0 or total_particles <= 0:
        return density

    for n in range(1, n_terms + 1):
        coeff = (2.0 * total_particles / (math.pi * n * slab_width_nm)) * np.sin(
            math.pi * n * slab_width_nm / box_length_nm
        )
        decay = np.exp(-diffusivity_nm2_s * (2.0 * math.pi * n / box_length_nm) ** 2 * time_s)
        density += coeff * np.cos(2.0 * math.pi * n * shifted / box_length_nm) * decay

    return density


def _collect_global_histogram(pool, comm: MPI.Comm, x_edges_a: np.ndarray):
    local_objects = pool.get_local_objects(active_only=True)
    local_hist = np.zeros(len(x_edges_a) - 1, dtype=np.int64)
    if len(local_objects) > 0:
        local_hist[:] = np.histogram(local_objects["x"], bins=x_edges_a)[0]

    global_hist = np.zeros_like(local_hist) if comm.rank == 0 else None
    comm.Reduce(local_hist, global_hist, op=MPI.SUM, root=0)

    local_count = int(len(local_objects))
    rank_counts = comm.gather(local_count, root=0)

    if comm.rank == 0:
        return global_hist, np.asarray(rank_counts, dtype=np.float64)
    return None, None


def _profile_stats_from_histogram(
    global_hist: np.ndarray,
    rank_counts: np.ndarray,
    time_s: float,
    x_centers_nm: np.ndarray,
    x_bin_width_nm: float,
    slab_width_nm: float,
    box_length_nm: float,
    diffusivity_nm2_s: float,
):
    total_particles = int(np.sum(global_hist))

    if total_particles > 0:
        sim_density = global_hist.astype(np.float64) / x_bin_width_nm
        ana_density = _periodic_slab_density(
            x_centers_nm,
            time_s,
            total_particles,
            box_length_nm,
            slab_width_nm,
            diffusivity_nm2_s,
        )
        x_cv = float(np.std(global_hist) / np.mean(global_hist))
        denom = float(np.linalg.norm(ana_density))
        theory_rel_l2 = float(np.linalg.norm(sim_density - ana_density) / denom) if denom > 0.0 else 0.0
    else:
        sim_density = np.zeros_like(x_centers_nm)
        ana_density = np.zeros_like(x_centers_nm)
        x_cv = 0.0
        theory_rel_l2 = 0.0

    if len(rank_counts) > 0 and np.mean(rank_counts) > 0.0:
        load_cv = float(np.std(rank_counts) / np.mean(rank_counts))
        min_rank_load = int(np.min(rank_counts))
        max_rank_load = int(np.max(rank_counts))
    else:
        load_cv = 0.0
        min_rank_load = 0
        max_rank_load = 0

    return {
        "total_particles": total_particles,
        "sim_density": sim_density,
        "ana_density": ana_density,
        "x_cv": x_cv,
        "theory_rel_l2": theory_rel_l2,
        "load_cv": load_cv,
        "min_rank_load": min_rank_load,
        "max_rank_load": max_rank_load,
    }


def create_initial_objects_func(
    num_particles: int,
    seed: int,
    box_size_a: np.ndarray,
    seed_slab_width_a: float,
    ic_mode: str = "slab",
    cube_frac: float = 0.1,
):
    def initial_objects_func(rank, size, config):
        saved_state = np.random.get_state()
        np.random.seed(config.get("_user_seed", seed))

        x0 = 0.5 * (box_size_a[0] - seed_slab_width_a)
        x1 = 0.5 * (box_size_a[0] + seed_slab_width_a)
        if ic_mode == "cube":
            # centered hotspot box: cube_frac of every box edge
            y_half = 0.5 * cube_frac * box_size_a[1]
            z_half = 0.5 * cube_frac * box_size_a[2]
            y0, y1 = 0.5 * box_size_a[1] - y_half, 0.5 * box_size_a[1] + y_half
            z0, z1 = 0.5 * box_size_a[2] - z_half, 0.5 * box_size_a[2] + z_half
        else:
            y0, y1 = 0.0, box_size_a[1]
            z0, z1 = 0.0, box_size_a[2]
        domain_decomp = config["_domain_decomp"]

        local_objects = []
        owner_counters = {}

        for _ in range(num_particles):
            pos = np.array(
                [
                    np.random.uniform(x0, x1),
                    np.random.uniform(y0, y1),
                    np.random.uniform(z0, z1),
                ],
                dtype=np.float64,
            )
            owner_rank = int(domain_decomp.get_owner_rank(pos.tolist()))
            owner_counters[owner_rank] = owner_counters.get(owner_rank, 0) + 1
            obj_id = (owner_rank << 48) | owner_counters[owner_rank]

            if owner_rank == rank:
                local_objects.append(
                    {
                        "id": obj_id,
                        "type": "Tracer",
                        "x": float(pos[0]),
                        "y": float(pos[1]),
                        "z": float(pos[2]),
                        "size": 0.0,
                        "float_props": [],
                    }
                )

        np.random.set_state(saved_state)
        return local_objects

    return initial_objects_func


def create_diffusion_tracker(
    output_dir: str,
    num_x_bins: int,
    box_size_a: np.ndarray,
    box_size_nm: np.ndarray,
    slab_width_nm: float,
    diffusivity_nm2_s: float,
):
    x_edges_a = np.linspace(0.0, box_size_a[0], num_x_bins + 1)
    x_bin_width_nm = float(box_size_nm[0] / num_x_bins)
    x_centers_nm = (np.arange(num_x_bins, dtype=np.float64) + 0.5) * x_bin_width_nm
    metrics_path = Path(output_dir) / "metrics.csv"
    profile_path = Path(output_dir) / "x_profile.csv"
    state = {"initialized": False}

    def _initialize_files(n_ranks: int):
        if state["initialized"]:
            return

        Path(output_dir).mkdir(parents=True, exist_ok=True)
        with metrics_path.open("w", encoding="ascii") as f:
            header = [
                "step",
                "time_s",
                "total_particles",
                "x_cv",
                "theory_rel_l2",
                "load_cv",
                "min_rank_load",
                "max_rank_load",
            ]
            header.extend(f"rank_{i}_objects" for i in range(n_ranks))
            f.write(",".join(header) + "\n")

        with profile_path.open("w", encoding="ascii") as f:
            header = ["step", "time_s"]
            header.extend(f"sim_bin_{i:03d}" for i in range(num_x_bins))
            header.extend(f"ana_bin_{i:03d}" for i in range(num_x_bins))
            f.write(",".join(header) + "\n")

        state["initialized"] = True

    def tracker(step: int, time_s: float, object_pool, comm: MPI.Comm):
        global_hist, rank_counts = _collect_global_histogram(object_pool, comm, x_edges_a)
        if comm.rank != 0:
            return

        _initialize_files(comm.size)
        stats = _profile_stats_from_histogram(
            global_hist=global_hist,
            rank_counts=rank_counts,
            time_s=time_s,
            x_centers_nm=x_centers_nm,
            x_bin_width_nm=x_bin_width_nm,
            slab_width_nm=slab_width_nm,
            box_length_nm=float(box_size_nm[0]),
            diffusivity_nm2_s=diffusivity_nm2_s,
        )

        with metrics_path.open("a", encoding="ascii") as f:
            row = [
                str(int(step)),
                f"{float(time_s):.12e}",
                str(int(stats['total_particles'])),
                f"{float(stats['x_cv']):.8e}",
                f"{float(stats['theory_rel_l2']):.8e}",
                f"{float(stats['load_cv']):.8e}",
                str(int(stats["min_rank_load"])),
                str(int(stats["max_rank_load"])),
            ]
            row.extend(str(int(v)) for v in rank_counts)
            f.write(",".join(row) + "\n")

        with profile_path.open("a", encoding="ascii") as f:
            row = [str(int(step)), f"{float(time_s):.12e}"]
            row.extend(f"{float(v):.8e}" for v in stats["sim_density"])
            row.extend(f"{float(v):.8e}" for v in stats["ana_density"])
            f.write(",".join(row) + "\n")

        print(
            f"[e0] step={step:>8d} time={time_s:.4e}s total={int(stats['total_particles'])} "
            f"x_cv={float(stats['x_cv']):.3e} theory_l2={float(stats['theory_rel_l2']):.3e} "
            f"load[min/avg/max]={int(stats['min_rank_load'])}/{np.mean(rank_counts):.1f}/{int(stats['max_rank_load'])}",
            flush=True,
        )

    return tracker


def analyze(engine, num_x_bins: int, box_size_nm: np.ndarray, slab_width_nm: float, diffusivity_nm2_s: float):
    comm = engine.comm
    box_size_a = np.asarray(box_size_nm, dtype=np.float64) * 10.0
    x_edges_a = np.linspace(0.0, box_size_a[0], num_x_bins + 1)
    x_centers_nm = (np.arange(num_x_bins, dtype=np.float64) + 0.5) * (box_size_nm[0] / num_x_bins)

    global_hist, rank_counts = _collect_global_histogram(engine.rma_object_pool, comm, x_edges_a)
    if comm.rank != 0:
        return

    stats = _profile_stats_from_histogram(
        global_hist=global_hist,
        rank_counts=rank_counts,
        time_s=float(engine.current_time),
        x_centers_nm=x_centers_nm,
        x_bin_width_nm=float(box_size_nm[0] / num_x_bins),
        slab_width_nm=slab_width_nm,
        box_length_nm=float(box_size_nm[0]),
        diffusivity_nm2_s=diffusivity_nm2_s,
    )

    print("\nFinal statistics:")
    print(f"  total particles: {int(stats['total_particles'])}")
    print(f"  x-profile CV: {float(stats['x_cv']):.4e}")
    print(f"  relative L2 error vs exact solution: {float(stats['theory_rel_l2']):.4e}")
    print(
        f"  rank load: min={int(stats['min_rank_load'])} "
        f"avg={np.mean(rank_counts):.2f} max={int(stats['max_rank_load'])} "
        f"CV={float(stats['load_cv']):.4e}"
    )


def _validate_inputs(
    num_particles: int,
    jump_length_nm: float,
    seed_slab_width_nm: float,
    grid_size: Tuple[int, int, int],
    num_x_bins: int,
):
    if num_particles <= 0:
        raise ValueError("num_particles must be positive")
    if jump_length_nm <= 0.0:
        raise ValueError("jump_length_nm must be positive")
    if seed_slab_width_nm <= 0.0:
        raise ValueError("seed_slab_width_nm must be positive")
    if len(grid_size) != 3 or any(v <= 0 for v in grid_size):
        raise ValueError("grid_size must contain three positive integers")
    if seed_slab_width_nm > float(grid_size[0]) * LATTICE_UNIT_NM:
        raise ValueError("seed_slab_width_nm cannot exceed the box length in x")
    if num_x_bins <= 0:
        raise ValueError("num_x_bins must be positive")


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Pure diffusion benchmark with analytical profile comparison."
    )
    parser.add_argument(
        "--num-particles",
        type=int,
        default=DEFAULT_NUM_PARTICLES,
        help="default particle count for fast daily runs",
    )
    parser.add_argument(
        "--article-particles",
        action="store_true",
        help="use the article particle count 448820",
    )
    parser.add_argument(
        "--box",
        type=int,
        nargs=3,
        metavar=("LX_NM", "LY_NM", "LZ_NM"),
        default=[100, 300, 300],
        help="box size in nm; the article uses 100 300 300",
    )
    parser.add_argument(
        "--diffusivity",
        type=float,
        default=ARTICLE_DIFFUSIVITY_NM2_S,
        help="tracer diffusivity in nm^2/s",
    )
    parser.add_argument(
        "--jump-length",
        type=float,
        default=DEFAULT_JUMP_LENGTH_NM,
        help="single jump length in nm",
    )
    parser.add_argument(
        "--seed-width",
        type=float,
        default=DEFAULT_SEED_SLAB_WIDTH_NM,
        help="initial center slab width in nm",
    )
    parser.add_argument(
        "--ic",
        choices=("slab", "cube"),
        default="slab",
        help="initial condition: periodic slab (default) or a centered hotspot box",
    )
    parser.add_argument(
        "--cube-frac",
        type=float,
        default=0.1,
        help="hotspot box edge as a fraction of each box edge (--ic cube); its x extent overrides --seed-width",
    )
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--max-steps", type=int, default=2_000_000, help="maximum KMC steps")
    parser.add_argument("--max-time", type=float, default=ARTICLE_SIM_TIME_S, help="maximum simulated time in s")
    parser.add_argument("--progress-interval", type=int, default=1000, help="progress print interval")
    parser.add_argument("--report-interval", type=int, default=1000, help="CSV output interval")
    parser.add_argument("--x-bins", type=int, default=DEFAULT_NUM_X_BINS, help="number of x bins")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="directory for metrics.csv and x_profile.csv")
    parser.add_argument("--dump-dir", default=None, help="optional OVITO dump directory")
    parser.add_argument("--disable-object-lb", action="store_true", help="disable dynamic object load balancing")
    parser.add_argument("--event-lb", action="store_true", help="enable event-level predictive redistribution (level 2)")
    parser.add_argument("--event-lb-threshold", type=float, default=0.2, help="event-level imbalance threshold")
    parser.add_argument("--event-lb-no-timer", action="store_true", help="enable event LB without the per-pass timing wrapper (control run)")
    parser.add_argument("--check-invariants", action="store_true",
                        help="every step: global single-event count == N, global total rate == N*jump_rate, no duplicate object ids (every 100 steps)")
    parser.add_argument("--balance-interval", type=int, default=250, help="object rebalance check interval")
    parser.add_argument("--imbalance-threshold", type=float, default=0.15, help="object rebalance threshold")
    return parser.parse_args()


def main():
    args = _parse_args()
    if args.article_particles:
        args.num_particles = ARTICLE_NUM_PARTICLES

    grid_size = tuple(int(v) for v in args.box)
    if args.ic == "cube":
        if not (0.0 < float(args.cube_frac) <= 1.0):
            raise ValueError("cube_frac must lie in (0, 1]")
        # the x-marginal of the hotspot box is a slab of the same x extent,
        # so the Fourier reference profile stays valid
        args.seed_width = float(args.cube_frac) * float(grid_size[0]) * LATTICE_UNIT_NM
    _validate_inputs(
        num_particles=args.num_particles,
        jump_length_nm=args.jump_length,
        seed_slab_width_nm=args.seed_width,
        grid_size=grid_size,
        num_x_bins=args.x_bins,
    )

    comm = MPI.COMM_WORLD
    jump_rate_hz = jump_rate_from_diffusivity(args.diffusivity, args.jump_length)
    box_size_nm = np.asarray(grid_size, dtype=np.float64) * LATTICE_UNIT_NM
    box_size_a = box_size_nm * 10.0
    jump_length_a = 10.0 * float(args.jump_length)
    seed_slab_width_a = 10.0 * float(args.seed_width)
    approx_local = max(64, int(np.ceil(float(args.num_particles) / max(1, comm.size))))
    pool_capacity = max(256, approx_local * 4)

    _RUNTIME["jump_length_a"] = float(jump_length_a)

    if comm.rank == 0:
        print("=" * 72)
        print("e0: Diffusion Balance Benchmark")
        print("=" * 72)
        print("Reference: Martin-Bragado et al. CPC 184 (2013) 2703-2710, Sec. 4.1")
        if args.ic == "cube":
            print(
                f"IC = centered hotspot box, frac {args.cube_frac:.3f} per axis "
                f"(volume {100.0 * float(args.cube_frac) ** 3:.3f}% of box)"
            )
        else:
            print("IC = periodic slab (x-centered, full y/z span)")
        print(
            f"box = {args.box[0]} x {args.box[1]} x {args.box[2]} nm^3 | "
            f"N = {args.num_particles} (article: {ARTICLE_NUM_PARTICLES})"
        )
        print(
            f"D = {args.diffusivity:.3f} nm^2/s | jump = {args.jump_length:.3f} nm | "
            f"rate = {jump_rate_hz:.3f} s^-1"
        )
        print(
            f"seed slab width = {args.seed_width:.3f} nm | "
            f"object LB = {'off' if args.disable_object_lb else 'on'} | "
            f"event LB = {'on (thr %.2f)' % args.event_lb_threshold if args.event_lb else 'off'}"
        )
        print("=" * 72, flush=True)

    outputs = {
        "interval": max(1, int(args.report_interval)),
        "postprocessors": [
            create_diffusion_tracker(
                output_dir=args.output_dir,
                num_x_bins=args.x_bins,
                box_size_a=box_size_a,
                box_size_nm=box_size_nm,
                slab_width_nm=args.seed_width,
                diffusivity_nm2_s=args.diffusivity,
            )
        ],
    }
    if args.dump_dir:
        outputs["ovito_dump"] = {
            "enabled": True,
            "output_dir": args.dump_dir,
            "prefix": "tracer",
            "properties": ["id", "type", "x", "y", "z"],
        }

    config = {
        "domain": {
            "lattice_constant": LATTICE_CONSTANT_A,
            "lattice_grid": list(grid_size),
            "h_factor": 1,
            "periodic": [True, True, True],
            "max_interaction_distance": float(jump_length_a),
        },
        "simulation": {
            "max_steps": int(args.max_steps),
            "max_time": float(args.max_time),
            "progress_interval": int(args.progress_interval),
            "verbose": False,
            "random_seed": int(args.seed),
            "rma_object_pool": {"initial_capacity": pool_capacity, "auto_expand": True},
            "rma_event_pool": {
                "capacity_zero": 4,
                "capacity_single": pool_capacity,
                "capacity_pair": 4,
                "auto_expand": True,
            },
            "predictive_rebalance": {"enabled": bool(args.event_lb), "threshold": float(args.event_lb_threshold)},
            "adaptive_weight": {"enabled": False},
            "runtime_load_balancer": {"enabled": False},
            "object_load_balancer": {
                "enabled": not args.disable_object_lb,
                "balance_interval": int(args.balance_interval),
                "imbalance_threshold": float(args.imbalance_threshold),
            },
        },
        "objects": {
            "Tracer": {
                "properties": {},
            }
        },
        "ic": {
            "mode": "function",
            "func": create_initial_objects_func(
                num_particles=int(args.num_particles),
                seed=int(args.seed),
                box_size_a=box_size_a,
                seed_slab_width_a=float(seed_slab_width_a),
                ic_mode=str(args.ic),
                cube_frac=float(args.cube_frac),
            ),
        },
        "events": {
            "tracer_migration": TracerMigration,
        },
        "physics": {
            "params": {
                "jump_rate_hz": float(jump_rate_hz),
            },
        },
        "outputs": outputs,
    }

    engine = OKMCengine(config)

    # ---- per-pass cost accounting for the event-level (predictive) redistribution ----
    # The library times level 2 only as a cumulative engine stat, so wrap rebalance():
    # a pass "executed" iff stats['total_rebalances'] advanced (uniform across ranks,
    # because the decision follows a broadcast), so the allreduce below is collective-safe.
    _ev_acc = {"check_calls": 0, "check_time": 0.0, "passes": 0,
               "pass_time": 0.0, "pass_max": 0.0, "events_moved": 0}
    if args.event_lb and not args.event_lb_no_timer:
        _pr = engine.predictive_rebalancer
        _orig_rebalance = _pr.rebalance

        _dbg = os.environ.get("E0_DEBUG_COUNTS", "") == "1"

        def _dbg_counts(tag):
            c = comm.gather(int(engine.rma_event_pool.count_single), root=0)
            if comm.rank == 0:
                print(f"[DBG] step={engine.current_step} {tag}: total={sum(c)} per_rank={c}", flush=True)

        def _timed_rebalance(event_pool, incremental=False):
            import time as _t
            if _dbg and engine.current_step <= 3:
                _dbg_counts("after build / before event-LB")
            before = _pr.stats["total_rebalances"]
            t0 = _t.perf_counter()
            res = _orig_rebalance(event_pool, incremental=incremental)
            dt = _t.perf_counter() - t0
            if _dbg and engine.current_step <= 3:
                _dbg_counts("after event-LB")
            if _pr.stats["total_rebalances"] > before:
                dt_max = comm.allreduce(dt, op=MPI.MAX)
                # the library already allreduces (SUM) the moved count: it is global on every rank
                moved_tot = int(res[0] if isinstance(res, tuple) else res)
                _ev_acc["passes"] += 1
                _ev_acc["pass_time"] += dt_max
                _ev_acc["pass_max"] = max(_ev_acc["pass_max"], dt_max)
                _ev_acc["events_moved"] += moved_tot
                if comm.rank == 0:
                    print(f"[EventLB] Step {engine.current_step}: redistribution completed in "
                          f"{dt_max:.3f}s, events moved={moved_tot}, "
                          f"mode={'incremental' if incremental else 'full'}")
            else:
                _ev_acc["check_calls"] += 1
                _ev_acc["check_time"] += dt
            return res

        _pr.rebalance = _timed_rebalance

    # ---- invariant checks (validation runs only): wrap the engine's total-rate query, which
    # every rank calls once per step right after the rate phase, so the collectives are safe ----
    _inv = {"checks": 0, "violations": 0}
    if args.check_invariants:
        _orig_gtr = engine._get_total_rate
        _n_expected = int(args.num_particles)
        _rate_expected = float(args.num_particles) * float(jump_rate_hz)

        def _checked_total_rate(*a, **k):
            local_rate = _orig_gtr(*a, **k)
            pool = engine.rma_event_pool
            n_single = int(pool.count_single)
            tot = comm.allreduce(np.array([n_single, float(local_rate)], dtype=np.float64), op=MPI.SUM)
            ok_count = int(round(tot[0])) == _n_expected
            ok_rate = abs(tot[1] - _rate_expected) <= 1e-9 * max(1.0, _rate_expected)
            ok_dup = True
            if engine.current_step % 100 == 0 or _n_expected <= 20000:
                seg = pool.single_obj_array[:n_single]
                ids = np.ascontiguousarray(seg['obj_ids'][:, 0], dtype=np.int64) if n_single > 0 else np.empty(0, np.int64)
                eids = (np.stack([seg['id_high'].astype(np.int64), seg['id_low'].astype(np.int64)], axis=1)
                        if (n_single > 0 and 'id_high' in seg.dtype.names) else np.empty((0, 2), np.int64))
                all_ids = comm.gather(ids, root=0); all_eids = comm.gather(eids, root=0)
                if comm.rank == 0:
                    cat = np.concatenate(all_ids); cate = np.concatenate(all_eids)
                    n_uobj = len(np.unique(cat)); n_ueid = len(np.unique(cate, axis=0)) if len(cate) else 0
                    ok_dup = bool(n_uobj == len(cat) and n_ueid == len(cate))
                    if not ok_dup and _inv["violations"] < 20:
                        print(f"[INVARIANT DETAIL] step={engine.current_step} events={len(cat)} unique_obj_ids={n_uobj} unique_event_ids={n_ueid}", flush=True)
                ok_dup = comm.bcast(ok_dup, root=0)
            _inv["checks"] += 1
            if not (ok_count and ok_rate and ok_dup):
                _inv["violations"] += 1
                if comm.rank == 0 and _inv["violations"] <= 20:
                    print(f"[INVARIANT VIOLATION] step={engine.current_step} events={int(tot[0])} expected={_n_expected} "
                          f"rate={tot[1]:.6f} expected={_rate_expected:.6f} unique_obj_ids_ok={ok_dup}", flush=True)
            return local_rate

        engine._get_total_rate = _checked_total_rate

    try:
        stats = engine.run()
        if args.check_invariants and comm.rank == 0:
            print(f"[Invariants] checks={_inv['checks']} violations={_inv['violations']}", flush=True)
        if args.event_lb and not args.event_lb_no_timer:
            _ct = comm.allreduce(_ev_acc["check_time"], op=MPI.MAX)
            if comm.rank == 0:
                print(f"[EventLB] Summary: passes={_ev_acc['passes']} "
                      f"pass_time_total={_ev_acc['pass_time']:.3f}s "
                      f"pass_time_max={_ev_acc['pass_max']:.3f}s "
                      f"events_moved={_ev_acc['events_moved']} "
                      f"check_calls={_ev_acc['check_calls']} "
                      f"check_time_total_maxrank={_ct:.3f}s "
                      f"engine_rebalance_time={engine.stats.get('rebalance_time', 0.0):.3f}s")
        analyze(
            engine,
            num_x_bins=int(args.x_bins),
            box_size_nm=box_size_nm,
            slab_width_nm=float(args.seed_width),
            diffusivity_nm2_s=float(args.diffusivity),
        )
    finally:
        engine.destroy()

    if comm.rank == 0:
        print(f"\ntotal steps: {stats['total_steps']}")
        print(f"simulated time: {stats['total_time']:.6e} s")
        print(f"output dir: {args.output_dir}")
        if args.dump_dir:
            print(f"dump dir: {args.dump_dir}")
        print("=" * 72)


if __name__ == "__main__":
    main()
