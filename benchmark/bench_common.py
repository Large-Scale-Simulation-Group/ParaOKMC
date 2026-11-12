"""
Shared benchmark module - event definitions, configuration, hotspot distribution, timing
"""

import numpy as np
import time
import json
from mpi4py import MPI
from numba import njit
from paraokmc import OKMCengine, SingleObjectEvent, RateDeps, change_obj

# ==================== Physical constants ====================
k_B   = 8.617e-5
a_0   = 3.14
E_m_H = 0.20
nu_0  = 6e12

# ==================== Event class (H migration only) ====================

@njit
def _h_migration_kernel(temperature):
    return nu_0 * np.exp(-E_m_H / (k_B * temperature))

class HMigration(SingleObjectEvent):
    """H jumps along bcc <111> - the only event, so the particle count is conserved"""
    object_types = ['H']
    params = []
    rate_deps = RateDeps(physics=['temperature'])
    rate_kernel = _h_migration_kernel

    @classmethod
    def execute(cls, event, objs, config):
        obj = objs[0]
        lc = config.get('lattice_constant', a_0)
        gs = config.get('grid_size', (100, 100, 100))
        bx, by, bz = gs[0]*lc, gs[1]*lc, gs[2]*lc
        dirs = [(0.5,0.5,0.5),(0.5,0.5,-0.5),(0.5,-0.5,0.5),(0.5,-0.5,-0.5),
                (-0.5,0.5,0.5),(-0.5,0.5,-0.5),(-0.5,-0.5,0.5),(-0.5,-0.5,-0.5)]
        d = dirs[config['rng'].randint(0, 8)]      # the engine's reproducible generator
        jump = lc * np.sqrt(3) / 2
        return [change_obj(obj,
                           x=(float(obj['x']) + d[0]*jump) % bx,
                           y=(float(obj['y']) + d[1]*jump) % by,
                           z=(float(obj['z']) + d[2]*jump) % bz)]


# ==================== Hotspot distribution ====================

def create_hotspot_init(n_H, grid_size, lc,
                        n_hotspots=3, hotspot_ratio=0.7):
    """
    70% of the particles sit in three Gaussian hotspots, 30% form a uniform background.
    Every rank generates the same configuration from one seed; the engine keeps the
    objects inside each rank's subdomain.
    """
    def init_func(rank, size, config):
        bx, by, bz = grid_size[0]*lc, grid_size[1]*lc, grid_size[2]*lc
        rng = np.random.RandomState(42)

        centers = [(rng.uniform(0.2*bx, 0.8*bx),
                     rng.uniform(0.2*by, 0.8*by),
                     rng.uniform(0.2*bz, 0.8*bz))
                    for _ in range(n_hotspots)]

        sigma = min(bx, by, bz) * 0.1 / 3.0
        n_hot = int(n_H * hotspot_ratio)
        n_bg  = n_H - n_hot

        objects = []
        # Hotspot particles
        per_hs = n_hot // n_hotspots
        for cx, cy, cz in centers:
            for _ in range(per_hs):
                objects.append({
                    'id': len(objects) + 1, 'type': 'H',
                    'x': rng.normal(cx, sigma) % bx,
                    'y': rng.normal(cy, sigma) % by,
                    'z': rng.normal(cz, sigma) % bz,
                })
        # Background particles
        for _ in range(n_bg):
            objects.append({
                'id': len(objects) + 1, 'type': 'H',
                'x': rng.uniform(0, bx),
                'y': rng.uniform(0, by),
                'z': rng.uniform(0, bz),
            })
        return objects

    return init_func


# ==================== Configuration ====================

def make_config(n_H, grid_size, max_steps=10000, enable_lb=True):
    lc  = a_0
    cap = n_H + 1000

    return {
        "domain": {
            "lattice_constant": lc,
            "lattice_grid": list(grid_size),
            "h_factor": 1,
            "periodic": [True, True, True],
            "max_interaction_distance": lc * 3,
        },
        "simulation": {
            "max_steps": max_steps,
            "max_time": 1e10,
            "verbose": False,
            "random_seed": 42015,
            "rma_object_pool": {"initial_capacity": cap, "auto_expand": True},
            "rma_event_pool": {"capacity_zero": 10, "capacity_single": cap * 2,
                               "capacity_pair": 100, "auto_expand": True},
            "predictive_rebalance": {"enabled": enable_lb, "threshold": 0.2},
            "adaptive_weight": {"enabled": enable_lb, "update_interval": 1000},
            "runtime_load_balancer": {"enabled": enable_lb, "check_interval": 32},
            "object_load_balancer": {"enabled": enable_lb, "balance_interval": 2000,
                                     "imbalance_threshold": 0.3},
        },
        "objects": {"H": {}},
        "ic": {"mode": "function", "func": create_hotspot_init(n_H, grid_size, lc)},
        "events": {"H_migration": HMigration},
        "physics": {"params": {"temperature": 300.0}},
        "outputs": {},
    }


# ==================== Run and time ====================

def run_bench(label, n_H, grid_size, max_steps=10000, enable_lb=True,
              output_file='benchmark_results.json'):
    comm = MPI.COMM_WORLD
    rank, nprocs = comm.rank, comm.size

    if rank == 0:
        print('=' * 65)
        print(f' {label}')
        print(f' ranks={nprocs}  H={n_H:,}  grid={grid_size}  '
              f'steps={max_steps}  LB={"ON" if enable_lb else "OFF"}')
        print('=' * 65)

    config = make_config(n_H, grid_size, max_steps, enable_lb)
    engine = OKMCengine(config)

    comm.Barrier()
    t0 = time.time()
    stats = engine.run()
    comm.Barrier()
    wall = time.time() - t0

    all_wall = comm.gather(wall, root=0)

    if rank == 0:
        total_steps = stats.get('total_steps', 0)
        wmax = max(all_wall)
        sps  = total_steps / wmax if wmax > 0 else 0

        print(f' steps={total_steps}  wall={wmax:.2f}s  '
              f'steps/s={sps:.1f}  atoms/rank={n_H//nprocs}')
        print('=' * 65 + '\n')

        result = {
            'label': label, 'n_ranks': nprocs, 'n_atoms': n_H,
            'atoms_per_rank': n_H // nprocs,
            'grid_size': list(grid_size), 'max_steps': max_steps,
            'load_balance': enable_lb,
            'total_steps': total_steps, 'wall_time': wmax,
            'steps_per_sec': sps,
            'per_rank_times': all_wall,
        }
        try:
            with open(output_file, 'r') as f:
                results = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            results = []
        results.append(result)
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)

    engine.destroy()
