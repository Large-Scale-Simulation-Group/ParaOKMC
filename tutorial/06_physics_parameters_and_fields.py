"""
Tutorial 6 - physics parameters that change in time, and a spatial field.

physics.params are named scalars every rate may read (through rate_deps or
config['physics'].get_param). physics.fields are 3-D grids spanning the box
that the engine interpolates at an object's position. physics.ic_func(physics,
config) runs once before the first step and is the place to fill a field with
data and to register a scheduled parameter update.

This script anneals a population of interstitials:
  * temperature rises 300 -> 900 K during the run (a parameter update every
    200 steps; the engine invalidates every rate that depends on it);
  * a static stress field lowers the migration barrier towards +x, so
    interstitials hop faster there. Jumps stay isotropic, so the population
    slowly piles up on the slow side (<x>/L drifts below 0.5): a
    position-dependent hop rate changes the diffusivity, not the direction.

The same field is read in two ways: the compiled kernel receives the
interpolated value as an argument (rate_deps fields=[...]); the Python rate
class calls get_physics_field(config, 'stress')(x, y, z).

Run:
    python 06_physics_parameters_and_fields.py
    python 06_physics_parameters_and_fields.py --style python
"""
import argparse
import numpy as np
from mpi4py import MPI
from numba import njit

from paraokmc import OKMCengine, SingleObjectEvent, RateDeps, change_obj, get_physics_param, get_physics_field

K_B = 8.617333262e-5
COUPLING = 0.05          # eV per unit of stress: E_eff = E_m - COUPLING * stress


def box_of(config):
    return np.asarray(config['grid_size'], dtype=float) * config['lattice_constant']


def jump(o, config):
    rng, a0 = config['rng'], config['lattice_constant']
    theta, phi = 2.0 * np.pi * rng.random(), np.arccos(2.0 * rng.random() - 1.0)
    step = a0 * np.array([np.sin(phi) * np.cos(theta), np.sin(phi) * np.sin(theta), np.cos(phi)])
    new = (np.array([o['x'], o['y'], o['z']]) + step) % box_of(config)
    return [change_obj(o, x=new[0], y=new[1], z=new[2])]


class MigrationKernel(SingleObjectEvent):
    object_types = ['I']
    # argument order: obj1 properties, fields (interpolated at the object), physics scalars
    rate_deps = RateDeps(obj1=['E_m', 'nu'], fields=['stress'], physics=['temperature'])

    @staticmethod
    @njit
    def rate_kernel(E_m, nu, stress, temperature):
        return nu * np.exp(-(E_m - 0.05 * stress) / (8.617333262e-5 * temperature))

    @classmethod
    def execute(cls, event, objs, config):
        return jump(objs[0], config)


class MigrationPython(SingleObjectEvent):
    object_types = ['I']

    @classmethod
    def rate(cls, objs, params, config):
        o = objs[0]
        T = get_physics_param(config, 'temperature')
        stress = float(get_physics_field(config, 'stress')(o['x'], o['y'], o['z']))
        return o['nu'] * np.exp(-(o['E_m'] - COUPLING * stress) / (K_B * T))

    @classmethod
    def execute(cls, event, objs, config):
        return jump(objs[0], config)


def physics_ic(physics, config):
    """Runs once before step 0 on every rank."""
    # 1. a stress field that grows linearly along x (0 at x=0, 1 at x=L; periodic box, so a sawtooth)
    nx, ny, nz = physics.get_field('stress').shape
    x = np.linspace(0.0, 1.0, nx)
    physics.set_field('stress', np.broadcast_to(x[:, None, None], (nx, ny, nz)).copy())

    # 2. a temperature ramp: re-evaluated every 200 steps
    def ramp(physics, step, time):
        return min(900.0, 300.0 + 0.1 * step)
    physics.set_param_update('temperature', ramp, step_interval=200)


def report(event_type, changed, step, time, pool, comm, config):
    """Event hook (called on every rank after every step): print T and the mean x every 1000 steps."""
    if step % 1000 != 0:
        return
    local = pool.get_local_objects()
    sx = comm.reduce(float(np.sum(local['x'])) if len(local) else 0.0, op=MPI.SUM, root=0)
    n = comm.reduce(len(local), op=MPI.SUM, root=0)
    if comm.rank == 0:
        T = get_physics_param(config, 'temperature')
        print(f"  step {step:>6d}  t = {time:.3e} s  T = {T:6.1f} K  <x>/L = {sx / n / box_of(config)[0]:.3f}")


def make_config(style):
    mig = MigrationKernel if style == 'kernel' else MigrationPython
    return {
        "domain": {"lattice_constant": 2.87, "lattice_grid": [40, 10, 10],
                   "periodic": [True, True, True], "max_interaction_distance": 1.0},
        "simulation": {"max_steps": 6000, "random_seed": 2, "progress_interval": 0},
        "objects": {"I": {"size": 1.0, "properties": {"E_m": 0.34, "nu": 6.0e12}}},
        "ic": {"mode": "builtin", "initial_conditions": {"I": {"count": 300, "distribution": "random"}}},
        "events": {"migration": mig},
        "physics": {
            "params": {"temperature": 300.0},
            "fields": {"stress": {"shape": [41, 11, 11]}},          # grid over the whole box
            "ic_func": physics_ic,
        },
        "outputs": {"event_hooks": [report]},
    }


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--style', choices=['kernel', 'python'], default='kernel')
    args = p.parse_args()
    engine = OKMCengine(make_config(args.style))
    stats = engine.run()
    engine.destroy()
    if MPI.COMM_WORLD.rank == 0:
        print(f"done ({args.style}): {stats['total_steps']} steps, t = {stats['total_time']:.3e} s")
