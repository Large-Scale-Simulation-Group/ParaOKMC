"""
Tutorial 2 - the two ways to write a rate.

Every event class gives its rate either as

  (a) rate(cls, objs, params, config): plain Python, called once per candidate
      event. It may read anything: object coordinates, the configuration,
      class attributes, external models.
  (b) rate_deps + rate_kernel: a Numba kernel that the engine applies to the
      whole event array in one compiled call. The kernel receives exactly what
      rate_deps declares, in this order:
          obj1 properties, obj2 properties (pair events), distance (pair
          events), field values, physics parameters.
      Object coordinates are not among them.

Both styles appear below for the same two events. Switch with --style; the
trajectory is identical because the numbers are identical.

Run:
    python 02_two_rate_styles.py --style python
    python 02_two_rate_styles.py --style kernel
"""
import argparse
import numpy as np
from mpi4py import MPI
from numba import njit

from paraokmc import OKMCengine, SingleObjectEvent, PairObjectEvent, RateDeps, change_obj, delete_obj

K_B = 8.617333262e-5


def _jump(obj, config):
    rng, a0 = config['rng'], config['lattice_constant']
    box = np.asarray(config['grid_size'], dtype=float) * a0
    theta, phi = 2.0 * np.pi * rng.random(), np.arccos(2.0 * rng.random() - 1.0)
    step = a0 * np.array([np.sin(phi) * np.cos(theta), np.sin(phi) * np.sin(theta), np.cos(phi)])
    new = (np.array([obj['x'], obj['y'], obj['z']]) + step) % box
    return [change_obj(obj, x=new[0], y=new[1], z=new[2])]


# ----------------------------------------------------------------- (a) Python rates

class MigrationPy(SingleObjectEvent):
    object_types = ['I', 'V']

    @classmethod
    def rate(cls, objs, params, config):
        o = objs[0]                                # a dict: x, y, z, size, type and every declared property
        T = config['physics'].get_param('temperature')
        return o['nu'] * np.exp(-o['E_m'] / (K_B * T))

    @classmethod
    def execute(cls, event, objs, config):
        return _jump(objs[0], config)


class CapturePy(PairObjectEvent):
    """Distance-dependent capture: rate = k0 * (r_c / r)."""
    type_pairs = [('I', 'V')]
    max_distance = 6.0
    params = ['distance']                          # ask for the pair distance in params

    @classmethod
    def rate(cls, objs, params, config):
        r = max(params['distance'], 0.5)
        return 1.0e12 * (2.0 / r)

    @classmethod
    def execute(cls, event, objs, config):
        return [delete_obj(objs[0]), delete_obj(objs[1])]


# ----------------------------------------------------------------- (b) compiled rates

class MigrationKernel(SingleObjectEvent):
    object_types = ['I', 'V']
    rate_deps = RateDeps(obj1=['E_m', 'nu'], physics=['temperature'])

    @staticmethod
    @njit
    def rate_kernel(E_m, nu, temperature):         # arguments follow the rate_deps order
        return nu * np.exp(-E_m / (8.617333262e-5 * temperature))

    @classmethod
    def execute(cls, event, objs, config):
        return _jump(objs[0], config)


class CaptureKernel(PairObjectEvent):
    type_pairs = [('I', 'V')]
    max_distance = 6.0
    # No object property is needed, so the event kind cannot be inferred from
    # the declaration: say kind='pair'. The kernel then receives the distance only.
    rate_deps = RateDeps(kind='pair')

    @staticmethod
    @njit
    def rate_kernel(distance):
        r = max(distance, 0.5)
        return 1.0e12 * (2.0 / r)

    @classmethod
    def execute(cls, event, objs, config):
        return [delete_obj(objs[0]), delete_obj(objs[1])]


# ----------------------------------------------------------------- configuration

def make_config(style):
    mig, cap = (MigrationPy, CapturePy) if style == 'python' else (MigrationKernel, CaptureKernel)
    return {
        "domain": {"lattice_constant": 2.87, "lattice_grid": [20, 20, 20],
                   "periodic": [True, True, True], "max_interaction_distance": 6.0},
        "simulation": {"max_steps": 3000, "random_seed": 11, "progress_interval": 0},
        "objects": {"I": {"size": 1.0, "properties": {"E_m": 0.34, "nu": 6.0e12}},
                    "V": {"size": 1.0, "properties": {"E_m": 0.67, "nu": 6.0e12}}},
        "ic": {"mode": "builtin", "initial_conditions": {"I": {"count": 60, "distribution": "random"},
                                                         "V": {"count": 60, "distribution": "random"}}},
        "events": {"migration": mig, "capture": cap},
        "physics": {"params": {"temperature": 600.0}},
        "outputs": {},
    }


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--style', choices=['python', 'kernel'], default='kernel')
    args = p.parse_args()
    engine = OKMCengine(make_config(args.style))
    stats = engine.run()
    engine.destroy()
    if MPI.COMM_WORLD.rank == 0:
        print(f"style={args.style}: steps={stats['total_steps']} time={stats['total_time']:.6e} s "
              f"objects={stats['object_count']}  rate-eval {1e3 * stats['avg_rate_time']:.3f} ms/step")
