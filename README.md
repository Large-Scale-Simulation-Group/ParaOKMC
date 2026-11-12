# ParaOKMC

**Parallel Object Kinetic Monte Carlo** for the long-term evolution of defects in
irradiated materials: migration, reaction, clustering and absorption at sinks, run on
distributed memory without weakening the residence-time algorithm.

ParaOKMC keeps the serial statistics exact — one event is drawn from the global rate
distribution and one clock advances per step, whatever the rank count — and parallelizes
the work around it: a spatial domain decomposition, a two-collective per-step protocol
(one allgather of partial rates that also carries the selection random number, one
broadcast of the resulting state change), and three levels of hotspot-aware load
balancing that act on self-contained event records rather than on the objects.

[中文说明 / Chinese README](README.zh-CN.md) · Version 2.0.0 · Apache-2.0

## Installation

Python ≥ 3.9 with NumPy ≥ 1.19, SciPy ≥ 1.5, Numba ≥ 0.55 and mpi4py ≥ 3.0 on an **MPI-3**
library (the object and event pools use one-sided RMA windows).

```bash
conda env create -f environment.yml   # one step, from the release root: dependencies, MPI and ParaOKMC
conda activate paraokmc
```

Or with pip, on a machine that already has an MPI library:

```bash
pip install -e .                 # from the release root
pip install -e ".[dedalus]"      # optional field-solver coupling (Dedalus, GPL-3.0; see Licensing)
```

See `INSTALL.md` for the MPI library pip needs and for cluster notes. Run one MPI rank per core and keep the
kernels single-threaded:

```bash
export NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1
mpirun -np 8 python my_model.py
```

## A complete model in one file

A model is a plain Python dictionary of seven blocks plus the event classes it registers;
`OKMCengine(config).run()` is the whole driver and runs unchanged on any number of ranks.

```python
import numpy as np
from numba import njit
from paraokmc import OKMCengine, SingleObjectEvent, PairObjectEvent, RateDeps, change_obj, delete_obj

@njit
def arrhenius(E_m, nu, temperature, k_B):
    return nu * np.exp(-E_m / (k_B * temperature))

class Migration(SingleObjectEvent):
    object_types = ['I', 'V']
    rate_deps = RateDeps(obj1=['E_m', 'nu'], physics=['temperature', 'k_B'])  # compiled batch path
    rate_kernel = arrhenius
    incremental_rate = True        # rates depend only on object properties and constants: reuse them

    @classmethod
    def execute(cls, event, objs, config):
        o, u, lc = objs[0], config['rng'], config['lattice_constant']
        th, ph = 2 * np.pi * u.random(), np.arccos(2 * u.random() - 1)
        return [change_obj(o, x=o['x'] + lc * np.sin(ph) * np.cos(th),
                              y=o['y'] + lc * np.sin(ph) * np.sin(th),
                              z=o['z'] + lc * np.cos(ph))]

class Recombination(PairObjectEvent):
    type_pairs = [('I', 'V')]
    max_distance = 2.0

    @classmethod
    def rate(cls, objs, params, config):   # plain-Python rate: anything goes, including an ML model
        return 1.0e10

    @classmethod
    def execute(cls, event, objs, config):
        return [delete_obj(objs[0]), delete_obj(objs[1])]

def on_event(event_type, changed, step, time, pool, comm, config):
    if event_type == 'recombination' and comm.rank == 0:
        print(f"step {step}: recombination at t = {time:.3e} s")

config = {
    "domain":     {"lattice_constant": 1.0, "lattice_grid": [30, 30, 30],
                   "periodic": [True, True, True], "max_interaction_distance": 3.0},
    "simulation": {"max_steps": 1000, "random_seed": 42},
    "objects":    {"I": {"size": 1.0, "properties": {"E_m": 0.3, "nu": 1e13}},
                   "V": {"size": 1.0, "properties": {"E_m": 1.0, "nu": 1e13}}},
    "ic":         {"mode": "builtin", "initial_conditions": {"I": {"count": 20, "distribution": "random"},
                                                             "V": {"count": 20, "distribution": "random"}}},
    "events":     {"migration": Migration, "recombination": Recombination},
    "physics":    {"params": {"temperature": 500.0, "k_B": 8.617e-5}},
    "outputs":    {"interval": 200,
                   "ovito_dump": {"enabled": True, "output_dir": "dump", "prefix": "state",
                                  "properties": ["id", "type", "x", "y", "z", "E_m"]},
                   "event_hooks": [on_event]},
}

engine = OKMCengine(config)
stats = engine.run()        # {'total_steps', 'total_time', 'total_events', 'object_count', 'avg_*_time'}
engine.destroy()
```

### The seven blocks

| Block | Contents |
|---|---|
| `domain` | `lattice_constant`, `lattice_grid` `[nx, ny, nz]`, `periodic` per axis, `max_interaction_distance` (sets the ghost-region width). The rank count is factored into a cuboid grid over this box. |
| `simulation` | `max_steps`, `max_time`, `random_seed`; the load-balancing switches below; pool capacities (`rma_object_pool`, `rma_event_pool`, auto-expanding). |
| `objects` | one entry per type: `size` (radius entering the influence range) and `properties`, scalar user attributes stored on every object and readable by any rate function. |
| `ic` | `builtin` (random placement, `count` per type), `function` (`func(rank, size, config) -> list[dict]`), or `dump` (restart from a saved frame, see below). |
| `events` | `{name: EventClass}`. |
| `physics` | `params` (named scalars such as temperature), `fields` on 3-D grids with `update_func`/`update_policy`/`queries`, optional external `solvers` and event `triggers`. Rates read them through `rate_deps` or `get_physics_param`; a change invalidates the rates that depend on it. |
| `outputs` | `interval`, `ovito_dump`, `postprocessors` (callables `f(step, time, pool, comm)`), `event_hooks`. |

Full reference: `docs/USER_GUIDE.md`. A guided introduction: `docs/TUTORIAL.md`.

## Events

An event class derives from `ZeroObjectEvent` (global or per-cell processes, e.g. a source),
`SingleObjectEvent` (migration, emission, attribute change) or `PairObjectEvent`
(recombination, reaction, coalescence, dissociation), names the types it applies to
(`object_types` / `type_pairs` + `max_distance`), and supplies

- a **rate**, either as `rate(cls, objs, params, config)` in plain Python, or as
  `rate_deps` + a Numba `rate_kernel` evaluated for the whole event array in one compiled call;
- its **consequence** as `execute(cls, event, objs, config)`, returning a list built from
  `new_obj`, `change_obj` and `delete_obj`. The engine applies the list as one state-change
  message on every rank.

Rules the parallel engine relies on: a rate must be deterministic and must not draw random
numbers; random numbers inside `execute` come from `config['rng']`, the counter-keyed
generator described below. `incremental_rate = True` on a class lets the engine keep a rate
whose objects did not change (safe when the rate depends only on object properties and
constant parameters; leave it off for rates that follow a time-varying field).

## What the engine does per step

1. **Build** — the catalogue of candidate events is updated structurally and incrementally
   for all three categories (pair events through a cell-level invalidation zone derived by
   every rank from the broadcast old/new positions); a full rebuild happens only at the
   first step, after a repartition or a global change.
2. **Rate** — rates are evaluated locally (JIT batch or Python), reused where
   `incremental_rate` allows.
3. **Select** — one `allgather` collects the partial rates and the selection random number
   drawn on rank 0; every rank identifies the same winning event from the cumulative sum.
4. **Execute** — the owning rank executes the event, draws the time advance and broadcasts
   one state-change message; every rank applies it, refreshes its ghost cache from it and
   runs the event hooks.

**Random numbers.** Every draw is a pure function of `(user_seed, step, state)`, passed as
an entropy tuple to NumPy's `SeedSequence` → PCG64. A run is therefore one reproducible
sequence that no rank can drift out of; the authoritative values are still exchanged in the
collectives, so floating-point differences between ranks cannot change the decision.

**Load balancing** (`simulation` block, all optional):

| Level | Mechanism | Key | Default |
|---|---|---|---|
| 1 | object-level repartition: RCB re-cuts the box by object density when the measured imbalance `max/mean − 1` exceeds a threshold | `object_load_balancer: {enabled, balance_interval, imbalance_threshold}` | off (interval 500, threshold 0.2) |
| 2 | event-level redistribution: weighted event records move to under-loaded ranks before rate evaluation; off by itself when every event class reuses its rate | `predictive_rebalance: {enabled, threshold, check_interval}` | on (threshold 1.5, every 100 steps) |
| 3 | runtime work stealing: an idle rank takes pending Python-rate events from a busy one through a one-sided window | `runtime_load_balancer: {enabled, check_interval}` | on (check every 32 evaluated events) |

All three preserve the sampled statistics by construction (every event is counted in exactly
one partial rate). Levels 2 and 3 react to measured timings, so bitwise trajectory
reproducibility across repeated runs holds under a fixed partition only.

## Output and restart

`outputs.ovito_dump` writes LAMMPS-style dump frames that OVITO opens directly. By default
(`mode: "distributed"`) every rank writes its own `prefix.<step>.rank_<r>.dump` with no
gather; load them in OVITO with the wildcard `prefix.<step>.rank_*.dump`. `mode: "collect"`
gathers everything to rank 0 into one file. The `properties` list may name any declared
object property — it becomes a column. A frame is written every `interval` steps and once
more after the last step, whatever ended the run.

A saved frame is an initial condition:

```python
"ic": {"mode": "dump", "file": "run1/dump", "frame": -1, "id_mode": "keep"}
```

reads a file or a whole dump directory (per-rank files merged, `-1` = last frame), restores
positions, sizes and all declared properties, and distributes the objects to whatever ranks
the new run has — a restart may use a different rank count.

`outputs.event_hooks` are called on every rank after each applied state-change message
with `(event_type, changed_objects, step, time, pool, comm, config)`; the deleted records
carry their full data, so observables such as the jump count of an absorbed defect can be
accumulated without any user-side MPI.

## Examples and benchmarks

`example/` holds every validation case and performance test of the accompanying paper,
one directory per section (`3.1_free_diffusion` … `4.5_step_time_budget`), each with the
model script and the exact run commands. The computed results (logs, CSV tables and OVITO
frames) are distributed with the program package of the paper in the CPC Program Library;
with them, `example/figures/make_pub_figures.py` regenerates Figs. 4–11. See
`example/README.md`.

`tutorial/` contains eight short scripts, one per chapter of `docs/TUTORIAL.md`
(`01_first_model.py` … `08_parallel_runs_and_load_balancing.py`). `benchmark/` holds
strong/weak-scaling drivers.

## Tests

```bash
python -m pytest -q tests
mpirun -np 2 python tests/test_dump_restart.py     # multi-rank tests are plain scripts; run them under mpirun as well
```

## Documentation

A Chinese translation of every document below is in `docs_zh_cn/`.

- `docs/TUTORIAL.md` — eight chapters, each backed by a script in `tutorial/`
- `docs/USER_GUIDE.md` — reference: the seven blocks key by key, the event-class contract, run-time `config` entries, validation errors
- `docs/ARCHITECTURE.md` — data design, per-step protocol, the three balancing levels
- `docs/RANDOM_NUMBERS.md` — the counter-keyed random-number generator
- `CHANGELOG.md`

## Licensing

ParaOKMC is released under the Apache License 2.0 (`LICENSE`), copyright Beihang University (`NOTICE`);
the authors are listed in `AUTHORS`.
The core dependencies (NumPy, SciPy, Numba, mpi4py) carry permissive BSD/Apache licenses.
Dedalus, used only by the optional `[dedalus]` extra for field-solver coupling, is GPL-3.0:
it is not installed by default, and a build that bundles it is subject to the GPL.
