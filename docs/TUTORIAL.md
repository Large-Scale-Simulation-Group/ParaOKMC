# ParaOKMC tutorial

*中文版：[`docs_zh_cn/TUTORIAL.md`](../docs_zh_cn/TUTORIAL.md)*

A guided tour of the library in eight short scripts. Each chapter introduces one idea,
shows the code that carries it, and tells you what to look at when you run it. The scripts
live in `tutorial/` and run in seconds on one core; every one of them also runs unchanged
under `mpirun`. Read the chapters in order the first time; afterwards use `docs/USER_GUIDE.md`
as the reference.

```bash
cd paraokmc_release
pip install -e .
export PYTHONPATH=$PWD NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1   # one kernel thread per MPI rank
cd tutorial
python 01_first_model.py
mpirun -np 2 python 01_first_model.py
```

| Chapter | Script | What you learn |
|---|---|---|
| 1 | `01_first_model.py` | the two kinds of class you write, the seven configuration blocks, what a run returns |
| 2 | `02_two_rate_styles.py` | Python `rate()` versus a compiled `rate_kernel`; what a kernel can and cannot see |
| 3 | `03_objects_and_execute.py` | object properties; `change_obj`, `delete_obj`, `new_obj`; post-processors |
| 4 | `04_initial_condition_and_random_numbers.py` | writing your own initial condition; how random numbers stay reproducible |
| 5 | `05_zero_object_events.py` | events attached to no object: a source of defects |
| 6 | `06_physics_parameters_and_fields.py` | parameters that change in time; a spatial field read by a rate |
| 7 | `07_outputs_and_restart.py` | OVITO dumps, event hooks, restarting from a saved frame |
| 8 | `08_parallel_runs_and_load_balancing.py` | what the rank count changes; the three load-balancing levels |

After the tutorial, `example/` holds the complete validation cases and benchmarks of the
accompanying paper, one directory per section; their computed results come with the paper's
program package in the CPC Program Library.

---

## 1. A first model

**The idea.** An OKMC model is a set of *objects* (point defects, clusters, sinks) and a set
of *events* each object may undergo. ParaOKMC asks you for two things: a Python class per
event type and a dictionary of seven blocks that describes the box, the objects, the initial
state, the physics parameters and the output. `OKMCengine(config).run()` does the rest.

`01_first_model.py` simulates Frenkel pairs: interstitials (`I`) and vacancies (`V`) migrate
by random jumps; an `I` and a `V` closer than 2 Å recombine.

**An event class** derives from `SingleObjectEvent` (one object takes part) or
`PairObjectEvent` (two objects within `max_distance` of each other), says which types it
applies to, and provides a rate and a consequence:

```python
class Migration(SingleObjectEvent):
    object_types = ['I', 'V']
    rate_deps = RateDeps(obj1=['E_m', 'nu'], physics=['temperature', 'k_B'])

    @staticmethod
    @njit
    def rate_kernel(E_m, nu, temperature, k_B):
        return nu * np.exp(-E_m / (k_B * temperature))

    @classmethod
    def execute(cls, event, objs, config):
        obj, rng = objs[0], config['rng']
        ...                                    # pick a random direction with rng
        return [change_obj(obj, x=new[0], y=new[1], z=new[2])]
```

* `rate_deps` names the inputs of the rate: here two object properties and two physics
  parameters. `rate_kernel` is a Numba function whose arguments follow that declaration.
  The engine evaluates it for every candidate event at once, so this is the fast path.
* `execute` is plain Python. It receives the objects of the event as dictionaries (`x`, `y`,
  `z`, `size`, `type`, and every declared property by name) and returns a list of changes.
  Random numbers come from `config['rng']` (chapter 4 explains why).

```python
class Recombination(PairObjectEvent):
    type_pairs = [('I', 'V')]
    max_distance = 2.0

    @classmethod
    def rate(cls, objs, params, config):
        return 1.0e12

    @classmethod
    def execute(cls, event, objs, config):
        return [delete_obj(objs[0]), delete_obj(objs[1])]
```

A pair event lists the allowed type pairs and a cut-off; a candidate event exists for every
pair of objects of those types within `max_distance`. This one writes its rate as a plain
Python method instead of a kernel; chapter 2 compares the two forms.

**The seven blocks.**

| Block | Role in this script |
|---|---|
| `domain` | the box is `lattice_grid × lattice_constant` = 20 × 2.87 Å per side, periodic; `max_interaction_distance` must be at least the largest `max_distance` of any pair event, because it sets the width of the halo a rank sees of its neighbours |
| `simulation` | 2000 steps or 1 s of simulated time, whichever comes first; the seed; a progress line every 500 steps |
| `objects` | two types, each with a `size` and the scalar `properties` the rates read |
| `ic` | 40 + 40 objects at random positions |
| `events` | the registered classes, under the names the output will use |
| `physics` | named scalars any rate may depend on |
| `outputs` | the cadence of post-processing (nothing registered yet) |

**Try it.** Run the script on one rank and then on two. Both print a line like

```
steps = 2000, simulated time = 6.737e-09 s, objects left = 64
```

The simulated time is the sum of the residence times of the 2000 steps; it is a few
nanoseconds because the interstitials jump at about 10¹⁰ Hz at 600 K. The two runs differ
slightly in time and in the number of objects left: the trajectories are two samples of the
same stochastic process (chapter 8 says more). Run the one-rank case twice and the numbers
are identical.

---

## 2. The two ways to write a rate

**The idea.** A rate can be a Python method or a compiled kernel. Both are evaluated for
every candidate event, every step, unless the engine can prove that the event did not change.
The Python form is the general one; the kernel form is the fast one and comes with a strict
rule about what it may read.

`02_two_rate_styles.py` contains the Frenkel-pair model twice. `--style python` uses

```python
class MigrationPy(SingleObjectEvent):
    object_types = ['I', 'V']

    @classmethod
    def rate(cls, objs, params, config):
        o = objs[0]
        T = config['physics'].get_param('temperature')
        return o['nu'] * np.exp(-o['E_m'] / (K_B * T))


class CapturePy(PairObjectEvent):
    type_pairs = [('I', 'V')]
    max_distance = 6.0
    params = ['distance']            # ask the engine to put the pair distance in params

    @classmethod
    def rate(cls, objs, params, config):
        r = max(params['distance'], 0.5)
        return 1.0e12 * (2.0 / r)
```

and `--style kernel` uses

```python
class MigrationKernel(SingleObjectEvent):
    object_types = ['I', 'V']
    rate_deps = RateDeps(obj1=['E_m', 'nu'], physics=['temperature'])

    @staticmethod
    @njit
    def rate_kernel(E_m, nu, temperature):
        return nu * np.exp(-E_m / (8.617333262e-5 * temperature))


class CaptureKernel(PairObjectEvent):
    type_pairs = [('I', 'V')]
    max_distance = 6.0
    rate_deps = RateDeps(kind='pair')

    @staticmethod
    @njit
    def rate_kernel(distance):
        r = max(distance, 0.5)
        return 1.0e12 * (2.0 / r)
```

**What a kernel receives.** The arguments of `rate_kernel` are, in this order:

1. the properties of the first object named in `rate_deps.obj1`,
2. for pair events, the properties of the second object named in `rate_deps.obj2`,
3. for pair events, `distance`: the minimum-image distance between the two objects,
4. the field values named in `rate_deps.fields`, interpolated at the object position
   (chapter 6),
5. the physics parameters named in `rate_deps.physics`.

Two consequences:

* **Coordinates are not passed.** A kernel cannot compute a geometric condition from the
  object positions. If a rate needs a position (distance to a plane, to a line, to a
  surface), either write it in Python, or mirror the coordinates it needs into object
  properties that `execute()` keeps current. The dislocation case in
  `example/3.2_sink_strength/run_dislocation_sink_strength.py` shows both forms side by
  side: the Python `rate()` tests the capture cylinder from `x, y, z`; the commented-out
  compiled variant reads mirrored `pos_y`, `pos_z` properties.
* **A kernel with no object property must say what it is.** `RateDeps` infers the event
  kind from the declared properties (`obj1` only → single, `obj1` and `obj2` → pair,
  neither → zero-object). `CaptureKernel` declares none, so it writes `RateDeps(kind='pair')`
  and its kernel receives `distance` alone.

For a pair kernel, the first object is always the one whose type comes first in
`type_pairs` ( `('I', 'V')` → `obj1` is the `I` ), whatever the order in which the pair was
found.

**Try it.** Run both styles:

```
style=python: steps=3000 time=3.868133e-08 s objects=8  rate-eval 0.640 ms/step
style=kernel: steps=3000 time=3.868133e-08 s objects=8  rate-eval 0.422 ms/step
```

The trajectories are identical to the last digit: the numbers computed are the same, only
the way they are computed differs. The kernel is faster per step even for this tiny
catalogue; with 10⁴ to 10⁵ candidate events per step the difference is one to two orders of
magnitude, because the Python form calls one function per event.

**When to use which.** Use kernels for the rates that are evaluated most often (migration
of the mobile species, the common pair reactions). Use Python for rates that need
coordinates, string logic, lookups in tables or an external model, and for rates that are
evaluated rarely. Mixing the two in one model is normal.

---

## 3. Objects, properties and the consequence of an event

**The idea.** An object is a record with a position, a `size`, a type and the scalar
`properties` declared for its type. `execute()` describes what an event does to objects with
three helpers:

| Helper | Meaning |
|---|---|
| `change_obj(obj, **fields)` | update coordinates and/or properties of an existing object |
| `delete_obj(obj)` | remove it |
| `new_obj(type=..., x=, y=, z=, size=, float_props=[...])` | create one |

`03_objects_and_execute.py` forms and dissolves clusters:

```
H + V  -> HV        capture (a new object)
HV     -> H + V     dissociation (the cluster disappears, two objects appear)
H                   migrates; V and HV do not
```

**Changing a property.** The migration event counts the jumps of each `H` in a property:

```python
return [change_obj(o, x=new[0], y=new[1], z=new[2], jumps=o['jumps'] + 1)]
```

Any declared property can be passed to `change_obj` by name.

**Creating an object.** A new object must be complete: type, position, `size`, and its
whole property vector. Properties are stored in a fixed-length array (`float_props`) whose
slot order the engine assigns at start-up; the mapping is available at run time as
`config['_props_mapping']['float_props'][type]`. The script wraps this in a helper:

```python
def make_props(config, obj_type, **overrides):
    n = config['_object_dtype'].fields['float_props'][0].shape[0]
    slots = config['_props_mapping']['float_props'][obj_type]
    props = np.zeros(n)
    for name, value in config['objects'][obj_type].get('properties', {}).items():
        props[slots[name]] = value                     # declared defaults
    for name, value in overrides.items():
        props[slots[name]] = value                     # what this event decides
    return props.tolist()
```

so that capture reads

```python
h, v = (objs[0], objs[1]) if objs[0]['type'] == 'H' else (objs[1], objs[0])
return [delete_obj(h), delete_obj(v),
        new_obj(type='HV', x=v['x'], y=v['y'], z=v['z'],
                size=config['objects']['HV']['size'],
                float_props=make_props(config, 'HV', h_jumps=h['jumps']))]
```

Note the first line: in a pair event the order of `objs` follows `type_pairs` for compiled
rates but is not something `execute()` should rely on, so test the type. New objects get a
unique id from the engine (ids are 64-bit integers that encode the creating rank).

**A post-processor** is any callable `f(step, time, pool, comm)` listed under
`outputs.postprocessors`; it runs every `outputs.interval` steps on every rank.
`pool.get_local_objects()` returns the objects this rank owns as a NumPy structured array;
reduce over `comm` for a global number:

```python
def count_types(step, time, pool, comm):
    local = {}
    for o in pool.get_local_objects():
        t = pool.get_type_name(o)
        local[t] = local.get(t, 0) + 1
    totals = comm.gather(local, root=0)
    ...
```

**Try it.**

```
  step      0  t = 2.424e-14 s  H=49  HV=1  V=24
  step   1000  t = 6.165e-10 s  H=38  HV=12  V=13
  step   2000  t = 1.315e-09 s  H=34  HV=16  V=9
  step   4000  t = 2.876e-09 s  H=32  HV=18  V=7
```

The populations reach a steady balance between capture and dissociation. Open the script
and lower `E_b` of `HV` to 0.6 eV: dissociation becomes fast and few clusters survive.

---

## 4. Your own initial condition, and how random numbers work

**Initial condition from a function.** `ic.mode = "function"` hands the engine

```python
def initial_objects(rank, size, config) -> list[dict]
```

Every rank calls it once and must return **only the objects inside its own subdomain**.
The simplest correct pattern generates the whole configuration identically on every rank from
a seeded generator and keeps what the rank owns:

```python
def two_slabs_ic(n_per_slab):
    def initial_objects(rank, size, config):
        dd = config['_domain_decomp']
        box = np.asarray(config['grid_size'], dtype=float) * config['lattice_constant']
        gen = np.random.default_rng(config['_user_seed'])      # same stream on every rank
        mine = []
        for k in range(2 * n_per_slab):
            pos = ...                                           # drawn from gen
            if dd.get_owner_rank(pos) == rank:
                mine.append({'id': k + 1, 'type': 'A', 'x': pos[0], 'y': pos[1], 'z': pos[2],
                             'size': 1.0, 'float_props': []})
        return mine
    return initial_objects
```

Each dictionary needs `id` (unique across all ranks), `type`, `x`, `y`, `z`; give `size` and
`float_props` when the type has them (see chapter 3 for the slot order). The generator above
is seeded with `config['_user_seed']`, which is `simulation.random_seed`, so a different seed
gives a different initial state as well as a different trajectory. For millions of objects
generate the coordinates as arrays and mask them with the subdomain bounds instead of
calling `get_owner_rank` per object (`example/4.1_benchmark_workloads/pair_scaling.py` does
this).

**Random numbers inside events.** `config['rng']` is the only generator an event should use.
Every draw is a pure function of `(random_seed, step, draw index)`: the tuple seeds NumPy's
`SeedSequence`, which seeds a PCG64 generator, whose first output is the number. Nothing is
carried between steps, so

* a run is reproducible from its seed: the same seed on the same number of ranks gives the
  same trajectory (a different rank count gives a different sample of the same process,
  chapter 8);
* every rank that asks for the *k*-th number of step *n* gets the same value, so no random
  state ever has to be communicated;
* `np.random` inside an event would break this: each rank would see its own stream and
  the ranks would disagree about what happened.

The methods are `rng.random()`, `rng.uniform(a, b)`, `rng.normal(mu, sigma)`,
`rng.randint(lo, hi)` (half-open) and `rng.choice(seq)`. Rates must not draw random numbers
at all: they are recomputed at the engine's discretion and a stochastic rate would make the
total rate depend on when it was last evaluated.

**Try it.** The script prints a fingerprint (a weighted sum of all coordinates) at steps 0,
1000 and 2000:

```
python 04_initial_condition_and_random_numbers.py --seed 5
  step  2000  t = 5.053982e-12  fingerprint = 27968.301200
python 04_initial_condition_and_random_numbers.py --seed 5
  step  2000  t = 5.053982e-12  fingerprint = 27968.301200
python 04_initial_condition_and_random_numbers.py --seed 6
  step  2000  t = 5.070687e-12  fingerprint = 28106.869662
```

Same seed, same numbers to the last digit. Under `mpirun -np 2` the fingerprint differs from
the one-rank value but is again identical from run to run. See `docs/RANDOM_NUMBERS.md` for the full account.

---

## 5. Zero-object events

**The idea.** Some processes belong to no object: irradiation creating defects, a thermal
source, a global switch. A `ZeroObjectEvent` is a candidate event that exists by itself. By
default exactly one copy exists, on rank 0 (`owner_rank_mode = 'rank0'`), and its rate is
added to the global total like any other. When it is selected, its `execute()` receives an
empty object list and may create objects anywhere in the box.

`05_zero_object_events.py` starts with an **empty** box and lets a source create Frenkel
pairs at `pair_production_rate` per second while `I` and `V` migrate and recombine:

```python
class FrenkelSource(ZeroObjectEvent):
    rate_deps = RateDeps(physics=['pair_production_rate'])   # no objects: kind = zero

    @staticmethod
    @njit
    def rate_kernel(pair_production_rate):
        return pair_production_rate

    @classmethod
    def execute(cls, event, objs, config):
        rng, box = config['rng'], box_of(config)
        pos = np.array([rng.random(), rng.random(), rng.random()]) * box
        ...
        return [new_obj(type='V', x=..., size=1.0, float_props=default_props(config, 'V')),
                new_obj(type='I', x=..., size=1.0, float_props=default_props(config, 'I'))]
```

The new objects are given the property defaults declared for their type (`default_props`
builds the vector as in chapter 3). An empty initial condition is written
`"ic": {"mode": "builtin", "initial_conditions": {}}`.

**Try it.**

```
  step      0  t = 1.000e-10 s  objects = 2
  step   1000  t = 6.136e-09 s  objects = 210
  ...
  step   6000  t = 1.922e-08 s  objects = 386
```

The population grows and then saturates as recombination catches up with production. The
first residence time is 1/(2·10¹⁰ Hz) = 5·10⁻¹¹ s on average: with nothing else in the box
the source is the only event.

A second spatial mode exists for processes that depend on a local field: with
`spatial_mode = 'per_cell'` one candidate event is created per cell of the decomposition
and the fields named in `field_deps` are passed in `params`. See the `ZeroObjectEvent`
docstring and `docs/USER_GUIDE.md`.

---

## 6. Physics parameters that change in time, and a spatial field

**The idea.** `physics.params` are named scalars that rates read by name; `physics.fields`
are 3-D grids spanning the box, interpolated at an object's position when a rate needs
them. `physics.ic_func(physics, config)` runs once before the first step on every rank and
is where you fill a field with data and register scheduled updates.

`06_physics_parameters_and_fields.py` anneals a population of interstitials while the
temperature ramps from 300 to 900 K, in a static stress field that lowers the migration
barrier towards +x.

```python
def physics_ic(physics, config):
    nx, ny, nz = physics.get_field('stress').shape
    x = np.linspace(0.0, 1.0, nx)
    physics.set_field('stress', np.broadcast_to(x[:, None, None], (nx, ny, nz)).copy())

    def ramp(physics, step, time):
        return min(900.0, 300.0 + 0.1 * step)
    physics.set_param_update('temperature', ramp, step_interval=200)
```

```python
"physics": {
    "params": {"temperature": 300.0},
    "fields": {"stress": {"shape": [41, 11, 11]}},
    "ic_func": physics_ic,
},
```

A field of shape `[nx, ny, nz]` has its first and last grid points on the box faces. When a
scheduled update changes a parameter or a field, the engine invalidates every rate that
depends on it, so the next step sees the new value everywhere.

**Reading the field in a rate.** The compiled form declares it and receives the
interpolated value as an argument, after the object properties and before the physics
parameters:

```python
class MigrationKernel(SingleObjectEvent):
    object_types = ['I']
    rate_deps = RateDeps(obj1=['E_m', 'nu'], fields=['stress'], physics=['temperature'])

    @staticmethod
    @njit
    def rate_kernel(E_m, nu, stress, temperature):
        return nu * np.exp(-(E_m - 0.05 * stress) / (8.617333262e-5 * temperature))
```

The Python form asks for an interpolator:

```python
    @classmethod
    def rate(cls, objs, params, config):
        o = objs[0]
        T = get_physics_param(config, 'temperature')
        stress = float(get_physics_field(config, 'stress')(o['x'], o['y'], o['z']))
        return o['nu'] * np.exp(-(o['E_m'] - COUPLING * stress) / (K_B * T))
```

**An event hook** is a callable `f(event_type, changed, step, time, pool, comm, config)`
run on every rank after every applied event (chapter 7 says more). Here one prints the
temperature and the mean position every 1000 steps; it is the right place because, unlike a
post-processor, it receives `config` and so can read the physics.

**Try it.**

```
  step      0  t = 0.000e+00 s  T =  300.0 K  <x>/L = 0.488
  step   1000  t = 3.655e-08 s  T =  400.0 K  <x>/L = 0.466
  step   2000  t = 3.927e-08 s  T =  500.0 K  <x>/L = 0.473
  ...
```

The simulated time per 1000 steps shrinks by orders of magnitude as T rises. `<x>/L` drifts
below 0.5: the hop *rate* grows towards +x but the jumps stay isotropic, so objects spend
less time where they move fast and the population piles up on the slow side. A
position-dependent hop rate changes the diffusivity, not the direction of motion; a drift
needs direction-dependent rates.

Run `--style python` and compare: identical output, because identical numbers.

**Further coupling.** A field or a parameter may also be written *by an event*
(`physics.set_param_in_event`, `physics.add_field_delta_in_event`, applied on every rank with
the broadcast state change), updated on a step or time interval from the objects
(`fields.<name>.update_func` with `queries`), or produced by an external solver
(`physics.solvers`, `physics.triggers`). `docs/USER_GUIDE.md` lists the options.

---

## 7. Output, observers and restarting from a saved frame

Three ways to look at a running simulation, all declared in `outputs`:

```python
"outputs": {
    "interval": 500,
    "ovito_dump": {"enabled": True, "output_dir": DUMP_DIR, "prefix": "frenkel",
                   "mode": "distributed",
                   "properties": ["id", "type", "x", "y", "z", "size", "jumps"]},
    "postprocessors": [summary],
    "event_hooks": [count_recombinations],
},
```

**OVITO dumps.** A LAMMPS-style text frame is written every `interval` steps and once more
after the last step. By default (`mode: "distributed"`) each rank writes the objects it owns
to `prefix.<step>.rank_<r>.dump`, with no gather; OVITO loads a frame through the wildcard
`frenkel.1500.rank_*.dump`. `mode: "collect"` gathers everything to rank 0 into one file.
The columns are those listed in `properties`: the standard `id type x y z size owner_rank
type_id` plus any declared object property. `type` is written as the type name.

**Post-processors** run every `interval` steps (chapter 3).

**Event hooks** run on every rank after every applied event:

```python
def count_recombinations(event_type, changed, step, time, pool, comm, config):
    if event_type == 'recombination':
        STATS['recombinations'] += 1
```

`event_type` is the registered name of the class that fired; `changed` is the broadcast
record of what the event did, a structured array with one row per created, modified or
deleted object, and the deleted rows still carry their full data. Every rank receives the
same record, so a count kept in a hook is the same on every rank without any MPI of your own.
The Section 3 cases of the paper use a hook to collect the jump count of every absorbed defect.

**Restart.** A dump is an initial condition:

```python
"ic": {"mode": "dump", "file": DUMP_DIR, "frame": -1, "id_mode": "keep"}
```

`file` is a single dump file or a dump directory; `frame` is a step number, `-1` the last
frame present; per-rank files of one frame are merged. Positions, sizes and every declared
property present in the file are restored, and each rank keeps the objects inside its own
subdomain, so the restart may use **a different number of ranks**. `id_mode: "keep"` keeps
the ids from the file (a true continuation); `"renumber"` assigns fresh ones.

**Try it.**

```bash
python 07_outputs_and_restart.py                        # 1500 steps -> out/07_dump/
mpirun -np 2 python 07_outputs_and_restart.py --restart   # 1000 more steps on two ranks
```

```
  step   1500  t = 5.136e-10 s  objects = 258  recombinations so far = 21
done: 1500 steps, 258 objects; frames ['0', '500', '1000', '1500'] in .../out/07_dump
  step      0  t = 1.167e-13 s  objects = 258  recombinations so far = 0
  ...
done: 1000 steps, 222 objects; frames ['0', '500', '1000'] in .../out/07_dump_restart
```

The restarted run begins with the 258 objects the first run ended with. Open
`out/07_dump/frenkel.1500.rank_*.dump` in OVITO and colour by the `jumps` column.

---

## 8. Running in parallel and the load-balancing switches

**What the rank count changes.** Nothing in a model refers to ranks. The engine factors the
rank count into a cuboid grid of subdomains, gives every rank the objects inside its own and
a halo of width `max_interaction_distance` around it, and runs the residence-time algorithm
*exactly*: every step, every rank evaluates the rates of its own candidate events, one
`allgather` collects the partial sums together with the selection random number drawn on
rank 0, every rank identifies the same winning event, the owner executes it and broadcasts
one state-change message that every rank applies. One event, one clock advance per step, on
any number of ranks.

The statistics are therefore the same on 1 and on 64 ranks. The trajectory is not bitwise
identical, because the catalogue of candidate events is enumerated in a different order and
the selection random number lands on a different event. A model that gives different
*averages* on different rank counts has a bug (usually an initial condition that is not
decomposition-safe, or `np.random` inside an event).

**Load balancing.** Defects cluster; a plain decomposition then leaves most ranks idle. Three
levels redistribute the work, all switched in the `simulation` block:

| Level | Key | What it does | Default |
|---|---|---|---|
| 1 | `object_load_balancer` | every `balance_interval` steps, re-cuts the box by object density (recursive coordinate bisection) when max/mean − 1 of the object counts exceeds `imbalance_threshold` | off; `balance_interval` 500, `imbalance_threshold` 0.2 |
| 2 | `predictive_rebalance` | every `check_interval` steps, moves the records of events whose rate is still to be computed from over- to under-loaded ranks, weighted by their cost, when their imbalance exceeds `threshold`; off by itself when every event class has `incremental_rate = True` | on; `threshold` 1.5, `check_interval` 100 |
| 3 | `runtime_load_balancer` | during the Python-rate loop, an idle rank takes pending events from a busy one; a busy rank looks for idle ranks every `check_interval` evaluated events | on; `check_interval` 32 |

Level 1 moves objects. Levels 2 and 3 move *event records*, which are self-contained (they
carry copies of their objects); a moved record stays with the rank that received it, and the
broadcast state change invalidates it there when one of its objects changes. None of the
levels changes the sampled statistics, because every event is counted in exactly one partial
sum (`tests/test_lb_rate_conservation.py` checks this at every step).

Two properties decide which level can help a given model:

* Levels 2 and 3 balance the *rates still to be computed*. With `incremental_rate = True`
  almost every rate is reused, so there is little left to balance; level 2 switches itself
  off when every class reuses its rate.
* Level 3 steals only events of Python-rate classes. Compiled kernels evaluate all their
  events in one batch, which leaves nothing to steal.

`08_parallel_runs_and_load_balancing.py` puts the particles in a 10 % cube near one corner
of the box, a hotspot that a single rank owns. `--levels` picks the levels and `--rate`
picks the hop class: `kernel` is a compiled rate with rate reuse, `python` a Python rate
recomputed every step.

```bash
mpirun -np 4 python 08_parallel_runs_and_load_balancing.py --levels ""  --steps 600
mpirun -np 4 python 08_parallel_runs_and_load_balancing.py --levels 1   --steps 600
mpirun -np 4 python 08_parallel_runs_and_load_balancing.py --levels 23  --steps 600
mpirun -np 4 python 08_parallel_runs_and_load_balancing.py --rate python --n 5000 --steps 200 --levels 3
```

Measured on a 10-core workstation (wall time per step):

| Ranks | Levels | `--rate kernel`, 20 000 particles | `--rate python`, 5 000 particles |
|---|---|---|---|
| 1 | none | 25.8 ms | 91.3 ms |
| 4 | none | 27.6 ms | 99.0 ms |
| 4 | 1 | 17.7 ms | 41.2 ms |
| 4 | 3 | not run | 84.9 ms |
| 4 | 2 + 3 | 28.0 ms | 88.0 ms |
| 4 | 1 + 3 | not run | 68.7 ms |

The kernel case was not run with level 3 alone or with levels 1 + 3: level 3 has nothing to
steal there (see below).

What the numbers say:

* Without balancing, four ranks are no faster than one: the rank that owns the hotspot does
  all the work and the others wait.
* Level 1 moves the cuts into the hotspot and gives the largest gain in both cases.
* With the compiled, reused rate, levels 2 and 3 change nothing: level 2 is off by itself and
  level 3 finds nothing to steal.
* With the Python rate, level 3 helps a little on its own. Once level 1 has balanced the
  objects, adding level 3 makes the step slower: the waiting it removes is smaller than the
  messages and the end-of-stage barrier it adds.

Start with level 1 for a hotspot that persists. Enable levels 2 and 3 for models whose
expensive rates are Python and recomputed every step, and measure before keeping them.

The per-step averages that the script prints add up to less than the wall time when level 2
is on: the redistribution itself is timed separately.

The measured strong- and weak-scaling curves and the load-balancing study of the paper, with
their analysis scripts, are in `example/4.*`.

**Practical settings.**

* One MPI rank per core, kernels single-threaded: `NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1`.
* `incremental_rate = True` on an event class lets the engine keep the rate of an event whose
  objects did not change instead of recomputing it. It is safe when the rate depends only on
  object properties, the pair distance and parameters that do not change in time; leave it
  off for rates that follow a scheduled parameter or a field that is updated.
* Pool capacities (`rma_object_pool`, `rma_event_pool`) auto-expand, but an expansion rebuilds
  an MPI window; size them for the expected population per rank, not for the global count.
* `stats = engine.run()` returns `total_steps`, `total_time`, `total_events`, `object_count`
  and the per-step averages `avg_build_time`, `avg_rate_time`, `avg_select_time`,
  `avg_execute_time` (seconds, rank 0 only). They are the first thing to look at when a run
  is slower than expected.

---

## Where to go next

* `docs/USER_GUIDE.md`: every key of the seven blocks, the event-class contract, the
  run-time `config` entries available to your functions, validation errors.
* `docs/RANDOM_NUMBERS.md`: the counter-keyed generator in detail.
* `docs/ARCHITECTURE.md`: data layout, the per-step protocol and the balancing levels.
* `example/`: the validation cases of the paper (free diffusion, sink strengths, the 1D→3D
  transition, invariance under parallel execution) and its benchmarks, each with the model
  script and the run commands.
