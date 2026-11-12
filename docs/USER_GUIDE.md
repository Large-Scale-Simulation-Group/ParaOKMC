# ParaOKMC user guide

*中文版：[`docs_zh_cn/USER_GUIDE.md`](../docs_zh_cn/USER_GUIDE.md)*

The reference for writing a model. For a guided introduction read `docs/TUTORIAL.md` first;
for the design see `docs/ARCHITECTURE.md`.

A model is

1. one Python class per event type (section 2),
2. a dictionary of seven blocks (section 3),
3. `engine = OKMCengine(config); stats = engine.run(); engine.destroy()`.

The same script runs on one rank or under `mpirun -np N`. Run one rank per core with the
kernels single-threaded: `export NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1`.

```python
from paraokmc import (OKMCengine, ZeroObjectEvent, SingleObjectEvent, PairObjectEvent,
                      RateDeps, change_obj, delete_obj, new_obj,
                      get_physics_param, get_physics_field, ConfigValidationError)
```

---

## 1. Objects

An object is a record with

| Field | Meaning |
|---|---|
| `id` | 64-bit integer, unique in the run; assigned by the initial condition or by the engine for created objects |
| `type` | the type name (a string in user code; an integer `type_id` internally) |
| `x`, `y`, `z` | position in the box, in the length unit of `lattice_constant` (Å throughout the examples) |
| `size` | a diameter: the influence range of an object is `size/2 + max_interaction_distance`; an object whose influence range exceeds what the neighbour halo can hold is a *large object*, stored on every rank |
| `float_props` | the scalar properties declared for its type, in a fixed slot order assigned at start-up |

Inside `rate()` and `execute()` an object arrives as a dictionary with these fields plus
every declared property by name (`obj['E_m']`). Inside post-processors and hooks the
objects are NumPy structured arrays with the raw fields; `pool.get_type_name(rec)` gives the
type name and `config['_props_mapping']['float_props'][type][prop]` the slot of a property.

---

## 2. Event classes

### 2.1 Which base class

| Base class | Objects involved | Candidate events | Typical use |
|---|---|---|---|
| `ZeroObjectEvent` | none | one per run (default) or one per cell | a source, a global process |
| `SingleObjectEvent` | one | one per object of the listed types | migration, emission, transformation |
| `PairObjectEvent` | two | one per pair of the listed types within `max_distance` | recombination, capture, coalescence |

### 2.2 Class attributes

| Attribute | On | Meaning | Default |
|---|---|---|---|
| `object_types` | single | list of type names the event applies to | required |
| `type_pairs` | pair | list of `(typeA, typeB)` tuples; the first member of each tuple is `obj1` for the compiled rate | required |
| `max_distance` | pair | cut-off for candidate pairs (minimum-image distance); must not exceed `domain.max_interaction_distance` | required |
| `params` | all | names the engine puts into the `params` dict of `rate()`; resolved when the event is built, from the object properties first, then `physics.params`; for pair events the name `distance` yields the pair distance; at most 8 names | `[]` |
| `rate_deps`, `rate_kernel` | all | the compiled rate (2.3) | none |
| `incremental_rate` | all | `True`: keep the rate of an event whose objects did not change instead of recomputing it | `False` |
| `weight` | all | relative cost used by the balancing levels; detected automatically (compiled kernel 1, Python rate 1000) unless set | auto |
| `spatial_mode` | zero | `'global'` (one event) or `'per_cell'` (one per decomposition cell, fields in `field_deps` passed in `params`) | `'global'` |
| `owner_rank_mode` | zero | `'rank0'` (the event lives on rank 0) or `'round_robin'` (one per rank) | `'rank0'` |
| `field_deps` | zero, per_cell | field names interpolated at the cell centre into `params` | `[]` |

### 2.3 The rate

Exactly one of the two forms must be present.

**Python.** `rate(cls, objs, params, config) -> float`, called once per candidate event.
`objs` is the list of object dictionaries (empty for zero-object events), `params` the dict
built from the `params` attribute, `config` the run-time configuration (section 5). It may
read coordinates, call `get_physics_param(config, name)` and `get_physics_field(config, name)`
(an interpolator `f(x, y, z)`), or anything else. It must be deterministic and must not draw
random numbers.

**Compiled.** `rate_deps = RateDeps(obj1=[...], obj2=[...], fields=[...], physics=[...],
kind=None)` declares the inputs; `rate_kernel` is a Numba `@njit` scalar function (also
accepted: a plain Python function, slower) whose positional arguments are, in order,

1. the `obj1` properties,
2. the `obj2` properties (pair events),
3. `distance` (pair events),
4. the `fields` values interpolated at the position of `obj1`,
5. the `physics` scalars.

The engine evaluates it for the whole event array in one call. Coordinates are not passed;
a geometric condition must either be written in Python or read mirrored coordinates kept in
properties by `execute()` (`docs/TUTORIAL.md`, chapter 2). `kind` is inferred from the
declaration (`obj1` only: single; `obj1` and `obj2`: pair; neither: zero) and must be given
explicitly when a pair or single kernel declares no object property (`RateDeps(kind='pair')`).
Every name in `obj1`/`obj2` must be a declared property of every type the event applies to;
every name in `physics` a key of `physics.params`; every name in `fields` a declared field.
A kernel defined inside the class is written

```python
    @staticmethod
    @njit
    def rate_kernel(E_m, nu, temperature):
        ...
```

A negative or non-finite rate is clamped to zero.

### 2.4 The consequence

`execute(cls, event, objs, config) -> list[dict]` is plain Python. It returns the changes as
a list built from

| Helper | Effect |
|---|---|
| `change_obj(obj, **fields)` | update `x`, `y`, `z`, `size` and/or declared properties (by name) of an existing object |
| `delete_obj(obj)` | remove the object |
| `new_obj(type=, x=, y=, z=, size=, float_props=[...])` | create an object; the engine assigns the id; `float_props` must be the complete property vector in slot order (see `docs/TUTORIAL.md`, chapter 3, for a helper) |

Random numbers come from `config['rng']` only (`docs/RANDOM_NUMBERS.md`). The returned list
is broadcast as one state-change message and applied on every rank; an object created and
deleted in the same list never exists. An event may also change the physics
(`config['physics'].set_param_in_event(name, value)`,
`config['physics'].add_field_delta_in_event(field, position, delta, radius)`); the change
travels with the message and is applied on every rank. Declare every parameter an event may
set in `physics.params` beforehand: a name unknown to the physics module is created only on
the rank that executes the event, and the other ranks receive its index without its name.

### 2.5 Filtering candidate events

`filter()` may be overridden to exclude objects from a single-object event. It is called as
`filter(cls, obj_types, obj_props, config)` with the type array of the candidate objects;
`obj_props` is currently passed as an empty dict, so the filter can decide by type and by
`config` only. It returns a boolean mask over the candidates, or `False` to suppress the
event entirely. For a condition on positions or properties, return a rate of zero from the
rate instead.

---

## 3. The seven configuration blocks

Every block must be present (an empty dictionary is allowed where noted). Unknown top-level
keys are rejected.

### 3.1 `domain`

| Key | Type | Meaning |
|---|---|---|
| `lattice_constant` | float > 0 | length unit of the grid; the box is `lattice_grid × lattice_constant` |
| `lattice_grid` | `[nx, ny, nz]` ints > 0 | cells per axis |
| `periodic` | `[bool, bool, bool]` | periodic boundary per axis |
| `max_interaction_distance` | float > 0 | halo width each rank keeps of its neighbours; must be ≥ every `max_distance` |
| `h_factor` | int ≥ 1, optional | subdivides every lattice cell into `h³` decomposition cells (finer cell lists) |

The rank count is factored into a cuboid grid of subdomains over this box; nothing else is
configurable about the decomposition. `box_size` is derived and available at run time as
`config['grid_size'] * config['lattice_constant']`.

### 3.2 `simulation`

| Key | Default | Meaning |
|---|---|---|
| `max_steps` | required | stop after this many steps |
| `max_time` | `inf` | stop when the simulated time exceeds this (s) |
| `random_seed` | 42 | the seed of every random number in the run |
| `verbose` | `False` | engine diagnostics and the end-of-run physics summary |
| `progress_interval` | 500 | steps between progress lines (0: none) |
| `predictive_rebalance` | `{"enabled": True, "threshold": 1.5, "check_interval": 100}` | level 2: every `check_interval` steps, move weighted event records before rate evaluation when the imbalance of the events still to be evaluated exceeds `threshold`; switches itself off when every event class has `incremental_rate = True` |
| `runtime_load_balancer` | `{"enabled": True, "check_interval": 32}` | level 3: work stealing during the Python-rate loop (events with a compiled kernel are evaluated in one batch and are never stolen); a stolen event stays with the thief, like an event moved by level 2; when one of its objects changes, the broadcast state-change message invalidates it on the thief |
| `object_load_balancer` | `{"enabled": False, "balance_interval": 500, "imbalance_threshold": 0.2, "min_improvement": 0.05}` | level 1: every `balance_interval` steps, when max/mean − 1 of the per-rank object counts exceeds the threshold, recompute the cuts by distributed RCB and adopt them if the imbalance drops by at least `min_improvement` |
| `adaptive_weight` | `{"enabled": False, "update_interval": 1000}` | re-measure event weights from wall-clock timings |
| `rma_object_pool` | `{"initial_capacity": 1000, "auto_expand": True}` | per-rank object pool |
| `rma_event_pool` | `{"capacity_zero": 10, "capacity_single": 5000, "capacity_pair": 5000, "auto_expand": True}` | per-rank event pools |

The structural incremental update of the event catalogue is always on. Pool expansion rebuilds an MPI
window; size the pools for the expected per-rank population.

### 3.3 `objects`

```python
"objects": {
    "I": {"size": 1.0, "properties": {"E_m": 0.34, "nu": 6.0e12}},
    "S": {"size": 20.0},
}
```

One entry per type name. `size` (float ≥ 0, default 0) and `properties` (dict of scalars)
are optional. Properties are the only per-object data a compiled rate can read; their
declared values are the defaults the built-in initial condition assigns and the values
restored when a dump lacks the column.

### 3.4 `ic`

| `mode` | Keys | Behaviour |
|---|---|---|
| `"builtin"` | `initial_conditions: {type: {"count": n, "distribution": "random" \| "lattice"}}` | every rank generates the same configuration from the seed and keeps its own share; `"random"` places uniformly, `"lattice"` fills cell centres in index order; an empty dict starts with no objects |
| `"function"` | `func(rank, size, config) -> list[dict]` | called once per rank; must return only the objects inside this rank's subdomain (`config['_domain_decomp'].get_owner_rank([x, y, z]) == rank`), each as `{'id', 'type', 'x', 'y', 'z', 'size', 'float_props'}` with ids unique across ranks |
| `"dump"` | `file` (dump file or directory), `frame` (step; −1 = last), `id_mode` (`"keep"` \| `"renumber"`) | restore a saved frame; per-rank files merged; the rank count may differ from the writing run |

### 3.5 `events`

`{"name": EventClass, ...}`. The name is what `event_type` carries in hooks and what the
output reports. A class may be registered under several names; an unregistered class does
not run. Validation checks that every type named by a class is declared, that pair classes
have a positive `max_distance`, and that every `rate_deps` name exists.

### 3.6 `physics`

May be empty. Keys:

| Key | Meaning |
|---|---|
| `params` | `{name: scalar}`; readable by `rate_deps.physics`, `get_physics_param`, `config['physics'].get_param(name)` |
| `fields` | `{name: {"shape": [nx, ny, nz], "initial_value": v, "update_func": f, "update_policy": {...}, "queries": {...}}}`; a uniform grid over the box with grid points on the faces, interpolated trilinearly |
| `ic_func` | `f(physics, config)`: runs once before step 0 on every rank; use it to fill fields (`physics.set_field(name, array)`) and to register scheduled parameter updates (`physics.set_param_update(name, f(physics, step, time) -> value, step_interval=…, time_interval=…)`) |
| `update_func` | `f(physics, step, time, config) -> bool`: a global update called every step; return `True` when something changed |
| `solvers` | `[{"name": str, "solver": ExternalSolver, "fields": [...]}]`; an object with `initialize(physics, config)`, `solve(physics, step, time, object_data=None, trigger_source=None) -> {name: array \| scalar}` and `finalize()` |
| `triggers` | `[{"event": name, "fields": [...], "solver": name (optional), "condition": f (optional)}]`: request a field update after the named event fires |
| `field_coupling` | `{"mode": "inward" \| "outward" \| "two_way", "enable_cache": bool}` |

`update_policy` of a field: `step_interval`, `time_interval`, `enabled`, `trigger_events`
(list of event names), `trigger_on_object_create/change/delete` (bool) with `object_types`
(list). `queries` names object snapshots the update function receives:
`{"sinks": {"types": ["S"]}, "all": {"all": True}}`.

A field `update_func` is recognised by its positional parameters:

| Signature | Called as |
|---|---|
| `f(physics, snapshots, ctx)` (second parameter named `snapshots`, `snapshot`, `objects` or `object_snapshots`) | the query snapshots and a context with `step`, `time`, `field_name`, `trigger_source`, `trigger_reason` |
| `f(physics, step, time, config)` | four or more positional parameters otherwise |
| `f(physics, step, time)` | three |
| `f(physics, step)` / `f(physics)` | two / one |

It returns an array (the new field), a dict of arrays and scalars (fields and parameters to
set), `True`/`False` (changed or not, after setting fields itself), or `None`.

**What a physics change invalidates.** A change never rebuilds the event structure; it marks
for recomputation (status TODO) the rates of the classes that depend on the changed name:

* a class with `rate_deps`: the names in `rate_deps.physics` and `rate_deps.fields`;
* a Python-rate class: the names in its `params` that resolve to physics parameters (the
  value cached in the event is rewritten as well);
* a Python-rate class with `incremental_rate = True` and no `params`: every parameter and
  field, since its `rate()` can only read them directly.

A Python-rate class without `incremental_rate` recomputes every rate every step anyway. A
global `update_func` invalidates every class.

### 3.7 `outputs`

May be empty. Keys:

| Key | Meaning |
|---|---|
| `interval` | steps between post-processing calls (default 100) |
| `ovito_dump` | `{"enabled": bool, "output_dir": str, "prefix": str, "mode": "distributed" \| "collect", "properties": [...]}`; a LAMMPS-style text frame every `interval` steps and after the last step; `distributed` (default) writes `prefix.<step>.rank_<r>.dump` per rank, `collect` one file from rank 0; `properties` may list `id type x y z size owner_rank type_id` and any declared property |
| `postprocessors` | callables `f(step, time, pool, comm)` run every `interval` steps on every rank |
| `event_hooks` | callables `f(event_type, changed, step, time, pool, comm, config)` run on every rank after every applied event; `changed` is the broadcast structured array of created/modified/deleted records |

`pool.get_local_objects()` returns the active objects this rank owns; `pool.get_local_count()`
their number; `pool.get_type_name(rec)` the type name of a record.

---

## 4. Running

```python
engine = OKMCengine(config)      # validate, compile, decompose, build the pools (collective)
stats = engine.run()             # the step loop (collective)
engine.destroy()                 # free the MPI windows (collective)
```

`run()` ends at `max_steps`, at `max_time`, or when the total rate is zero. It returns a
dictionary (values on rank 0; zeros elsewhere):

| Key | Meaning |
|---|---|
| `total_steps` | steps executed |
| `total_time` | simulated time (s) |
| `total_events` | events executed (equals the steps) |
| `object_count` | active objects at the end |
| `avg_build_time`, `avg_rate_time`, `avg_select_time`, `avg_execute_time` | per-step averages over ranks (s) of the four phases |

Environment: `NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1` (one kernel thread per rank);
`PARAOKMC_VERBOSE_DUMP=1` prints the type map of every dump frame.

---

## 5. What user code finds in `config` at run time

The dictionary passed to `rate()`, `execute()`, the initial-condition function, the physics
functions and the event hooks is the compiled run-time configuration, not the seven-block
dictionary. Useful entries:

| Entry | Content |
|---|---|
| `config['rng']` | the reproducible random generator (`random`, `uniform`, `normal`, `randint`, `choice`) |
| `config['lattice_constant']`, `config['grid_size']` | the box is their product |
| `config['objects']` | the `objects` block as written (declared sizes and property defaults) |
| `config['physics']` | the `UnifiedPhysics` instance: `get_param`, `set_param_in_event`, `get_field`, `set_field`, `get_field_interpolator`, `add_field_delta_in_event`, `set_param_update` |
| `config['_domain_decomp']` | the decomposition: `get_owner_rank([x, y, z])`, `subdomain_bounds`, `box_size`, `periodic` |
| `config['_props_mapping']['float_props'][type]` | `{property: slot}` |
| `config['_object_dtype']` | the NumPy dtype of an object record (`fields['float_props'][0].shape[0]` is the number of slots) |
| `config['_type_to_int']`, `config['_int_to_type']` | type name ↔ integer id |
| `config['_user_seed']` | `simulation.random_seed` |
| `config['_current_time']` | the simulated time |

Entries starting with an underscore are injected by the engine and read-only.

---

## 6. Validation errors

`OKMCengine(config)` raises `ConfigValidationError` before anything is allocated when the
configuration is inconsistent. The message names the block and the key. Common cases:

| Message (abridged) | Cause |
|---|---|
| `Invalid top-level blocks: {...}` | a key outside the seven blocks |
| `domain.max_interaction_distance is required` | a missing required key |
| `ic.initial_conditions references unknown object type 'B'` | a type not declared in `objects` |
| `Event 'm': SingleObjectEvent must define non-empty object_types` / `PairObjectEvent must define non-empty type_pairs` | a class without its type list |
| `Event 'c': PairObjectEvent must define max_distance` / `max_distance must be positive number` | missing or non-positive cut-off |
| `Event 'm': missing object property 'E_m' in objects['I']['properties'] for rate_deps.obj1` | the kernel asks for an undeclared property (coordinates are not properties) |
| `Event 'm': physics quantity 'temperature' requested by rate_deps.physics is not defined in config['physics']['params']` | missing parameter |
| `Event 'm': field 's' requested by rate_deps.fields is not defined in config['physics']` | missing field |
| `physics.fields['s'].shape must be [nx, ny, nz]` | malformed field |
| `outputs.ovito_dump.mode must be 'distributed' ... or 'collect'` | bad output mode |
| `ic.mode must be 'builtin', 'function' or 'dump'` | bad initial-condition mode |

---

## 7. Checklist for a parallel-safe model

* The initial condition returns, on each rank, exactly the objects inside that rank's
  subdomain, with globally unique ids, and would give the same global configuration on any
  rank count.
* Events draw random numbers from `config['rng']` only; rates draw none.
* `domain.max_interaction_distance` ≥ every pair `max_distance`.
* `incremental_rate = True` only on classes whose rate depends on object properties, the
  pair distance and constant parameters.
* Observables that need the whole system are accumulated in an event hook (identical on
  every rank) or reduced over `comm` in a post-processor.
* Pool capacities match the expected per-rank population.

See `docs/TUTORIAL.md` for worked examples of each point and `example/` for complete,
validated models.
