# ParaOKMC architecture

*中文版：[`docs_zh_cn/ARCHITECTURE.md`](../docs_zh_cn/ARCHITECTURE.md)*

How the engine is organised and what happens in one step. For using the library see
`docs/TUTORIAL.md` and `docs/USER_GUIDE.md`.

ParaOKMC runs the residence-time (BKL) algorithm *exactly* on distributed memory: one event
is drawn from the global rate distribution and one clock advances per step, whatever the
rank count. The parallel work is everything around that draw. It has three sources:

1. **Spatial domain decomposition.** Every rank owns the objects of one subdomain and builds
   and evaluates the candidate events of those objects.
2. **A two-collective step protocol.** The partial rates are combined by one `Allgather`
   that also carries the selection random number; the consequence of the selected event
   travels in one broadcast state-change message that every rank applies. Every rank
   derives the rest of its work (which events are stale, which rates must be recomputed,
   which physics changed) from that message.
3. **Load balancing at three levels**, acting on objects (level 1) or on self-contained
   event records (levels 2 and 3).

---

## 1. Modules

| Module | Role |
|---|---|
| `user_config.py` | validates the seven-block configuration and compiles it into the internal flat configuration; `OKMCengine()` |
| `event_class.py`, `event_compiler.py` | event base classes, registry, helpers (`change_obj`, `delete_obj`, `new_obj`, `get_physics_param`, …) |
| `v16_engine.py` | the engine: initialisation, the step loop, selection, execution, the state-change broadcast |
| `domain_decomposition.py` | the cuboid rank grid, the cell grid (`lattice_grid × h_factor`), ownership map, ghost ranks |
| `rma_object_pool.py` | object storage in three classes, the boundary RMA window, the ghost cache, application of state changes |
| `rma_event_pool.py` | the three event arrays (zero, single, pair), statuses, rate-sum mirrors |
| `parallel_event_builder.py`, `vectorized_event_builder.py`, `jit_event_builder.py`, `jit_cell_list.py` | event construction: full and incremental, Numba kernels, cell lists |
| `jit_rate_engine.py` | `RateDeps`, code generation of the batched rate launchers |
| `incremental_update_v16.py` | bookkeeping of changed, new and deleted objects between steps |
| `load_balancer.py` | level 1: object repartitioning by recursive coordinate bisection (RCB) |
| `predictive_rebalance_v16.py` | level 2: event redistribution before rate evaluation |
| `runtime_load_balancer_v16.py` | level 3: work stealing during rate evaluation |
| `unified_physics.py` | parameters, fields, interpolation, scheduled updates, solvers, triggers |
| `random_generator.py` | the counter-keyed generator (`docs/RANDOM_NUMBERS.md`) |
| `post_processor.py` | post-processors, event hooks, OVITO dump writers and the dump reader |

---

## 2. Data layout

### 2.1 Objects

An object is a record of a NumPy structured dtype generated at start-up from the
`objects` block: `id` (int64), `type_id`, `x y z`, `size`, `owner_rank`, the flags
`active new changed`, and `float_props`, a fixed-length float array holding the declared
properties of every type in slot order (`config['_props_mapping']`).

The object pool of a rank keeps three classes:

| Class | Content | Visible to neighbours through |
|---|---|---|
| center | owned objects farther than the cut-off from every subdomain face | nothing (they need no exchange) |
| boundary | owned objects within the cut-off of a face | an MPI RMA window, read by the neighbours with one-sided `Get` |
| large | objects whose influence range (size/2 + cut-off) exceeds a threshold derived from the ghost capacity | a copy on every rank |

Each rank also holds a **ghost cache**: copies of the neighbours' boundary objects. It is
built once by one-sided reads and then kept current incrementally from the broadcast
state-change message, so a step does not re-read the windows.

### 2.2 Events

An event record is self-contained: it carries a 128-bit id, its type id, `status`, `rate`,
`weight`, a `params` array, and **copies of the objects it involves**. A rank can therefore
evaluate the rate of an event it does not own; this is what lets levels 2 and 3 move work
without moving objects.

The event pool has one array per category (zero, single, pair). Statuses:

| Status | Meaning |
|---|---|
| TODO (0) | the rate must be (re)computed |
| BUSY (1) | being evaluated (transient, within the rate stage) |
| DONE (2) | the rate is valid and counted in the local rate sum |
| SENT (3) | handed to another rank by work stealing during this step; rate 0; removed before the next build |

The pool keeps a mirror of the local rate sum per category, updated incrementally as rates
are written or events removed.

---

## 3. One step

```
build events  →  (level 2)  →  evaluate rates (level 3)  →  select  →  execute + broadcast  →  apply
     ↑                                                                                         │
     └──────────── changed / new / deleted objects, pair zone, physics changes ←───────────────┘
```

### 3.1 Build

The catalogue is updated **incrementally** for all three categories; a full rebuild happens
only at the first step, after a repartition (level 1) or on a user request.

* Before building, the SENT husks of the previous rate stage are dropped, and the events of
  the objects deleted (and, when level 2 or 3 is on, changed) in the previous step are
  removed on every rank, since such an event may live on any rank.
* **Single-object events** are updated in place: the events of changed or new owned objects
  are rebuilt, the others are kept.
* **Pair events** are kept outside a cell-level invalidation zone derived by every rank
  from the old and new positions in the previous broadcast; pairs inside the zone are
  rebuilt with Numba cell lists.
* **Zero-object events** are kept unless a global change requires a rebuild.

A retained event keeps its DONE rate only if its class has `incremental_rate = True`;
otherwise it goes back to TODO. Structure and rate are thus separate: the structure
increment is always on, rate reuse is a per-class choice.

### 3.2 Evaluate

Events of classes with `rate_deps` + `rate_kernel` are grouped by type and evaluated by a
generated Numba launcher in one call per type (properties, distances, interpolated field
values and physics scalars are gathered into contiguous arrays first). The remaining TODO
events go through the Python `rate()` one by one. Rates are clamped to finite, non-negative
values and added to the rate-sum mirror.

### 3.3 Select

Rank 0 draws the selection number. One `Allgather` collects `(local rate sum, number)` from
every rank; every rank computes the global sum and the winner rank from the same data. The
winner locates the event inside its own arrays by a cumulative sum.

### 3.4 Execute, broadcast, apply

The winner runs `execute()`, completes the returned changes into full object records, draws
the time increment, and decides the physics update flags. It broadcasts one state-change
message: a fixed-size header (event type, time step, category, physics flags, counts),
followed by the arrays it announces (changed objects with their old positions, parameter
changes, field increments). Every rank then

1. applies the object changes to its pool and to its ghost cache, migrating objects that
   crossed a subdomain face;
2. records changed, new and deleted ids for the next build and keeps the message for the
   pair invalidation zone;
3. applies the physics changes, runs any scheduled or triggered physics updates, and drops
   the rates that depend on what changed (section 5);
4. calls the event hooks with the message;
5. advances the clock by the broadcast time step.

Because every rank applies the same message, no rank ever needs to ask another what
happened.

---

## 4. Load balancing

All three levels preserve the sampled statistics: every live event is counted in exactly one
partial rate sum. `tests/test_lb_rate_conservation.py` checks this identity at every step for
every combination of levels.

### Level 1: object repartitioning (`object_load_balancer`, off by default)

Every `balance_interval` steps the object counts per rank are compared. When
max/mean − 1 exceeds `imbalance_threshold`, the cuts are recomputed by a distributed
recursive coordinate bisection on the cell grid: the per-cell object counts are summed with
`Allreduce` and each bisection level is decided collectively, so no rank holds a special
role. Subdomains stay cuboids aligned with cells. The new partition is adopted only if it
lowers the imbalance by at least `min_improvement` (default 0.05); objects then move with one
`Alltoallv`, and a full event rebuild follows on every rank.

### Level 2: event redistribution (`predictive_rebalance`, on by default)

Every `check_interval` steps (and always after a repartition) the ranks compare the total
weight of their TODO events with one small `Allgather`. Above `threshold`, rank 0 plans a
greedy redistribution from the gathered `(id, weight)` lists and the planned event records
move with one `Alltoallv`. A moved event stays with the receiving rank; the broadcast
state-change message invalidates it there when one of its objects changes, and the owner
rebuilds it.

Weights come from the rate path (compiled kernel 1, Python rate 1000, or an explicit
`weight`) and, with `adaptive_weight`, from measured timings. The level switches itself off
when every registered class has `incremental_rate = True`: then almost no event is TODO and
there is nothing worth moving.

### Level 3: work stealing (`runtime_load_balancer`, on by default)

During the Python-rate loop a busy rank checks a one-sided window of idle flags every N
evaluated events, where N is `runtime_load_balancer.check_interval` (default 32), raised
automatically to one eighth of the local catalogue and capped at 2048. If ranks are idle it
sends them part of its TODO events with non-blocking point-to-point messages, marks them
SENT, and continues. A rank that has finished its own events keeps receiving and evaluating
until every batch it sent is acknowledged and a non-blocking barrier over all ranks has
completed, so no stolen event is missing from the partial sums gathered for selection. Like
a level-2 move, a stolen event then stays with the thief.

Events of compiled-kernel classes are evaluated in one batch before this loop and are never
stolen.

---

## 5. Physics coupling

`UnifiedPhysics` holds named scalar parameters and 3-D fields on rectilinear grids spanning
the box, with trilinear interpolation. Changes arrive in four ways: scheduled updates of a
parameter or field (step or time interval), a global `update_func`, external solvers called
by triggers, and changes made inside `execute()` (`set_param_in_event`,
`add_field_delta_in_event`) that travel in the state-change message. The update decisions
are taken by the winner rank and broadcast, so every rank updates the same things at the
same step.

A physics change never touches the event structure. The engine keeps a static table of
which classes depend on which names: `rate_deps.physics` and `rate_deps.fields` for kernels;
`params` entries that resolve to physics parameters for Python rates (their cached column is
rewritten in place); and, for a Python class with `incremental_rate = True` and no declared
`params`, every name. Only those classes' rates go back to TODO.

---

## 6. Communication summary

| Where | Pattern | Volume |
|---|---|---|
| selection | one `Allgather` of two doubles per rank | every step |
| state change | broadcast of a header plus the announced arrays | every step; the size of what the event changed |
| ghost cache | one-sided `Get` from the neighbours' boundary windows | at start-up and after a repartition |
| level 1 | `Allreduce` of per-cell counts per bisection level, `Alltoallv` of objects | when triggered |
| level 2 | `Allgather` of totals; gather/broadcast of the plan; `Alltoallv` of events | every `check_interval` steps, moves only above the threshold |
| level 3 | one-sided reads of idle flags, `Isend`/`Recv` of event batches, ACKs, `Ibarrier` | during the Python-rate loop |

All arrays travel as raw bytes of their structured dtypes (`MPI.BYTE`); nothing is pickled
on the step path.
