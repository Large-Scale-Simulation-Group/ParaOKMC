# Random numbers in ParaOKMC

*中文版：[`docs_zh_cn/RANDOM_NUMBERS.md`](../docs_zh_cn/RANDOM_NUMBERS.md)*

ParaOKMC draws every random number from a counter-keyed generator so that a run is a pure
function of its seed for a given number of ranks, and no random state is ever communicated.

## The construction

A draw is identified by three integers: the user seed (`simulation.random_seed`), the
current step, and a *state* index that tells draws of one step apart. The engine uses
state 1 for the event selection (drawn on rank 0) and state 2 for the time advance (drawn
by the winner rank); user draws inside `execute()` take states 3, 4, 5, … in the order they
are made. The value is

```python
np.random.default_rng((user_seed, step, state)).random()
```

that is, the tuple is passed as an entropy sequence to NumPy's `SeedSequence`, which seeds a
PCG64 generator, and the first output of that generator is the number. `SeedSequence`
guarantees that distinct integer tuples give statistically independent streams; this is
NumPy's recommended construction for independent parallel streams.

Consequences:

* **Reproducible.** The same seed gives the same sequence of decisions and the same
  trajectory, run after run.
* **Rank-independent.** The construction does not involve the rank. The k-th user draw of
  step n has the same value on every rank, so no random state travels in any message. The
  values that decide a step (the selection number, the time advance) are still exchanged in
  the per-step collectives, so a floating-point difference between ranks cannot change the
  decision.
* **Stateless.** Nothing is carried from step to step; a restart from a dump at step n with
  the same seed reproduces the original draws of step n onward if the state is otherwise
  identical.

References: M.E. O'Neill, *PCG: A family of simple fast space-efficient statistically good
algorithms for random number generation*, HMC-CS-2014-0905 (2014); C.R. Harris et al.,
*Array programming with NumPy*, Nature 585, 357–362 (2020).

## Using it

Inside `execute()` the generator is `config['rng']`:

```python
@classmethod
def execute(cls, event, objs, config):
    rng = config['rng']
    r = rng.random()                 # float in [0, 1)
    u = rng.uniform(-1.0, 1.0)       # float in [low, high)
    g = rng.normal(0.0, 1.0)         # normal deviate
    k = rng.randint(0, 4)            # integer in [0, 4)
    s = rng.choice([-1.0, 1.0])      # one element of a sequence
    ...
```

Rules:

1. **Never use `np.random` or Python's `random` inside an event.** Each rank would see its
   own stream; the ranks would disagree about what happened.
2. **Rates must not draw random numbers.** A rate is recomputed at the engine's discretion
   (and reused when `incremental_rate` allows); a stochastic rate would make the total rate
   depend on when it was last evaluated.
3. **Initial conditions** written as a function may use any generator, but every rank must
   produce the same global configuration before keeping its own share; seed it with
   `config['_user_seed']` (see `docs/TUTORIAL.md`, chapter 4).

## Determinism across rank counts

Changing the number of ranks changes the spatial decomposition and the order in which
candidate events are enumerated, so the selection number lands on a different event and the
trajectories differ. The sampled statistics do not: the residence-time algorithm is executed
exactly on every rank count (see `docs/TUTORIAL.md`, chapter 8, and Section 3.4 of the accompanying CPC paper). With
a fixed rank count and a fixed partition, repeated runs are bitwise identical; balancing
levels 2 and 3 react to measured timings, so they can change the partition between runs and
should be off when bitwise reproducibility is required.

## API

`paraokmc.RandomGenerator(user_seed, rank=0)`: the class the engine instantiates and injects
as `config['rng']`. `rank` is accepted for backward compatibility and ignored.

| Method | Returns |
|---|---|
| `random()` | float in [0, 1) |
| `uniform(low, high)` | float in [low, high) |
| `normal(mu, sigma, size=None)` | normal deviate(s) |
| `randint(low, high)` | integer in [low, high) |
| `choice(seq)` | one element of `seq` |
| `set_step(step)` | selects the step and resets the user state counter (called by the engine every step) |

Tests: `tests/test_random_generator.py` (determinism, rank independence, seed and step
sensitivity).
