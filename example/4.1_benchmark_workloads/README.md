# Performance tests (§4)

All timings are the engine's own wall clock per KMC step, taken from the `Perf:` summary
line that every run prints (`build`, `rate`, `select`, `exec`, `per_step`, in ms). Python
import and Numba compilation are excluded. Each data point is one run. Every job uses
`NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1` and one MPI rank per core.

Rate reuse (`incremental_rate` on the event classes) is off in every run, so each step
re-evaluates the rate of every event in the catalogue.

## §4.1–4.3, Figs. 9 and 10 — this directory, `../4.2_strong_scaling/`, `../4.3_weak_scaling/`

One cluster node with two AMD EPYC 7452 (64 cores), campaigns v4 / v5 / v7 of 2026-07-17.

**Diffusion workload** — `../3.1_free_diffusion/run_diffusion_balance.py`, slab initial
condition (`--seed-width 10`, all objects inside a 10-nm slab spanning y and z), 1000 steps,
seed 42, statistics written every step (`--report-interval 0`). Level-1 object
repartitioning is at the script's default (on); levels 2 and 3 are off.
Strong scaling: N = 10³, 10⁴, 10⁵ at n_p = 1 … 64. Weak scaling: 10³ objects per rank in
the fixed 100 × 300 × 300 nm³ box.

**Pair workload** — `pair_scaling.py`: H and V objects distributed uniformly at constant
density (the box edge scales with N^{1/3}; about 2.6 H–V neighbours per object), a
migration event for both species (rate 10⁶, small random step) and an H–V capture
pair-object event with `max_distance = 6 Å` whose rate is 10⁻⁶, so the pair catalogue is
built and kept consistent every step but a capture is never selected and the structure
stays stationary. 300 steps, seed 9, all load-balancing levels off. Strong scaling:
N = 2 × 10⁴ at n_p = 1 … 64 (and N = 10⁵ at n_p = 1, 8, 64); weak scaling: 2 × 10⁴ objects
per rank. Three arms were run: `old` (library before the pair structural update),
`new_off` (current library, pair structural increment disabled as a control) and `new_on`
(current library, the released path). Figs. 9–10 use `new` / `new_on`.

Campaign v5 adds n_p = 24/40/48/56 and the cross-node 128/256 attempts; v7 adds the N = 10³
grid and both weak-scaling series. Files: `logs_v4_v5_v7.tar.gz` (all 143 `.out` files), `bscc_a_measured_20260717.csv` (one row per job, the master table; `../4.2_strong_scaling/strong_scaling.csv` and `../4.3_weak_scaling/weak_scaling.csv` are its two halves; `status=failed` marks the n_p ≥ 128 jobs, which died in the one-sided RDMA layer across
nodes) and `parse_perf_logs.py`, which rebuilds that CSV from the logs:

```bash
tar xzf logs_v4_v5_v7.tar.gz && python parse_perf_logs.py logs/*.out > rebuilt.csv
```

n_p = 40 and 56 are present in the CSV but omitted from the figures (elongated 2 × 4 × 5 and
2 × 4 × 7 decompositions).

## `../4.4_load_balancing/fig11_hotspot_lb_levels/` — §4.4, Fig. 11

Same node, 2026-09-16. Diffusion workload with the hotspot initial condition
`--ic cube --cube-frac 0.1` (all objects inside a central cube with edge 0.1 of the box,
0.1 % of the volume), 1000 steps, seed 42, `--report-interval 0`; strong scaling with 10⁵
objects and weak scaling with 10³ objects per rank, n_p = 1 … 64.

`bscc_a_lb_levels_20260916.csv` collects eight arms (`arm` column); the two used in Fig. 11
are

| arm | campaign | flags | levels |
|---|---|---|---|
| `none` | v9 | `--disable-object-lb` | all off |
| `event` | v14 | `--disable-object-lb --event-lb` | level 2 only (event-level redistribution) |

Level 3 (runtime work stealing) is off in every arm; the e0 script never enables it. The
other arms (`rcb`, `rcb+event`, and the intermediate `*_prepatch` / `*_gatheronly` /
`*_fullgather` variants) document the development of the level-2 balancer; the library
changes made for v14 are part of this release. `bscc_a_hotspot_20260916.csv` is the v9
campaign alone, with per-run RCB pass counts and object counts per rank.

Files: `logs_v9.tar.gz`, `logs_v14.tar.gz`, `parse_v9.py`, `parse_lb.py` (log → CSV), `plot_lb_compare.py`.

## `../4.5_step_time_budget/` — §4.5, Table 1

Group workstation, two Intel Xeon Gold 6132 @ 2.60 GHz, 2026-07-18. Diffusion workload
(`../3.1_free_diffusion/run_diffusion_balance.py`), 10⁵ objects, uniform start, all
load-balancing levels off, 1000 steps, seed 42, statistics every step, n_p = 1 and 4.
`step_time_budget_20260718.csv` has two
blocks: the rows with `category` ending in `_fixed` are the published Table 1 (library at
the released version); the earlier rows record the same measurement before the
ghost-event cleanup and `apply` fixes, kept for comparison. The `build`/`rate`/`select`/
`exec` phases are the engine's own counters; the finer entries (statistics output, RMA
window check, the breakdown of `exec` into winner execution, broadcast, apply, ghost
refresh) were obtained with timing wrappers placed around the main-loop stages for that
measurement, which are not part of the release; less than 0.05 ms per step is unaccounted.

The logs are the scheduler output of the original jobs. Cluster paths, the account name and
node names are replaced by placeholders (`<workdir>`, `<user>`, `<node>`, …); the measured
values are untouched, and the parsers rebuild the published CSV files from them unchanged.
The Slurm job scripts themselves are not distributed: each job ran one of the commands
documented above, with the rank count given by `n_p`.
