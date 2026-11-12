# e0 — free diffusion of non-interacting objects (§3.1, Fig. 4)

Reference: I. Martin-Bragado et al., *MMonCa*, Comput. Phys. Commun. 184 (2013) 2703, §4.1.

## Model

One object type (`Tracer`) and one single-object event (`TracerMigration`): a jump of
length *l* = 5 nm in a random direction at rate 6D/l² = 24 s⁻¹, which gives the article's
diffusivity D = 100 nm² s⁻¹. No other event is registered, so the case isolates migration
and the residence-time clock.

| Quantity | Value |
|---|---|
| Box | 100 × 300 × 300 nm³, periodic in x, y, z |
| Objects | 1024 (`--num-particles`; the article used 448 820) |
| Initial condition | uniform inside a 10-nm slab centred in x (`--seed-width 10`) |
| Simulated time | 10 s (`--max-time 10`) |
| Analysis | 32 slices of 3.125 nm along x (`--x-bins 32`); the exact profile is the Fourier series of the 1D periodic diffusion equation |

The script evaluates the analytic profile itself and reports the relative L2 error.

## Run

```bash
./run_validation.sh          # seeds 41-45, one after the other -> data/e0_s<seed>/
```

`run_validation.sh` is the exact command used for Fig. 4. Other options of
`run_diffusion_balance.py` that the performance tests use:

| Option | Meaning |
|---|---|
| `--ic slab` (default) / `--ic cube --cube-frac f` | slab initial condition, or a hotspot cube centred in the box with edge *f* × box edge (Fig. 11 uses `f = 0.1`, i.e. 0.1 % of the volume) |
| `--disable-object-lb`, `--balance-interval`, `--imbalance-threshold` | level 1, object-level RCB repartitioning (on by default, checked every 250 steps, threshold 0.15) |
| `--event-lb [--event-lb-threshold t]` | level 2, event-level redistribution (off by default) |
| `--report-interval n` | write the x profile and metrics every *n* steps (`0` = every step, used for the timing runs) |
| `--check-invariants` | assert per step that the global event count, total rate and object/event ids are consistent |

Level 3 (runtime work stealing) is left off by this script.

## Data

`data/e0_s<seed>/x_profile.csv` — one row per report step: step, time, then the 32 simulated
slice densities followed by the 32 analytic densities (header names the columns).
`data/e0_s<seed>/metrics.csv` — step, time, total particles, profile CV, relative L2 error.
`data/e0_s<seed>.log` — the run log, including the engine's `Perf:` summary.
`average_profiles.py` pools the five seeds.
