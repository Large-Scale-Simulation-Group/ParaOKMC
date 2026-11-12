# e1 — sink strength of spherical sinks (§3.2, Fig. 5)

Reference: L. Malerba, C.S. Becquart, C. Domain, J. Nucl. Mater. 360 (2007) 159.

## Model

bcc iron, a₀ = 2.87 Å. A self-interstitial atom (`SIA`) jumps to a nearest neighbour along
⟨111⟩ (jump distance d_j = √3 a₀/2) in 3D (`--regime 3d`). Spherical sinks (`S`) are inert
objects: no event is registered for them, they never move. Absorption is a pair-object
event `SphereAbsorption` between an SIA and a sink with `max_distance = R` (the capture
radius) and a rate far above the jump rate, so an SIA is absorbed on the first jump that
brings it within R.

Each SIA carries a user attribute `jump_count`, incremented by its migration event. The
post-processor hook `_absorption_hook` runs on every rank after each state-change
message, reads `jump_count` from the record of the deleted SIA and accumulates the sum
and the sum of squares. The sink strength follows from the mean number of jumps:

    k² = 6 / (d_j² ⟨n_j⟩)          (n = 3, i.e. the 3D diffusion coefficient throughout)

and is compared with the second-order expression k² = 4πNR [1 + R √(4πNR)].

| Quantity | Value |
|---|---|
| SIAs per seed | 250 (`--target-absorbed 250`), placed at random outside every capture sphere |
| Sinks | random, non-overlapping capture spheres, number density N |
| Capture radius R | 2.25, 4.25, 6.25, 8.25, 10.25 nm at N = 3 × 10¹⁶ cm⁻³; 2.25, 4.25, 6.25 nm at N = 1.5 × 10¹⁷ cm⁻³ |
| End of run | when the last SIA is absorbed (total rate zero) |
| Load balancing | all three levels off |

## Run

```bash
./run_spherical.sh      # 8 conditions x 5 seeds -> data/spherical/e1_R<R>_N<N>_s<seed>.log
```

## Data

Each log ends with a `Results` block: `absorbed`, `<nj> = mean +/- sem`, the simulated
k² and the analytic k². The header of the log records the box, the realised sink density,
the sink volume fraction f_V and d_j. `figures/make_pub_figures.py` parses `<nj>` from the
five seeds of each condition, averages them and propagates the standard error.

## Two ways to write the absorption rate

`SIAMigration` uses a compiled rate (`rate_deps` + Numba `rate_kernel`); `SphereAbsorption` keeps a plain
Python `rate()` so that both forms appear in one script. The same class in the compiled form
(`RateDeps(kind='pair')`, kernel argument `distance` only) is included commented out; swapping
the two gives the same trajectory.
