# e3 — 1D-to-3D transition of the sink strength (§3.3, Fig. 7)

Reference: H. Trinkaus, H.L. Heinisch, A.V. Barashev, S.I. Golubov, B.N. Singh,
Phys. Rev. B 66 (2002) 060105(R).

## Model

fcc copper, a₀ = 3.615 Å, jump distance a = a₀/√2 = 0.256 nm along ⟨110⟩. A defect (`DEF`)
performs a 1D walk along its current ⟨110⟩ axis and draws a new axis after every n_ch
jumps (`--n-ch`); n_ch = 1 is the 3D limit, n_ch = 10⁵ is strongly 1D. Spherical sinks
(`S`, inert) of capture radius R = 15 nm are placed at random at N = 10²² m⁻³ in a
500 × 500 × 500 nm³ periodic box (1250 sinks). Absorption and jump counting are as in e1.

The measured k² = 6/(a² ⟨n_j⟩) is compared with the master curve of Trinkaus et al.,

    y = k²/k²₁,   x² = l_ch² k²₁ / 12 + k⁴₁ / k⁴₃,   l_ch = a √n_ch,   k²₁ = 6 (πR²N)²,

where the 3D limit k²₃ is taken from the n_ch = 1 simulation itself, as the reference
prescribes (`--k3-sim-m2` passes it explicitly; otherwise the script uses the analytic 3D
value for the x axis of its own CSV and the figure script substitutes the measured one).

| n_ch | 1, 3, 10, 30, 100, 300, 1000, 3000 | 10⁴ | 3 × 10⁴ | 10⁵ |
|---|---|---|---|---|
| defects absorbed per seed | 60 | 40 | 45 | 30 |

Fewer defects are followed in the most 1D cases because each walk is long. Five seeds, all
load-balancing levels off.

## Run

```bash
./run_validation.sh     # 11 n_ch x 5 seeds -> data/e3_s<seed>_nch<n>.{csv,log}, dump/e3_s<seed>_nch<n>/
```

## Data

Each `.csv` has one row per report (the last row is the final state) with the columns
`n_ch, R_nm, N_m3, absorbed, avg_jumps, sem_jumps, k2_sim_m2, k2_err_m2, k2_1d_m2,
k2_3d_used_m2, x, y_sim, y_theory`. `dump_s41_s42.tar.gz`, `dump_s43_s44.tar.gz` and `dump_s45.tar.gz` unpack to `dump/e3_s<seed>_nch<n>/`, the OVITO frames of
each run (sinks `S` and the walking defects `DEF`, with their `jump_count`).

## Two ways to write the absorption rate

`DefectHop` uses a compiled rate (`rate_deps` + Numba `rate_kernel`); `SphereAbsorption` keeps a plain
Python `rate()` so that both forms appear in one script. The same class in the compiled form
(`RateDeps(kind='pair')`, kernel argument `distance` only) is included commented out; swapping
the two gives the same trajectory.
