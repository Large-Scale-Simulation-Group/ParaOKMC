# e2 — sink strength of a straight dislocation (§3.2, Fig. 6) and invariance under parallel execution (§3.4, Fig. 8)

Reference: V. Jansson, L. Malerba, A. De Backer, C.S. Becquart, J. Nucl. Mater. 442 (2013) 341
(Wiedersich expression).

## Model

bcc iron, a₀ = 2.87 Å, 3D ⟨111⟩ migration of SIAs as in e1. One straight dislocation runs
along x through the centre of the box; with periodic boundaries it is a square array of
parallel lines whose density is ρ_d = 1/(L_y L_z). The dislocation is an inert object
(`DISL`). Absorption is the pair-object event `DislocationAbsorption`: its `max_distance`
is the radius of a sphere that contains the whole capture cylinder (√((L_x/2)² + r_d²)),
and the rate function returns the absorption rate only when the minimum-image distance
from the SIA to the line is below the capture radius r_d, zero otherwise. Jump counts are
collected by the same event hook as in e1, and

    k² = 6 / (d_j² ⟨n_j⟩)

is compared with the Wiedersich expression

    k² = 2πρ_d (1 − q²) / [ln(1/q) − 3/4 + q² (1 − q²/4)],   q = r_d √(πρ_d).

| Quantity | Fig. 6 | Fig. 8 |
|---|---|---|
| SIAs per seed | 250 | 60 |
| r_d | 2.25, 4.25, 6.25, 9.25 nm at ρ_d = 3 × 10¹⁴ m⁻²; 2.25, 4.25, 6.25 nm at 10¹⁵ m⁻² | 6.25 nm |
| ρ_d | as above | 10¹⁵ m⁻² (box 28.7 × 31.6 × 31.6 nm³) |
| Ranks | 1 | 1, 2, 4, 8 |
| Load balancing | all three levels off | all three levels off |

r_d = 9.25 nm at ρ_d = 10¹⁵ m⁻² is not run: the pair search radius would exceed half the
box cross-section and the script refuses it.

## Run

```bash
./run_linear.sh         # Fig. 6: 7 conditions x 5 seeds -> data/linear/
../3.4_parallel_invariance/run_invariance.sh   # Fig. 8: 4 rank counts x 5 seeds (same script)
```

## Data

Logs end with the `Results` block (`absorbed`, `<nj> = mean +/- sem`, simulated k², and the
Wiedersich k²); the header records r_d, the target and realised ρ_d, and the box.
`../3.4_parallel_invariance/data/e2_np<P>_s<seed>.log` — the invariance runs, five seeds per rank count.

## Two ways to write the absorption rate

The migration event uses a compiled rate (`rate_deps` + Numba `rate_kernel`); the
absorption event keeps a plain Python `rate()` so that both forms appear in one script.
The script also contains, commented out, the same `DislocationAbsorption` class in the
compiled form. A kernel sees only declared object properties, the pair distance, field and
physics values, not x/y/z, so that variant mirrors the transverse coordinates of the SIA and
the dislocation axis into properties (the script adds them automatically when the compiled
class is active). Swapping the two classes gives the same trajectory and the same k².
