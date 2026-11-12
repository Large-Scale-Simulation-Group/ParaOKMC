# §3.4 — invariance of the sink strength under parallel execution (Fig. 8)

The linear-sink case of §3.2 (`../3.2_sink_strength/run_dislocation_sink_strength.py`) is
repeated on 1, 2, 4 and 8 MPI ranks with the physics fixed: ρ_d = 10¹⁵ m⁻², r_d = 6.25 nm,
box 28.7 × 31.6 × 31.6 nm³, 60 SIAs per seed, five seeds per rank count. All three
load-balancing levels are off (the script disables them), so only the spatial domain
decomposition acts. Nothing is changed between rank counts except `mpirun -np`.

```bash
./run_invariance.sh      # 4 rank counts x 5 seeds -> data/e2_np<P>_s<seed>.log
```

Each log ends with the `Results` block (`absorbed`, `<nj> = mean +/- sem`, simulated k² and
the Wiedersich k²). `dump.tar.gz` unpacks to `dump/e2_np<P>_s<seed>/`, the frames, one file per rank
(`e2.<step>.rank_<r>.dump`), so the spatial decomposition itself is visible in OVITO.
`../figures/make_pub_figures.py` averages ⟨n_j⟩ over the five seeds of each rank count and
plots k² against P.
