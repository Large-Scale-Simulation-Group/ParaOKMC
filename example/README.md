# ParaOKMC examples — the benchmarks of the CPC manuscript, one directory per section

Each directory holds the **model script** and the **exact run commands** (`run_*.sh`). The
**raw data** from which each published figure or table was computed (the files listed in the
"Raw data" column) are not in this repository: they are distributed with the program package
of the paper in the CPC Program Library. Unpack that package over this tree to add them;
`figures/make_pub_figures.py` then regenerates Figs. 4–11.

| Section | Directory | Model script | Raw data | Produced on |
|---|---|---|---|---|
| §3.1, Fig. 4 — free diffusion | `3.1_free_diffusion/` | `run_diffusion_balance.py` | `data/e0_s41…45/{x_profile,metrics}.csv`, `data/e0_s*.log`, `dump.tar.gz` (`dump/e0_s*/`, a frame every 1000 steps) | development machine, Oct 2026 |
| §3.2, Fig. 5 — spherical sinks | `3.2_sink_strength/` | `run_sphere_sink_strength.py` (`run_spherical.sh`) | `data/spherical/e1_R*_N*_s*.log` (8 conditions × 5 seeds), `dump.tar.gz` (`dump/spherical/`) | group workstation + development machine, Oct 2026 |
| §3.2, Fig. 6 — linear sinks | `3.2_sink_strength/` | `run_dislocation_sink_strength.py` (`run_linear.sh`) | `data/linear/e2_rd*_rho*_s*.log` (7 × 5), `dump.tar.gz` (`dump/linear/`) | same |
| §3.3, Fig. 7 — 1D→3D transition | `3.3_1d3d_transition/` | `run_1d3d_master_curve.py` | `data/e3_s*_nch*.csv` + `.log` (11 n_ch × 5 seeds), `dump_s41_s42.tar.gz`, `dump_s43_s44.tar.gz`, `dump_s45.tar.gz` | development machine + group workstation, Oct 2026 |
| §3.4, Fig. 8 — invariance under parallel execution | `3.4_parallel_invariance/` | `../3.2_sink_strength/run_dislocation_sink_strength.py` on 1/2/4/8 ranks (`run_invariance.sh`) | `data/e2_np*_s*.log` (4 × 5), `dump.tar.gz` (one file per rank) | group workstation, Oct 2026 |
| §4.1 — benchmark workloads and hardware | `4.1_benchmark_workloads/` | `pair_scaling.py` (pair workload; the diffusion workload is the §3.1 script), log parser | `bscc_a_measured_20260717.csv` (master table), `logs_v4_v5_v7.tar.gz` (143 `.out`) | cluster node, 2 × AMD EPYC 7452, 2026-07-17 |
| §4.2, Fig. 9 — strong scaling | `4.2_strong_scaling/` | see `4.1_benchmark_workloads/README.md` | `strong_scaling.csv` | same |
| §4.3, Fig. 10 — weak scaling | `4.3_weak_scaling/` | see `4.1_benchmark_workloads/README.md` | `weak_scaling.csv` | same |
| §4.4, Fig. 11 — load balancing under a hotspot | `4.4_load_balancing/fig11_hotspot_lb_levels/` | §3.1 script with `--ic cube` | `bscc_a_lb_levels_20260916.csv`, `logs_v9.tar.gz`, `logs_v14.tar.gz` | cluster node, 2026-09-16 |
| §4.5, Table 1 — per-step time budget | `4.5_step_time_budget/` | see its README | `step_time_budget_20260718.csv` | group workstation, 2026-07-18 |

Each directory has a `README.md` with the physics, the command-line options and the meaning
of every data file (`3.2_sink_strength/` has one per sink type; `4.1_benchmark_workloads/README.md`
covers all of §4).

## License

Everything in this directory, scripts and data alike, is released under the Apache License
2.0, like the library (`LICENSE`, a copy of the one at the release root). The scripts carry
the license header.

## Running

```bash
# from the release root
export PYTHONPATH=$PWD                              # the library of this release
export NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1        # one kernel thread per MPI rank
cd example/3.1_free_diffusion && ./run_validation.sh    # five seeds -> data/
mpirun -np 8 python run_diffusion_balance.py --num-particles 100000 --max-steps 1000   # any case runs on any rank count
```

Requirements: Python ≥ 3.9, NumPy, SciPy, Numba, mpi4py on an MPI-3 library. The `run_*.sh`
scripts reproduce the published runs serially and may take hours.

Conventions shared by all validation cases: five seeds `41 … 45`; reported values are the
seed mean with one standard error of that mean; sink strengths in m⁻²; the jump count of an
absorbed defect is accumulated by a post-processor event hook (`outputs.event_hooks`), so the
scripts contain no MPI calls of their own. All §3 runs were made with this release
(October 2026), with rate reuse declared on the event classes (`incremental_rate = True`) and OVITO dumps on
(`--dump-dir`, about 100 frames per run plus the final state, one file per rank; the
`jump_count` of every defect rides along as a column). Rate reuse and dumping do not change
a trajectory: the sink-strength runs reproduce the earlier logs byte for byte.

In the program package the OVITO frames (24 035 files, 380 MB) are packed as one
`dump.tar.gz` per directory (three archives split by seed in `3.3_1d3d_transition/`);
`data/` stays unpacked and is all the figure script needs. To look at the frames, unpack in
place, for example

```bash
cd example/3.2_sink_strength && tar xzf dump.tar.gz      # -> dump/spherical/, dump/linear/
```

Rerunning a case writes fresh `data/` and `dump/` directories (ignored by git).

## Regenerating the figures

```bash
cd example/figures && python make_pub_figures.py    # -> figures/out/fig04 … fig11 (.png + .pdf)
```
