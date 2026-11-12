# Figures of the manuscript from the example data

The data files listed below come with the program package of the paper in the CPC Program
Library (see `../README.md`); they are not in this repository.

```bash
python make_pub_figures.py        # writes out/fig01 … fig11 as .png and .pdf
```

| Figure | Reads |
|---|---|
| fig01–fig03 | schematic diagrams, no data |
| fig04 | `../3.1_free_diffusion/data/e0_s*/x_profile.csv` |
| fig05 | `../3.2_sink_strength/data/spherical/e1_*_s*.log` |
| fig06 | `../3.2_sink_strength/data/linear/e2_*_s*.log` |
| fig07 | `../3.3_1d3d_transition/data/e3_*nch*.csv` |
| fig08 | `../3.4_parallel_invariance/data/e2_np*.log` |
| fig09, fig10 | `../4.1_benchmark_workloads/bscc_a_measured_20260717.csv` |
| fig11 | `../4.4_load_balancing/fig11_hotspot_lb_levels/bscc_a_lb_levels_20260916.csv` |

`pubstyle.py` holds the colours and the figure widths. The script prints, per figure, how
many conditions it found and the range of deviation from the analytic reference.
