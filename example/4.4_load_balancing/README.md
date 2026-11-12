# §4.4 — contribution of the load-balancing mechanisms (Fig. 11)

`fig11_hotspot_lb_levels/` holds the experiment of Fig. 11: the diffusion workload
(`../3.1_free_diffusion/run_diffusion_balance.py`) with the hotspot initial condition
`--ic cube --cube-frac 0.1` (all objects inside a central cube with edge 0.1 of the box,
0.1 % of the volume), run on one cluster node (2 × AMD EPYC 7452). It compares all levels off
(`none`, campaign v9) with level 2, event-level redistribution (`event`, campaign v14);
level 3 is off in every arm.

Protocol, arms, CSV columns and the parsers are described in
`../4.1_benchmark_workloads/README.md` (section "Fig. 11").
