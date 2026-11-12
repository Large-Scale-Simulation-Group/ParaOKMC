#!/usr/bin/env python3
# Copyright (c) 2026 Beihang University
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Three-arm comparison (none / RCB only / RCB + event-level LB), hotspot IC.
Panels: (a) 100k strong speed-up, (b) 100k per-step time, (c) weak efficiency 1k/rank,
        (d) cost of one RCB pass vs one event-level pass vs per-step event-LB check."""
import csv, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

rows = [r for r in csv.DictReader(open(sys.argv[1])) if r["status"] == "ok"]
out = sys.argv[2]
ARMS = [("none", "#55534e", "s", (0, (4, 2)), "no LB"), ("rcb", "#c85a19", "o", "-", "RCB only (level 1)"),
        ("event", "#1baf7a", "v", "-", "event LB only (level 2)"),
        ("rcb+event", "#2a78d6", "^", "-", "RCB + event LB (levels 1+2)"),
        ("event_fullgather", "#9fd9c0", "v", ":", "event LB only, full gather every step (v13)"),
        ("rcb+event_fullgather", "#9ec5f4", "^", ":", "RCB + event LB, full gather every step (v12)")]
def ser(arm, scaling, n=None):
    s = sorted([r for r in rows if r["arm"] == arm and r["scaling_type"] == scaling and (n is None or r["n_objects"] == n)], key=lambda r: int(r["mpi_ranks"]))
    return s
fig, ax = plt.subplots(2, 2, figsize=(11, 8))
ax = ax.ravel()
for arm, c, m, ls, lab in ARMS:
    s = ser(arm, "strong", "100000")
    if s:
        x = [int(r["mpi_ranks"]) for r in s]
        ax[0].plot(x, [float(r["speedup"]) for r in s], ls=ls, marker=m, color=c, label=lab)
        ax[1].plot(x, [float(r["per_step_ms"]) for r in s], ls=ls, marker=m, color=c, label=lab)
    w = ser(arm, "weak_1k_per_rank")
    if w:
        ax[2].plot([int(r["mpi_ranks"]) for r in w], [float(r["efficiency"]) for r in w], ls=ls, marker=m, color=c, label=lab)
ax[0].plot([1, 64], [1, 64], ":", color="gray", label="ideal"); ax[0].set_xscale("log", base=2); ax[0].set_yscale("log", base=2)
ax[0].set_title("(a) strong scaling, 100k objects"); ax[0].set_ylabel("speed-up $T_1/T_{n_p}$")
ax[1].set_xscale("log", base=2); ax[1].set_yscale("log"); ax[1].set_title("(b) per-step wall time, 100k objects (absolute)"); ax[1].set_ylabel("ms per step")
ax[2].set_xscale("log", base=2); ax[2].set_ylim(0, 1.15); ax[2].axhline(1, ls=":", color="gray"); ax[2].set_title("(c) weak scaling, 1k objects per rank"); ax[2].set_ylabel("efficiency $T_1/T_{n_p}$")
for a in ax[:3]: a.set_xlabel("MPI ranks"); a.grid(alpha=0.3, which="both"); a.legend(fontsize=8)
# (d) cost per pass at 100k strong: RCB first pass (rcb arm), event first pass + per-step check (rcb+event arm)
r1 = ser("rcb", "strong", "100000"); r2 = ser("rcb+event", "strong", "100000")
x1 = [int(r["mpi_ranks"]) for r in r1 if r["rcb_first_pass_s"]]; y1 = [1000*float(r["rcb_first_pass_s"]) for r in r1 if r["rcb_first_pass_s"]]
x2 = [int(r["mpi_ranks"]) for r in r2 if r["rcb_first_pass_s"]]; y2 = [1000*float(r["rcb_first_pass_s"]) for r in r2 if r["rcb_first_pass_s"]]
x3 = [int(r["mpi_ranks"]) for r in r2 if r["event_first_pass_s"]]; y3 = [1000*float(r["event_first_pass_s"]) for r in r2 if r["event_first_pass_s"]]
x4 = [int(r["mpi_ranks"]) for r in r2 if r["event_check_ms_per_step"]]; y4 = [float(r["event_check_ms_per_step"]) for r in r2 if r["event_check_ms_per_step"]]
ax[3].plot(x1, y1, "-o", color="#c85a19", label="one RCB pass (RCB-only run)")
ax[3].plot(x2, y2, "--o", color="#c85a19", alpha=0.6, label="one RCB pass (RCB+event run)")
ax[3].plot(x3, y3, "-^", color="#2a78d6", label="one event-LB pass (full)")
x5 = [int(r["mpi_ranks"]) for r in r2 if int(r["event_passes"]) >= 2]; y5 = [1000*(float(r["event_total_s"]) - float(r["event_first_pass_s"])) / (int(r["event_passes"]) - 1) for r in r2 if int(r["event_passes"]) >= 2]
ax[3].plot(x5, y5, "--^", color="#2a78d6", alpha=0.6, label="one event-LB pass (incremental, after each RCB)")
x4 = [int(r["mpi_ranks"]) for r in r2 if int(r["event_passes"]) < 10 and float(r["event_check_ms_per_step"]) > 0]; y4 = [float(r["event_check_ms_per_step"]) for r in r2 if int(r["event_passes"]) < 10 and float(r["event_check_ms_per_step"]) > 0]
ax[3].plot(x4, y4, "-.v", color="#1baf7a", label="event-LB check only, per step")
r3 = ser("event", "strong", "100000")
x6 = [int(r["mpi_ranks"]) for r in r3 if r["event_first_pass_s"]]; y6 = [1000*float(r["event_first_pass_s"]) for r in r3 if r["event_first_pass_s"]]
ax[3].plot(x6, y6, "-v", color="#1baf7a", alpha=0.7, label="one event-LB pass (full), event-only run")
ax[3].set_xscale("log", base=2); ax[3].set_yscale("log"); ax[3].set_xlabel("MPI ranks"); ax[3].set_ylabel("ms"); ax[3].set_title("(d) cost, 100k objects"); ax[3].grid(alpha=0.3, which="both"); ax[3].legend(fontsize=8)
fig.tight_layout(); fig.savefig(out, dpi=150); print("saved", out)
