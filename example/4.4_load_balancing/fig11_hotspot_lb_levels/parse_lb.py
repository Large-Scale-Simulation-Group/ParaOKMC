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

"""Parse scaling_v9 (RCB on/off) and scaling_v10 (RCB + event LB) logs into one CSV.

usage: parse_lb.py OUT.csv LOGDIR [LOGDIR ...]
arm: none = no LB, rcb = object-level RCB only, rcb+event = RCB + event-level predictive LB.
"""
import csv, glob, os, re, sys

out_csv, logdirs = sys.argv[1], sys.argv[2:]
name_re = re.compile(r"v(9|10|11|12|13|14)_e0cube_(1k|10k|100k|weak)_lb(on|off|2|ev)_np(\d+)\.out$")
inv_re = re.compile(r"v1[234]_inv_(\S+)\.out$")
perf_re = re.compile(r"Perf: build=([\d.]+)ms rate=([\d.]+)ms select=([\d.]+)ms exec=([\d.]+)ms per_step=([\d.]+)ms")
step0_re = re.compile(r"\[e0\] step=\s*0 .*load\[min/avg/max\]=(\d+)/([\d.]+)/(\d+)")
final_re = re.compile(r"rank load: min=(\d+) avg=([\d.]+) max=(\d+)")
ev_sum_re = re.compile(r"\[EventLB\] Summary: passes=(\d+) pass_time_total=([\d.]+)s pass_time_max=([\d.]+)s events_moved=(\d+) check_calls=(\d+) check_time_total_maxrank=([\d.]+)s engine_rebalance_time=([\d.]+)s")
ARM = {"off": "none", "on": "rcb", "2": "rcb+event", "ev": "event"}   # v10/v11 (older libraries) relabelled below
rows = []
for d in logdirs:
    for f in sorted(glob.glob(os.path.join(d, "v*_e0cube_*.out"))):
        m = name_re.search(f)
        if not m:
            continue
        camp, size, lb, np_ = m.groups(); np_ = int(np_)
        txt = open(f, errors="ignore").read()
        p = perf_re.search(txt); s0 = step0_re.search(txt); fl = final_re.search(txt); ev = ev_sum_re.search(txt)
        rcb = [float(x) for x in re.findall(r"\[ObjectLB\] Step \d+: rebalance completed in ([\d.]+)s", txt)]
        evp = [float(x) for x in re.findall(r"\[EventLB\] Step \d+: redistribution completed in ([\d.]+)s", txt)]
        steps = re.search(r"Done: step=(\d+)", txt)
        r = {
            "campaign": "v" + camp, "arm": ({"10": "rcb+event_prepatch", "11": "rcb+event_gatheronly", "12": "rcb+event_fullgather"}.get(camp, ARM[lb]) if lb == "2"
                                          else ("event_fullgather" if (lb == "ev" and camp == "13") else ARM[lb])),
            "object_lb": "on" if lb in ("on", "2") else "off", "event_lb": "on" if lb in ("2", "ev") else "off",
            "scaling_type": "weak_1k_per_rank" if size == "weak" else "strong",
            "n_objects": 1000 * np_ if size == "weak" else {"1k": 1000, "10k": 10000, "100k": 100000}[size], "mpi_ranks": np_,
            "status": "ok" if p else ("failed" if any(k in txt for k in ("Traceback", "MPI_ABORT", "Segmentation", "CANCELLED", "DUE TO TIME LIMIT")) else "incomplete"),
            "per_step_ms": p.group(5) if p else "", "build_ms": p.group(1) if p else "", "rate_ms": p.group(2) if p else "",
            "select_ms": p.group(3) if p else "", "exec_ms": p.group(4) if p else "",
            "steps": steps.group(1) if steps else "",
            "step0_load_min": s0.group(1) if s0 else "", "step0_load_max": s0.group(3) if s0 else "",
            "step0_imbalance_max_over_avg": f"{int(s0.group(3))/float(s0.group(2)):.2f}" if s0 else "",
            "final_load_min": fl.group(1) if fl else "", "final_load_max": fl.group(3) if fl else "",
            "final_imbalance_max_over_avg": f"{int(fl.group(3))/float(fl.group(2)):.2f}" if fl else "",
            "rcb_passes": len(rcb), "rcb_total_s": f"{sum(rcb):.3f}", "rcb_first_pass_s": f"{rcb[0]:.3f}" if rcb else "",
            "rcb_mean_pass_s": f"{sum(rcb)/len(rcb):.3f}" if rcb else "",
            "event_passes": len(evp), "event_total_s": f"{sum(evp):.3f}", "event_first_pass_s": f"{evp[0]:.3f}" if evp else "",
            "event_mean_pass_s": f"{sum(evp)/len(evp):.3f}" if evp else "",
            "event_events_moved": ev.group(4) if ev else "", "event_check_calls": ev.group(5) if ev else "",
            "event_check_total_s": ev.group(6) if ev else "",
            "event_check_ms_per_step": f"{1000*float(ev.group(6))/max(1,int(ev.group(5))):.2f}" if ev else "",
            "engine_rebalance_time_s": ev.group(7) if ev else "",
        }
        rows.append(r)

def key(r): return (r["arm"], r["scaling_type"], r["n_objects"] if r["scaling_type"] == "strong" else "weak")
base = {key(r): float(r["per_step_ms"]) for r in rows if r["mpi_ranks"] == 1 and r["per_step_ms"]}
for r in rows:
    b = base.get(key(r)); r["speedup"] = r["efficiency"] = ""
    if b and r["per_step_ms"]:
        sp = b / float(r["per_step_ms"]); r["speedup"] = f"{sp:.2f}"
        r["efficiency"] = f"{sp/r['mpi_ranks']:.2f}" if r["scaling_type"] == "strong" else f"{sp:.2f}"
order = {"none": 0, "rcb": 1, "event": 2, "rcb+event": 3, "event_fullgather": 4, "rcb+event_fullgather": 5, "rcb+event_gatheronly": 6, "rcb+event_prepatch": 7}
rows.sort(key=lambda r: (r["scaling_type"], r["n_objects"] if r["scaling_type"] == "strong" else 0, order[r["arm"]], r["mpi_ranks"]))
cols = list(rows[0].keys()) if rows else []
with open(out_csv, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=cols); w.writeheader(); w.writerows(rows)
from collections import Counter
print(f"{len(rows)} rows -> {out_csv}; status={dict(Counter(r['status'] for r in rows))}; arms={dict(Counter(r['arm'] for r in rows))}")
for d in logdirs:
    for f in sorted(glob.glob(os.path.join(d, "v1[234]_inv_*.out"))):
        txt = open(f, errors="ignore").read()
        m = re.search(r"\[Invariants\] checks=(\d+) violations=(\d+)", txt)
        p = perf_re.search(txt)
        print(f"  invariants {os.path.basename(f)}: " + (f"checks={m.group(1)} violations={m.group(2)}" if m else "NO SUMMARY (incomplete/failed)") + (f" per_step={p.group(5)}ms" if p else ""))
