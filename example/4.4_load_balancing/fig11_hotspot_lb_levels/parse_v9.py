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

"""Parse scaling_v9 logs into a CSV in the style of bscc_a_measured_20260717.csv."""
import csv, glob, os, re, sys

logdir = sys.argv[1]
out_csv = sys.argv[2]
rows = []
name_re = re.compile(r"v9_e0cube_(1k|10k|100k|weak)_lb(on|off)_np(\d+)\.out$")
perf_re = re.compile(r"Perf: build=([\d.]+)ms rate=([\d.]+)ms select=([\d.]+)ms exec=([\d.]+)ms per_step=([\d.]+)ms")
step0_re = re.compile(r"\[e0\] step=\s*0 .*load\[min/avg/max\]=(\d+)/([\d.]+)/(\d+)")
final_re = re.compile(r"rank load: min=(\d+) avg=([\d.]+) max=(\d+)")
for f in sorted(glob.glob(os.path.join(logdir, "v9_e0cube_*.out"))):
    m = name_re.search(f)
    if not m:
        continue
    size, lb, np_ = m.group(1), m.group(2), int(m.group(3))
    txt = open(f, errors="ignore").read()
    row = {
        "campaign": "v9", "workload": "e0_diffusion", "ic": "cube_frac0.1",
        "scaling_type": "weak_1k_per_rank" if size == "weak" else "strong",
        "object_lb": lb, "mpi_ranks": np_,
        "n_objects": 1000 * np_ if size == "weak" else {"1k": 1000, "10k": 10000, "100k": 100000}[size],
        "status": "ok" if "Perf:" in txt else ("failed" if any(k in txt for k in ("Traceback", "MPI_ABORT", "Segmentation", "CANCELLED", "srun: error", "DUE TO TIME LIMIT")) else "incomplete"),
        "build_ms": "", "rate_ms": "", "select_ms": "", "exec_ms": "", "per_step_ms": "",
        "step0_load_min": "", "step0_load_avg": "", "step0_load_max": "", "step0_imbalance_max_over_avg": "",
        "final_load_min": "", "final_load_max": "",
        "rebalances": "", "objects_migrated": "", "rcb_passes": "", "rcb_total_s": "", "per_step_net_ms": "", "wall_s": "", "note": "",
    }
    p = perf_re.search(txt)
    if p:
        row.update(build_ms=p.group(1), rate_ms=p.group(2), select_ms=p.group(3), exec_ms=p.group(4), per_step_ms=p.group(5))
    s0 = step0_re.search(txt)
    if s0:
        mn, av, mx = int(s0.group(1)), float(s0.group(2)), int(s0.group(3))
        row.update(step0_load_min=mn, step0_load_avg=av, step0_load_max=mx, step0_imbalance_max_over_avg=f"{mx/av:.2f}" if av else "")
    fl = final_re.search(txt)
    if fl:
        row.update(final_load_min=fl.group(1), final_load_max=fl.group(3))
    rb = re.search(r"Total rebalances: (\d+)", txt); mg = re.search(r"Total objects migrated: (\d+)", txt)
    if rb: row["rebalances"] = rb.group(1)
    if mg: row["objects_migrated"] = mg.group(1)
    passes = [float(x) for x in re.findall(r"rebalance completed in ([\d.]+)s", txt)]
    row["rcb_passes"] = len(passes)
    row["rcb_total_s"] = f"{sum(passes):.3f}"
    steps = re.search(r"Done: step=(\d+)", txt)
    if p and steps:
        row["per_step_net_ms"] = f"{float(p.group(5)) - 1000.0 * sum(passes) / int(steps.group(1)):.1f}"
    w = re.search(r"Done: step=\d+\s+sim_t=\S+\s+wall=([\d.]+)s", txt)
    if w: row["wall_s"] = w.group(1)
    rows.append(row)

# speed-up / efficiency relative to np=1 of the same (size, lb) series
def key(r): return (r["scaling_type"], r["object_lb"], r["n_objects"] if r["scaling_type"] == "strong" else "weak")
base = {}
for r in rows:
    if r["mpi_ranks"] == 1 and r["per_step_ms"]:
        base[key(r)] = float(r["per_step_ms"])
for r in rows:
    r["speedup"] = r["efficiency"] = r["speedup_net"] = r["efficiency_net"] = ""
    b = base.get(key(r))
    if b and r["per_step_ms"]:
        sp = b / float(r["per_step_ms"])
        r["speedup"] = f"{sp:.2f}"
        r["efficiency"] = f"{sp / r['mpi_ranks']:.2f}" if r["scaling_type"] == "strong" else f"{sp:.2f}"
    if b and r["per_step_net_ms"]:
        sp = b / float(r["per_step_net_ms"])
        r["speedup_net"] = f"{sp:.2f}"
        r["efficiency_net"] = f"{sp / r['mpi_ranks']:.2f}" if r["scaling_type"] == "strong" else f"{sp:.2f}"
rows.sort(key=lambda r: (r["scaling_type"], r["n_objects"] if r["scaling_type"] == "strong" else 0, r["object_lb"], r["mpi_ranks"]))
cols = ["campaign","workload","ic","scaling_type","object_lb","n_objects","mpi_ranks","status","per_step_ms","speedup","efficiency","per_step_net_ms","speedup_net","efficiency_net","rcb_passes","rcb_total_s",
        "build_ms","rate_ms","select_ms","exec_ms","step0_load_min","step0_load_avg","step0_load_max","step0_imbalance_max_over_avg",
        "final_load_min","final_load_max","rebalances","objects_migrated","wall_s","note"]
with open(out_csv, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=cols); w.writeheader(); w.writerows(rows)
print(f"{len(rows)} rows -> {out_csv}; ok={sum(r['status']=='ok' for r in rows)} failed={sum(r['status']=='failed' for r in rows)} incomplete={sum(r['status']=='incomplete' for r in rows)}")
