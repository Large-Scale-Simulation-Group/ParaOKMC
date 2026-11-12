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

"""Regenerate all ten manuscript figures at final print size.

Paper figure -> output file -> data source
  Fig 1  fig01_architecture.png   (drawn)
  Fig 2  fig02_visibility.png     (drawn)
  Fig 3  fig03_protocol.png       (drawn)
  Fig 4  fig04_e0_diffusion.png   validation_results/e0_s41..45/
  Fig 5  fig05_e1_spheres.png     validation_results/grid/e1_*.log
  Fig 6  fig06_e2_dislocation.png validation_results/grid/e2_*.log
  Fig 7  fig07_e3_crossover.png   validation_results/{grid,remote,extra,.}/e3_*.csv
  Fig 8  fig08_invariance.png     validation_results/postfix_np/e2_np*.log
  Fig 9  fig09_strong.png         docs/scaling/results/bscc_a_measured_20260717.csv
  Fig 10 fig10_weak.png           same CSV
  Fig 11 fig11_hotspot.png        docs/scaling/results/bscc_a_lb_levels_20260916.csv
                                  (hotspot IC: no LB = scaling_v9, LB on = event-level
                                  redistribution only, scaling_v14)

Data handling is ported unchanged from validation_results/figures/
(make_extra_figures.py, make_diag_figures.py, make_parallel_figure.py,
make_scaling_figures_measured.py) so every plotted number stays identical to
the figures the text quotes; only sizes, fonts and layout change.
"""

import csv
import glob
import math
import os
import re
import statistics as st
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle, FancyArrowPatch, Rectangle
from matplotlib.ticker import FixedLocator, FixedFormatter, NullFormatter, LogLocator, MultipleLocator

from pubstyle import apply, INK, MUTED, SURFACE, DATA, DATA2, THEORY, \
    C1K, C10K, C100K, CPAIR, FULLW, HALFW

HERE = os.path.dirname(os.path.abspath(__file__))
EX = os.path.dirname(HERE)   # paraokmc_release/example
E0 = f"{EX}/3.1_free_diffusion/data"
E3 = f"{EX}/3.3_1d3d_transition/data"
SINK = {"e1": f"{EX}/3.2_sink_strength/data/spherical", "e2": f"{EX}/3.2_sink_strength/data/linear"}
SCAL = f"{EX}/4.1_benchmark_workloads/bscc_a_measured_20260717.csv"
HOT = f"{EX}/4.4_load_balancing/fig11_hotspot_lb_levels/bscc_a_lb_levels_20260916.csv"
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = f"{HERE}/out"
os.makedirs(OUT, exist_ok=True)

DJ = math.sqrt(3.0) / 2.0 * 2.87e-10      # bcc Fe <111> jump distance
A_CU = 3.615e-10 / math.sqrt(2.0)         # fcc Cu <110> jump distance
SEEDS = [41, 42, 43, 44, 45]

apply({"xtick.minor.visible": True, "ytick.minor.visible": True})


def save(fig, name):
    fig.savefig(f"{OUT}/{name}.png", dpi=300)
    fig.savefig(f"{OUT}/{name}.pdf")
    plt.close(fig)
    print(f"{name} written")


# ================================================================ Fig 4 (e0)
def fig04():
    TIMES = [0.05, 0.5, 2.0, 9.0]
    NBIN = 32
    xc = (np.arange(NBIN) + 0.5) * 100.0 / NBIN
    RAMP = ["#9ec5f4", "#5598e7", "#2a78d6", "#123a6b"]

    profiles = {}
    for t in TIMES:
        sims, ana = [], None
        for s in SEEDS:
            d = np.genfromtxt(f"{E0}/e0_s{s}/x_profile.csv", delimiter=",", skip_header=1)
            i = int(np.argmin(np.abs(d[:, 1] - t)))
            sims.append(d[i, 2:2 + NBIN])
            if ana is None:
                ana = d[i, 2 + NBIN:2 + 2 * NBIN]
        profiles[t] = (np.mean(sims, axis=0), ana)

    fig, ax = plt.subplots(figsize=(FULLW, 2.90))
    fig.subplots_adjust(left=0.085, right=0.975, bottom=0.155, top=0.965)

    MARKS = ["o", "s", "^", "D"]   # one marker shape per time, on top of the colour ramp
    MS = {"o": 5.2, "s": 4.8, "^": 5.6, "D": 4.4}
    for (t, (sim, ana)), c, mk in zip(profiles.items(), RAMP, MARKS):
        ax.plot(xc, ana, "-", color=c, lw=1.8, zorder=2)
        ax.plot(xc, sim, mk, color=c, ms=MS[mk], mec="white", mew=0.6, zorder=3)

    # two legend columns inside the axes: lines (analytic) | markers (ParaOKMC)
    from matplotlib.lines import Line2D
    line_h = [Line2D([], [], color=c, lw=1.8, label=f"$t = {t:g}$ s")
              for t, c in zip(TIMES, RAMP)]
    mark_h = [Line2D([], [], color=c, marker=mk, ls="none", ms=MS[mk],
                     mec="white", mew=0.6, label=f"$t = {t:g}$ s")
              for t, c, mk in zip(TIMES, RAMP, MARKS)]
    legA = ax.legend(handles=line_h, title="Analytical", loc="upper right",
                     bbox_to_anchor=(0.815, 0.985), fontsize=8.5,
                     handlelength=1.5, borderaxespad=0.0,
                     title_fontproperties={"size": 8.5, "weight": "bold"})
    ax.add_artist(legA)
    ax.legend(handles=mark_h, title="ParaOKMC", loc="upper right",
              bbox_to_anchor=(0.985, 0.985), fontsize=8.5,
              handlelength=1.5, borderaxespad=0.0,
              title_fontproperties={"size": 8.5, "weight": "bold"})

    ax.set_xlim(0, 100)
    ax.set_ylim(-3, 97)
    ax.xaxis.set_major_locator(MultipleLocator(10))
    ax.xaxis.set_minor_locator(MultipleLocator(2))
    ax.set_xlabel("Position $x$ (nm)")
    ax.set_ylabel("The number of defects per 3.125 nm")
    save(fig, "fig04_e0_diffusion")


# ========================================================== Fig 5/6 helpers
def read(f):
    return open(f, encoding="utf-8", errors="replace").read()


def pool_sink(prefix):
    out = {}
    for f0 in sorted(glob.glob(f"{SINK[prefix]}/{prefix}_*_s41.log")):
        tag = re.search(rf"{prefix}_(.+)_s41\.log", os.path.basename(f0)).group(1)
        logs = [f"{SINK[prefix]}/{prefix}_{tag}_s{s}.log" for s in SEEDS]
        logs = [l for l in logs if os.path.exists(l)]
        texts = [read(l) for l in logs]
        if any(re.search(r"RuntimeError|ValueError", t) for t in texts):
            continue
        njs = []
        rad = dens = None
        for t in texts:
            m = re.search(r"<nj> = ([\d.eE+]+)", t)
            if not m:
                continue
            njs.append(float(m.group(1)))
            if prefix == "e1":
                rad = float(re.search(r"R\(capture\)=([\d.]+)", t).group(1))
                dens = float(re.search(r"N=([\d.eE+]+) cm", t).group(1))
            else:
                rad = float(re.search(r"rd\(capture\)=([\d.]+)", t).group(1))
                dens = float(re.search(r"rho_d\(actual\)=([\d.eE+]+)", t).group(1))
        if len(njs) < 2:
            continue
        m = sum(njs) / len(njs)
        sem = st.stdev(njs) / math.sqrt(len(njs))
        k2 = 6.0 / (DJ * DJ * m)
        out[tag] = dict(radius=rad, density=dens, k2=k2, err=k2 * sem / m)
    return out


def e1_theory(R_nm, N_cm3):
    R = R_nm * 1e-9
    N = N_cm3 * 1e6
    k0 = 4 * math.pi * N * R
    return k0 * (1 + R * math.sqrt(k0))


def e2_theory(rd_nm, rho):
    rd = rd_nm * 1e-9
    q = rd * math.sqrt(math.pi * rho)
    return 2 * math.pi * rho * (1 - q * q) / (math.log(1 / q) - 0.75 + q * q * (1 - 0.25 * q * q))


def _sci(d):
    e = int(math.floor(math.log10(d)))
    m = d / 10 ** e
    ms = f"{m:.1f}".rstrip("0").rstrip(".")
    return rf"{ms}\times10^{{{e}}}"


def sink_figure(data, theory_fn, sym, sym_unit, xlabel, theory_label, fname,
                xlim, ylim, legend_loc="upper left", dens_scale=1.0, ymajor=None,
                figsize=(HALFW, 3.45), legend_fs=9.5, theory_full=False):
    """One panel, both density families together (markers + matching curves).
    theory_full=True draws each theory curve over the whole x range instead of
    the range of its own data points."""
    groups = defaultdict(list)
    for v in data.values():
        groups[f"{v['density']:.3g}"].append(v)
    dens_keys = sorted(groups, key=lambda k: float(k))

    fig, ax = plt.subplots(figsize=figsize)
    fam = [(DATA, "o"), (DATA2, "s")]
    ranges = []
    for (dk, (c, mk)) in zip(dens_keys, fam):
        pts = sorted(groups[dk], key=lambda p: p["radius"])
        dens = pts[0]["density"]
        Rs = np.array([p["radius"] for p in pts])
        sim = np.array([p["k2"] for p in pts]) / 1e15
        err = np.array([p["err"] for p in pts]) / 1e15
        Rgrid = (np.linspace(xlim[0], xlim[1], 240) if theory_full
                 else np.linspace(Rs.min() * 0.90, Rs.max() * 1.06, 240))
        thy = np.array([theory_fn(r, dens) for r in Rgrid]) / 1e15
        ax.plot(Rgrid, thy, "-", color=THEORY, lw=1.2, zorder=2)
        ax.errorbar(Rs, sim, yerr=err, fmt=mk, color=c, ms=5.2, mfc=c,
                    mec="white", mew=0.6, elinewidth=0.9, capsize=2.0,
                    capthick=0.9, ls="none", zorder=3,
                    label=rf"ParaOKMC, ${sym} = {_sci(dens * dens_scale)}$ {sym_unit}")
        ranges.append((Rs.min(), Rs.max(), dens))
    ax.plot([], [], "-", color=THEORY, lw=1.2, label=theory_label)

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    if ymajor:
        ax.yaxis.set_major_locator(MultipleLocator(ymajor))
    ax.tick_params(labelsize=10)
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(r"Sink strength $k^2$ ($10^{15}\,\mathrm{m^{-2}}$)", fontsize=11)
    ax.legend(loc=legend_loc, fontsize=legend_fs, handlelength=1.3, borderaxespad=0.5)
    fig.tight_layout(pad=0.8)   # pad large enough that the last tick label is never clipped
    save(fig, fname)
    devs = [100 * (p["k2"] - theory_fn(p["radius"], p["density"])) / theory_fn(p["radius"], p["density"])
            for p in data.values()]
    print(f"  {fname}: {len(data)} conditions, dev [{min(devs):+.0f}%, {max(devs):+.0f}%], "
          f"median|dev| {np.median(np.abs(devs)):.0f}%")
    for lo, hi, dens in ranges:
        print(f"    density {dens:.4g}: radii {lo} .. {hi} nm")


def fig05(e1):
    sink_figure(e1, e1_theory, "N", r"m$^{-3}$",
                "Capture radius $R_c$ (nm)", "Malerba’s theory, Eq. (5)",
                "fig05_e1_spheres", xlim=(2, 11), ylim=(0, 35), dens_scale=1e6,
                legend_loc="center right", figsize=(FULLW, 4.0), legend_fs=10.5,
                theory_full=True)


def fig06(e2):
    sink_figure(e2, e2_theory, r"\rho_d", r"m$^{-2}$",
                "Capture radius $r_d$ (nm)", "Wiedersich’s theory, Eq. (6)",
                "fig06_e2_dislocation", xlim=(2, 10), ylim=(0, 15), ymajor=5,
                legend_loc="center right")


# ================================================================ Fig 7 (e3)
def pool_e3():
    per, meta = {}, {}
    for f in (glob.glob(f"{E3}/e3_s*_nch*.csv") + glob.glob(f"{E3}/e3_nch*_s*.csv")):
        row = np.atleast_1d(np.genfromtxt(f, delimiter=",", names=True))[-1]
        n = int(row["n_ch"])
        per.setdefault(n, []).append(float(row["avg_jumps"]))
        meta[n] = float(row["k2_1d_m2"])
    out = {}
    for n, ajs in per.items():
        m = sum(ajs) / len(ajs)
        sem = st.stdev(ajs) / math.sqrt(len(ajs)) if len(ajs) > 1 else 0.0
        k2 = 6.0 / (A_CU * A_CU * m)
        out[n] = dict(k2=k2, err=k2 * sem / m, k2_1=meta[n])
    return out


def fig07():
    e3 = pool_e3()
    K2_1 = e3[1]["k2_1"]
    K3 = e3[1]["k2"]

    def e3_theory(n):
        lch = A_CU * math.sqrt(n)
        x2 = lch * lch * K2_1 / 12.0 + (K2_1 / K3) ** 2
        return 0.5 * (1.0 + math.sqrt(1.0 + 4.0 / x2)) * K2_1

    ns = sorted(e3)
    fig, ax = plt.subplots(figsize=(HALFW, 3.45))
    xmax = max(ns) * 1.7
    ngrid = np.logspace(0, math.log10(xmax), 280)
    ax.plot(ngrid, np.array([e3_theory(n) for n in ngrid]) / 1e15, "-",
            color=THEORY, lw=1.2, zorder=2, label="Trinkaus’s theory, Eq. (7)")
    sim = np.array([e3[n]["k2"] for n in ns]) / 1e15
    err = np.array([e3[n]["err"] for n in ns]) / 1e15
    ax.errorbar(ns, sim, yerr=err, fmt="o", color=DATA, ms=5.2, mfc=DATA,
                mec="white", mew=0.6, elinewidth=0.9, capsize=2.0, capthick=0.9,
                ls="none", zorder=3, label="ParaOKMC")
    ax.set_xscale("log")
    ax.set_xlim(0.8, xmax)
    ax.set_ylim(0, 5.0)
    ax.tick_params(labelsize=10)
    ax.set_xlabel(r"Number of jumps between direction changes $n_{\mathrm{ch}}$", fontsize=11)
    ax.set_ylabel(r"Sink strength $k^2$ ($10^{15}\,\mathrm{m^{-2}}$)", fontsize=11)
    ax.legend(loc="lower left", fontsize=10, borderaxespad=0.6)
    ax.annotate("3D limit", (1.3, 4.35), fontsize=12, color=MUTED, va="center",
                ha="left")
    ax.annotate("1D limit", (2.5e4, 0.28), fontsize=12, color=MUTED, ha="center",
                va="center")
    fig.tight_layout(pad=0.4)
    save(fig, "fig07_e3_crossover")
    devs = [100 * (e3[n]["k2"] - e3_theory(n)) / e3_theory(n) for n in ns]
    print(f"  fig07: {len(ns)} conditions, dev [{min(devs):+.0f}%, {max(devs):+.0f}%], "
          f"median|dev| {np.median(np.abs(devs)):.0f}%")


# ============================================================ Fig 8 (np inv)
def fig08():
    PF = f"{EX}/3.4_parallel_invariance/data"

    def nj_of(f):
        t = read(f)
        m = re.search(r"<nj> = ([\d.eE+]+)", t)
        th = re.search(r"Wiedersich\) = ([\d.eE+]+)", t)
        return (float(m.group(1)) if m else None, float(th.group(1)) if th else None)

    theory_k2 = None
    nps = [1, 2, 4, 8]
    k2s, errs = [], []
    for np_ in nps:
        files = sorted(glob.glob(f"{PF}/e2_np{np_}_s*.log")) + \
            ([f"{PF}/e2_np{np_}.log"] if os.path.exists(f"{PF}/e2_np{np_}.log") else [])
        njs = []
        for f in files:
            nj, th = nj_of(f)
            if nj:
                njs.append(nj)
                theory_k2 = th
        m = sum(njs) / len(njs)
        sem = st.stdev(njs) / math.sqrt(len(njs))
        k2 = 6.0 / (DJ * DJ * m)
        k2s.append(k2 / 1e15)
        errs.append(k2 / 1e15 * sem / m)
        print(f"  np={np_}: {len(njs)} seeds, k2/theory = {k2/theory_k2:.3f}")
    th = theory_k2 / 1e15

    fig, ax = plt.subplots(figsize=(HALFW, 3.45))
    x = np.arange(len(nps))
    ax.axhline(th, color=THEORY, lw=1.1, ls="--", zorder=1)
    ax.annotate("Wiedersich’s\ntheory", (2.5, th + 0.10), ha="center", va="bottom",
                fontsize=12, color=THEORY, linespacing=1.15)
    ax.errorbar(x, k2s, yerr=errs, fmt="o", color=DATA, ms=5.5, mfc=DATA,
                mec="white", mew=0.7, elinewidth=1.0, capsize=2.6, capthick=1.0,
                ls="none", zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels([str(n) for n in nps])
    ax.xaxis.set_minor_locator(FixedLocator([]))
    ax.set_xlim(-0.45, 3.45)
    ax.set_ylim(11.2, 15.4)
    ax.tick_params(labelsize=10)
    ax.set_xlabel("MPI rank count $n_p$", fontsize=11)
    ax.set_ylabel(r"Sink strength $k^2$ ($10^{15}\,\mathrm{m^{-2}}$)", fontsize=11)
    fig.tight_layout(pad=0.4)
    save(fig, "fig08_invariance")


# ========================================================= Fig 9/10 (scaling)
def load_scaling():
    rows = [r for r in csv.DictReader(open(SCAL)) if r["status"] == "ok"]

    def series(workload, scaling, arm, n_objects=None):
        out = []
        for r in rows:
            if r["workload"] != workload or r["scaling_type"] != scaling or r["lib_arm"] != arm:
                continue
            if n_objects is not None and r["n_objects"] != n_objects:
                continue
            out.append((int(r["mpi_ranks"]), float(r["per_step_ms"])))
        return sorted(out)

    keep = {1, 2, 4, 8, 16, 32, 64}
    return lambda *a, **k: [(n, t) for n, t in series(*a, **k) if n in keep]


def fig09(series):
    fig, ax = plt.subplots(figsize=(HALFW, 3.7))
    ideal = np.array([1, 64])
    ax.plot(ideal, ideal, "--", color=INK, lw=1.0, zorder=2, label="ideal")

    STRONG = [
        ("e0_diffusion", "1000", "new", C1K, "^", "-", True, r"diffusion, $N_{\mathrm{obj}} = 1$k"),
        ("e0_diffusion", "10000", "new", C10K, "s", "-", True, r"diffusion, $N_{\mathrm{obj}} = 10$k"),
        ("e0_diffusion", "100000", "new", C100K, "o", "-", True, r"diffusion, $N_{\mathrm{obj}} = 100$k"),
        ("pair_HV", "20000", "new_on", CPAIR, "D", (0, (4, 2)), False, r"pair, $N_{\mathrm{obj}} = 20$k"),
    ]
    for wl, N, arm, c, m, ls, filled, lab in STRONG:
        s = series(wl, "strong", arm, N)
        n = np.array([x[0] for x in s])
        t = np.array([x[1] for x in s])
        sp = t[0] / t
        ax.plot(n, sp, ls=ls, marker=m, color=c, lw=1.2, ms=4.4,
                mfc=(c if filled else "white"), mec=("white" if filled else c),
                mew=(0.6 if filled else 1.0), zorder=4, label=lab)
        print(f"  strong {lab}: " + " ".join(f"{ni}:{si:.1f}x" for ni, si in zip(n, sp)))

    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=2)
    TICKS = [1, 2, 4, 8, 16, 32, 64]
    ax.xaxis.set_major_locator(FixedLocator(TICKS))
    ax.xaxis.set_major_formatter(FixedFormatter([str(v) for v in TICKS]))
    ax.yaxis.set_major_locator(FixedLocator([1, 2, 4, 8, 16, 32, 64]))
    ax.yaxis.set_major_formatter(FixedFormatter(["1", "2", "4", "8", "16", "32", "64"]))
    ax.xaxis.set_minor_locator(FixedLocator([]))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_xlim(0.85, 76)
    ax.set_ylim(0.55, 84)
    ax.set_xlabel("MPI ranks $n_p$")
    ax.set_ylabel(r"Speed-up $T_1 / T_{n_p}$")
    ax.legend(loc="upper left", fontsize=12, handlelength=2.0, borderaxespad=0.5)
    fig.tight_layout(pad=0.4)
    save(fig, "fig09_strong")


def fig10(series):
    fig, ax = plt.subplots(figsize=(HALFW, 3.35))
    ax.axhline(1.0, ls="--", color=INK, lw=1.0, zorder=2, label="ideal")
    WEAK = [
        ("e0_diffusion", "weak_1k_per_rank", "new", C100K, "o",
         "diffusion, 1k objects per rank"),
        ("pair_HV", "weak_20k_per_rank", "new", CPAIR, "D",
         "pair, 20k objects per rank"),
    ]
    for wl, sc, arm, c, m, lab in WEAK:
        s = series(wl, sc, arm)
        n = np.array([x[0] for x in s])
        t = np.array([x[1] for x in s])
        eff = t[0] / t
        ax.plot(n, eff, "-", marker=m, color=c, lw=1.2, ms=4.4, mec="white",
                mew=0.6, zorder=4, label=lab)
        print(f"  weak {lab}: " + " ".join(f"{ni}:{100*ei:.0f}%" for ni, ei in zip(n, eff)))
    ax.set_xscale("log", base=2)
    TICKS = [1, 2, 4, 8, 16, 32, 64]
    ax.xaxis.set_major_locator(FixedLocator(TICKS))
    ax.xaxis.set_major_formatter(FixedFormatter([str(v) for v in TICKS]))
    ax.xaxis.set_minor_locator(FixedLocator([]))
    ax.set_xlim(0.85, 76)
    ax.set_ylim(0.68, 1.045)
    ax.set_xlabel("MPI ranks $n_p$")
    ax.set_ylabel(r"Weak-scaling efficiency $T_1 / T_{n_p}$")
    ax.legend(loc="lower left", fontsize=12, handlelength=2.0, borderaxespad=0.5)
    fig.tight_layout(pad=0.4)
    save(fig, "fig10_weak")


# ================================================= Fig 11 (hotspot vs slab)
def fig11(series):
    """Local-hotspot effect: the diffusion workload started from the Table-1 hotspot
    box (0.1 of every box edge, 0.1 % of the volume) with load balancing off (arm
    "none", scaling_v9) and with event-level redistribution only (arm "event",
    scaling_v14: in-place structural increment kept under event migration, two-stage
    imbalance check).  Each series is normalised by its own single-rank time
    (T1 = 198 and 191 ms at 100k)."""
    keep = {1, 2, 4, 8, 16, 32, 64}
    hot_rows = [r for r in csv.DictReader(open(HOT)) if r["status"] == "ok"]

    def hot(scaling, arm, n_objects=None):
        out = [(int(r["mpi_ranks"]), float(r["per_step_ms"])) for r in hot_rows
               if r["scaling_type"] == scaling and r["arm"] == arm
               and (n_objects is None or r["n_objects"] == n_objects)]
        return sorted(x for x in out if x[0] in keep)

    HOFF, HON = DATA2, DATA2
    ROWS = [  # (strong series, weak series, colour, marker, ls, filled, label)
        (hot("strong", "none", "100000"), hot("weak_1k_per_rank", "none"),
         HOFF, "s", (0, (4, 2)), False, "load balancing off"),
        (hot("strong", "event", "100000"), hot("weak_1k_per_rank", "event"),
         HON, "s", "-", True, "load balancing on"),
    ]

    fig, (axa, axb) = plt.subplots(1, 2, figsize=(FULLW, 2.95))
    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.16, top=0.90, wspace=0.30)

    ideal = np.array([1, 64])
    axa.plot(ideal, ideal, "--", color=INK, lw=1.0, zorder=2, label="ideal")
    axb.axhline(1.0, ls="--", color=INK, lw=1.0, zorder=2, label="ideal")
    DY = (-7, 7)   # end-point label offsets (pt): LB off / LB on
    for (st, wk, c, m, ls, filled, lab), dy in zip(ROWS, DY):
        n = np.array([x[0] for x in st]); t = np.array([x[1] for x in st]); sp = t[0] / t
        axa.plot(n, sp, ls=ls, marker=m, color=c, lw=1.2, ms=4.4,
                 mfc=(c if filled else "white"), mec=("white" if filled else c),
                 mew=(0.6 if filled else 1.0), zorder=4, label=lab)
        print(f"  fig11 strong {lab}: " + " ".join(f"{ni}:{si:.2f}x" for ni, si in zip(n, sp)))
        n = np.array([x[0] for x in wk]); t = np.array([x[1] for x in wk]); eff = t[0] / t
        axb.plot(n, eff, ls=ls, marker=m, color=c, lw=1.2, ms=4.4,
                 mfc=(c if filled else "white"), mec=("white" if filled else c),
                 mew=(0.6 if filled else 1.0), zorder=4, label=lab)
        print(f"  fig11 weak {lab}: " + " ".join(f"{ni}:{100*ei:.0f}%" for ni, ei in zip(n, eff)))

    axa.set_xscale("log", base=2)
    axa.set_yscale("log", base=2)
    TICKS = [1, 2, 4, 8, 16, 32, 64]
    axa.xaxis.set_major_locator(FixedLocator(TICKS))
    axa.xaxis.set_major_formatter(FixedFormatter([str(v) for v in TICKS]))
    axa.yaxis.set_major_locator(FixedLocator([1, 2, 4, 8, 16, 32, 64]))
    axa.yaxis.set_major_formatter(FixedFormatter(["1", "2", "4", "8", "16", "32", "64"]))
    axa.xaxis.set_minor_locator(FixedLocator([]))
    axa.yaxis.set_minor_formatter(NullFormatter())
    axa.set_xlim(0.85, 100)
    axa.set_ylim(0.8, 84)
    axa.set_xlabel("MPI ranks $n_p$")
    axa.set_ylabel(r"Speed-up $T_1 / T_{n_p}$")
    axa.set_title("(a) strong scaling, 100k objects", loc="left", fontsize=10.5)
    axa.legend(loc="upper left", fontsize=10.5, handlelength=2.0, borderaxespad=0.4)

    axb.set_xscale("log", base=2)
    TICKS = [1, 2, 4, 8, 16, 32, 64]
    axb.xaxis.set_major_locator(FixedLocator(TICKS))
    axb.xaxis.set_major_formatter(FixedFormatter([str(v) for v in TICKS]))
    axb.xaxis.set_minor_locator(FixedLocator([]))
    axb.set_xlim(0.85, 100)
    axb.set_ylim(0.22, 1.12)
    axb.set_xlabel("MPI ranks $n_p$")
    axb.set_ylabel(r"Weak-scaling efficiency $T_1 / T_{n_p}$")
    axb.set_title("(b) weak scaling, 1k objects per rank", loc="left", fontsize=10.5)
    axb.legend(loc="lower left", fontsize=10.5, handlelength=2.0, borderaxespad=0.4)
    save(fig, "fig11_hotspot")


if __name__ == "__main__":
    fig04()
    e1 = pool_sink("e1")
    e2 = pool_sink("e2")
    fig05(e1)
    fig06(e2)
    fig07()
    fig08()
    series = load_scaling()
    fig09(series)
    fig10(series)
    fig11(series)
