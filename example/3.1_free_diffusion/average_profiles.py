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

"""
Average multiple e0 profile outputs and compare against the analytical solution.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _load_profile_csv(path: Path):
    data = np.genfromtxt(path, delimiter=",", names=True)
    if data.ndim == 0:
        data = np.array([data], dtype=data.dtype)

    sim_cols = [name for name in data.dtype.names if name.startswith("sim_bin_")]
    ana_cols = [name for name in data.dtype.names if name.startswith("ana_bin_")]
    sim = np.column_stack([data[name] for name in sim_cols])
    ana = np.column_stack([data[name] for name in ana_cols])
    steps = data["step"].astype(np.int64)
    times = data["time_s"].astype(np.float64)
    return steps, times, sim, ana


def _common_step_average(profile_paths):
    loaded = [_load_profile_csv(Path(p)) for p in profile_paths]
    common_steps = sorted(set(loaded[0][0]).intersection(*[set(item[0]) for item in loaded[1:]]))
    if not common_steps:
        raise ValueError("No common steps found across the provided x_profile.csv files")

    avg_steps = []
    avg_times = []
    avg_sim = []
    avg_ana = []

    for step in common_steps:
        times = []
        sim_rows = []
        ana_rows = []
        for steps, times_arr, sim, ana in loaded:
            idx = int(np.where(steps == step)[0][0])
            times.append(times_arr[idx])
            sim_rows.append(sim[idx])
            ana_rows.append(ana[idx])
        avg_steps.append(step)
        avg_times.append(float(np.mean(times)))
        avg_sim.append(np.mean(np.vstack(sim_rows), axis=0))
        avg_ana.append(np.mean(np.vstack(ana_rows), axis=0))

    return (
        np.asarray(avg_steps, dtype=np.int64),
        np.asarray(avg_times, dtype=np.float64),
        np.vstack(avg_sim),
        np.vstack(avg_ana),
    )


def _write_average_csv(out_csv: Path, steps, times, sim, ana):
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    n_bins = sim.shape[1]
    header = ["step", "time_s"]
    header.extend(f"sim_bin_{i:03d}" for i in range(n_bins))
    header.extend(f"ana_bin_{i:03d}" for i in range(n_bins))

    arr = np.column_stack([steps, times, sim, ana])
    np.savetxt(out_csv, arr, delimiter=",", header=",".join(header), comments="", fmt="%.10e")


def _plot_profiles(out_png: Path, steps, times, sim, ana, box_length_nm: float):
    out_png.parent.mkdir(parents=True, exist_ok=True)
    x_centers_nm = (np.arange(sim.shape[1], dtype=np.float64) + 0.5) * (box_length_nm / sim.shape[1])

    if len(steps) >= 3:
        pick = [0, len(steps) // 2, len(steps) - 1]
    else:
        pick = list(range(len(steps)))

    fig, axes = plt.subplots(1, len(pick), figsize=(5.0 * len(pick), 4.2), sharey=True)
    if len(pick) == 1:
        axes = [axes]

    colors = ["tab:blue", "tab:orange", "tab:green"]
    for ax, idx, color in zip(axes, pick, colors):
        ax.plot(x_centers_nm, ana[idx], color=color, linewidth=2.2)
        ax.plot(x_centers_nm, sim[idx], "o", color=color, markersize=4.0)
        ax.set_title(f"step {int(steps[idx])}\nt = {times[idx]:.3e} s")
        ax.set_xlabel("Depth x (nm)")
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel("Line density (particles / nm)")
    fig.suptitle("e0 averaged diffusion profile vs analytical solution")
    fig.tight_layout()
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_overlay_profiles(out_png: Path, steps, times, sim, ana, box_length_nm: float, target_times):
    out_png.parent.mkdir(parents=True, exist_ok=True)
    x_centers_nm = (np.arange(sim.shape[1], dtype=np.float64) + 0.5) * (box_length_nm / sim.shape[1])

    target_times = [float(t) for t in target_times]
    chosen = []
    used = set()
    for target in target_times:
        idx = int(np.argmin(np.abs(times - target)))
        if idx not in used:
            chosen.append(idx)
            used.add(idx)

    fig, ax = plt.subplots(figsize=(8.0, 5.2))
    cmap = plt.get_cmap("tab10")

    for i, idx in enumerate(chosen):
        color = cmap(i % 10)
        label = f"t={times[idx]:.2f} s"
        ax.plot(x_centers_nm, sim[idx], color=color, linewidth=2.0, label=f"{label} (KMC avg)")
        ax.plot(x_centers_nm, ana[idx], color=color, linestyle="--", linewidth=2.0, label=f"{label} (analytic)")

    ax.set_title("e0 averaged diffusion profiles at multiple times")
    ax.set_xlabel("Depth x (nm)")
    ax.set_ylabel("Line density (particles / nm)")
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=2, fontsize=9)
    fig.tight_layout()
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profiles", nargs="+", required=True, help="Input x_profile.csv files")
    parser.add_argument("--out-csv", required=True, help="Output averaged CSV path")
    parser.add_argument("--out-png", required=True, help="Output figure path")
    parser.add_argument("--box-length-nm", type=float, default=100.0, help="Box length in x (nm)")
    parser.add_argument(
        "--target-times",
        type=float,
        nargs="*",
        default=None,
        help="If provided, plot these times on one overlay figure; analytic curves are dashed",
    )
    args = parser.parse_args()

    steps, times, sim, ana = _common_step_average(args.profiles)
    _write_average_csv(Path(args.out_csv), steps, times, sim, ana)
    if args.target_times:
        _plot_overlay_profiles(
            Path(args.out_png),
            steps,
            times,
            sim,
            ana,
            float(args.box_length_nm),
            args.target_times,
        )
    else:
        _plot_profiles(Path(args.out_png), steps, times, sim, ana, float(args.box_length_nm))


if __name__ == "__main__":
    main()
