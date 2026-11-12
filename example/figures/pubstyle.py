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

"""Shared style for the manuscript figures, drawn at final print size.

Every figure is created at the exact width it occupies in the manuscript
(6.0 in full-column or 4.6 in reduced) and inserted at 100 %, so a point of
type in the script is a point of type on the page.  Font sizes therefore sit
directly in the 8.5-10 pt band that a 12 pt body text calls for.
"""

TIMES = ["Times New Roman", "Nimbus Roman", "Liberation Serif",
         "Tinos", "FreeSerif", "DejaVu Serif"]

INK, MUTED, SURFACE = "#1a1a19", "#55534e", "#ffffff"
DATA = "#2a78d6"          # OKMC results (identity colour of the suite)
DATA2 = "#c85a19"         # second parameter family (vermillion, CVD-safe pair)
THEORY = "#1a1a19"        # analytic reference curves

# strong-scaling series (kept from the measured-campaign figures)
C1K, C10K, C100K, CPAIR = "#eda100", "#1baf7a", "#2a78d6", "#c2426d"

FULLW, HALFW = 6.0, 4.6   # inserted widths (in), matching the 6.0 in text column


def apply(extra=None):
    import matplotlib.pyplot as plt
    rc = {
        "figure.dpi": 150, "savefig.dpi": 300,
        "font.family": "serif",
        "font.serif": TIMES,
        "mathtext.fontset": "stix",
        "font.size": 9,
        "axes.labelsize": 10,
        "axes.titlesize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "text.color": INK, "axes.labelcolor": INK, "axes.edgecolor": INK,
        "axes.linewidth": 0.8,
        "xtick.color": INK, "ytick.color": INK,
        "xtick.direction": "in", "ytick.direction": "in",
        "xtick.top": True, "ytick.right": True,
        "xtick.major.size": 4.0, "ytick.major.size": 4.0,
        "xtick.minor.size": 2.2, "ytick.minor.size": 2.2,
        "xtick.major.width": 0.8, "ytick.major.width": 0.8,
        "xtick.minor.width": 0.6, "ytick.minor.width": 0.6,
        "axes.grid": False,
        "legend.frameon": False,
        "legend.handlelength": 1.7,
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
    }
    if extra:
        rc.update(extra)
    plt.rcParams.update(rc)
