#!/bin/bash
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

# Section 3.3 / Fig. 7 -- 1D-to-3D transition of the sink strength (Trinkaus et al. 2002).
# fcc Cu, a0 = 3.615 A, <110> 1D walk with a direction change every n_ch jumps;
# spherical sinks R = 15 nm at N = 1e22 m^-3 in a 500^3 nm^3 periodic box.
# n_ch from 1 (3D limit) to 1e5 (1D limit); five seeds. The number of absorbed
# defects per run is reduced for the most 1D cases because each walk is long.
# --- environment: the release library + single-threaded kernels per MPI rank ---
HERE="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="$(cd "$HERE/../.." && pwd)"
export NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
SEEDS="41 42 43 44 45"
mkdir -p "$HERE/data"

declare -A TGT=( [1]=60 [3]=60 [10]=60 [30]=60 [100]=60 [300]=60 [1000]=60 [3000]=60 [10000]=40 [30000]=45 [100000]=30 )
for N in 1 3 10 30 100 300 1000 3000 10000 30000 100000; do
  for S in $SEEDS; do
    python -u "$HERE/run_1d3d_master_curve.py" --n-ch $N --R-nm 15 \
        --box-nm 500 500 500 --target-absorbed ${TGT[$N]} --seed $S \
        --csv "$HERE/data/e3_s${S}_nch$N.csv" --max-steps 12000000 \
        --report-interval 0 --progress-interval 500000 \
        --dump-dir "$HERE/dump/e3_s${S}_nch$N" --dump-interval 15000 \
        > "$HERE/data/e3_s${S}_nch$N.log" 2>&1
  done
done
