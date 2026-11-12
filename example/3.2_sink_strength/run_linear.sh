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

# Section 3.2 / Fig. 6 -- sink strength of a straight dislocation (Jansson et al. 2013,
# Wiedersich expression). bcc Fe, 3D migration; 250 SIAs per seed, run until the last
# SIA is absorbed. Capture radius rd x dislocation density rho_d; five seeds each.
# (rd = 9.25 nm at rho_d = 1e15 m^-2 exceeds half the box cross-section and is omitted.)
# --- environment: the release library + single-threaded kernels per MPI rank ---
HERE="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="$(cd "$HERE/../.." && pwd)"
export NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
SEEDS="41 42 43 44 45"
mkdir -p "$HERE/data"

mkdir -p "$HERE/data/linear"
run() {  # $1=rd[nm] $2=rho[m^-2] $3=tag
  for S in $SEEDS; do
    python -u "$HERE/run_dislocation_sink_strength.py" \
        --rd-nm $1 --rho-m2 $2 --target-absorbed 250 --seed $S \
        --max-steps 15000000 --report-interval 500000 --progress-interval 0 --dump-dir "$HERE/dump/linear/e2_$3_s$S" --dump-interval 50000 \
        > "$HERE/data/linear/e2_$3_s$S.log" 2>&1
  done
}
for RD in 2.25 4.25 6.25 9.25; do run $RD 3e14 rd${RD/./p}_rho3e14; done
for RD in 2.25 4.25 6.25;      do run $RD 1e15 rd${RD/./p}_rho1e15; done
