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

# Section 3.2 / Fig. 5 -- sink strength of spherical sinks (Malerba et al. 2007).
# bcc Fe, a0 = 2.87 A, <111> jumps, 3D migration; 250 SIAs per seed, run until the
# last SIA is absorbed. Capture radius R x number density N; five seeds each.
# (R = 8.25 and 10.25 nm at N = 1.5e17 cm^-3 are not part of the published figure.)
# --- environment: the release library + single-threaded kernels per MPI rank ---
HERE="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="$(cd "$HERE/../.." && pwd)"
export NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
SEEDS="41 42 43 44 45"
mkdir -p "$HERE/data/spherical"

run() {  # $1=R[nm] $2=N[cm^-3] $3=tag
  for S in $SEEDS; do
    python -u "$HERE/run_sphere_sink_strength.py" --regime 3d \
        --R-nm $1 --N-cm3 $2 --target-absorbed 250 --seed $S \
        --max-steps 15000000 --report-interval 500000 --progress-interval 0 --dump-dir "$HERE/dump/spherical/e1_$3_s$S" --dump-interval 50000 \
        > "$HERE/data/spherical/e1_$3_s$S.log" 2>&1
  done
}
for R in 2.25 4.25 6.25 8.25 10.25; do run $R 3e16   R${R/./p}_N3e16; done
for R in 2.25 4.25 6.25;            do run $R 1.5e17 R${R/./p}_N2e17; done
