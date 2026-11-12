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

# Section 3.4 / Fig. 8 -- invariance of the sink strength under parallel execution.
# Linear-sink case rho_d = 1e15 m^-2, rd = 6.25 nm, box 28.7 x 31.6 x 31.6 nm^3,
# 60 SIAs per seed, 1/2/4/8 MPI ranks, five seeds each. All three load-balancing
# levels are off (the script disables them), so only the domain decomposition acts.
# --- environment: the release library + single-threaded kernels per MPI rank ---
HERE="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="$(cd "$HERE/../.." && pwd)"
export NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
SEEDS="41 42 43 44 45"
mkdir -p "$HERE/data"

mkdir -p "$HERE/data"
for NP in 1 2 4 8; do
  for S in $SEEDS; do
    mpirun -np $NP python -u "$HERE/../3.2_sink_strength/run_dislocation_sink_strength.py" \
        --rho-m2 1e15 --rd-nm 6.25 --target-absorbed 60 --seed $S \
        --max-steps 15000000 --report-interval 500000 --progress-interval 0 --dump-dir "$HERE/dump/e2_np${NP}_s$S" --dump-interval 4000 \
        > "$HERE/data/e2_np${NP}_s$S.log" 2>&1
  done
done
