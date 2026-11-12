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

# Section 3.1 / Fig. 4 -- free diffusion of 1024 non-interacting objects.
# Box 100 x 300 x 300 nm^3 (periodic), D = 100 nm^2/s, 10-nm slab at the box centre,
# 32 slices of 3.125 nm along x, advanced to t = 10 s. Five seeds, run serially.
# --- environment: the release library + single-threaded kernels per MPI rank ---
HERE="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="$(cd "$HERE/../.." && pwd)"
export NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
SEEDS="41 42 43 44 45"
mkdir -p "$HERE/data"

for S in $SEEDS; do
  python -u "$HERE/run_diffusion_balance.py" --seed $S --num-particles 1024 \
      --seed-width 10 --x-bins 32 --max-time 10 \
      --report-interval 1000 --progress-interval 50000 --dump-dir "$HERE/dump/e0_s$S" \
      --output-dir "$HERE/data/e0_s$S" > "$HERE/data/e0_s$S.log" 2>&1
done
