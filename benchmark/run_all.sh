#!/bin/bash
# ParaOKMC benchmark batch runner
#
# Usage:
#   bash run_all.sh          # everything
#   bash run_all.sh strong   # strong scaling only
#   bash run_all.sh weak     # weak scaling only
#   bash run_all.sh compare  # load-balancing comparison only
#
# Set NP_LIST to the rank counts your machine supports

set -e
cd "$(dirname "$0")"

NP_LIST="1 2 4 8"          # <- rank counts for this machine
# NP_LIST="1 2 4 8 16 32"  # on a larger machine

MODE=${1:-all}

echo "============================================"
echo " ParaOKMC Benchmark Suite"
echo " Processes: $NP_LIST"
echo " Mode: $MODE"
echo "============================================"

run_test() {
    local script=$1
    for np in $NP_LIST; do
        echo ""
        echo ">>> mpirun -np $np python $script"
        mpirun -np $np python "$script"
    done
}

# ---------- strong scaling ----------
if [ "$MODE" = "all" ] || [ "$MODE" = "strong" ]; then
    echo ""
    echo "========== Strong Scaling =========="
    run_test strong_50k.py
    run_test strong_100k.py
    run_test strong_200k.py
    run_test strong_400k.py
fi

# ---------- weak scaling ----------
if [ "$MODE" = "all" ] || [ "$MODE" = "weak" ]; then
    echo ""
    echo "========== Weak Scaling =========="
    run_test weak_5k.py
    run_test weak_10k.py
    run_test weak_25k.py
fi

# ---------- load-balancing comparison ----------
if [ "$MODE" = "all" ] || [ "$MODE" = "compare" ]; then
    echo ""
    echo "========== LB Comparison =========="
    run_test strong_100k_nolb.py
    run_test weak_10k_nolb.py
fi

echo ""
echo "============================================"
echo " All done! Results in:"
echo "   benchmark_results.json"
echo "   benchmark_nolb.json"
echo "============================================"
