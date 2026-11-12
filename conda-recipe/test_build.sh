#!/bin/bash
# Build the conda package from this tree and check that it installs and imports.
set -e

echo "=========================================="
echo "ParaOKMC conda package build test"
echo "=========================================="

if ! command -v conda-build &> /dev/null; then
    echo "conda-build not found; installing it"
    conda install -y conda-build
fi

echo ""
echo "Step 1: building the package"
conda build conda-recipe
PACKAGE_PATH=$(conda build conda-recipe --output)
echo "  $PACKAGE_PATH"

echo ""
echo "Step 2: creating a test environment"
eval "$(conda shell.bash hook)"          # conda activate needs this in a non-interactive shell
conda create -n test-paraokmc-conda python=3.11 -y || true
conda activate test-paraokmc-conda

echo ""
echo "Step 3: installing the package"
conda install --use-local paraokmc -y

echo ""
echo "Step 4: import test"
python -c "import paraokmc; print('version', paraokmc.__version__)"
python -c "from paraokmc import OKMCengine, SingleObjectEvent, PairObjectEvent, ZeroObjectEvent, RateDeps; print('public API ok')"

echo ""
echo "All checks passed."
echo "Clean up: conda deactivate && conda env remove -n test-paraokmc-conda"
