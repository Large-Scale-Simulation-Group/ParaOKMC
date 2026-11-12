# Building the conda package locally

```bash
conda install conda-build
cd paraokmc_release              # the directory that holds pyproject.toml
conda build conda-recipe
```

`conda build conda-recipe --output` prints the path of the built package. `test_build.sh`
runs the build, installs the package into a fresh environment and checks the imports:

```bash
conda create -n test-paraokmc python=3.11
conda activate test-paraokmc
conda install --use-local paraokmc
python -c "import paraokmc; print(paraokmc.__version__)"
```

To build a new version, bump it in `paraokmc/_version.py` and in `meta.yaml`.

Notes:

* **MPI.** `mpi4py` must match the MPI library used at run time; install it from the same
  channel as that MPI, or build it against the system MPI (see `INSTALL.md`).
* **Python versions.** `build: skip: true  # [py<39]` keeps the minimum at 3.9.
* **Import test failures** usually mean a missing entry under `requirements: run`.
