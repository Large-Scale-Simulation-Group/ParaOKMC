# Installing ParaOKMC

## Requirements

* Python ≥ 3.9
* NumPy ≥ 1.19, SciPy ≥ 1.5, Numba ≥ 0.55
* mpi4py ≥ 3.0 built against an **MPI-3** library (Open MPI, MPICH, Intel MPI). The object
  and event pools use one-sided RMA windows, so MPI-3 support is required.
* Optional: Dedalus ≥ 3.0 for the spectral field-solver coupling (GPL-3.0; see Licensing in
  `README.md`).

## One step, with conda (recommended)

From the release root (the directory that holds `environment.yml`):

```bash
conda env create -f environment.yml
conda activate paraokmc
```

This creates the environment `paraokmc` with Python, NumPy, SciPy, Numba, an MPI library
(MPICH) and mpi4py linked against it, all from conda-forge, and installs ParaOKMC from the
release directory. Nothing else is needed: `mpirun` comes with the environment. If conda
needs a proxy, set `http_proxy` / `https_proxy` before running the command.

On a cluster whose MPI must be used instead of MPICH from conda (for the interconnect),
use pip as below.

## With pip

From the release root:

```bash
pip install .                 # or: pip install -e .   (editable, for development)
pip install -e ".[dedalus]"   # with the optional solver coupling
```

`pip` installs a binary mpi4py wheel that does **not** contain an MPI library: it looks for
`libmpi.so` of an MPI already on the machine when it is first imported. If
`import paraokmc` fails with `RuntimeError: cannot load MPI library`, make that library
visible, for example

```bash
module load mpich                          # where MPI is provided as a module
export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH   # MPI installed under /usr/local
```

or rebuild mpi4py against the MPI compiler wrapper, which records the library path:

```bash
env MPICC=$(which mpicc) pip install --no-binary=mpi4py --force-reinstall mpi4py
```

## Building a conda package

`conda build conda-recipe && conda install --use-local paraokmc` builds and installs a
package from this tree; `conda-recipe/README.md` has the details. For using the library,
`environment.yml` above is simpler.

## Check the installation

```bash
python -c "import paraokmc; print(paraokmc.__version__)"
export NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1
cd tutorial
python 01_first_model.py
mpirun -np 4 python 01_first_model.py
```

Then read `docs/TUTORIAL.md`; `docs/USER_GUIDE.md` is the reference.

## On a cluster

* Load the same MPI module for building `mpi4py` and for running, for example
  `module load python; module load intel/mpi/2021.17`.
* Run one MPI rank per core and keep the compiled kernels single-threaded:
  `export NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1`.
* If the package is not installed on the compute nodes, point `PYTHONPATH` at the release
  root, and give scheduler scripts absolute paths.
* Numba compiles the kernels on the first step of every run (a few seconds). Set
  `NUMBA_CACHE_DIR` to a directory the job can write to if the default cache location is
  read-only.
* A Slurm job looks like

  ```bash
  #!/bin/bash
  #SBATCH -N 1
  #SBATCH --ntasks-per-node=32
  module load python intel/mpi/2021.17
  export NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1
  export PYTHONPATH=$HOME/paraokmc_release
  srun python $HOME/runs/my_model.py
  ```

