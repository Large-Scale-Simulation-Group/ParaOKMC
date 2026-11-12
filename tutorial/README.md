# tutorial/

Eight short, self-contained scripts that introduce the library one idea at a time. They are
the companion of `docs/TUTORIAL.md` (Chinese: `docs_zh_cn/TUTORIAL.md`), which explains each of them; read the chapter, run the
script, change something.

```bash
export PYTHONPATH=/path/to/paraokmc_release NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1
python 01_first_model.py
mpirun -np 2 python 01_first_model.py          # every script runs on any rank count
```

| Script | Topic | Runs in |
|---|---|---|
| `01_first_model.py` | Frenkel pairs: two event classes and the seven configuration blocks | seconds |
| `02_two_rate_styles.py` | the same model with Python rates and with compiled kernels (`--style`) | seconds |
| `03_objects_and_execute.py` | properties, `change_obj` / `delete_obj` / `new_obj`, a post-processor | seconds |
| `04_initial_condition_and_random_numbers.py` | an initial condition from a function; reproducibility (`--seed`) | seconds |
| `05_zero_object_events.py` | a source of Frenkel pairs as a `ZeroObjectEvent` | seconds |
| `06_physics_parameters_and_fields.py` | a temperature ramp and a stress field read by the rate (`--style`) | seconds |
| `07_outputs_and_restart.py` | OVITO dumps, event hooks, restart from a frame (`--restart`) | seconds |
| `08_parallel_runs_and_load_balancing.py` | a hotspot on several ranks; `--levels` selects the balancing levels, `--rate` a compiled or a Python hop rate | a minute |

Scripts write into `out/` (ignored by git).
