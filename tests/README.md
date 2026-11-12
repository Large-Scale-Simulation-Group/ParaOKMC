# tests/

```bash
python -m pytest -q tests                          # single-rank tests
mpirun -np 2 python tests/test_dump_restart.py     # multi-rank tests are plain scripts; run them under mpirun too
mpirun -np 4 python tests/test_dump_restart.py
```

Multi-rank scripts: `test_dump_restart.py`, `test_dump_user_properties.py`, `test_event_hook.py`,
`test_ghost_event_after_delete.py`, `test_physics_in_event.py`, `test_physics_sync.py`,
`test_distributed_rcb.py`, `test_object_lb_incremental.py`, `test_random_generator.py`. Each prints `PASSED` lines and exits non-zero on failure.

What the main groups cover:

| Files | Covers |
|---|---|
| `test_new_config.py`, `test_new_api.py` | validation and compilation of the seven-block configuration |
| `test_jit_rate.py`, `test_jit_event_builder.py`, `test_vectorized_builder.py`, `test_cell_list.py` | compiled rate path, event construction |
| `test_pair_incremental.py`, `test_rate_invalidation.py`, `test_ghost_event_after_delete.py` | incremental structure update, rate reuse and invalidation, ghost consistency |
| `test_distributed_rcb.py`, `test_object_lb_incremental.py`, `object_lb_test/`, `rebalance_test/`, `runtime_lb_test/` | the three load-balancing levels |
| `test_unified_physics.py`, `test_adaptive_field.py`, `test_param_coupling.py`, `test_physics_*.py`, `test_event_trigger.py`, `test_dedalus_coupler.py` | parameters, fields, solvers, triggers |
| `test_random_generator.py` | reproducibility of the counter-keyed generator |
| `test_dump_restart.py`, `test_dump_user_properties.py`, `test_event_hook.py` | OVITO output, restart from a frame, event hooks |

Known: `test_pair_incremental.py::test_direction_entry_points` imports a helper that no
longer exists and fails; the other tests in that file pass.
