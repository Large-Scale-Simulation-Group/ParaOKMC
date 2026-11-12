#!/usr/bin/env python3
"""Round trip through the dump: write (distributed, the default) -> restart from it.

A frame is written inside the step loop every `interval` steps, after the step's event
has been applied, and once more after the loop for the final state (whatever stopped the
run), so the last frame of a run is always its final state. The step counter advances
after the in-loop write, so an in-loop frame k holds the state after event k+1 while the
final frame is labelled with the number of steps completed (30 here). With interval=7 the
in-loop frames are 0, 7, 14, 21, 28 and the final frame 30 exists only thanks to that
extra write; it is never a duplicate (frame 28 is the state after 29 events).
The restart is checked by continuity: run2 starts from the last frame of run1, and its
own frame 0 must equal that frame with exactly one object advanced by one migration
(one more jump_count, a displaced position) and everything else bit-for-bit the same:
  * the set of object ids is identical (true restart, id_mode='keep'),
  * sizes and every user-declared property (jump_count, E_m) are restored,
  * the default writer is the distributed one (one file per rank, type column = name).
Run:  python tests/test_dump_restart.py ; mpirun -np 2 python tests/test_dump_restart.py
"""
import glob, os, shutil, tempfile
import numpy as np
from mpi4py import MPI
from paraokmc import OKMCengine, SingleObjectEvent, change_obj
from paraokmc.post_processor import read_dump_frame


class Mig(SingleObjectEvent):
    object_types = ['V']

    @classmethod
    def rate(cls, objs, params, config):
        return 1.0e3

    @classmethod
    def execute(cls, event, objs, config):
        v = objs[0]; u = config['rng']; lc = config['lattice_constant']
        th, ph = 2 * np.pi * u.random(), np.arccos(2 * u.random() - 1)
        return [change_obj(v, x=v['x'] + lc * np.sin(ph) * np.cos(th),
                           y=v['y'] + lc * np.sin(ph) * np.sin(th),
                           z=v['z'] + lc * np.cos(ph),
                           jump_count=v['jump_count'] + 1.0)]


PROPS = ["id", "type", "x", "y", "z", "size", "jump_count", "E_m"]


def base_config(ic, dump_dir, steps, interval=7):
    return {
        "domain": {"lattice_constant": 1.0, "lattice_grid": [20, 20, 20],
                   "periodic": [True, True, True], "max_interaction_distance": 3.0},
        "simulation": {"max_steps": steps, "random_seed": 7},
        "objects": {"V": {"size": 1.0, "properties": {"E_m": 0.5, "jump_count": 0.0}},
                    "S": {"size": 2.0, "properties": {"radius": 2.0}}},
        "ic": ic,
        "events": {"mig": Mig},
        "physics": {"params": {"temperature": 500.0}},
        "outputs": {"interval": interval, "ovito_dump": {
            "enabled": True, "output_dir": dump_dir, "prefix": "s", "properties": PROPS}},
    }


def load_frame(dump_dir, step):
    """All objects of one frame, merged over the per-rank files: {id: (type, x, y, z, size, jump, E_m)}."""
    out = {}
    for f in sorted(glob.glob(f"{dump_dir}/s.{step}.rank_*.dump")):
        _, cols, rows = read_dump_frame(f)
        c = {k: i for i, k in enumerate(cols)}
        for r in rows:
            out[int(r[c['id']])] = (r[c['type']], float(r[c['x']]), float(r[c['y']]), float(r[c['z']]),
                                    float(r[c['size']]), float(r[c['jump_count']]), float(r[c['E_m']]))
    return out


def main():
    comm = MPI.COMM_WORLD
    root = comm.bcast(tempfile.mkdtemp(prefix='restart_') if comm.rank == 0 else None, root=0)
    d1, d2 = f"{root}/run1", f"{root}/run2"
    STEPS = 30

    # phase 1: builtin IC, run, dump at step 0 and at step STEPS
    OKMCengine(base_config({"mode": "builtin", "initial_conditions": {
        "V": {"count": 24, "distribution": "random"}, "S": {"count": 3, "distribution": "random"}}},
        d1, STEPS)).run()
    comm.Barrier()

    # phase 2: restart from the LAST frame of run1, dump again at step 0 (before any event)
    OKMCengine(base_config({"mode": "dump", "file": d1, "frame": -1}, d2, STEPS)).run()
    comm.Barrier()

    ok = True
    if comm.rank == 0:
        files1 = sorted(os.path.basename(f) for f in glob.glob(f"{d1}/*.dump"))
        frames = sorted({int(f.split('.')[1]) for f in files1})
        last = frames[-1]
        # interval=7, 30 steps: in-loop frames at 0,7,14,21,28 plus the forced final frame 30
        expect_frames = [0, 7, 14, 21, 28, 30]
        a = load_frame(d1, last)      # state that was dumped at the end of run1
        b = load_frame(d2, 0)         # state the restart started from
        per_rank = all('.rank_' in f for f in files1)
        same_ids = set(a) == set(b)
        same = [i for i in a if same_ids and a[i][0] == b[i][0]
                and np.allclose(a[i][1:], b[i][1:], rtol=0, atol=1e-5)]
        moved = [i for i in a if same_ids and i not in same]
        # exactly one V moved by one event: jump_count +1, size and E_m untouched, S never moves
        one_event = (len(moved) == 1 and a[moved[0]][0] == 'V'
                     and b[moved[0]][5] == a[moved[0]][5] + 1.0
                     and b[moved[0]][4] == a[moved[0]][4] and b[moved[0]][6] == a[moved[0]][6])
        types = {v[0] for v in a.values()}
        jumps = sum(v[5] for v in a.values() if v[0] == 'V')
        ok = (per_rank and same_ids and one_event and types == {'V', 'S'}
              and jumps == float(last) and len(a) == 27 and frames == expect_frames)
        print(("OK" if ok else "FAIL") +
              f": np={comm.size} frames={frames} per-rank files={per_rank} objects={len(a)} "
              f"ids equal={same_ids} unchanged={len(same)} moved by one event={moved} "
              f"types={sorted(types)} sum jump_count={jumps:.0f}")
        if not ok:
            for i in sorted(set(a) ^ set(b))[:5]:
                print("   id only on one side:", i, a.get(i), b.get(i))
        shutil.rmtree(root)
    ok = comm.bcast(ok, root=0)
    comm.Barrier()
    if not ok:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
