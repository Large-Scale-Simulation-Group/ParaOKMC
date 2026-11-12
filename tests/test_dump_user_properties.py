#!/usr/bin/env python3
"""ovito_dump must write user-declared object properties (objects[type].properties)
as real values, not zeros, when they are listed under outputs.ovito_dump.properties.
Run:  python tests/test_dump_user_properties.py   (also under mpirun -np 2)"""
import glob, shutil, tempfile
import numpy as np
from mpi4py import MPI
from paraokmc import OKMCengine, SingleObjectEvent, change_obj


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


def main():
    comm = MPI.COMM_WORLD
    out = comm.bcast(tempfile.mkdtemp(prefix='dumptest_') if comm.rank == 0 else None, root=0)
    config = {
        "domain": {"lattice_constant": 1.0, "lattice_grid": [20, 20, 20],
                   "periodic": [True, True, True], "max_interaction_distance": 3.0},
        "simulation": {"max_steps": 40, "random_seed": 1},
        "objects": {"V": {"size": 1.0, "properties": {"E_m": 0.5, "jump_count": 0.0}}},
        "ic": {"mode": "builtin", "initial_conditions": {"V": {"count": 20, "distribution": "random"}}},
        "events": {"mig": Mig},
        "physics": {"params": {"temperature": 500.0}},
        "outputs": {"interval": 20, "ovito_dump": {
            "enabled": True, "output_dir": out, "prefix": "v",
            "properties": ["id", "type", "x", "y", "z", "jump_count", "E_m"]}},
    }
    OKMCengine(config).run()
    comm.Barrier()
    ok = True
    if comm.rank == 0:
        # default writer is distributed: v.<step>.rank_<r>.dump (rank 0's file carries all objects on np=1;
        # on np>1 the frame is split, so the step is taken from the name and all rank files are merged)
        files = glob.glob(f"{out}/v.*.rank_*.dump")
        last = max(int(p.split('.')[-3]) for p in files)
        f = sorted(p for p in files if int(p.split('.')[-3]) == last)
        rows, hdr, step = [], None, None
        for fp in f:
            lines = open(fp).read().splitlines()
            step = int(lines[1]); hdr = lines[8].split()[2:]
            rows += [l.split() for l in lines[9:]]
        em = {float(r[6]) for r in rows}
        jumps = sum(float(r[5]) for r in rows)
        # One migration per KMC step, so the jump_count column sums to the frame's step
        # (the frame is written after that step's event has been applied: step + 1).
        ok = (hdr == ["id", "type", "x", "y", "z", "jump_count", "E_m"]
              and em == {0.5} and jumps in (float(step), float(step + 1)) and jumps > 0)
        print(("OK" if ok else "FAIL") +
              f": np={comm.size} files={len(f)} frame step={step}, header={hdr}, E_m values={sorted(em)}, sum jump_count={jumps:.0f}")
        shutil.rmtree(out)
    # Every rank takes rank 0's verdict, so no rank is left behind in a collective.
    ok = comm.bcast(ok, root=0)
    comm.Barrier()
    if not ok:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
