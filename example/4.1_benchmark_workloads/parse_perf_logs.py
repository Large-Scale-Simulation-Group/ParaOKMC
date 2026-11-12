#!/usr/bin/env python3
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

"""Rebuild bscc_a_measured_*.csv from the Slurm .out logs of campaigns v4/v5/v7.

Each log ends with the engine's own summary line
    [ParaOKMC] Perf: build=..ms rate=..ms select=..ms exec=..ms per_step=..ms
and, for the pair workload, a line '[pair_scaling] ... pairs=NNNN ...'.
A log without a Perf line is a job that died (all n_p >= 128 runs: one-sided RDMA
across nodes), and is written with status=failed.

usage: python parse_perf_logs.py logs/*.out > rebuilt.csv
"""
import re, sys, csv
from pathlib import Path

SIZE = {'1k': 1000, '10k': 10000, '100k': 100000}
NAME = re.compile(
    r'^(?P<camp>v[457])_'
    r'(?:(?P<e0>e0(?:old|new)?)_(?P<size>1k|10k|100k)'
    r'|(?P<e0w>e0weak)_(?P<e0warm>new|old)'
    r'|pair_(?P<pn>\d+)_(?P<parm>new_on|new_off|old)'
    r'|(?P<pw>pairweak)_(?P<pwarm>new_on|old))'
    r'_np(?P<np>\d+)$')
PERF = re.compile(r'Perf: build=([\d.]+)ms rate=([\d.]+)ms select=([\d.]+)ms exec=([\d.]+)ms per_step=([\d.]+)ms')
PAIRS = re.compile(r'\[pair_scaling\].*?pairs=(\d+)')

def parse(path):
    m = NAME.match(Path(path).stem)
    if not m:
        return None
    g = m.groupdict(); np_ = int(g['np'])
    if g['e0']:
        wl, st, arm, n = 'e0_diffusion', 'strong', ('old' if g['e0'].endswith('old') else 'new'), SIZE[g['size']]
    elif g['e0w']:
        wl, st, arm, n = 'e0_diffusion', 'weak_1k_per_rank', g['e0warm'], 1000 * np_
    elif g['pn']:
        wl, st, arm, n = 'pair_HV', 'strong', g['parm'], int(g['pn'])
    else:
        wl, st, arm, n = 'pair_HV', 'weak_20k_per_rank', ('new' if g['pwarm'] == 'new_on' else 'old'), 20000 * np_
    txt = open(path, errors='replace').read()
    perf = PERF.findall(txt); pairs = PAIRS.findall(txt)
    row = dict(campaign=g['camp'], workload=wl, scaling_type=st, lib_arm=arm, n_objects=n, mpi_ranks=np_)
    if perf:
        b, r, s, e, p = perf[-1]
        row.update(status='ok', build_ms=b, rate_ms=r, select_ms=s, exec_ms=e, per_step_ms=p,
                   pairs=(pairs[-1] if pairs else ''), note='')
    else:
        row.update(status='failed', build_ms='', rate_ms='', select_ms='', exec_ms='', per_step_ms='',
                   pairs='', note='cross-node(osc/rdma)' if np_ >= 128 else 'no Perf line')
    return row

if __name__ == '__main__':
    cols = ['campaign','workload','scaling_type','lib_arm','n_objects','mpi_ranks','status',
            'build_ms','rate_ms','select_ms','exec_ms','per_step_ms','pairs','note']
    out = csv.DictWriter(sys.stdout, fieldnames=cols); out.writeheader()
    rows = [r for r in (parse(p) for p in sys.argv[1:]) if r]
    rows.sort(key=lambda r: (r['campaign'], r['workload'], r['scaling_type'], r['lib_arm'], r['n_objects'], r['mpi_ranks']))
    for r in rows: out.writerow(r)
