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

"""
负载均衡 × 事件构建路径 回归测试（2026-07 审计条 1/4 + 复审条 2）

[契约更新] 全局 incremental_update.enabled 已废除：结构增量恒启用，
原地路径即主路径。两个通道共用同一断言（每个活跃对象必须有
single 事件，全局不丢事件）：

  A. ObjectLB（原审计冻结场景）
     RCB 迁移后的对象不带变化标志，且原地路径的不变式
     "池 == 本 rank 对象的事件"被迁移打破。修复 = ObjectLB 实际迁移
     时 mark_global_state_changed() → allow_single_inplace=False
     → 下一步全量重建。

  B. predictive 重平衡（复审条 2 丢失场景）
     predictive 把事件搬到别的 rank（owner 不留副本）。原地路径若仍
     改写本地池，会重建被搬走事件的副本或丢失事件。修复 =
     事件搬运型 LB 开启时原地路径整体禁用（single 全量重建）。
     事件合法地寄居外 rank，冻结检查须用全局并集。
     （差分验证：去掉 allow_single_inplace 修复后本通道 59/60 冻结）

  另有 gate 接线白盒检查：记录引擎每步传给 build_all_events 的
  allow_single_inplace——常态步必须 True（原地路径可达），
  global_state_changed / force_rebuild 在场的构建必须 False。

运行：
    mpirun -np 2 python test_object_lb_incremental.py
    mpirun -np 4 python test_object_lb_incremental.py
"""

import sys
import numpy as np
from mpi4py import MPI

sys.path.insert(0, '..')

from paraokmc.v16_engine import create_okmc_engine_v16
from paraokmc.event_class import SingleObjectEvent
from paraokmc.event_compiler import merge_event_config

LC = 3.17
GRID = (40, 40, 40)
BOX = [GRID[i] * LC for i in range(3)]

N_OBJECTS = 60
BALANCE_INTERVAL = 4
# 至少覆盖两次重平衡窗口，且末步之后仍有事件构建
MAX_STEPS = BALANCE_INTERVAL * 3 + 2


class TracerMigration(SingleObjectEvent):
    """常速率随机迁移（保证每步都有对象变化 → 走增量路径）"""
    object_types = ['Tracer']
    params = []

    @classmethod
    def rate(cls, objs, params, config):
        return 1.0e6

    @classmethod
    def execute(cls, event, objs, config):
        obj = objs[0]
        step = 1.5
        rng = config['rng']
        new_pos = {}
        for axis, box_len in zip(('x', 'y', 'z'), BOX):
            val = float(obj[axis]) + rng.uniform(-step, step)
            new_pos[axis] = val % box_len
        return [{'id': int(obj['id']), **new_pos}]


def create_hotspot_ic(n_objects):
    """全部对象集中在盒子一角（仅 rank 0 生成）→ 触发 RCB/predictive 迁移"""
    def initial_objects_func(rank, size, config):
        if rank != 0:
            return []
        rng = np.random.default_rng(42)
        hot = BOX[0] * 0.12
        return [{
            'id': i + 1,
            'type': 'Tracer',
            'x': float(rng.uniform(0, hot)),
            'y': float(rng.uniform(0, hot)),
            'z': float(rng.uniform(0, hot)),
        } for i in range(n_objects)]
    return initial_objects_func


def create_config(*, object_lb, predictive):
    cap = N_OBJECTS + 64
    base_config = {
        'physics': {'temperature': 300.0, 'lattice_constant': LC},
        'objects': {'Tracer': {'properties': {}}},
        'domain_decomposition': {
            'lattice_constant': LC,
            'lattice_grid': list(GRID),
            'h_factor': 1,
            'periodic': [True, True, True],
            'max_interaction_distance': LC * 2,
        },
        'initial_objects_func': create_hotspot_ic(N_OBJECTS),
        'lattice_constant': LC,
        'grid_size': GRID,
        'temperature': 300.0,
        'max_steps': MAX_STEPS,
        'max_time': 1.0,
        'verbose': False,
        'random_seed': 7,
        'progress_interval': 0,
        'rma_object_pool': {'initial_capacity': cap, 'auto_expand': True},
        'rma_event_pool': {
            'capacity_zero': 4,
            'capacity_single': cap,
            'capacity_pair': 4,
            'auto_expand': True,
        },
        'predictive_rebalance': {'enabled': predictive, 'threshold': 0.05},
        'adaptive_weight_tracker': {'enabled': False},
        'runtime_load_balancer': {'enabled': False},
        'object_load_balancer': {
            'enabled': object_lb,
            'balance_interval': BALANCE_INTERVAL,
            'imbalance_threshold': 0.05,
        },
    }
    return merge_event_config(base_config, {'tracer_migration': TracerMigration})


def run_pass(label, comm, *, object_lb, predictive):
    rank = comm.rank
    size = comm.size

    if rank == 0:
        print("-" * 60)
        print(f"[{label}] object_lb={object_lb} predictive={predictive}")

    config = create_config(object_lb=object_lb, predictive=predictive)
    engine = create_okmc_engine_v16(config)
    engine.run()

    pool = engine.rma_object_pool
    epool = engine.rma_event_pool

    owned = pool.get_local_objects()
    owned = owned[(owned['owner_rank'] == rank) & owned['active']]
    owned_ids = set(int(i) for i in owned['id'])

    events = epool.single_obj_array[:epool.count_single]
    local_event_ids = [int(i) for i in events['obj_ids'][:, 0]]
    if predictive:
        # 事件被 predictive 合法搬运寄居外 rank：冻结检查用全局并集
        event_obj_ids = set()
        for chunk in comm.allgather(local_event_ids):
            event_obj_ids.update(chunk)
    else:
        event_obj_ids = set(local_event_ids)

    # 末步执行的对象：其旧事件在执行时被清理，重建发生在"下一步"的
    # build——run 已结束不会再来。这是正常状态，不算冻结。
    last_step_changed = set(
        int(i) for i in engine.incremental_manager.get_changed_objects())

    frozen = sorted(owned_ids - event_obj_ids - last_step_changed)
    n_owned_all = comm.allreduce(len(owned_ids), op=MPI.SUM)
    n_frozen_all = comm.allreduce(len(frozen), op=MPI.SUM)

    if frozen:
        print(f"[{label}][Rank {rank}] FROZEN objects (owned, no single event): "
              f"{frozen[:10]}{'...' if len(frozen) > 10 else ''} "
              f"({len(frozen)}/{len(owned_ids)})", flush=True)

    counts = comm.allgather(len(owned_ids))
    moved_events = comm.allreduce(
        int(engine.predictive_rebalancer.stats['total_events_moved']),
        op=MPI.MAX) if predictive else 0

    if rank == 0:
        print(f"[{label}] 对象分布: {counts} (总计 {n_owned_all}/{N_OBJECTS}) "
              f"冻结: {n_frozen_all}"
              + (f" predictive 搬运事件: {moved_events}" if predictive else ""))

    failed = False
    if n_frozen_all > 0:
        failed = True
        if rank == 0:
            print(f"[{label}] FAIL: objects lost their single events (frozen)")
    if n_owned_all != N_OBJECTS:
        failed = True
        if rank == 0:
            print(f"[{label}] FAIL: object count not conserved: "
                  f"{n_owned_all} != {N_OBJECTS}")
    if object_lb and size > 1 and max(counts) == N_OBJECTS:
        # 所有对象仍挤在一个 rank → ObjectLB 从未迁移，没覆盖到目标路径
        failed = True
        if rank == 0:
            print(f"[{label}] FAIL: ObjectLB never migrated any object; "
                  "test did not exercise the target path")
    if predictive and size > 1 and moved_events == 0:
        failed = True
        if rank == 0:
            print(f"[{label}] FAIL: predictive rebalancer never moved events; "
                  "test did not exercise the target path")

    # 显式集体释放 RMA 窗口：留给 GC 会在各 rank 不同步的时刻触发
    # __del__ → Win.Free（集体调用）→ 与下一通道的集体调用互锁
    engine.destroy()
    return bool(comm.allreduce(1 if failed else 0, op=MPI.MAX))


def check_gate_wiring(comm):
    """白盒：allow_single_inplace 的引擎接线（复审条 2 析取支 a）

    常态非增量步 gate 必须为 True（快路径真实可达，检出不空洞）；
    global_state_changed / force_rebuild_single 在场的那次构建必须为 False。
    """
    rank = comm.rank
    if rank == 0:
        print("-" * 60)
        print("[gate] allow_single_inplace wiring check")

    config = create_config(object_lb=False, predictive=False)
    engine = create_okmc_engine_v16(config)

    records = []
    orig = engine.event_builder.build_all_events

    def recorder(*args, **kwargs):
        records.append((engine.current_step,
                        kwargs.get('allow_single_inplace')))
        return orig(*args, **kwargs)

    engine.event_builder.build_all_events = recorder
    engine.run()

    steady = [flag for step, flag in records if step > 0]
    ok_steady = len(steady) > 0 and all(f is True for f in steady)

    engine.incremental_manager.mark_global_state_changed()
    engine._build_events()
    ok_global = records[-1][1] is False
    engine.incremental_manager.step_completed()  # 清标志（run 循环同款）

    engine.incremental_manager.set_force_rebuild(single=True)
    engine._build_events()
    ok_force = records[-1][1] is False
    engine.incremental_manager.step_completed()
    engine.destroy()

    failed = not (ok_steady and ok_global and ok_force)
    if rank == 0:
        print(f"[gate] steady-state True: {ok_steady} "
              f"(steps recorded: {len(steady)}), "
              f"global_state_changed→False: {ok_global}, "
              f"force_rebuild→False: {ok_force}")
        if failed:
            print("[gate] FAIL: engine does not wire allow_single_inplace "
                  "as the fix requires")
    return bool(comm.allreduce(1 if failed else 0, op=MPI.MAX))


if __name__ == '__main__':
    comm = MPI.COMM_WORLD
    if comm.rank == 0:
        print("=" * 60)
        print("Load balancer × event build path regression test")
        print(f"  ranks={comm.size} objects={N_OBJECTS} "
              f"balance_interval={BALANCE_INTERVAL} max_steps={MAX_STEPS}")
        print("=" * 60)

    any_failed = False
    any_failed |= run_pass("A objlb", comm,
                           object_lb=True, predictive=False)
    any_failed |= run_pass("B predictive", comm,
                           object_lb=False, predictive=True)
    any_failed |= check_gate_wiring(comm)

    if comm.rank == 0:
        print("=" * 60)
        print("FAILED" if any_failed else "PASSED")
        print("=" * 60)
    sys.exit(1 if any_failed else 0)
