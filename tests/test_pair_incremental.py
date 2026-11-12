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
pair 结构增量更新（cell 级毒区）+ 类级速率增量开关 回归测试

设计要点（与实现一一对应）：
- 毒区 = 状态包(旧坐标 ∪ 新坐标)所在 cell ± 邻居环，全 rank 一致
- 失效判据（JIT）：任一嵌入 ID ∈ 状态包 ∨ 任一嵌入位置 ∈ 毒区
- 重建范围 = 至少一个端点在毒区、且本 rank 拥有的 pair（杀 ⟺ 重建恒等）
- 保留事件速率默认重算；incremental_rate=True 的类可保留 DONE 速率

运行：
    python test_pair_incremental.py
    mpirun -np 2 python test_pair_incremental.py
"""

import sys
import types
import numpy as np
from mpi4py import MPI

sys.path.insert(0, '..')

from paraokmc.jit_event_builder import jit_mark_invalid_pair_events
from paraokmc.rma_event_pool import (
    RMAEventPool, create_default_event_dtype, STATUS_TODO, STATUS_DONE,
)
from paraokmc.event_class import PairObjectEvent, SingleObjectEvent, EventRegistry
from paraokmc.vectorized_event_builder import VectorizedPairEventBuilder

COMM = MPI.COMM_WORLD
RANK = COMM.rank

BOX = [100.0, 100.0, 100.0]
MAX_DIST = 5.0

OBJ_DTYPE = np.dtype([
    ('id', np.int64), ('type_id', np.int32),
    ('x', np.float64), ('y', np.float64), ('z', np.float64),
    ('owner_rank', np.int32), ('active', np.bool_),
    ('size', np.float64), ('float_props', np.float64, (4,)),
], align=True)

EVENT_DTYPE = np.dtype(
    create_default_event_dtype().descr + [('objects', OBJ_DTYPE, (2,))],
    align=True)

CHANGED_DTYPE = np.dtype({
    'names': ['id', 'x', 'y', 'z', 'active',
              'old_x', 'old_y', 'old_z', 'old_valid'],
    'formats': [np.int64, np.float64, np.float64, np.float64, np.bool_,
                np.float64, np.float64, np.float64, np.bool_],
}, align=True)


class Capture(PairObjectEvent):
    type_pairs = [('H', 'V')]
    max_distance = MAX_DIST
    params = ['distance']

    @classmethod
    def rate(cls, objs, params, config):
        return 1.0

    @classmethod
    def execute(cls, event, objs, config):
        return []


class CaptureKeepRate(Capture):
    incremental_rate = True


def _make_builder(event_class=Capture, verify=False, rank=None):
    registry = EventRegistry()
    registry.register('capture', event_class)

    grid_system = types.SimpleNamespace(periodic=[True, True, True])
    domain_decomp = types.SimpleNamespace(
        comm=None, rank=RANK if rank is None else rank, size=COMM.size,
        max_interaction_distance=MAX_DIST,
        box_size=np.array(BOX),
        grid_system=grid_system,
    )
    config = {
        '_type_to_int': {'H': 0, 'V': 1},
        '_int_to_type': {0: 'H', 1: 'V'},
        '_event_registry': registry,
        'pair_type_filters': {'capture': lambda *a, **k: True},
        'pair_param_extractors': {},
        'event_weights': {'capture': 1.0},
        '_event_name_to_int': {'capture': 0},
        '_event_int_to_name': {0: 'capture'},
        'vectorized_pair_filters': {
            'capture': {'type_pairs': [('H', 'V')], 'max_distance': MAX_DIST},
        },
        'domain_decomposition': {'box_size': BOX, 'periodic': [True, True, True]},
        '_pair_structure_verify': verify,
    }
    builder = VectorizedPairEventBuilder(
        domain_decomp, config, OBJ_DTYPE, EVENT_DTYPE,
        verbose=False, use_cell_list=True)
    return builder


def _make_objects(n, rng):
    objs = np.zeros(n, dtype=OBJ_DTYPE)
    objs['id'] = np.arange(1, n + 1)
    objs['type_id'] = (np.arange(n) % 2).astype(np.int32)
    # 高密度小区域：保证生成几十个 pair（低密度下等价性测试形同虚设，
    # 且 rank 间 pair 数差异会诱发 rank 局部控制流分歧）
    objs['x'] = rng.uniform(0, 30.0, n)
    objs['y'] = rng.uniform(0, 30.0, n)
    objs['z'] = rng.uniform(0, 30.0, n)
    objs['owner_rank'] = RANK
    objs['active'] = True
    return objs


def _make_pool_ns(objs):
    return types.SimpleNamespace(
        large_count=0,
        get_local_objects=lambda: objs[objs['active']],
        get_ghost_objects=lambda: np.array([], dtype=OBJ_DTYPE),
    )


def _fresh_event_pool():
    return RMAEventPool(COMM, dtype=EVENT_DTYPE,
                        capacity_zero=4, capacity_single=4,
                        capacity_pair=4096, auto_expand=True)


def _pair_keys(events):
    return set(zip(events['obj_ids'][:, 0].tolist(),
                   events['obj_ids'][:, 1].tolist(),
                   events['event_type_id'].tolist()))


# ============================================================
# Test 1: 失效判定内核
# ============================================================

def test_kill_kernel():
    """ID 判据与毒区判据的基本行为"""
    Gx = Gy = Gz = 20  # box/max_dist = 100/5
    cs = 5.0
    poison = np.zeros(Gx * Gy * Gz, dtype=np.bool_)
    # 毒化 cell (2,2,2)
    poison[2 + 2 * Gx + 2 * Gx * Gy] = True

    # 事件0：对象在毒区 cell 内 → 杀
    # 事件1：远离毒区、ID 无关 → 留
    # 事件2：远离毒区，但 ids2 在状态包 → 杀
    x1 = np.array([12.5, 60.0, 60.0]); y1 = np.array([12.5, 60.0, 60.0])
    z1 = np.array([12.5, 60.0, 60.0])
    x2 = np.array([13.0, 61.0, 61.0]); y2 = np.array([13.0, 61.0, 61.0])
    z2 = np.array([13.0, 61.0, 61.0])
    ids1 = np.array([1, 3, 5], dtype=np.int64)
    ids2 = np.array([2, 4, 6], dtype=np.int64)
    changed = np.array([6, 100], dtype=np.int64)

    keep = np.empty(3, dtype=np.int8)
    jit_mark_invalid_pair_events(
        x1, y1, z1, x2, y2, z2, ids1, ids2, changed, poison,
        np.int64(Gx), np.int64(Gy), np.int64(Gz),
        cs, cs, cs, BOX[0], BOX[1], BOX[2],
        True, True, True, keep)

    assert keep.tolist() == [0, 1, 0], f"kill verdicts wrong: {keep.tolist()}"
    print("  PASSED: kill kernel verdicts")


# ============================================================
# Test 2: 增量 == 全量（模糊等价，多轮随机变异）
# ============================================================

def _apply_mutation(objs, rng, next_id):
    """随机变异对象集，返回 (new_objs, changed_records)"""
    changed = []
    active_idx = np.flatnonzero(objs['active'])

    # 移动 1-2 个（其一可能长跳）
    for jump in (1.0, 40.0):
        i = int(rng.choice(active_idx))
        rec = {'id': int(objs['id'][i]),
               'old_x': float(objs['x'][i]), 'old_y': float(objs['y'][i]),
               'old_z': float(objs['z'][i]), 'old_valid': True, 'active': True}
        objs['x'][i] = (objs['x'][i] + rng.uniform(-jump, jump)) % BOX[0]
        objs['y'][i] = (objs['y'][i] + rng.uniform(-jump, jump)) % BOX[1]
        rec.update({'x': float(objs['x'][i]), 'y': float(objs['y'][i]),
                    'z': float(objs['z'][i])})
        changed.append(rec)

    # 删除 1 个
    i = int(rng.choice(active_idx))
    changed.append({'id': int(objs['id'][i]), 'active': False,
                    'x': 0.0, 'y': 0.0, 'z': 0.0,
                    'old_x': float(objs['x'][i]), 'old_y': float(objs['y'][i]),
                    'old_z': float(objs['z'][i]), 'old_valid': True})
    objs['active'][i] = False

    # 新增 1 个
    new = np.zeros(1, dtype=OBJ_DTYPE)[0]
    new['id'] = next_id
    new['type_id'] = int(rng.integers(0, 2))
    new['x'] = rng.uniform(0, 30.0)
    new['y'] = rng.uniform(0, 30.0)
    new['z'] = rng.uniform(0, 30.0)
    new['owner_rank'] = RANK
    new['active'] = True
    objs = np.append(objs, new)
    changed.append({'id': int(next_id), 'active': True, 'old_valid': False,
                    'x': float(new['x']), 'y': float(new['y']),
                    'z': float(new['z']),
                    'old_x': 0.0, 'old_y': 0.0, 'old_z': 0.0})

    arr = np.zeros(len(changed), dtype=CHANGED_DTYPE)
    for k, rec in enumerate(changed):
        for f in CHANGED_DTYPE.names:
            arr[k][f] = rec.get(f, 0)
    return objs, arr


def test_incremental_equals_full():
    """10 轮随机 移动/长跳/删除/新增 后，增量池与全量重建逐集合一致"""
    rng = np.random.default_rng(1234 + RANK)
    builder = _make_builder()
    epool = _fresh_event_pool()

    objs = _make_objects(80, rng)
    pool_ns = _make_pool_ns(objs)

    # 步 0：全量播种
    full0 = builder.build_pair_events(pool_ns, current_step=0)
    epool.set_pair_obj_events(full0, recompute_rate_sum=False)

    next_id = 1000
    for step in range(1, 11):
        objs, changed_arr = _apply_mutation(objs, rng, next_id)
        next_id += 1
        pool_ns = _make_pool_ns(objs)

        applied = builder.build_pair_events_incremental(
            pool_ns, epool, changed_arr, current_step=step)

        # 对照：全量重建
        full = builder.build_pair_events(pool_ns, current_step=step)
        pool_keys = _pair_keys(epool.pair_obj_array[:epool.count_pair])
        full_keys = _pair_keys(full) if len(full) else set()
        ok = (pool_keys == full_keys)
        if not ok:
            print(f"[Rank {RANK}] step {step}: incremental != full "
                  f"(missing={len(full_keys - pool_keys)}, "
                  f"extra={len(pool_keys - full_keys)})", flush=True)
        # [MPI] 集合化判定：失败时所有 rank 一起 raise，避免单边挂死
        assert COMM.allreduce(ok, op=MPI.LAND), f"step {step}: mismatch"
        assert COMM.allreduce(bool(applied), op=MPI.LAND)

        # 结构增量语义：全部为 TODO、速率清零
        n = epool.count_pair
        if n > 0:
            assert np.all(epool.pair_obj_array[:n]['status'] == STATUS_TODO)
            assert np.all(epool.pair_obj_array[:n]['rate'] == 0.0)
        assert epool.rate_sum_pair == 0.0

    epool.destroy()
    print("  PASSED: incremental == full over 10 fuzz rounds")


# ============================================================
# Test 3: 空状态包 → 纯保留；类级速率保留（feat-2 pair 侧）
# ============================================================

def test_retention_and_rate_keep():
    rng = np.random.default_rng(77 + RANK)

    # 默认类（incremental_rate=False）：空包保留结构，但速率必须重算
    builder = _make_builder(Capture)
    epool = _fresh_event_pool()
    objs = _make_objects(60, rng)
    pool_ns = _make_pool_ns(objs)
    full0 = builder.build_pair_events(pool_ns, current_step=0)
    epool.set_pair_obj_events(full0, recompute_rate_sum=False)
    n0 = epool.count_pair
    # [MPI] 不做 rank 局部提前返回（会跨集合操作失配）；密度已保证有 pair
    assert COMM.allreduce(n0, op=MPI.MIN) > 0, "test setup must generate pairs"
    # 模拟已算完
    epool.pair_obj_array['status'][:n0] = STATUS_DONE
    epool.pair_obj_array['rate'][:n0] = 3.5

    empty = np.zeros(0, dtype=CHANGED_DTYPE)
    applied = builder.build_pair_events_incremental(
        pool_ns, epool, empty, current_step=1, allow_rate_keep=True)
    assert applied
    assert epool.count_pair == n0, "empty packet must retain full structure"
    assert np.all(epool.pair_obj_array[:n0]['status'] == STATUS_TODO), \
        "incremental_rate=False class must recompute rates"
    assert epool.rate_sum_pair == 0.0
    epool.destroy()

    # incremental_rate=True 类：允许保留 DONE 速率
    builder2 = _make_builder(CaptureKeepRate)
    epool2 = _fresh_event_pool()
    full0 = builder2.build_pair_events(pool_ns, current_step=0)
    epool2.set_pair_obj_events(full0, recompute_rate_sum=False)
    n0 = epool2.count_pair
    epool2.pair_obj_array['status'][:n0] = STATUS_DONE
    epool2.pair_obj_array['rate'][:n0] = 3.5

    applied = builder2.build_pair_events_incremental(
        pool_ns, epool2, empty, current_step=1, allow_rate_keep=True)
    assert applied
    assert np.all(epool2.pair_obj_array[:n0]['status'] == STATUS_DONE), \
        "incremental_rate=True class must keep DONE rates on retention"
    assert abs(epool2.rate_sum_pair - 3.5 * n0) < 1e-9
    epool2.destroy()
    print("  PASSED: retention semantics + class-level rate keep")


# ============================================================
# Test 4: 引擎级偏执校验（每步 增量 vs 全量 自动对照）
# ============================================================

LC = 3.17
GRID = (30, 30, 30)
EBOX = [GRID[i] * LC for i in range(3)]


class EngMigration(SingleObjectEvent):
    object_types = ['H', 'V']
    params = []

    @classmethod
    def rate(cls, objs, params, config):
        return 1.0e6

    @classmethod
    def execute(cls, event, objs, config):
        obj = objs[0]
        rng = config['rng']
        out = {'id': int(obj['id'])}
        for axis, box_len in zip(('x', 'y', 'z'), EBOX):
            out[axis] = (float(obj[axis]) + rng.uniform(-1.5, 1.5)) % box_len
        return [out]


class EngCapture(PairObjectEvent):
    type_pairs = [('H', 'V')]
    max_distance = 6.0
    params = ['distance']
    # predictive 贪心按权重降序搬事件：pair 权重显式调大，保证均衡时
    # pair 被优先搬运 → verify 期间必然出现寄居 pair（测试前提确定性成立）。
    # 旧版此前提"碰巧"成立，实为 enabled+predictive 下 single 事件全局
    # 丢失 bug 的副产品（single 池蒸发 → pair 成为可搬多数）；契约重构
    # （predictive 在场时 single 每步全量重建）修掉了丢失，也就需要显式权重
    weight = 5000.0

    @classmethod
    def rate(cls, objs, params, config):
        return 5.0e5

    @classmethod
    def execute(cls, event, objs, config):
        # 删除 V、移动 H：同时演练删除(旧坐标)与修改两类状态包记录
        h, v = objs[0], objs[1]
        return [
            {'id': int(v['id']), 'deleted': True},
            {'id': int(h['id']), 'x': (float(h['x']) + 0.5) % EBOX[0]},
        ]


def test_engine_paranoid_verify():
    """真实引擎逐步跑，builder 内部每步用全量重建对照增量结果"""
    from paraokmc.v16_engine import create_okmc_engine_v16
    from paraokmc.event_compiler import merge_event_config

    def make_ic(n):
        def initial_objects_func(rank, size, config):
            if rank != 0:
                return []
            r = np.random.default_rng(5)
            # 热点集中在盒子一角：np>=2 时制造强不均衡，
            # 保证 predictive 实际搬运 pair 事件（寄居场景覆盖）
            return [{
                'id': i + 1,
                'type': 'H' if i % 2 == 0 else 'V',
                'x': float(r.uniform(5, 40)),
                'y': float(r.uniform(5, 40)),
                'z': float(r.uniform(5, 40)),
            } for i in range(n)]
        return initial_objects_func

    cap = 300
    base = {
        'physics': {'temperature': 500.0, 'lattice_constant': LC},
        'objects': {'H': {'properties': {}}, 'V': {'properties': {}}},
        'domain_decomposition': {
            'lattice_constant': LC,
            'lattice_grid': list(GRID),
            'h_factor': 1,
            'periodic': [True, True, True],
            'max_interaction_distance': 6.0,
        },
        'initial_objects_func': make_ic(60),
        'lattice_constant': LC,
        'grid_size': GRID,
        'temperature': 500.0,
        'max_steps': 25,
        'max_time': 1.0,
        'verbose': False,
        'random_seed': 9,
        'progress_interval': 0,
        'rma_object_pool': {'initial_capacity': cap, 'auto_expand': True},
        'rma_event_pool': {
            'capacity_zero': 4, 'capacity_single': cap,
            'capacity_pair': cap * 10, 'auto_expand': True,
        },
        '_pair_structure_verify': True,   # 偏执模式：每步增量 vs 全量对照
        # [fix-5 覆盖] predictive 开启：np>=2 时 pair 事件会被搬运寄居，
        # 全局并集校验必须仍然通过（旧的本地对照在此场景必然误报）
        'predictive_rebalance': {'enabled': True, 'threshold': 0.05},
        'adaptive_weight_tracker': {'enabled': False},
        'runtime_load_balancer': {'enabled': False},
        'object_load_balancer': {'enabled': False},
    }
    config = merge_event_config(base, {
        'migration': EngMigration,
        'capture': EngCapture,
    })
    engine = create_okmc_engine_v16(config)

    # [覆盖自证] 仅靠参数配置不能保证 predictive 真的搬运了 pair：
    # 包一层 verifier，在它被调用的那一刻记录"全局存在 live 寄居 pair"
    # （status != SENT 且 owner_rank != 本 rank）。verifier 内部做
    # allgather（集体调用），所以这里的 allreduce 与之同步安全。
    from paraokmc.rma_event_pool import STATUS_SENT
    builder = engine.event_builder._vectorized_builder
    orig_verify = builder._verify_pair_pool_against_full
    lodged_seen = {'flag': False}

    def recording_verify(*args, **kwargs):
        ep = engine.rma_event_pool
        ev = ep.pair_obj_array[:ep.count_pair]
        local = bool(np.any((ev['status'] != STATUS_SENT)
                            & (ev['owner_rank'] != RANK))) if len(ev) else False
        lodged_seen['flag'] |= bool(COMM.allreduce(local, op=MPI.LOR))
        return orig_verify(*args, **kwargs)

    builder._verify_pair_pool_against_full = recording_verify

    engine.run()   # 偏执校验失败会直接 raise

    steps = COMM.allreduce(engine.current_step, op=MPI.MAX)
    assert steps > 0
    if COMM.size > 1:
        moved = COMM.allreduce(
            int(engine.predictive_rebalancer.stats['total_events_moved']),
            op=MPI.MAX)
        assert moved > 0, \
            "test premise: predictive must actually move events"
        # total_events_moved 含 zero/single/pair，单靠它不足以证明
        # 修复 5 被覆盖——必须在 verifier 执行期间见过 live 寄居 pair
        assert lodged_seen['flag'], \
            "test premise: a live lodged pair must exist during verification"
    # 显式集体释放 RMA 窗口：留给 GC 会在 numba 编译等时机各 rank
    # 不同步地触发 __del__ → Win.Free（集体调用）→ 死锁
    engine.destroy()
    if RANK == 0:
        print(f"  PASSED: engine paranoid verify ({steps} steps, "
              f"np={COMM.size}, incremental pair pool validated every step, "
              f"lodged-pair-at-verify={'yes' if lodged_seen['flag'] else 'n/a np=1'})")


# ============================================================
# Test 5: 类级速率增量（feat-2 single 侧，rate 调用计数）
# ============================================================

def test_single_rate_incremental_flag():
    from paraokmc.v16_engine import create_okmc_engine_v16
    from paraokmc.event_compiler import merge_event_config

    class MigKeep(SingleObjectEvent):
        object_types = ['A']
        params = []
        incremental_rate = True
        calls = 0

        @classmethod
        def rate(cls, objs, params, config):
            MigKeep.calls += 1
            return 1.0e6

        @classmethod
        def execute(cls, event, objs, config):
            obj = objs[0]
            rng = config['rng']
            return [{'id': int(obj['id']),
                     'x': (float(obj['x']) + rng.uniform(-1, 1)) % EBOX[0]}]

    class MigRecompute(SingleObjectEvent):
        object_types = ['A']
        params = []
        incremental_rate = False
        calls = 0

        @classmethod
        def rate(cls, objs, params, config):
            MigRecompute.calls += 1
            return 1.0e6

        @classmethod
        def execute(cls, event, objs, config):
            obj = objs[0]
            rng = config['rng']
            return [{'id': int(obj['id']),
                     'y': (float(obj['y']) + rng.uniform(-1, 1)) % EBOX[1]}]

    def make_ic(n):
        # builtin 约定：每 rank 同种子生成全量对象、只保留本域——
        # rank0-only IC 在 np>=4 时本域只剩 3 个对象，触发原地路径的
        # ×4 启发式回退全量，保率契约无从体现（测试前提坍塌）
        def initial_objects_func(rank, size, config):
            r = np.random.default_rng(3)
            objs = [{'id': i + 1, 'type': 'A',
                     'x': float(r.uniform(10, 80)),
                     'y': float(r.uniform(10, 80)),
                     'z': float(r.uniform(10, 80))} for i in range(n)]
            dd = config.get('_domain_decomp')
            if dd is None:
                return objs if rank == 0 else []
            b = dd.subdomain_bounds
            return [o for o in objs
                    if b['x_min'] <= o['x'] < b['x_max']
                    and b['y_min'] <= o['y'] < b['y_max']
                    and b['z_min'] <= o['z'] < b['z_max']]
        return initial_objects_func

    n_objects, n_steps = 48, 15
    cap = 200
    base = {
        'physics': {'temperature': 500.0, 'lattice_constant': LC},
        'objects': {'A': {'properties': {}}},
        'domain_decomposition': {
            'lattice_constant': LC, 'lattice_grid': list(GRID),
            'h_factor': 1, 'periodic': [True, True, True],
            'max_interaction_distance': 6.0,
        },
        'initial_objects_func': make_ic(n_objects),
        'lattice_constant': LC, 'grid_size': GRID, 'temperature': 500.0,
        'max_steps': n_steps, 'max_time': 1.0,
        'verbose': False, 'random_seed': 11, 'progress_interval': 0,
        'rma_object_pool': {'initial_capacity': cap, 'auto_expand': True},
        'rma_event_pool': {
            'capacity_zero': 4, 'capacity_single': cap * 2,
            'capacity_pair': 4, 'auto_expand': True,
        },
        'predictive_rebalance': {'enabled': False},
        'adaptive_weight_tracker': {'enabled': False},
        'runtime_load_balancer': {'enabled': False},
        'object_load_balancer': {'enabled': False},
    }
    MigKeep.calls = 0
    MigRecompute.calls = 0
    config = merge_event_config(base, {
        'mig_keep': MigKeep,
        'mig_recompute': MigRecompute,
    })
    engine = create_okmc_engine_v16(config)
    engine.run()

    calls_keep = COMM.allreduce(MigKeep.calls, op=MPI.SUM)
    calls_recompute = COMM.allreduce(MigRecompute.calls, op=MPI.SUM)
    if RANK == 0:
        print(f"  rate() calls: incremental_rate=True → {calls_keep}, "
              f"False → {calls_recompute}")
    # True 类：首步全量 + 每步 O(1)；False 类：每步全量重算
    assert calls_recompute > calls_keep * 2, (
        f"incremental_rate gating ineffective: "
        f"keep={calls_keep}, recompute={calls_recompute}")
    engine.destroy()
    if RANK == 0:
        print("  PASSED: class-level incremental_rate gating (single)")


# ============================================================
# Test 5.5: 真空 host —— 从未全量构建的 rank 也必须能增量处理寄居事件
# ============================================================

def test_lodged_empty_host():
    """[fix-1] 空 rank（对象<2、网格未初始化）收到 predictive 迁入的 pair 后，
    下一步必须走增量（按需建网格），不得回退全量把寄居副本抹掉"""
    objs, bO, bT_unused, eO, eT, pool_ns = _lodged_setup()

    # 真空 host：全新 builder（从未调用过全量构建，_cl_grid 未初始化），
    # 且本地对象集为空
    bT = _make_builder(rank=1)
    empty_objs = np.zeros(0, dtype=OBJ_DTYPE)
    pool_ns_T = _make_pool_ns(empty_objs)

    # predictive 物理搬运 A、B 到真空 host
    moved_idx = np.concatenate([_find_pair(eO, 1, 2), _find_pair(eO, 3, 4)])
    assert len(moved_idx) == 2
    moved = eO.pair_obj_array[moved_idx].copy()
    keep = np.ones(eO.count_pair, dtype=bool)
    keep[moved_idx] = False
    eO.set_pair_obj_events(eO.pair_obj_array[:eO.count_pair][keep],
                           recompute_rate_sum=False)
    eT.set_pair_obj_events(moved, recompute_rate_sum=False)

    packet = _move_obj3_packet(objs)
    pool_ns = _make_pool_ns(objs)

    assert bO.build_pair_events_incremental(pool_ns, eO, packet, 1)
    applied_T = bT.build_pair_events_incremental(pool_ns_T, eT, packet, 1)
    assert applied_T, \
        "empty host must take the incremental path (on-demand grid init)"

    # 情况4：有效寄居 A 保留在真空 host
    assert len(_find_pair(eT, 1, 2)) == 1, \
        "empty host must retain valid lodged pair"
    assert _live_count([eO, eT], 1, 2) == 1
    # 情况1：失效寄居 B 被清、owner 重建
    assert len(_find_pair(eT, 3, 4)) == 0
    assert len(_find_pair(eO, 3, 4)) == 1
    assert _live_count([eO, eT], 3, 4) == 1

    eO.destroy(); eT.destroy()
    print("  PASSED: empty host handles lodged pairs incrementally")


# ============================================================
# Test 6: 毒区掩码构造（白盒单测）
# ============================================================

def test_poison_mask_construction():
    """新/旧坐标侧、删除记录零值新坐标不参与、邻居环、周期绕回"""
    builder = _make_builder()
    # 先做一次全量以初始化邻居表（20^3 网格）
    rng = np.random.default_rng(2)
    pool_ns = _make_pool_ns(_make_objects(40, rng))
    builder.build_pair_events(pool_ns, current_step=0)

    grid_dims, cell_sizes, box_size, periodic = builder._pair_grid_params()
    Gx, Gy, Gz = grid_dims
    assert (Gx, Gy, Gz) == (20, 20, 20)

    def cell(ix, iy, iz):
        return (ix % Gx) + (iy % Gy) * Gx + (iz % Gz) * Gx * Gy

    # 记录1：移动 old c(6,6,6) → new c(10,10,10)
    # 记录2：删除 old c(9,9,9)（new 坐标为序列化零值，不得参与毒区）
    # 记录3：新增 new c(14,14,14)（无旧坐标）
    # 记录4：移动 new 落在 c(19,10,10)（邻居环需周期绕回到 c(0,10,10)）
    arr = np.zeros(4, dtype=CHANGED_DTYPE)
    arr[0] = (101, 52.5, 52.5, 52.5, True, 32.5, 32.5, 32.5, True)
    arr[1] = (102, 0.0, 0.0, 0.0, False, 47.5, 47.5, 47.5, True)
    arr[2] = (103, 72.5, 72.5, 72.5, True, 0.0, 0.0, 0.0, False)
    arr[3] = (104, 98.0, 52.5, 52.5, True, 52.5, 52.5, 52.5, True)

    mask = builder._compute_poison_mask(arr, grid_dims, cell_sizes,
                                        box_size, periodic)

    # 种子 cell 及其环
    for c in [cell(10, 10, 10), cell(6, 6, 6), cell(9, 9, 9),
              cell(14, 14, 14), cell(19, 10, 10)]:
        assert mask[c], f"seed cell {c} must be poisoned"
    assert mask[cell(9, 9, 10)], "neighbor ring must be poisoned"
    assert mask[cell(11, 11, 11)], "neighbor ring must be poisoned"
    # 周期绕回：c(19,10,10) 的邻居含 c(0,10,10)
    assert mask[cell(0, 10, 10)], "periodic wrap ring must be poisoned"
    # 删除记录的零值新坐标不得毒化原点区域
    assert not mask[cell(0, 0, 0)], \
        "deletion's zeroed new coords must not poison origin"
    # 远处干净
    assert not mask[cell(3, 15, 3)], "far cell must stay clean"
    print("  PASSED: poison mask construction")


# ============================================================
# Test 7: 杀 ⟺ 重建恒等（内核层：zone 产出 == 全量按毒区过滤）
# ============================================================

def test_zone_kernel_matches_full_subset():
    from paraokmc.jit_cell_list import (
        build_neighbor_map, build_cell_bins,
        jit_find_pairs_cell_list, jit_find_pairs_cell_list_zone,
    )
    from paraokmc.jit_event_builder import build_match_tables

    rng = np.random.default_rng(31)
    n = 120
    x = rng.uniform(0, 40.0, n)
    y = rng.uniform(0, 40.0, n)
    z = rng.uniform(0, 40.0, n)
    obj_ids = np.arange(1, n + 1, dtype=np.int64)
    type_ids = (np.arange(n) % 2).astype(np.int32)
    owners = np.zeros(n, dtype=np.int32)

    Gx = Gy = Gz = 20
    cs = 5.0
    nb_map, nb_counts = build_neighbor_map(
        Gx, Gy, Gz, MAX_DIST, cs, cs, cs, True, True, True)
    heads, nxt = build_cell_bins(x, y, z, n, cs, cs, cs, Gx, Gy, Gz,
                                 BOX[0], BOX[1], BOX[2], True, True, True)
    mt, mdt = build_match_tables(
        {'capture': {'type_pairs': [('H', 'V')], 'max_distance': MAX_DIST}},
        {'H': 0, 'V': 1}, {'capture': 0})

    # 伪随机毒区（~1/3 的 cell）
    zone = np.zeros(Gx * Gy * Gz, dtype=np.bool_)
    zone[np.arange(0, Gx * Gy * Gz, 3)] = True

    cap = 8192
    def run(kernel, *extra):
        o1 = np.empty(cap, np.int64); o2 = np.empty(cap, np.int64)
        mn = np.empty(cap, np.int64); mx = np.empty(cap, np.int64)
        ow = np.empty(cap, np.int32); di = np.empty(cap, np.float64)
        et = np.empty(cap, np.int32)
        nf = kernel(x, y, z, obj_ids, type_ids, owners, n, heads, nxt,
                    nb_map, nb_counts, *extra, mt, mdt,
                    BOX[0], BOX[1], BOX[2], True, True, True, np.int32(0),
                    o1, o2, mn, mx, ow, di, et, cap)
        assert nf <= cap
        return set(zip(mn[:nf].tolist(), mx[:nf].tolist())), o1[:nf], o2[:nf]

    full_keys, f1, f2 = run(jit_find_pairs_cell_list)
    zone_keys, _, _ = run(jit_find_pairs_cell_list_zone, zone)

    # 期望：全量产出中"任一端点在毒区 cell"的子集
    def cell_of(i):
        return (int(x[i] // cs) + int(y[i] // cs) * Gx
                + int(z[i] // cs) * Gx * Gy)
    expected = set()
    for i1, i2 in zip(f1.tolist(), f2.tolist()):
        if zone[cell_of(i1)] or zone[cell_of(i2)]:
            key = (min(obj_ids[i1], obj_ids[i2]),
                   max(obj_ids[i1], obj_ids[i2]))
            expected.add(key)

    assert len(full_keys) > 20, f"test needs pairs, got {len(full_keys)}"
    assert zone_keys == expected, (
        f"zone kernel != filtered full: missing={len(expected - zone_keys)}, "
        f"extra={len(zone_keys - expected)}")
    print(f"  PASSED: zone kernel == full∩zone ({len(zone_keys)}/{len(full_keys)} pairs)")


# ============================================================
# Test 8: update_pair_events_inplace 的 SENT / rate-keep 语义
# ============================================================

def test_inplace_update_sent_semantics():
    epool = _fresh_event_pool()
    ev = np.zeros(4, dtype=EVENT_DTYPE)
    from paraokmc.rma_event_pool import STATUS_SENT
    # e0: SENT 壳（保留 → 必须仍是 SENT）
    # e1: DONE，允许保率（保留 → 保持 DONE + 速率）
    # e2: DONE，将被杀
    # e3: DONE，不允许保率（保留 → TODO、速率清零）
    for k in range(4):
        ev[k]['obj_ids'][:] = -1
        ev[k]['obj_ids'][0] = 10 + k
        ev[k]['obj_ids'][1] = 20 + k
        ev[k]['status'] = STATUS_DONE
        ev[k]['rate'] = 2.0 + k
    ev[0]['status'] = STATUS_SENT
    ev[0]['rate'] = 0.0
    epool.set_pair_obj_events(ev, recompute_rate_sum=False)

    keep_mask = np.array([True, True, False, True])
    keep_rate = np.array([True, True, False, False])  # SENT 的 True 必须被忽略
    fresh = np.zeros(1, dtype=EVENT_DTYPE)
    fresh[0]['obj_ids'][:] = -1
    fresh[0]['obj_ids'][0] = 99
    fresh[0]['status'] = STATUS_TODO

    total = epool.update_pair_events_inplace(keep_mask, fresh, keep_rate)
    assert total == 4  # 3 保留 + 1 新增
    arr = epool.pair_obj_array[:total]

    assert arr[0]['status'] == STATUS_SENT and arr[0]['rate'] == 0.0, \
        "SENT husk must stay SENT"
    assert arr[1]['status'] == STATUS_DONE and arr[1]['rate'] == 3.0, \
        "rate-keep DONE event must keep its rate"
    assert arr[2]['status'] == STATUS_TODO and arr[2]['rate'] == 0.0, \
        "non-keep event must reset to TODO"
    assert arr[3]['obj_ids'][0] == 99 and arr[3]['status'] == STATUS_TODO
    assert abs(epool.rate_sum_pair - 3.0) < 1e-12
    epool.destroy()
    print("  PASSED: in-place update SENT / rate-keep semantics")


# ============================================================
# Test 9/10: 寄居 pair 的情况 1/4（runtime 窃取 与 predictive 搬运）
# ============================================================

def _lodged_setup():
    """owner(rank0) / host(rank1) 双视角环境。

    对象全部归 rank0：pair A=(1,2) 本轮保持有效；pair B=(3,4) 将因
    obj3 移动而失效；远处 filler 提供背景事件。
    """
    specs = [
        (1, 0, 10.0, 10.0, 10.0), (2, 1, 11.0, 10.0, 10.0),   # pair A
        (3, 0, 30.0, 30.0, 30.0), (4, 1, 31.0, 30.0, 30.0),   # pair B
    ]
    rng = np.random.default_rng(8)
    for i in range(20):  # filler
        specs.append((100 + i, i % 2,
                      60.0 + rng.uniform(0, 10), 60.0 + rng.uniform(0, 10),
                      60.0 + rng.uniform(0, 10)))
    objs = np.zeros(len(specs), dtype=OBJ_DTYPE)
    for k, (oid, tid, ox, oy, oz) in enumerate(specs):
        objs[k] = (oid, tid, ox, oy, oz, 0, True, 0.0, (0, 0, 0, 0))

    builder_O = _make_builder(rank=0)
    builder_T = _make_builder(rank=1)
    epool_O = _fresh_event_pool()
    epool_T = _fresh_event_pool()

    pool_ns = _make_pool_ns(objs)
    events_O = builder_O.build_pair_events(pool_ns, current_step=0)
    builder_T.build_pair_events(pool_ns, current_step=0)  # 仅初始化网格缓存
    epool_O.set_pair_obj_events(events_O, recompute_rate_sum=False)

    return objs, builder_O, builder_T, epool_O, epool_T, pool_ns


def _move_obj3_packet(objs):
    """状态包：obj3 移动 0.8（pair B 失效但几何上仍成对）"""
    i3 = int(np.flatnonzero(objs['id'] == 3)[0])
    old = (float(objs['x'][i3]), float(objs['y'][i3]), float(objs['z'][i3]))
    objs['x'][i3] += 0.8
    arr = np.zeros(1, dtype=CHANGED_DTYPE)
    arr[0] = (3, float(objs['x'][i3]), float(objs['y'][i3]),
              float(objs['z'][i3]), True, old[0], old[1], old[2], True)
    return arr


def _find_pair(epool, a, b):
    n = epool.count_pair
    ev = epool.pair_obj_array[:n]
    hit = np.flatnonzero((ev['obj_ids'][:, 0] == a) & (ev['obj_ids'][:, 1] == b))
    return hit


def _live_count(pools, a, b):
    from paraokmc.rma_event_pool import STATUS_SENT
    total = 0
    for p in pools:
        idx = _find_pair(p, a, b)
        ev = p.pair_obj_array
        total += int(np.sum(ev['status'][idx] != STATUS_SENT))
    return total


def test_lodged_runtime_steal_cases():
    """runtime LB 窃取：情况4（有效寄居保留）+ 情况1（失效寄居双清+owner重建）"""
    from paraokmc.rma_event_pool import STATUS_SENT
    objs, bO, bT, eO, eT, pool_ns = _lodged_setup()

    # 模拟窃取 pair A 与 pair B：owner 标 SENT 壳，host 持活副本（已算完）
    stolen_idx = np.concatenate([_find_pair(eO, 1, 2), _find_pair(eO, 3, 4)])
    assert len(stolen_idx) == 2, "setup must produce pairs A and B"
    stolen = eO.pair_obj_array[stolen_idx].copy()
    stolen['status'] = STATUS_DONE
    stolen['rate'] = 7.0
    eT.set_pair_obj_events(stolen, recompute_rate_sum=False)
    eO.pair_obj_array['status'][stolen_idx] = STATUS_SENT
    eO.pair_obj_array['rate'][stolen_idx] = 0.0

    # 状态包：obj3 移动 → pair B 失效、pair A 保持有效
    packet = _move_obj3_packet(objs)
    pool_ns = _make_pool_ns(objs)

    assert bO.build_pair_events_incremental(pool_ns, eO, packet, 1)
    assert bT.build_pair_events_incremental(pool_ns, eT, packet, 1)

    # 情况4（pair A 有效）：host 活副本保留并重算，owner 壳原样
    iA_T = _find_pair(eT, 1, 2)
    assert len(iA_T) == 1 and eT.pair_obj_array['status'][iA_T[0]] == STATUS_TODO, \
        "host must retain valid lodged pair (reset to TODO)"
    iA_O = _find_pair(eO, 1, 2)
    assert len(iA_O) == 1 and eO.pair_obj_array['status'][iA_O[0]] == STATUS_SENT, \
        "owner must keep SENT husk for valid lodged pair"
    assert _live_count([eO, eT], 1, 2) == 1

    # 情况1（pair B 失效）：host 清除；owner 壳清除且毒区重建出新副本
    assert len(_find_pair(eT, 3, 4)) == 0, "host must clear invalidated lodged pair"
    iB_O = _find_pair(eO, 3, 4)
    assert len(iB_O) == 1 and eO.pair_obj_array['status'][iB_O[0]] == STATUS_TODO, \
        "owner must rebuild invalidated pair as fresh TODO"
    # 重建副本嵌入的是 obj3 的新位置
    emb3 = eO.pair_obj_array['objects'][iB_O[0], 0]
    assert abs(float(emb3['x']) - 30.8) < 1e-9, \
        "rebuilt event must embed the updated position"
    assert _live_count([eO, eT], 3, 4) == 1

    eO.destroy(); eT.destroy()
    print("  PASSED: lodged pairs under runtime-LB steal (cases 1 & 4)")


def test_lodged_predictive_move_cases():
    """predictive 搬运：情况4（有效仅宿主持有）+ 情况1（宿主清、owner无副本重建）"""
    objs, bO, bT, eO, eT, pool_ns = _lodged_setup()

    # 模拟 predictive 物理搬运：owner 移除 A、B，host 收到活副本（TODO）
    moved_idx = np.concatenate([_find_pair(eO, 1, 2), _find_pair(eO, 3, 4)])
    assert len(moved_idx) == 2
    moved = eO.pair_obj_array[moved_idx].copy()
    keep = np.ones(eO.count_pair, dtype=bool)
    keep[moved_idx] = False
    eO.set_pair_obj_events(eO.pair_obj_array[:eO.count_pair][keep],
                           recompute_rate_sum=False)
    eT.set_pair_obj_events(moved, recompute_rate_sum=False)

    packet = _move_obj3_packet(objs)
    pool_ns = _make_pool_ns(objs)

    assert bO.build_pair_events_incremental(pool_ns, eO, packet, 1)
    assert bT.build_pair_events_incremental(pool_ns, eT, packet, 1)

    # 情况4（pair A 有效）：仅宿主持有活副本，owner 不重建
    assert len(_find_pair(eT, 1, 2)) == 1, "host must retain valid moved pair"
    assert len(_find_pair(eO, 1, 2)) == 0, \
        "owner must not duplicate a valid moved pair"
    assert _live_count([eO, eT], 1, 2) == 1

    # 情况1（pair B 失效）：宿主清除；owner 无旧副本仍从对象数组重建
    assert len(_find_pair(eT, 3, 4)) == 0, "host must clear invalidated moved pair"
    iB_O = _find_pair(eO, 3, 4)
    assert len(iB_O) == 1 and eO.pair_obj_array['status'][iB_O[0]] == STATUS_TODO, \
        "owner must rebuild invalidated pair from object arrays"
    assert _live_count([eO, eT], 3, 4) == 1

    eO.destroy(); eT.destroy()
    print("  PASSED: lodged pairs under predictive move (cases 1 & 4)")


# ============================================================
# Test 11: physics 强制重建 → pair/single 全量（params 不陈旧）
# ============================================================

def _make_engine_config(max_steps=25, seed=9, verify=True, predictive=True):
    def make_ic(n):
        def initial_objects_func(rank, size, config):
            if rank != 0:
                return []
            r = np.random.default_rng(5)
            return [{
                'id': i + 1,
                'type': 'H' if i % 2 == 0 else 'V',
                'x': float(r.uniform(5, 40)),
                'y': float(r.uniform(5, 40)),
                'z': float(r.uniform(5, 40)),
            } for i in range(n)]
        return initial_objects_func

    cap = 300
    return {
        'physics': {'temperature': 500.0, 'lattice_constant': LC},
        'objects': {'H': {'properties': {}}, 'V': {'properties': {}}},
        'domain_decomposition': {
            'lattice_constant': LC,
            'lattice_grid': list(GRID),
            'h_factor': 1,
            'periodic': [True, True, True],
            'max_interaction_distance': 6.0,
        },
        'initial_objects_func': make_ic(60),
        'lattice_constant': LC,
        'grid_size': GRID,
        'temperature': 500.0,
        'max_steps': max_steps,
        'max_time': 1.0,
        'verbose': False,
        'random_seed': seed,
        'progress_interval': 0,
        'rma_object_pool': {'initial_capacity': cap, 'auto_expand': True},
        'rma_event_pool': {
            'capacity_zero': 4, 'capacity_single': cap,
            'capacity_pair': cap * 10, 'auto_expand': True,
        },
        '_pair_structure_verify': verify,
        'predictive_rebalance': {'enabled': predictive, 'threshold': 0.05},
        'adaptive_weight_tracker': {'enabled': False},
        'runtime_load_balancer': {'enabled': False},
        'object_load_balancer': {'enabled': False},
    }


def _id_high_steps(events):
    """从 128 位事件 ID 中解出构建步数（[prefix:2][step:35][rank:16][type:10]）"""
    return (events['id_high'].astype(np.int64) >> 26) & ((1 << 35) - 1)


def test_force_rebuild_goes_full():
    """[fix-3] physics 强制重建标志在场时，pair 结构增量与 single 增量复用
    都必须转为全量（否则保留事件缓存的旧 params 参与速率计算）"""
    from paraokmc.v16_engine import create_okmc_engine_v16
    from paraokmc.event_compiler import merge_event_config

    config = merge_event_config(
        _make_engine_config(max_steps=20, verify=False, predictive=False),
        {'migration': EngMigration, 'capture': EngCapture})
    engine = create_okmc_engine_v16(config)
    engine.run()

    ep = engine.rma_event_pool
    n_pair = ep.count_pair
    # 前提自检：结构增量确实在工作（存在早于当前步构建的保留事件）
    retained_before = int(np.sum(
        _id_high_steps(ep.pair_obj_array[:n_pair]) < engine.current_step)) \
        if n_pair > 0 else 0
    retained_ok = COMM.allreduce(retained_before, op=MPI.SUM) > 0

    # 模拟 physics 更新：设置强制重建标志后手动触发一次构建
    engine.incremental_manager.set_force_rebuild(single=True, zero=True)
    engine._build_events()

    n_pair = ep.count_pair
    pair_fresh = bool(np.all(
        _id_high_steps(ep.pair_obj_array[:n_pair]) == engine.current_step)) \
        if n_pair > 0 else True
    n_single = ep.count_single
    single_fresh = bool(np.all(
        _id_high_steps(ep.single_obj_array[:n_single]) == engine.current_step)) \
        if n_single > 0 else True

    assert COMM.allreduce(retained_ok, op=MPI.LAND), \
        "test premise: structural retention must have occurred before forcing"
    assert COMM.allreduce(pair_fresh, op=MPI.LAND), \
        "force_rebuild must fully rebuild pair events (stale params risk)"
    assert COMM.allreduce(single_fresh, op=MPI.LAND), \
        "force_rebuild must fully rebuild single events (stale params risk)"
    engine.destroy()
    if RANK == 0:
        print("  PASSED: physics force flags convert both pools to full rebuild")


# ============================================================
# Test 12: random_migration 方向重抽样（fix-4）
# ============================================================

def test_random_migration_resamples():
    from paraokmc.config_system import random_migration
    from paraokmc.random_generator import RandomGenerator

    obj = {'id': 1, 'type': 'A', 'x': 50.0, 'y': 50.0, 'z': 50.0}
    config = {
        'lattice_constant': 1.0,
        'domain_decomposition': {'box_size': [100.0, 100.0, 100.0]},
        'rng': RandomGenerator(user_seed=42, rank=0),
        '_user_seed': 42,
    }
    ev = np.zeros(1, dtype=EVENT_DTYPE)[0]  # id 字段恒为 0（128 位模式）

    # 统一 RNG 路径：重复执行必须重新抽样，方向不得恒定
    config['rng'].set_step(0)
    moves = set()
    for _ in range(30):
        out = random_migration(ev, [obj], config)[0]
        moves.add((round(out['x'], 9), round(out['y'], 9), round(out['z'], 9)))
    assert len(moves) > 20, \
        f"random_migration must resample per call, got {len(moves)} unique moves"

    # 回退路径（无 rng）：不同 128 位事件 ID 给出不同方向；id 字段为 0 不再致命
    cfg_no_rng = {k: v for k, v in config.items() if k != 'rng'}
    ev1 = np.zeros(1, dtype=EVENT_DTYPE)[0]
    ev2 = np.zeros(1, dtype=EVENT_DTYPE)[0]
    ev1['id_high'], ev1['id_low'] = 12345, 7
    ev2['id_high'], ev2['id_low'] = 12345, 8
    m1 = random_migration(ev1, [obj], cfg_no_rng)[0]
    m2 = random_migration(ev2, [obj], cfg_no_rng)[0]
    assert (m1['x'], m1['y']) != (m2['x'], m2['y']), \
        "fallback must derive direction from the real 128-bit event ID"
    print("  PASSED: random_migration resamples (rng path + 128-bit fallback)")


# ============================================================
# Test 13: 大种子（fix-6）
# ============================================================

def test_big_seed_engine():
    from paraokmc.v16_engine import create_okmc_engine_v16
    from paraokmc.event_compiler import merge_event_config

    config = merge_event_config(
        _make_engine_config(max_steps=3, seed=2**40 + 123,
                            verify=False, predictive=False),
        {'migration': EngMigration})
    engine = create_okmc_engine_v16(config)   # 旧代码在 np.random.seed 处 ValueError
    engine.run()
    assert COMM.allreduce(engine.current_step, op=MPI.MAX) > 0
    engine.destroy()
    if RANK == 0:
        print("  PASSED: seeds >= 2^32 accepted end-to-end")


# ============================================================
# Test 14: builtin IC + 大种子（复审二轮条 1）
# ============================================================

def test_big_seed_builtin_ic():
    """七块配置 ic.mode='builtin'、非空 IC、random_seed >= 2^32：
    引擎入口掩码之外，create_initial_objects_func 里的第二处
    np.random.seed(ic_seed) 也必须掩码，否则 IC 执行时 ValueError"""
    from paraokmc.user_config import validate_user_config, compile_user_config
    from paraokmc.v16_engine import create_okmc_engine_v16
    from paraokmc.event_compiler import merge_event_config

    cap = 200
    user_cfg = {
        'domain': {
            'lattice_constant': LC,
            'lattice_grid': list(GRID),
            'h_factor': 1,
            'periodic': [True, True, True],
            'max_interaction_distance': 6.0,
        },
        'simulation': {
            'max_steps': 3,
            'max_time': 1.0,
            'random_seed': 2**40 + 123,
            'verbose': False,
            'progress_interval': 0,
            'rma_object_pool': {'initial_capacity': cap, 'auto_expand': True},
            'rma_event_pool': {
                'capacity_zero': 4, 'capacity_single': cap,
                'capacity_pair': cap * 4, 'auto_expand': True,
            },
        },
        'objects': {'H': {'properties': {}}, 'V': {'properties': {}}},
        'ic': {'mode': 'builtin', 'initial_conditions': {
            'H': {'count': 20, 'distribution': 'random'},
            'V': {'count': 20, 'distribution': 'random'},
        }},
        'events': {'migration': EngMigration},
        'physics': {},
        'outputs': {},
    }
    validate_user_config(user_cfg)
    flat = compile_user_config(user_cfg)
    config = merge_event_config(flat, {'migration': EngMigration})
    engine = create_okmc_engine_v16(config)   # IC 在此执行
    engine.run()

    pool = engine.rma_object_pool
    owned = pool.get_local_objects()
    owned = owned[(owned['owner_rank'] == RANK) & owned['active']]
    n_all = COMM.allreduce(len(owned), op=MPI.SUM)
    # 40 初始对象 - 至多 3 步迁移不删对象 → 数量守恒（证明非空 IC 真的走通）
    assert n_all == 40, f"builtin IC must actually create objects, got {n_all}"
    assert COMM.allreduce(engine.current_step, op=MPI.MAX) > 0
    engine.destroy()
    if RANK == 0:
        print("  PASSED: builtin IC path accepts seeds >= 2^32")


# ============================================================
# Test 15: 三个旧迁移入口 + 字典事件（复审二轮条 2/3）
# ============================================================

def test_direction_entry_points():
    from paraokmc.config_system import deterministic_direction_rands
    from paraokmc.builtin_migrations import default_execute_migration
    from paraokmc.random_generator import RandomGenerator

    base_cfg = {
        'domain_decomposition': {'box_size': [100.0, 100.0, 100.0]},
        '_user_seed': 7,
        '_int_to_type': {0: 'A'},
    }

    # --- 字典形态：id_high/id_low 必须先于兼容 id 被识别 ---
    r1 = deterministic_direction_rands({'id_high': 111, 'id_low': 5, 'id': 0}, base_cfg)
    r2 = deterministic_direction_rands({'id_high': 222, 'id_low': 5, 'id': 0}, base_cfg)
    assert r1 != r2, "dict with id=0 must not swallow the 128-bit fields"
    # 只有 id_high/id_low、无 id 键：不得 KeyError，且与带 id=0 时一致
    r3 = deterministic_direction_rands({'id_high': 111, 'id_low': 5}, base_cfg)
    assert r3 == r1
    # 只有兼容 id
    assert (deterministic_direction_rands({'id': 9}, base_cfg)
            != deterministic_direction_rands({'id': 10}, base_cfg))

    # --- default_execute_migration：回退用 128 位 ID + jump_distance 语义 ---
    obj = {'id': 3, 'type_id': 0, 'x': 50.0, 'y': 50.0, 'z': 50.0}
    ev1 = np.zeros(1, dtype=EVENT_DTYPE)[0]
    ev2 = np.zeros(1, dtype=EVENT_DTYPE)[0]
    ev1['id_high'], ev1['id_low'] = 77, 1
    ev2['id_high'], ev2['id_low'] = 78, 1
    m1 = default_execute_migration(ev1, [obj], base_cfg, jump_distance=2.5)[0]
    m2 = default_execute_migration(ev2, [obj], base_cfg, jump_distance=2.5)[0]
    assert (m1['x'], m1['y'], m1['z']) != (m2['x'], m2['y'], m2['z']), \
        "fallback must derive direction from the 128-bit event ID"
    disp = np.sqrt((m1['x'] - 50.0)**2 + (m1['y'] - 50.0)**2 + (m1['z'] - 50.0)**2)
    assert abs(disp - 2.5) < 1e-9, \
        f"jump_distance semantics must be preserved, |dr|={disp}"

    # rng 路径：重复执行必须重新抽样
    rng_cfg = dict(base_cfg)
    rng_cfg['rng'] = RandomGenerator(user_seed=1, rank=0)
    rng_cfg['rng'].set_step(0)
    moves = set()
    for _ in range(20):
        out = default_execute_migration(ev1, [obj], rng_cfg, jump_distance=2.5)[0]
        moves.add((round(out['x'], 9), round(out['y'], 9), round(out['z'], 9)))
    assert len(moves) > 15, \
        f"rng path must resample per call, got {len(moves)} unique moves"

    # --- tutorial 的两个迁移事件类 ---
    sys.path.insert(0, '../tutorial')
    import example_event_filters as tut

    tut_cfg = {
        'domain_decomposition': {'box_size': [100.0, 100.0, 100.0]},
        '_user_seed': 7,
        'lattice_constant': 1.0,
    }
    for cls, otype in ((tut.ClusterMigrationEvent, 'Cluster'),
                       (tut.IMigrationEvent, 'I')):
        o = {'id': 1, 'type': otype, 'x': 50.0, 'y': 50.0, 'z': 50.0}
        a = cls.execute(ev1, [o], tut_cfg)[0]
        b = cls.execute(ev2, [o], tut_cfg)[0]
        assert (a['x'], a['y'], a['z']) != (b['x'], b['y'], b['z']), \
            f"{cls.__name__} fallback must vary with the 128-bit event ID"
        c1 = cls.execute(ev1, [o], rng_cfg)[0]
        c2 = cls.execute(ev1, [o], rng_cfg)[0]
        assert (c1['x'], c1['y'], c1['z']) != (c2['x'], c2['y'], c2['z']), \
            f"{cls.__name__} rng path must resample per call"

    print("  PASSED: legacy migration entries use rng/128-bit ID "
          "(builtin helper, tutorial classes, dict events)")


# ============================================================
# Test 16: 结构增量恒启用（全局开关废除）
# ============================================================

def test_single_structure_unconditional():
    """[契约] 全局 incremental_update.enabled 已废除：即使配置里写
    enabled=False 也被忽略——single 结构照样增量（未变对象的事件跨步
    保留，id_high 停在创建步，不再每步重生成）"""
    from paraokmc.v16_engine import create_okmc_engine_v16
    from paraokmc.event_compiler import merge_event_config

    base = _make_engine_config(max_steps=15, verify=False, predictive=False)
    base['incremental_update'] = {'enabled': False}   # 应被忽略
    config = merge_event_config(base, {'migration': EngMigration})
    engine = create_okmc_engine_v16(config)
    engine.run()

    ep = engine.rma_event_pool
    n = ep.count_single
    retained = int(np.sum(
        _id_high_steps(ep.single_obj_array[:n]) < engine.current_step)) \
        if n > 0 else 0
    total_retained = COMM.allreduce(retained, op=MPI.SUM)
    engine.destroy()
    assert total_retained > 0, (
        "single structural increment must be unconditional: expected "
        "retained events built at earlier steps, found none")
    if RANK == 0:
        print(f"  PASSED: single structure incremental is unconditional "
              f"({total_retained} retained events, enabled=False ignored)")


# ============================================================
# Test 17: zero 事件速率契约（结构保留 + 类级速率开关）
# ============================================================

def test_zero_rate_incremental_flag():
    """[契约] zero 结构整池保留，速率归类级开关：未开启的类每步重算
    （时间/物理场依赖的速率不再静默冻结），开启的类保 DONE 速率"""
    from paraokmc.v16_engine import create_okmc_engine_v16
    from paraokmc.event_compiler import merge_event_config
    from paraokmc.event_class import ZeroObjectEvent

    class SrcRecompute(ZeroObjectEvent):
        params = []
        calls = 0

        @classmethod
        def rate(cls, objs, params, config):
            SrcRecompute.calls += 1
            return 1.0e-6   # 只建不选

        @classmethod
        def execute(cls, event, objs, config):
            return []

    class SrcKeep(ZeroObjectEvent):
        params = []
        incremental_rate = True
        calls = 0

        @classmethod
        def rate(cls, objs, params, config):
            SrcKeep.calls += 1
            return 1.0e-6

        @classmethod
        def execute(cls, event, objs, config):
            return []

    n_steps = 15
    base = _make_engine_config(max_steps=n_steps, verify=False,
                               predictive=False)
    SrcRecompute.calls = 0
    SrcKeep.calls = 0
    config = merge_event_config(base, {
        'migration': EngMigration,       # 驱动模拟前进
        'src_recompute': SrcRecompute,
        'src_keep': SrcKeep,
    })
    engine = create_okmc_engine_v16(config)
    engine.run()
    steps = COMM.allreduce(engine.current_step, op=MPI.MAX)
    calls_r = COMM.allreduce(SrcRecompute.calls, op=MPI.SUM)
    calls_k = COMM.allreduce(SrcKeep.calls, op=MPI.SUM)
    engine.destroy()

    if RANK == 0:
        print(f"  zero rate() calls over {steps} steps: "
              f"recompute={calls_r}, keep={calls_k}")
    assert calls_r >= steps, (
        f"non-flagged zero class must recompute every step: "
        f"{calls_r} calls < {steps} steps")
    assert calls_k * 2 < calls_r, (
        f"incremental_rate=True zero class must hold DONE rates: "
        f"keep={calls_k}, recompute={calls_r}")
    if RANK == 0:
        print("  PASSED: zero structural retention + class-level rate gating")


# ============================================================

ALL_TESTS = [
    test_kill_kernel,
    test_poison_mask_construction,
    test_zone_kernel_matches_full_subset,
    test_inplace_update_sent_semantics,
    test_incremental_equals_full,
    test_retention_and_rate_keep,
    test_lodged_runtime_steal_cases,
    test_lodged_predictive_move_cases,
    test_lodged_empty_host,
    test_engine_paranoid_verify,
    test_single_rate_incremental_flag,
    # 2026-07 复审修复
    test_force_rebuild_goes_full,
    test_random_migration_resamples,
    test_big_seed_engine,
    # 2026-07 复审二轮
    test_big_seed_builtin_ic,
    test_direction_entry_points,
    # 2026-07 契约统一：结构增量恒启用，全局开关废除
    test_single_structure_unconditional,
    test_zero_rate_incremental_flag,
]

if __name__ == '__main__':
    if RANK == 0:
        print("=" * 60)
        print(f"pair structural incremental tests (np={COMM.size})")
        print("=" * 60)
    failures = 0
    for t in ALL_TESTS:
        if RANK == 0:
            print(f"[{t.__name__}]")
        try:
            t()
        except Exception:
            failures += 1
            import traceback
            traceback.print_exc()
    failures = COMM.allreduce(failures, op=MPI.MAX)
    if RANK == 0:
        print("=" * 60)
        print("FAILED" if failures else "ALL PASSED")
    sys.exit(1 if failures else 0)
