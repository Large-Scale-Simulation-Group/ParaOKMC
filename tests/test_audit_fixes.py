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
审计修复回归测试（保留项）

覆盖 2026-07 代码审计中经复现证据确认并保留的修复：
1. PBC pair 距离缺最小镜像（cell list + KDTree 距离内核）
2. pair 输出缓冲区溢出静默丢失（真实计数 + 扩容重跑）
3. 异构 pair obj1/obj2 规范化作用在 NumPy 副本上（改为索引级交换）
4. 'deleted': True 契约失效 + size 未序列化
5. 对象 ID 方案确定性碰撞 / ID=0 幽灵事件 / IDGenerator 负 ID
6. pair 命名参数静默归零（含 merge_event_config 真实路径）
7. 负载均衡按非负 obj_ids 计数错误分池（per-cell 零事件）
8. ExecuteResult.new_obj() 缺 new=True / 引擎识别死代码
9. NaN/Inf 速率防线（保留项 B5）
10. ghost 邻接按 rank 区域最小距离（多层邻居 + RCB 后重算）
"""

import sys
import types
import numpy as np

sys.path.insert(0, '..')

from paraokmc.jit_cell_list import (
    build_neighbor_map, build_cell_bins, jit_find_pairs_cell_list,
)
from paraokmc.jit_event_builder import (
    build_match_tables, jit_filter_pairs_lut, jit_compute_distances_pbc,
    jit_count_valid_event_rows, jit_compact_pair_rows,
)
from paraokmc.rma_event_pool import (
    classify_event_rows, classify_event_row,
    CATEGORY_SINGLE, CATEGORY_PAIR, CATEGORY_ZERO,
    create_default_event_dtype,
)
from paraokmc.id_generator import IDGenerator, generate_global_id


# ============================================================
# 公共构造
# ============================================================

def _make_cell_list_inputs(x, y, z, box, periodic, max_dist):
    n = len(x)
    Gx = max(1, int(np.floor(box[0] / max_dist)))
    Gy = max(1, int(np.floor(box[1] / max_dist)))
    Gz = max(1, int(np.floor(box[2] / max_dist)))
    csx, csy, csz = box[0] / Gx, box[1] / Gy, box[2] / Gz

    nb_map, nb_counts = build_neighbor_map(
        Gx, Gy, Gz, max_dist, csx, csy, csz,
        periodic[0], periodic[1], periodic[2])
    heads, nxt = build_cell_bins(
        x, y, z, n, csx, csy, csz, Gx, Gy, Gz,
        box[0], box[1], box[2], periodic[0], periodic[1], periodic[2])
    return nb_map, nb_counts, heads, nxt


def _run_cell_list(x, y, z, obj_ids, type_ids, owners, box, periodic,
                   match_table, max_dist_table, max_dist, max_pairs=1024):
    n = len(x)
    nb_map, nb_counts, heads, nxt = _make_cell_list_inputs(
        x, y, z, box, periodic, max_dist)

    out_idx1 = np.empty(max_pairs, dtype=np.int64)
    out_idx2 = np.empty(max_pairs, dtype=np.int64)
    out_min = np.empty(max_pairs, dtype=np.int64)
    out_max = np.empty(max_pairs, dtype=np.int64)
    out_own = np.empty(max_pairs, dtype=np.int32)
    out_dist = np.empty(max_pairs, dtype=np.float64)
    out_et = np.empty(max_pairs, dtype=np.int32)

    n_found = jit_find_pairs_cell_list(
        x, y, z, obj_ids, type_ids, owners, n,
        heads, nxt, nb_map, nb_counts,
        match_table, max_dist_table,
        float(box[0]), float(box[1]), float(box[2]),
        bool(periodic[0]), bool(periodic[1]), bool(periodic[2]),
        np.int32(0),
        out_idx1, out_idx2, out_min, out_max, out_own, out_dist, out_et,
        max_pairs)
    n_write = min(n_found, max_pairs)
    return n_found, (out_idx1[:n_write], out_idx2[:n_write],
                     out_dist[:n_write], out_et[:n_write])


# ============================================================
# Test 1: PBC 最小镜像
# ============================================================

def test_pbc_minimum_image_cell_list():
    """跨周期边界、物理相邻的一对对象必须被找到，且距离正确"""
    box = (100.0, 100.0, 100.0)
    periodic = (True, True, True)
    max_dist = 5.0

    x = np.array([0.1, 99.9])
    y = np.array([50.0, 50.0])
    z = np.array([50.0, 50.0])
    obj_ids = np.array([1, 2], dtype=np.int64)
    type_ids = np.array([0, 1], dtype=np.int32)
    owners = np.zeros(2, dtype=np.int32)

    vf = {'capture': {'type_pairs': [('H', 'V')], 'max_distance': 5.0}}
    mt, mdt = build_match_tables(vf, {'H': 0, 'V': 1}, {'capture': 0})

    n_found, (i1, i2, dist, et) = _run_cell_list(
        x, y, z, obj_ids, type_ids, owners, box, periodic, mt, mdt, max_dist)

    assert n_found == 1, f"cross-boundary pair not found (n={n_found})"
    assert abs(dist[0] - 0.2) < 1e-10, f"min-image distance wrong: {dist[0]}"

    # 非周期时同一构型不应产生 pair
    n_found_np, _ = _run_cell_list(
        x, y, z, obj_ids, type_ids, owners, box,
        (False, False, False), mt, mdt, max_dist)
    assert n_found_np == 0, "non-periodic box must not wrap"
    print("  PASSED: cell list PBC minimum image")


def test_pbc_minimum_image_distance_kernel():
    """KDTree 回退路径的 PBC 距离内核"""
    box = (100.0, 100.0, 100.0)
    x1 = np.array([0.5]); y1 = np.array([99.5]); z1 = np.array([50.0])
    x2 = np.array([99.5]); y2 = np.array([0.5]); z2 = np.array([50.0])
    ids1 = np.array([7], dtype=np.int64)
    ids2 = np.array([3], dtype=np.int64)
    d = np.empty(1); mn = np.empty(1, dtype=np.int64); mx = np.empty(1, dtype=np.int64)

    jit_compute_distances_pbc(x1, y1, z1, x2, y2, z2, ids1, ids2,
                              box[0], box[1], box[2], True, True, True,
                              d, mn, mx)
    expected = np.sqrt(1.0**2 + 1.0**2)
    assert abs(d[0] - expected) < 1e-10, f"PBC distance wrong: {d[0]} vs {expected}"
    assert mn[0] == 3 and mx[0] == 7
    print("  PASSED: PBC distance kernel")


# ============================================================
# Test 2: 溢出返回真实计数
# ============================================================

def test_overflow_reports_true_count():
    """内核写满缓冲区后继续计数，调用方可据此扩容重跑"""
    box = (10.0, 10.0, 10.0)
    periodic = (False, False, False)
    max_dist = 5.0
    n = 20
    rng = np.random.default_rng(7)
    # 全部挤在一个小区域内 → C(20,2)=190 对 H-V 组合中约一半匹配
    x = rng.uniform(4.0, 5.0, n)
    y = rng.uniform(4.0, 5.0, n)
    z = rng.uniform(4.0, 5.0, n)
    obj_ids = np.arange(1, n + 1, dtype=np.int64)
    type_ids = (np.arange(n) % 2).astype(np.int32)  # H/V 交替
    owners = np.zeros(n, dtype=np.int32)

    vf = {'capture': {'type_pairs': [('H', 'V')], 'max_distance': 5.0}}
    mt, mdt = build_match_tables(vf, {'H': 0, 'V': 1}, {'capture': 0})

    # 足够大的缓冲区 → 得到真实数量
    n_true, _ = _run_cell_list(x, y, z, obj_ids, type_ids, owners,
                               box, periodic, mt, mdt, max_dist, max_pairs=4096)
    assert n_true == 100, f"expected 10*10=100 H-V pairs, got {n_true}"

    # 缓冲区不足 → 返回值仍是真实数量（供扩容重试）
    n_reported, (i1, _, _, _) = _run_cell_list(
        x, y, z, obj_ids, type_ids, owners,
        box, periodic, mt, mdt, max_dist, max_pairs=16)
    assert n_reported == n_true, \
        f"overflow must report true count: {n_reported} vs {n_true}"
    assert len(i1) == 16
    print("  PASSED: overflow reports true count")


# ============================================================
# Test 3: 同一类型对的多事件通道
# ============================================================

# ============================================================
# Test 4: 完整 pair 构建管线（规范化 / 用户 filter / 参数 / PBC）
# ============================================================

def _make_pipeline(config_extra=None, use_cell_list=True,
                   periodic=(True, True, True)):
    """构造最小可用的 VectorizedPairEventBuilder 环境（单 rank，无 MPI 依赖）"""
    from paraokmc.vectorized_event_builder import VectorizedPairEventBuilder

    obj_dtype = np.dtype([
        ('id', np.int64), ('type_id', np.int32),
        ('x', np.float64), ('y', np.float64), ('z', np.float64),
        ('owner_rank', np.int32), ('active', np.bool_),
        ('size', np.float64), ('float_props', np.float64, (4,)),
    ], align=True)

    base = create_default_event_dtype()
    event_dtype = np.dtype(base.descr + [('objects', obj_dtype, (2,))], align=True)

    grid_system = types.SimpleNamespace(periodic=list(periodic))
    domain_decomp = types.SimpleNamespace(
        comm=None, rank=0, size=1,
        max_interaction_distance=5.0,
        box_size=np.array([100.0, 100.0, 100.0]),
        grid_system=grid_system,
    )

    config = {
        '_type_to_int': {'H': 0, 'V': 1},
        '_int_to_type': {0: 'H', 1: 'V'},
        'pair_type_filters': {'capture': lambda *a, **k: True},
        'pair_param_extractors': {},
        'event_weights': {'capture': 1.0},
        '_event_name_to_int': {'capture': 0},
        '_event_int_to_name': {0: 'capture'},
        'vectorized_pair_filters': {
            'capture': {'type_pairs': [('H', 'V')], 'max_distance': 5.0},
        },
        'domain_decomposition': {
            'box_size': [100.0, 100.0, 100.0],
            'periodic': list(periodic),
        },
    }
    if config_extra:
        config.update(config_extra)

    builder = VectorizedPairEventBuilder(
        domain_decomp, config, obj_dtype, event_dtype,
        verbose=False, use_cell_list=use_cell_list)
    return builder, obj_dtype, event_dtype, config


def _make_pool(obj_dtype, rows):
    objs = np.zeros(len(rows), dtype=obj_dtype)
    for i, r in enumerate(rows):
        for k, v in r.items():
            objs[i][k] = v
        objs[i]['active'] = True
    pool = types.SimpleNamespace(
        large_count=0,
        get_local_objects=lambda: objs,
        get_ghost_objects=lambda: np.array([], dtype=obj_dtype),
    )
    return pool


def test_pipeline_normalization_and_pbc():
    """objects[0] 必须是 type_pairs 首类型；obj_ids 与 objects 对齐；跨界 pair 找到"""
    builder, obj_dtype, event_dtype, _ = _make_pipeline()

    # 对 1: (V at 0.1, H at 99.8) 跨界，V 在前 → 需要交换
    # 对 2: (H, V) 常规顺序
    pool = _make_pool(obj_dtype, [
        {'id': 10, 'type_id': 1, 'x': 0.1, 'y': 50.0, 'z': 50.0, 'owner_rank': 0},
        {'id': 20, 'type_id': 0, 'x': 99.8, 'y': 50.0, 'z': 50.0, 'owner_rank': 0},
        {'id': 30, 'type_id': 0, 'x': 40.0, 'y': 50.0, 'z': 50.0, 'owner_rank': 0},
        {'id': 40, 'type_id': 1, 'x': 41.0, 'y': 50.0, 'z': 50.0, 'owner_rank': 0},
    ])

    events = builder.build_pair_events(pool, current_step=0)
    assert len(events) == 2, f"expected 2 pair events (1 cross-boundary), got {len(events)}"

    for ev in events:
        o1, o2 = ev['objects'][0], ev['objects'][1]
        assert o1['type_id'] == 0 and o2['type_id'] == 1, \
            f"normalization failed: types {o1['type_id']}, {o2['type_id']}"
        # obj_ids 保持 (min, max) 规范（objects 槽位按类型规范化，两者独立）
        pair_ids = sorted([int(o1['id']), int(o2['id'])])
        assert list(ev['obj_ids'][:2]) == pair_ids, \
            f"obj_ids must be (min, max): {ev['obj_ids'][:2]} vs {pair_ids}"

    dists = sorted(float(ev['params'][0]) for ev in events)
    assert abs(dists[0] - 0.3) < 1e-9, f"cross-boundary distance wrong: {dists[0]}"
    assert abs(dists[1] - 1.0) < 1e-9
    print("  PASSED: pipeline normalization + PBC + obj_ids alignment")


def test_pipeline_kdtree_fallback_pbc():
    """KDTree 回退路径也必须找到跨界 pair"""
    builder, obj_dtype, _, _ = _make_pipeline(use_cell_list=False)

    pool = _make_pool(obj_dtype, [
        {'id': 10, 'type_id': 0, 'x': 0.1, 'y': 50.0, 'z': 50.0, 'owner_rank': 0},
        {'id': 20, 'type_id': 1, 'x': 99.8, 'y': 50.0, 'z': 50.0, 'owner_rank': 0},
    ])

    events = builder.build_pair_events(pool, current_step=0)
    assert len(events) == 1, f"KDTree fallback missed cross-boundary pair ({len(events)})"
    assert abs(float(events[0]['params'][0]) - 0.3) < 1e-9
    print("  PASSED: KDTree fallback PBC")


# ============================================================
# Test 5: pair 参数提取
# ============================================================

def test_param_extractor_named_params():
    """命名参数必须从对象属性 / 事件配置正确提取，不再静默归零"""
    from paraokmc.event_class import PairObjectEvent, EventRegistry
    from paraokmc.event_compiler import EventCompiler, compile_event_config

    class BindingCapture(PairObjectEvent):
        type_pairs = [('H', 'V')]
        max_distance = 5.0
        params = ['distance', 'E_b', 'r_capture']

        @classmethod
        def rate(cls, objs, params, config):
            return 1.0

        @classmethod
        def execute(cls, event, objs, config):
            return []

    config = {
        '_int_to_type': {0: 'H', 1: 'V'},
        '_props_mapping': {'float_props': {'H': {'r_capture': 1}, 'V': {}}},
        'events': {'capture': {'E_b': 0.75}},
        'domain_decomposition': {
            'box_size': [100.0, 100.0, 100.0],
            'periodic': [True, True, True],
        },
    }
    registry = compile_event_config({'capture': BindingCapture})
    compiler = EventCompiler(registry, config)
    _, extractor = compiler._compile_pair_functions('capture', BindingCapture)

    obj_dtype = np.dtype([
        ('id', np.int64), ('type_id', np.int32),
        ('x', np.float64), ('y', np.float64), ('z', np.float64),
        ('float_props', np.float64, (4,)),
    ], align=True)
    objs = np.zeros(2, dtype=obj_dtype)
    objs[0] = (1, 0, 0.5, 50.0, 50.0, (0.0, 2.5, 0.0, 0.0))   # H, r_capture=2.5
    objs[1] = (2, 1, 99.7, 50.0, 50.0, (0.0, 0.0, 0.0, 0.0))  # V, 跨界

    params = extractor(objs[0], objs[1])
    assert abs(params[0] - 0.8) < 1e-9, f"min-image distance wrong: {params[0]}"
    assert abs(params[1] - 0.75) < 1e-12, f"E_b from event config wrong: {params[1]}"
    assert abs(params[2] - 2.5) < 1e-12, f"r_capture from float_props wrong: {params[2]}"
    print("  PASSED: named pair params extracted")


# ============================================================
# Test 6: deleted 契约 / size 序列化 / 新对象 ID
# ============================================================

def _make_engine_stub():
    from paraokmc.v16_engine import OKMCEngineV16

    obj_dtype = np.dtype([
        ('id', np.int64), ('type_id', np.int32),
        ('x', np.float64), ('y', np.float64), ('z', np.float64),
        ('owner_rank', np.int32), ('active', np.bool_),
        ('size', np.float64), ('float_props', np.float64, (4,)),
        ('new', np.bool_), ('changed', np.bool_),
    ], align=True)

    stub = types.SimpleNamespace(
        rank=0, size=1, current_step=0,
        _new_obj_local_counter=0,
        object_dtype=obj_dtype,
        changed_object_dtype=obj_dtype,
        config={
            '_type_to_int': {'H': 0, 'V': 1, 'HV': 2},
            '_int_to_type': {0: 'H', 1: 'V', 2: 'HV'},
            '_props_mapping': {},
        },
    )
    stub._object_type_name = types.MethodType(
        OKMCEngineV16._object_type_name, stub)
    stub._build_complete_objects = types.MethodType(
        OKMCEngineV16._build_complete_objects, stub)
    stub._serialize_changed_objects = types.MethodType(
        OKMCEngineV16._serialize_changed_objects, stub)
    stub._warn_once = types.MethodType(OKMCEngineV16._warn_once, stub)
    return stub, obj_dtype


def test_deleted_contract():
    """'deleted': True（公开契约、库自带湮灭/捕获事件所用）必须真正删除"""
    stub, _ = _make_engine_stub()

    objs_data = [
        {'id': 11, 'type': 'H', 'x': 1.0, 'y': 2.0, 'z': 3.0},
        {'id': 22, 'type': 'V', 'x': 4.0, 'y': 5.0, 'z': 6.0},
    ]
    # 库自带 Annihilation 风格的返回
    changed = [
        {'id': 11, 'deleted': True},
        {'id': 22, 'deleted': True},
        {'id': -1, 'type': 'HV', 'x': 2.5, 'y': 3.5, 'z': 4.5, 'new': True},
    ]
    complete = stub._build_complete_objects(changed, objs_data)
    assert len(complete) == 3

    deletions = [o for o in complete if not o.get('active', True)]
    assert len(deletions) == 2, f"'deleted': True must map to active=False: {complete}"
    assert {o['id'] for o in deletions} == {11, 22}

    arr = stub._serialize_changed_objects(complete)
    inactive = arr[~arr['active']]
    assert len(inactive) == 2, "serialized deletions must have active=False"

    # 同一事件里创建又删除：应直接丢弃
    complete2 = stub._build_complete_objects(
        [{'id': -1, 'new': True, 'deleted': True, 'type': 'H',
          'x': 0, 'y': 0, 'z': 0}], [])
    assert complete2 == [], "new+deleted object must be dropped"
    print("  PASSED: 'deleted' contract")


def test_size_serialization():
    """size 必须随 changed_objects 广播，避免大对象被错误降级"""
    stub, _ = _make_engine_stub()

    objs_data = [{'id': 5, 'type': 'H', 'x': 1.0, 'y': 1.0, 'z': 1.0, 'size': 12.5}]
    changed = [{'id': 5, 'x': 2.0}]
    complete = stub._build_complete_objects(changed, objs_data)
    assert complete[0].get('size') == 12.5, "size must be carried into complete object"

    arr = stub._serialize_changed_objects(complete)
    assert abs(float(arr[0]['size']) - 12.5) < 1e-12, \
        f"size lost in serialization: {arr[0]['size']}"
    print("  PASSED: size serialization")


def test_new_object_id_scheme():
    """运行期新 ID：bit62 判别位 + 持久计数器；与 IC ID 空间正交"""
    stub, _ = _make_engine_stub()

    ids = []
    for step in (0, 0, 1, 1):
        stub.current_step = step
        out = stub._build_complete_objects(
            [{'id': -1, 'new': True, 'type': 'H', 'x': 0, 'y': 0, 'z': 0}], [])
        ids.append(out[0]['id'])

    assert len(set(ids)) == 4, f"runtime IDs must be unique across steps: {ids}"
    for i in ids:
        assert i > 0 and (i >> 62) & 1 == 1, f"runtime ID must set bit 62: {i:#x}"

    # IC 空间（config_system: (rank<<48)|counter；builtin_ics: (rank<<48)+1+seq）
    # bit 62 恒为 0 → 与运行期空间正交
    ic_id_rank0 = (0 << 48) | 1
    ic_id_rank1 = (1 << 48) | 1
    for ic in (ic_id_rank0, ic_id_rank1):
        assert (ic >> 62) & 1 == 0
        assert ic not in ids
    print("  PASSED: runtime object ID scheme")


def test_id_generator_positive():
    """IDGenerator 不再产生负 ID / 零 ID"""
    gen = IDGenerator(birth_rank=3, birth_step=17)
    ids = gen.generate_batch(['V'] * 5000 + ['H'] * 5000)
    assert np.all(ids > 0), f"negative/zero IDs remain: min={ids.min()}"
    assert len(np.unique(ids)) == len(ids)
    assert generate_global_id('vacancy', 0, 10, 0) > 0
    print("  PASSED: IDGenerator positive IDs")


def test_builtin_ic_ids():
    """builtin_ics 的 ID：无 0、跨 rank 正交"""
    from paraokmc.builtin_ics import create_random_ic
    ic = create_random_ic(['H'], {'H': 10}, [50.0, 50.0, 50.0], seed=1)
    objs_r0 = ic(0, 2, {})
    objs_r1 = ic(1, 2, {})
    ids0 = {o['id'] for o in objs_r0}
    ids1 = {o['id'] for o in objs_r1}
    assert 0 not in ids0, "rank 0 must not produce object ID 0"
    assert min(ids0) >= 1
    assert not (ids0 & ids1), "cross-rank IC ID collision"
    print("  PASSED: builtin IC IDs")


# ============================================================
# Test 7: 事件分池
# ============================================================

def test_classify_by_prefix():
    """per-cell 零事件（obj_ids[0]=cell_id>=0）必须留在 zero 池"""
    dtype = create_default_event_dtype()
    events = np.zeros(4, dtype=dtype)
    events['obj_ids'][:] = -1
    events['id'] = 0

    # single: prefix 0（id_high 含 step/rank/type 非零）
    events[0]['id_high'] = (0 << 61) | (5 << 26) | (1 << 10) | 2
    events[0]['obj_ids'][0] = 42

    # pair: prefix 1
    events[1]['id_high'] = (1 << 61) | (5 << 26)
    events[1]['obj_ids'][0] = 42
    events[1]['obj_ids'][1] = 43

    # per-cell zero: prefix 2，obj_ids[0] = cell_id（可为 0）
    events[2]['id_high'] = (2 << 61) | (5 << 26)
    events[2]['obj_ids'][0] = 0  # cell 0

    # 无 128 位 ID 的 legacy 事件 → 回退计数
    events[3]['id'] = 12345
    events[3]['id_high'] = 12345
    events[3]['obj_ids'][0] = 7

    cats = classify_event_rows(events)
    assert cats[0] == CATEGORY_SINGLE
    assert cats[1] == CATEGORY_PAIR
    assert cats[2] == CATEGORY_ZERO, \
        f"per-cell zero event misrouted to category {cats[2]}"
    assert cats[3] == CATEGORY_SINGLE  # legacy 回退：1 个非负 obj_id

    for i in range(4):
        assert classify_event_row(events[i]) == cats[i]
    print("  PASSED: prefix-based pool classification")


def test_per_cell_zero_obj_ids_slots():
    """per-cell 零事件 obj_ids[2:] 不再残留 0"""
    dtype = create_default_event_dtype()
    ev = np.zeros(3, dtype=dtype)
    # 模拟 VectorizedZeroEventBuilder 的填充方式
    ev['obj_ids'][:, 1:] = -1
    ev['obj_ids'][:, 0] = [0, 5, 9]  # cell ids
    ev['id_high'] = (2 << 61) | (1 << 26)
    ev['id'] = 0
    cats = classify_event_rows(ev)
    assert np.all(cats == CATEGORY_ZERO)
    print("  PASSED: per-cell zero obj_ids slots")


# ============================================================
# Test 8: 单对象用户 filter
# ============================================================

# ============================================================
# 第二轮审计修复（2026-07-11）
# ============================================================

def test_execute_result_new_obj():
    """ExecuteResult.new_obj() 必须产出 new=True，并走新 ID 分配分支"""
    from paraokmc.event_class import ExecuteResult

    result = ExecuteResult()
    result.new_obj('SIA', x=1.0, y=2.0, z=3.0)
    result.delete_obj(np.array([(7,)], dtype=[('id', np.int64)])[0])
    changes = result.to_list()

    assert changes[0].get('new') is True, "new_obj must set new=True"
    assert changes[1] == {'id': 7, 'active': False}

    # 引擎路径：新对象拿到 bit62 运行期 ID
    stub, _ = _make_engine_stub()
    complete = stub._build_complete_objects(result.to_list(), [])
    new_recs = [o for o in complete if o.get('new')]
    assert len(new_recs) == 1
    assert new_recs[0]['id'] > 0 and (new_recs[0]['id'] >> 62) & 1 == 1, \
        f"new_obj must get a runtime ID, got {new_recs[0]['id']}"
    print("  PASSED: ExecuteResult.new_obj")


def test_named_params_through_merge_path():
    """命名参数提取必须在真实 merge_event_config 路径生效（闭包捕获 result）"""
    from paraokmc.event_class import PairObjectEvent
    from paraokmc.event_compiler import merge_event_config

    class BindingCapture(PairObjectEvent):
        type_pairs = [('H', 'V')]
        max_distance = 5.0
        params = ['distance', 'E_b', 'r_capture']

        @classmethod
        def rate(cls, objs, params, config):
            return 1.0

        @classmethod
        def execute(cls, event, objs, config):
            return []

    base_config = {
        'events': {'capture': {'E_b': 0.75}},
        'domain_decomposition': {
            'box_size': [100.0, 100.0, 100.0],
            'periodic': [True, True, True],
        },
    }
    merged = merge_event_config(base_config, {'capture': BindingCapture})

    # 模拟引擎 init 的运行时注入（注入的是 merged，即 engine.config）
    merged['_props_mapping'] = {'float_props': {'H': {'r_capture': 1}, 'V': {}}}
    merged['_int_to_type'] = {0: 'H', 1: 'V'}

    extractor = merged['pair_param_extractors']['capture']

    obj_dtype = np.dtype([
        ('id', np.int64), ('type_id', np.int32),
        ('x', np.float64), ('y', np.float64), ('z', np.float64),
        ('float_props', np.float64, (4,)),
    ], align=True)
    objs = np.zeros(2, dtype=obj_dtype)
    objs[0] = (1, 0, 1.0, 50.0, 50.0, (0.0, 2.5, 0.0, 0.0))
    objs[1] = (2, 1, 2.0, 50.0, 50.0, (0.0, 0.0, 0.0, 0.0))

    params = extractor(objs[0], objs[1])
    assert abs(params[1] - 0.75) < 1e-12, f"E_b lost on merge path: {params[1]}"
    assert abs(params[2] - 2.5) < 1e-12, \
        f"float_props named param lost on merge path: {params[2]}"
    print("  PASSED: named params through merge_event_config path")


def test_nonfinite_rate_clamped():
    """NaN/Inf 速率必须被夹到 0，不得污染总速率"""
    from paraokmc.event_class import SingleObjectEvent, EventRegistry
    from paraokmc.v16_engine import OKMCEngineV16

    class BadRate(SingleObjectEvent):
        object_types = ['H']
        params = []

        @classmethod
        def rate(cls, objs, params, config):
            return float('inf') if int(objs[0]['id']) == 1 else float('nan')

        @classmethod
        def execute(cls, event, objs, config):
            return []

    registry = EventRegistry()
    registry.register('bad', BadRate)

    obj_dtype = np.dtype([
        ('id', np.int64), ('type_id', np.int32),
        ('x', np.float64), ('y', np.float64), ('z', np.float64),
        ('owner_rank', np.int32), ('active', np.bool_),
        ('float_props', np.float64, (4,)),
    ], align=True)
    event_dtype = np.dtype(
        create_default_event_dtype().descr + [('objects', obj_dtype, (2,))],
        align=True)

    # 两个单对象事件，速率分别为 inf / nan
    pool_events = np.zeros(2, dtype=event_dtype)
    for i, oid in enumerate((1, 2)):
        pool_events[i]['obj_ids'][:] = -1
        pool_events[i]['obj_ids'][0] = oid
        pool_events[i]['event_type_id'] = 0
        pool_events[i]['objects'][0]['id'] = oid
        pool_events[i]['objects'][0]['active'] = True
        pool_events[i]['objects'][1]['id'] = -1

    rate_sums = {}
    fake_pool = types.SimpleNamespace(
        zero_obj_array=np.zeros(0, dtype=event_dtype), count_zero=0,
        single_obj_array=pool_events, count_single=2,
        pair_obj_array=np.zeros(0, dtype=event_dtype), count_pair=0,
        add_rate_sum_delta=lambda cat, d: rate_sums.__setitem__(
            cat, rate_sums.get(cat, 0.0) + d),
    )
    # 模拟"刚收到 LB 事件"的场景：check_incoming_tasks 返回非零
    fake_lb = types.SimpleNamespace(check_incoming_tasks=lambda pool: 1)
    stub = types.SimpleNamespace(
        rank=0, event_registry=registry,
        rma_event_pool=fake_pool,
        runtime_load_balancer=fake_lb,
        config={'_event_int_to_name': {0: 'bad'}, '_props_mapping': {}},
    )
    stub._warn_once = types.MethodType(OKMCEngineV16._warn_once, stub)
    stub._resolve_event_type_name = types.MethodType(
        OKMCEngineV16._resolve_event_type_name, stub)
    stub._get_objects_for_event = types.MethodType(
        OKMCEngineV16._get_objects_for_event, stub)
    stub._object_record_to_user_dict = types.MethodType(
        OKMCEngineV16._object_record_to_user_dict, stub)
    stub._object_type_name = types.MethodType(
        OKMCEngineV16._object_type_name, stub)
    stub._compute_received_events = types.MethodType(
        OKMCEngineV16._compute_received_events, stub)

    stub._compute_received_events()

    assert np.all(pool_events['status'] == 2)
    assert np.all(pool_events['rate'] == 0.0), \
        f"non-finite rates must clamp to 0: {pool_events['rate']}"
    assert np.isfinite(rate_sums.get('single', 0.0))
    print("  PASSED: non-finite rates clamped")


def test_param_truncation_follows_dtype():
    """参数截断上限跟随动态 dtype（不再硬编码 8）"""
    from paraokmc.event_class import PairObjectEvent
    from paraokmc.event_compiler import EventCompiler, compile_event_config

    names = ['distance'] + [f'p{i}' for i in range(9)]  # 10 个参数

    class ManyParams(PairObjectEvent):
        type_pairs = [('H', 'V')]
        max_distance = 5.0
        params = names

        @classmethod
        def rate(cls, objs, params, config):
            return 1.0

        @classmethod
        def execute(cls, event, objs, config):
            return []

    wide_dtype = np.dtype([
        ('id', np.int64), ('params', np.float64, (12,)),
    ], align=True)
    config = {
        '_event_dtype': wide_dtype,
        '_int_to_type': {0: 'H', 1: 'V'},
        'events': {'many': {f'p{i}': float(i + 1) for i in range(9)}},
        'domain_decomposition': {
            'box_size': [100.0, 100.0, 100.0],
            'periodic': [True, True, True],
        },
    }
    compiler = EventCompiler(compile_event_config({'many': ManyParams}), config)
    _, extractor = compiler._compile_pair_functions('many', ManyParams)

    obj_dtype = np.dtype([
        ('id', np.int64), ('type_id', np.int32),
        ('x', np.float64), ('y', np.float64), ('z', np.float64),
    ], align=True)
    objs = np.zeros(2, dtype=obj_dtype)
    objs[0] = (1, 0, 1.0, 50.0, 50.0)
    objs[1] = (2, 1, 2.0, 50.0, 50.0)

    params = extractor(objs[0], objs[1])
    assert len(params) == 10, f"10 params must survive (got {len(params)})"
    assert abs(params[9] - 9.0) < 1e-12, f"param p8 lost: {params}"
    print("  PASSED: param truncation follows dtype capacity")


# ============================================================
# 第四轮：ghost 邻接判据（rank 区域最小距离）
# ============================================================

def test_ghost_ranks_multilayer():
    """子域宽度 < r_cut 时必须包含两层以外的邻居（旧实现硬编码一层）"""
    from paraokmc.domain_decomposition import compute_ghost_ranks_from_bounds

    # 4 个 rank 沿 x 均分 box=100（宽 25），r_cut=30 > 25
    bounds = [
        {'x_min': 25.0 * r, 'x_max': 25.0 * (r + 1),
         'y_min': 0.0, 'y_max': 100.0, 'z_min': 0.0, 'z_max': 100.0}
        for r in range(4)
    ]
    box = [100.0, 100.0, 100.0]

    # 周期：rank0 与 rank2 间隙 25 < 30 → 邻居；rank3 经环绕相邻
    ghosts = compute_ghost_ranks_from_bounds(0, bounds, box, [True, True, True], 30.0)
    assert ghosts == [1, 2, 3], f"periodic thin-slab neighbors wrong: {ghosts}"

    # 非周期：rank0 → {1, 2}（rank3 间隙 50 > 30）
    ghosts_np = compute_ghost_ranks_from_bounds(0, bounds, box, [False, False, False], 30.0)
    assert ghosts_np == [1, 2], f"non-periodic thin-slab neighbors wrong: {ghosts_np}"

    # r_cut=10 < 25：退回单层行为（周期下 rank0 → {1, 3}）
    ghosts_thin = compute_ghost_ranks_from_bounds(0, bounds, box, [True, True, True], 10.0)
    assert ghosts_thin == [1, 3], f"single-layer case wrong: {ghosts_thin}"
    print("  PASSED: multi-layer ghost ranks")


def test_ghost_ranks_rcb_layout():
    """RCB 不规则划分：邻接必须按实际边界盒计算（旧实现依赖初始规则网格）"""
    from paraokmc.domain_decomposition import compute_ghost_ranks_from_bounds

    # 不规则 3-rank 划分：rank1 是夹在 0/2 之间的细条
    bounds = [
        {'x_min': 0.0, 'x_max': 10.0,
         'y_min': 0.0, 'y_max': 100.0, 'z_min': 0.0, 'z_max': 100.0},
        {'x_min': 10.0, 'x_max': 13.0,
         'y_min': 0.0, 'y_max': 100.0, 'z_min': 0.0, 'z_max': 100.0},
        {'x_min': 13.0, 'x_max': 100.0,
         'y_min': 0.0, 'y_max': 100.0, 'z_min': 0.0, 'z_max': 100.0},
    ]
    box = [100.0, 100.0, 100.0]

    # r_cut=5：rank0 与 rank2 间隙 3 < 5 → 跨过细条 rank1 相邻；
    # rank0 与 rank2 又经周期环绕直接接触
    ghosts = compute_ghost_ranks_from_bounds(0, bounds, box, [True, True, True], 5.0)
    assert ghosts == [1, 2], f"RCB layout neighbors wrong: {ghosts}"

    # 非周期下同样成立（靠间隙 3 < 5）
    ghosts_np = compute_ghost_ranks_from_bounds(0, bounds, box, [False, False, False], 5.0)
    assert ghosts_np == [1, 2], f"RCB layout (non-periodic) wrong: {ghosts_np}"

    # r_cut=2：rank0 只与 rank1 相邻（gap 到 rank2 为 3；周期环绕 rank2 直接接触？
    # rank0 x=[0,10)、rank2 x=[13,100)，环绕间隙 = 100-(100-0) = 0 → 接触）
    ghosts_small = compute_ghost_ranks_from_bounds(0, bounds, box, [True, True, True], 2.0)
    assert ghosts_small == [1, 2], f"wrap-contact case wrong: {ghosts_small}"
    print("  PASSED: RCB-layout ghost ranks")


# ============================================================
# 主入口
# ============================================================

ALL_TESTS = [
    test_pbc_minimum_image_cell_list,
    test_pbc_minimum_image_distance_kernel,
    test_overflow_reports_true_count,
    test_pipeline_normalization_and_pbc,
    test_pipeline_kdtree_fallback_pbc,
    test_param_extractor_named_params,
    test_deleted_contract,
    test_size_serialization,
    test_new_object_id_scheme,
    test_id_generator_positive,
    test_builtin_ic_ids,
    test_classify_by_prefix,
    test_per_cell_zero_obj_ids_slots,
    test_execute_result_new_obj,
    test_named_params_through_merge_path,
    test_nonfinite_rate_clamped,
    test_param_truncation_follows_dtype,
    test_ghost_ranks_multilayer,
    test_ghost_ranks_rcb_layout,
]

if __name__ == '__main__':
    print("=" * 60)
    print("Audit fix regression tests")
    print("=" * 60)
    failures = 0
    for t in ALL_TESTS:
        print(f"[{t.__name__}]")
        try:
            t()
        except Exception as exc:
            failures += 1
            import traceback
            traceback.print_exc()
            print(f"  FAILED: {exc}")
    print("=" * 60)
    if failures:
        print(f"{failures}/{len(ALL_TESTS)} tests FAILED")
        sys.exit(1)
    print(f"All {len(ALL_TESTS)} tests passed!")
