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
exec 相位热点修复的语义回归（2026-07 profiling）

1. remove_events_for_objects：花式索引全量拷贝 + 整池回写 + O(N) 重算
   → swap-remove 原地删除 + 速率增量扣减。本测试锚定新实现与旧语义等价：
   - 保留集合正确（被删对象的事件消失，其余原样保留）
   - 保留事件的 status/rate 逐位不变
   - rate_sum_single 与全量重算一致
2. _plan_changed_actions_jit 串行化后行为不变（由引擎测试覆盖，这里只
   做单条记录的烟测）。
"""

import sys
import numpy as np
from mpi4py import MPI

sys.path.insert(0, '..')

from paraokmc.rma_event_pool import (
    RMAEventPool, create_default_event_dtype, STATUS_TODO, STATUS_DONE,
)


def _make_pool(n_events):
    pool = RMAEventPool(MPI.COMM_WORLD, capacity_single=n_events + 16)
    events = np.zeros(n_events, dtype=pool.dtype)
    events['obj_ids'][:] = -1
    events['obj_ids'][:, 0] = np.arange(1, n_events + 1)
    events['status'] = STATUS_DONE
    events['rate'] = np.arange(1, n_events + 1, dtype=np.float64)
    # 混入几个 TODO（不计入速率和）
    events['status'][::7] = STATUS_TODO
    pool.set_single_obj_events(events)
    return pool, events


def _expected_rate_sum(pool):
    ev = pool.single_obj_array[:pool.count_single]
    m = (ev['status'] == STATUS_DONE) & (ev['rate'] >= 0.0)
    return float(np.sum(ev['rate'][m], dtype=np.float64))


def test_remove_semantics_small_m():
    pool, _ = _make_pool(50)
    before_ids = set(pool.single_obj_array[:50]['obj_ids'][:, 0].tolist())
    before_records = {
        int(e['obj_ids'][0]): (int(e['status']), float(e['rate']))
        for e in pool.single_obj_array[:50]
    }

    pool.remove_events_for_objects(np.array([10, 20, 50], dtype=np.int64))

    assert pool.count_single == 47, f"count wrong: {pool.count_single}"
    after = pool.single_obj_array[:pool.count_single]
    after_ids = set(after['obj_ids'][:, 0].tolist())
    assert after_ids == before_ids - {10, 20, 50}, "kept set wrong"

    # 保留事件的 status/rate 原样（顺序允许改变）
    for e in after:
        oid = int(e['obj_ids'][0])
        assert (int(e['status']), float(e['rate'])) == before_records[oid], \
            f"record for obj {oid} mutated"

    # 速率账目与全量重算一致
    assert abs(pool.rate_sum_single - _expected_rate_sum(pool)) < 1e-9, \
        f"rate accounting drifted: {pool.rate_sum_single} vs {_expected_rate_sum(pool)}"
    pool.destroy()
    print("  PASSED: swap-remove semantics (small M)")


def test_remove_semantics_large_m():
    """M >= 100 走 np.isin 分支"""
    pool, _ = _make_pool(500)
    remove = np.arange(2, 300, 2, dtype=np.int64)  # 149 个
    pool.remove_events_for_objects(remove)

    after = pool.single_obj_array[:pool.count_single]
    assert pool.count_single == 500 - len(remove)
    assert not (set(after['obj_ids'][:, 0].tolist()) & set(remove.tolist()))
    assert abs(pool.rate_sum_single - _expected_rate_sum(pool)) < 1e-9
    pool.destroy()
    print("  PASSED: swap-remove semantics (large M / isin)")


def test_remove_noop():
    """删除不存在的对象：零副作用"""
    pool, _ = _make_pool(30)
    rate_before = pool.rate_sum_single
    snapshot = pool.single_obj_array[:30].copy()

    pool.remove_events_for_objects(np.array([9999], dtype=np.int64))

    assert pool.count_single == 30
    assert np.array_equal(pool.single_obj_array[:30], snapshot)
    assert pool.rate_sum_single == rate_before
    pool.destroy()
    print("  PASSED: nonexistent-id no-op")


def test_plan_kernel_serial_smoke():
    """串行化后的 _plan_changed_actions_jit 单条记录行为烟测"""
    from paraokmc.rma_object_pool import (
        _plan_changed_actions_jit, ACTION_DELETE, ACTION_NEW, ACTION_UPDATE,
    )
    ownership = np.zeros(8, dtype=np.int32)
    n = 3
    x = np.array([1.0, 1.0, 1.0]); y = x.copy(); z = x.copy()
    size = np.zeros(n)
    active = np.array([False, True, True])
    new = np.array([False, True, False])
    a = np.empty(n, np.int32); o = np.empty(n, np.int32)
    c = np.empty(n, np.int32); v = np.empty(n, np.bool_)
    _plan_changed_actions_jit(
        x, y, z, size, active, new, ownership,
        np.int32(2), np.int32(2), np.int32(2),
        2.0, 2.0, 2.0, 1.0, 1.0, 1.0,
        True, True, True,
        0.0, 2.0, 0.0, 2.0, 0.0, 2.0,
        0.5, 1.0, np.int32(0),
        a, o, c, v)
    assert a[0] == ACTION_DELETE and a[1] == ACTION_NEW and a[2] == ACTION_UPDATE
    print("  PASSED: serialized plan kernel smoke")


if __name__ == '__main__':
    print("=" * 60)
    print("Hotspot fix semantics tests")
    print("=" * 60)
    test_remove_semantics_small_m()
    test_remove_semantics_large_m()
    test_remove_noop()
    test_plan_kernel_serial_smoke()
    print("All hotspot tests passed!")
