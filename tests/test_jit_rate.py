"""
test_jit_rate.py — JIT 双 Launcher 端到端测试

测试内容：
1. JIT 事件类定义（capture + migration）
2. EventRegistry JIT 检测
3. JITRateDispatcher 注册和计算
4. JIT vs NumPy 向量化正确性对比
5. 性能基准测试

运行方式：
    python test_jit_rate.py              # 单进程
    mpirun -np 2 python test_jit_rate.py # MPI
"""

import numpy as np
import time
import sys

from mpi4py import MPI

from numba import njit

from paraokmc.event_class import (
    SingleObjectEvent, PairObjectEvent,
    EventRegistry, compile_event_config,
    get_physics_param,
)
from paraokmc.jit_rate_engine import (
    RateDeps, JITRateDispatcher, build_type_to_int,
)

comm = MPI.COMM_WORLD
rank = comm.rank
size = comm.size


# ============================================================================
# 1. 定义带 JIT 内核的事件类
# ============================================================================

# ---- Capture 事件（pair，JIT 加速）----

@njit
def capture_jit_kernel(o1_diffusivity, o1_radius, o2_radius,
                       distance, temperature, k_B):
    """扩散控制捕获速率内核"""
    r_capture = o1_radius + o2_radius
    if distance < r_capture:
        return 0.0
    return 4.0 * 3.14159265 * o1_diffusivity * r_capture


class JITCaptureEvent(PairObjectEvent):
    """带 JIT 内核的捕获事件"""
    type_pairs = [('H', 'V'), ('SIA', 'V')]
    max_distance = 5.0
    params = ['distance']
    product_type = 'HV'

    # JIT 声明
    rate_deps = RateDeps(
        obj1=['diffusivity', 'radius'],
        obj2=['radius'],
        physics=['temperature', 'k_B'],
    )
    rate_kernel = capture_jit_kernel

    @classmethod
    def execute(cls, event, objs, config):
        obj1, obj2 = objs[0], objs[1]
        new_x = (float(obj1['x']) + float(obj2['x'])) / 2
        new_y = (float(obj1['y']) + float(obj2['y'])) / 2
        new_z = (float(obj1['z']) + float(obj2['z'])) / 2
        return [
            {'id': int(obj1['id']), 'deleted': True},
            {'id': int(obj2['id']), 'deleted': True},
            {'id': -1, 'type': 'HV', 'x': new_x, 'y': new_y, 'z': new_z, 'new': True},
        ]


# ---- Migration 事件（single，JIT 加速）----

@njit
def migration_jit_kernel(E_m, nu, temperature, k_B):
    """Arrhenius 迁移速率内核"""
    return nu * np.exp(-E_m / (k_B * temperature))


class JITMigrationEvent(SingleObjectEvent):
    """带 JIT 内核的迁移事件"""
    object_types = ['H', 'V', 'SIA']
    params = ['E_m', 'nu']

    # JIT 声明
    rate_deps = RateDeps(
        obj1=['E_m', 'nu'],
        physics=['temperature', 'k_B'],
    )
    rate_kernel = migration_jit_kernel

    @classmethod
    def execute(cls, event, objs, config):
        obj = objs[0]
        dx = np.random.uniform(-1.0, 1.0)
        dy = np.random.uniform(-1.0, 1.0)
        dz = np.random.uniform(-1.0, 1.0)
        return [{'id': int(obj['id']),
                 'x': float(obj['x']) + dx,
                 'y': float(obj['y']) + dy,
                 'z': float(obj['z']) + dz}]


# ---- 普通事件（无 JIT，对照组）----

class PlainMigrationEvent(SingleObjectEvent):
    """无 JIT 的迁移事件（对照组）"""
    object_types = ['H', 'V']
    params = ['E_m', 'nu']

    @classmethod
    def rate(cls, objs, params, config):
        """标量 Arrhenius 速率"""
        T = get_physics_param(config, 'temperature', default=500.0)
        k_B = 8.617e-5
        E_m = params.get('E_m', 0.5)
        nu = params.get('nu', 1e13)
        return float(nu * np.exp(-E_m / (k_B * T)))

    @classmethod
    def execute(cls, event, objs, config):
        return [{'id': int(objs[0]['id'])}]


# ============================================================================
# 2. 测试
# ============================================================================

def test_registry_jit_detection():
    """Test 1: EventRegistry 正确检测 JIT 事件"""
    registry = EventRegistry()
    registry.register('jit_capture', JITCaptureEvent)
    registry.register('jit_migration', JITMigrationEvent)
    registry.register('plain_migration', PlainMigrationEvent)

    jit_events = registry.get_jit_events()
    assert 'jit_capture' in jit_events, f"Expected jit_capture in JIT events: {jit_events}"
    assert 'jit_migration' in jit_events, f"Expected jit_migration in JIT events: {jit_events}"
    assert 'plain_migration' not in jit_events, f"plain_migration should not be JIT"

    assert registry.is_jit('jit_capture')
    assert registry.is_jit('jit_migration')
    assert not registry.is_jit('plain_migration')

    # JIT 事件不在 python_events 中（它们走 JIT 路径）
    assert not registry.is_python('jit_capture')
    assert not registry.is_python('jit_migration')

    return True


def test_jit_dispatcher_pair():
    """Test 2: JIT Dispatcher pair 事件正确性"""
    props_mapping = {
        'float_props': {
            'H':   {'diffusivity': 2, 'radius': 0, 'E_m': 3, 'nu': 4},
            'V':   {'diffusivity': 2, 'radius': 0, 'E_m': 3, 'nu': 4},
            'SIA': {'diffusivity': 2, 'radius': 0, 'E_m': 3, 'nu': 4},
        }
    }
    type_to_int = build_type_to_int(props_mapping)

    disp = JITRateDispatcher(props_mapping, type_to_int)
    mode = disp.register('capture', JITCaptureEvent.rate_deps,
                         JITCaptureEvent.rate_kernel,
                         type_pairs=JITCaptureEvent.type_pairs)

    N = 10000
    np.random.seed(42 + rank)
    N_COLS = 10
    o1_props = np.random.rand(N, N_COLS)
    o2_props = np.random.rand(N, N_COLS)
    o1_tids = np.zeros(N, dtype=np.int32)  # H
    o2_tids = np.full(N, type_to_int['V'], dtype=np.int32)
    dists = np.random.rand(N) * 5.0 + 0.1
    field_vals = np.empty((N, 0), dtype=np.float64)
    phys_vals = np.array([500.0, 8.617e-5], dtype=np.float64)

    # JIT 计算
    rates_jit = disp.compute_rates(
        'capture',
        o1_props=o1_props, o2_props=o2_props,
        o1_type_ids=o1_tids, o2_type_ids=o2_tids,
        distances=dists, field_vals=field_vals,
        physics_vals=phys_vals)

    # NumPy 参考
    D = o1_props[:, 2]  # diffusivity at col 2
    r1 = o1_props[:, 0]  # radius at col 0
    r2 = o2_props[:, 0]
    r_cap = r1 + r2
    rates_ref = 4.0 * 3.14159265 * D * r_cap
    rates_ref[dists < r_cap] = 0.0

    max_diff = np.max(np.abs(rates_jit - rates_ref))
    assert max_diff < 1e-10, f"Pair correctness FAILED: max_diff={max_diff}"

    return mode, max_diff


def test_jit_dispatcher_single():
    """Test 3: JIT Dispatcher single 事件正确性"""
    props_mapping = {
        'float_props': {
            'H':   {'E_m': 3, 'nu': 4, 'diffusivity': 2, 'radius': 0},
            'V':   {'E_m': 3, 'nu': 4, 'diffusivity': 2, 'radius': 0},
            'SIA': {'E_m': 3, 'nu': 4, 'diffusivity': 2, 'radius': 0},
        }
    }
    type_to_int = build_type_to_int(props_mapping)

    disp = JITRateDispatcher(props_mapping, type_to_int)
    mode = disp.register('migration', JITMigrationEvent.rate_deps,
                         JITMigrationEvent.rate_kernel)

    N = 10000
    np.random.seed(123 + rank)
    N_COLS = 10
    o1_props = np.random.rand(N, N_COLS)
    # 设置合理的 E_m 和 nu 值
    o1_props[:, 3] = np.random.uniform(0.3, 0.8, N)  # E_m
    o1_props[:, 4] = 1e13  # nu
    o1_tids = np.zeros(N, dtype=np.int32)
    field_vals = np.empty((N, 0), dtype=np.float64)
    phys_vals = np.array([500.0, 8.617e-5], dtype=np.float64)

    # JIT 计算
    rates_jit = disp.compute_rates(
        'migration',
        o1_props=o1_props, o1_type_ids=o1_tids,
        field_vals=field_vals, physics_vals=phys_vals)

    # NumPy 参考
    E_m = o1_props[:, 3]
    nu = o1_props[:, 4]
    T = 500.0
    k_B = 8.617e-5
    rates_ref = nu * np.exp(-E_m / (k_B * T))

    max_diff = np.max(np.abs(rates_jit - rates_ref))
    rel_diff = np.max(np.abs((rates_jit - rates_ref) / (rates_ref + 1e-30)))
    assert rel_diff < 1e-10, f"Single correctness FAILED: rel_diff={rel_diff}"

    return mode, max_diff


def test_performance():
    """Test 4: 性能基准测试"""
    N = 200_000
    N_COLS = 20

    props_mapping = {
        'float_props': {
            'H':   {'diffusivity': 2, 'radius': 0},
            'V':   {'diffusivity': 2, 'radius': 0},
            'SIA': {'diffusivity': 2, 'radius': 0},
        }
    }
    type_to_int = build_type_to_int(props_mapping)

    disp = JITRateDispatcher(props_mapping, type_to_int)
    disp.register('capture', JITCaptureEvent.rate_deps,
                  JITCaptureEvent.rate_kernel,
                  type_pairs=JITCaptureEvent.type_pairs)

    np.random.seed(42)
    o1_props = np.random.rand(N, N_COLS)
    o2_props = np.random.rand(N, N_COLS)
    o1_tids = np.zeros(N, dtype=np.int32)
    o1_tids[int(N * 0.6):] = type_to_int.get('SIA', 2)
    o2_tids = np.full(N, type_to_int['V'], dtype=np.int32)
    dists = np.random.rand(N) * 5.0 + 0.1
    field_vals = np.empty((N, 0), dtype=np.float64)
    phys_vals = np.array([500.0, 8.617e-5], dtype=np.float64)

    # Warmup
    _ = disp.compute_rates('capture',
                           o1_props=o1_props[:100], o2_props=o2_props[:100],
                           o1_type_ids=o1_tids[:100], o2_type_ids=o2_tids[:100],
                           distances=dists[:100], field_vals=field_vals[:100],
                           physics_vals=phys_vals)

    # JIT benchmark
    n_runs = 5
    times_jit = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        _ = disp.compute_rates('capture',
                               o1_props=o1_props, o2_props=o2_props,
                               o1_type_ids=o1_tids, o2_type_ids=o2_tids,
                               distances=dists, field_vals=field_vals,
                               physics_vals=phys_vals)
        times_jit.append(time.perf_counter() - t0)

    # NumPy benchmark
    fm = props_mapping['float_props']
    times_np = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        rates = np.zeros(N)
        for t1_name, t2_name in [('H', 'V'), ('SIA', 'V')]:
            t1_int = type_to_int[t1_name]
            t2_int = type_to_int[t2_name]
            mask = (o1_tids == t1_int) & (o2_tids == t2_int)
            D = o1_props[mask, fm[t1_name]['diffusivity']]
            r1 = o1_props[mask, fm[t1_name]['radius']]
            r2 = o2_props[mask, fm[t2_name]['radius']]
            d = dists[mask]
            r_cap = r1 + r2
            r = 4.0 * 3.14159265 * D * r_cap
            r[d < r_cap] = 0.0
            rates[mask] = r
        times_np.append(time.perf_counter() - t0)

    t_jit = np.median(times_jit)
    t_np = np.median(times_np)
    speedup = t_np / t_jit if t_jit > 0 else float('inf')

    return N, t_jit, t_np, speedup


# ============================================================================
# Main
# ============================================================================

if __name__ == '__main__':
    passed = 0
    failed = 0

    if rank == 0:
        print("=" * 60)
        print(f"JIT Dual Launcher Integration Test (rank {rank}/{size})")
        print("=" * 60)

    # Test 1
    if rank == 0:
        print("\n[Test 1] EventRegistry JIT detection")
    try:
        ok = test_registry_jit_detection()
        if rank == 0:
            print("  PASSED")
        passed += 1
    except Exception as e:
        if rank == 0:
            print(f"  FAILED: {e}")
        failed += 1

    # Test 2
    if rank == 0:
        print("\n[Test 2] JIT Dispatcher pair event correctness")
    try:
        mode, diff = test_jit_dispatcher_pair()
        if rank == 0:
            print(f"  Mode: {mode}, max_diff: {diff:.2e}")
            print("  PASSED")
        passed += 1
    except Exception as e:
        if rank == 0:
            print(f"  FAILED: {e}")
        failed += 1

    # Test 3
    if rank == 0:
        print("\n[Test 3] JIT Dispatcher single event correctness")
    try:
        mode, diff = test_jit_dispatcher_single()
        if rank == 0:
            print(f"  Mode: {mode}, max_diff: {diff:.2e}")
            print("  PASSED")
        passed += 1
    except Exception as e:
        if rank == 0:
            print(f"  FAILED: {e}")
        failed += 1

    # Test 4
    if rank == 0:
        print("\n[Test 4] Performance benchmark")
    try:
        N, t_jit, t_np, speedup = test_performance()
        if rank == 0:
            print(f"  N = {N:,}")
            print(f"  JIT:   {t_jit*1000:.2f} ms")
            print(f"  NumPy: {t_np*1000:.2f} ms")
            print(f"  Speedup: {speedup:.2f}x")
            print("  PASSED")
        passed += 1
    except Exception as e:
        if rank == 0:
            print(f"  FAILED: {e}")
            import traceback
            traceback.print_exc()
        failed += 1

    # Summary
    if rank == 0:
        print("\n" + "=" * 60)
        print(f"Results: {passed} passed, {failed} failed")
        print("=" * 60)

    if failed > 0:
        sys.exit(1)
