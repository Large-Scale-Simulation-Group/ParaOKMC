"""
测试 per_cell 零对象事件

验证：
1. ZeroObjectEvent.spatial_mode='per_cell' 属性
2. EventCompiler 正确分流 per_cell vs global
3. VectorizedZeroEventBuilder 构建 per_cell 事件
4. JIT 内核正确填充 params
5. 与 global 零对象事件共存
"""

import numpy as np
from numba import njit
from paraokmc.event_class import ZeroObjectEvent, SingleObjectEvent, compile_event_config
from paraokmc.event_compiler import compile_events, merge_event_config
from paraokmc.jit_rate_engine import RateDeps


# ============================================================================
# JIT 内核
# ============================================================================

@njit
def _cascade_rate_kernel(cascade_rate):
    """Cascade 常数速率内核"""
    return cascade_rate


# ============================================================================
# 测试事件类定义
# ============================================================================

class CascadeEvent(ZeroObjectEvent):
    """全局零对象事件（传统模式）"""
    spatial_mode = 'global'
    params = ['cascade_rate']
    weight = 1.0

    # JIT 声明
    rate_deps = RateDeps(physics=['cascade_rate'])
    rate_kernel = _cascade_rate_kernel

    @classmethod
    def rate(cls, objs, params, config):
        """标量速率"""
        return float(params.get('cascade_rate', 1e5))

    @classmethod
    def execute(cls, event, objs, config):
        return [{'id': -1, 'type': 'SIA', 'new': True,
                 'x': 50.0, 'y': 50.0, 'z': 50.0}]


class DislocationNucleation(ZeroObjectEvent):
    """per_cell 零对象事件"""
    spatial_mode = 'per_cell'
    field_deps = ['stress_xx']
    params = ['stress', 'cell_x', 'cell_y', 'cell_z']
    weight = 2.0

    @classmethod
    def rate(cls, objs, params, config):
        """标量速率：应力阈值逻辑"""
        stress = params.get('stress', 0.0)
        threshold = 1e9
        if stress > threshold:
            return float(1e10 * np.exp(-(threshold / stress)))
        return 0.0

    @classmethod
    def execute(cls, event, objs, config):
        cell_x = float(event['params'][1])
        cell_y = float(event['params'][2])
        cell_z = float(event['params'][3])
        return [{'id': -1, 'type': 'dislocation', 'new': True,
                 'x': cell_x, 'y': cell_y, 'z': cell_z}]


class SimpleMigration(SingleObjectEvent):
    """简单迁移事件（用于验证共存）"""
    object_types = ['H']
    params = ['E_m']
    weight = 1.0

    @classmethod
    def rate(cls, objs, params, config):
        """标量速率：常数 1e10"""
        return 1e10

    @classmethod
    def execute(cls, event, objs, config):
        return [{'id': int(objs[0]['id']), 'x': float(objs[0]['x']) + 0.1}]


# ============================================================================
# 测试
# ============================================================================

def test_1_class_attributes():
    """测试 ZeroObjectEvent 新属性"""
    print("\n[Test 1] ZeroObjectEvent class attributes")

    assert CascadeEvent.spatial_mode == 'global'
    assert CascadeEvent.field_deps == []
    print("  CascadeEvent: spatial_mode='global', field_deps=[] ✓")

    assert DislocationNucleation.spatial_mode == 'per_cell'
    assert DislocationNucleation.field_deps == ['stress_xx']
    print("  DislocationNucleation: spatial_mode='per_cell', field_deps=['stress_xx'] ✓")

    # 默认值
    class DefaultZero(ZeroObjectEvent):
        params = []
        @classmethod
        def rate(cls, objs, params, config):
            return 1.0
        @classmethod
        def execute(cls, event, objs, config):
            return []

    assert DefaultZero.spatial_mode == 'global'
    assert DefaultZero.field_deps == []
    print("  DefaultZero: defaults correct ✓")
    print("  PASSED")


def test_2_compiler_routing():
    """测试 EventCompiler 正确分流 per_cell vs global"""
    print("\n[Test 2] EventCompiler routing")

    base_config = {
        'physics': {'temperature': 500.0},
        'events': {
            'cascade': {'cascade_rate': 1e5},
        },
    }

    compiled, registry = compile_events({
        'cascade': CascadeEvent,
        'dislocation_nucleation': DislocationNucleation,
        'migration': SimpleMigration,
    }, base_config)

    # cascade 应该在 zero_obj_builders（global 模式）
    assert 'cascade' in compiled['zero_obj_builders'], \
        "cascade should be in zero_obj_builders"
    print("  cascade → zero_obj_builders ✓")

    # dislocation_nucleation 不应该在 zero_obj_builders（per_cell 模式）
    assert 'dislocation_nucleation' not in compiled['zero_obj_builders'], \
        "dislocation_nucleation should NOT be in zero_obj_builders"
    print("  dislocation_nucleation → NOT in zero_obj_builders ✓")

    # dislocation_nucleation 应该在 vectorized_zero_filters
    assert 'dislocation_nucleation' in compiled['vectorized_zero_filters'], \
        "dislocation_nucleation should be in vectorized_zero_filters"
    vzf = compiled['vectorized_zero_filters']['dislocation_nucleation']
    assert vzf['spatial_mode'] == 'per_cell'
    assert vzf['field_deps'] == ['stress_xx']
    print(f"  vectorized_zero_filters: {vzf} ✓")

    # migration 应该在 single_obj_builders
    assert 'migration' in compiled['single_obj_builders']
    print("  migration → single_obj_builders ✓")

    # 权重
    assert compiled['event_weights']['cascade'] == 1.0
    assert compiled['event_weights']['dislocation_nucleation'] == 2.0
    print("  event_weights correct ✓")

    print("  PASSED")


def test_3_merge_config():
    """测试 merge_event_config 传递 vectorized_zero_filters"""
    print("\n[Test 3] merge_event_config")

    base_config = {
        'physics': {'temperature': 500.0},
        'domain_decomposition': {
            'lattice_constant': 10.0,
            'lattice_grid': [10, 10, 10],
            'h_factor': 1,
            'box_size': [100, 100, 100],
        },
    }

    merged = merge_event_config(base_config, {
        'cascade': CascadeEvent,
        'dislocation_nucleation': DislocationNucleation,
    })

    assert 'vectorized_zero_filters' in merged
    assert 'dislocation_nucleation' in merged['vectorized_zero_filters']
    assert 'cascade' not in merged['vectorized_zero_filters']
    assert 'cascade' in merged['zero_obj_builders']
    print("  vectorized_zero_filters propagated ✓")
    print("  PASSED")


def test_4_rate_function():
    """测试 per_cell 事件的标量速率函数"""
    print("\n[Test 4] per_cell scalar rate()")

    # 模拟 params: [stress, cell_x, cell_y, cell_z]
    n = 100
    stress_vals = np.linspace(0, 2e9, n)

    config = {'physics': {'temperature': 500.0}}

    # 用标量 rate() 逐个计算
    rates = np.array([
        DislocationNucleation.rate(
            [], {'stress': s, 'cell_x': 0, 'cell_y': 0, 'cell_z': 0}, config)
        for s in stress_vals
    ])

    assert len(rates) == n
    # 低应力区域速率应为 0
    low_stress_mask = stress_vals <= 1e9
    assert np.all(rates[low_stress_mask] == 0.0), "Low stress should have zero rate"
    # 高应力区域速率应 > 0
    high_stress_mask = stress_vals > 1e9
    if np.any(high_stress_mask):
        assert np.all(rates[high_stress_mask] > 0.0), "High stress should have positive rate"
    print(f"  Rates: min={rates.min():.2e}, max={rates.max():.2e}")
    print(f"  Zero-rate cells: {np.sum(rates == 0)}/{n}")
    print("  PASSED")


def test_5_jit_kernel():
    """测试 JIT 内核填充"""
    print("\n[Test 5] jit_build_per_cell_zero_events kernel")

    from paraokmc.jit_event_builder import jit_build_per_cell_zero_events

    n_cells = 200
    n_field_deps = 1
    max_params = 8

    cell_ids = np.arange(n_cells, dtype=np.int64)
    cx = np.random.rand(n_cells) * 100
    cy = np.random.rand(n_cells) * 100
    cz = np.random.rand(n_cells) * 100
    field_values = np.random.rand(n_cells, n_field_deps) * 2e9

    out_oid = np.empty(n_cells, dtype=np.int64)
    out_params = np.zeros((n_cells, max_params), dtype=np.float64)

    jit_build_per_cell_zero_events(cell_ids, cx, cy, cz, field_values,
                                    out_oid, out_params, n_field_deps)

    # Verify
    assert np.array_equal(out_oid, cell_ids)
    assert np.allclose(out_params[:, 0], field_values[:, 0])
    assert np.allclose(out_params[:, 1], cx)
    assert np.allclose(out_params[:, 2], cy)
    assert np.allclose(out_params[:, 3], cz)
    print(f"  {n_cells} cells, {n_field_deps} field deps ✓")
    print("  PASSED")


if __name__ == '__main__':
    print("=" * 60)
    print("Per-Cell Zero Object Event - Test Suite")
    print("=" * 60)

    test_1_class_attributes()
    test_2_compiler_routing()
    test_3_merge_config()
    test_4_rate_function()
    test_5_jit_kernel()

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)
