"""
v1.6.8 事件触发场更新测试

测试场景：
1. 创建物理模块，注册事件触发
2. 模拟cascade事件触发温度场更新
3. 验证场更新被正确触发
"""

import numpy as np
from mpi4py import MPI
from paraokmc.unified_physics import (
    UnifiedPhysics, create_unified_physics,
    ExternalSolver, MODE_TWO_WAY
)
from paraokmc.rma_object_pool import create_default_object_dtype
from typing import Dict, List, Optional

# ===== 测试用求解器 =====

class TestHeatSolver(ExternalSolver):
    """测试用热传导求解器"""

    def __init__(self):
        self.call_count = 0
        self.last_trigger_source = None

    def initialize(self, physics: UnifiedPhysics, config: dict) -> None:
        pass

    def solve(self, physics: UnifiedPhysics,
              step: int,
              time: float,
              object_data: Optional[np.ndarray] = None,
              trigger_source: Optional[str] = None) -> Optional[Dict[str, np.ndarray]]:

        self.call_count += 1
        self.last_trigger_source = trigger_source

        if physics.rank != 0:
            return None

        # 获取温度场形状
        if 'temperature' in physics._field_name_to_idx:
            idx = physics._field_name_to_idx['temperature']
            shape = physics.fields[idx].shape
            # 模拟温度场更新：基础温度 + 时间衰减的热spike
            base_temp = 500.0
            spike = 100.0 * np.exp(-time / 10.0) if trigger_source == 'cascade' else 0.0
            return {'temperature': np.full(shape, base_temp + spike)}

        return None

    def finalize(self) -> None:
        pass

    def get_supported_fields(self) -> List[str]:
        return ['temperature']


class DummyObjectPool:
    """最小对象池桩，用于测试对象查询/快照服务。"""

    def __init__(self, objects: np.ndarray, type_map: Optional[Dict[str, int]] = None):
        self.dtype = objects.dtype
        self._objects = objects
        self._type_to_int = dict(type_map or {})
        self._int_to_type = {v: k for k, v in self._type_to_int.items()}

    def get_local_objects(self, active_only: bool = True) -> np.ndarray:
        objs = self._objects
        if active_only and 'active' in objs.dtype.names:
            objs = objs[objs['active']]
        return objs.copy()

    def get_type_name(self, obj) -> str:
        if 'type_id' in obj.dtype.names:
            return self._int_to_type.get(int(obj['type_id']), '')
        return ''


def _make_objects():
    dtype = create_default_object_dtype()
    objs = np.zeros(3, dtype=dtype)
    objs['active'] = True
    objs['owner_rank'] = 0
    objs['type_id'] = np.array([1, 2, 1], dtype=np.int32)
    objs['id'] = np.array([10, 11, 12], dtype=np.int64)
    objs['x'] = np.array([1.0, 2.0, 3.0])
    objs['y'] = np.array([1.0, 2.0, 3.0])
    objs['z'] = np.array([1.0, 2.0, 3.0])
    return objs


# ===== 测试函数 =====

def test_event_trigger_registration():
    """测试事件触发注册"""
    comm = MPI.COMM_WORLD
    rank = comm.rank

    config = {
        'box_size': [50.0, 50.0, 50.0],
        'verbose': False,
    }

    physics = UnifiedPhysics(comm, config)

    # 注册事件触发
    physics.register_event_trigger('cascade', ['temperature', 'stress'])
    physics.register_event_trigger(
        'precipitation',
        ['concentration'],
        condition=lambda e: e.get('size', 0) > 10
    )

    # 验证注册
    assert 'cascade' in physics._event_triggers
    assert 'precipitation' in physics._event_triggers
    assert physics._event_triggers['cascade']['fields'] == ['temperature', 'stress']

    if rank == 0:
        print("[Test 1] Event trigger registration: PASSED")

    return True


def test_event_trigger_notification():
    """测试事件触发通知"""
    comm = MPI.COMM_WORLD
    rank = comm.rank

    config = {
        'box_size': [50.0, 50.0, 50.0],
        'verbose': False,
    }

    physics = UnifiedPhysics(comm, config)

    # 定义场和注册触发
    physics.define_field('temperature', (10, 10, 10))
    physics.register_event_trigger('cascade', ['temperature'])

    # 验证初始状态
    assert not physics.has_pending_triggers()

    # 通知事件执行
    physics.notify_event_executed('cascade', {'defect_count': 100})

    # 验证待处理触发
    assert physics.has_pending_triggers()
    assert 'temperature' in physics._pending_triggered_fields

    if rank == 0:
        print("[Test 2] Event trigger notification: PASSED")

    return True


def test_event_trigger_with_condition():
    """测试带条件的事件触发"""
    comm = MPI.COMM_WORLD
    rank = comm.rank

    config = {
        'box_size': [50.0, 50.0, 50.0],
        'verbose': False,
    }

    physics = UnifiedPhysics(comm, config)

    # 注册带条件的触发
    physics.register_event_trigger(
        'precipitation',
        ['concentration'],
        condition=lambda e: e.get('size', 0) > 10
    )

    # 小尺寸不触发
    physics.notify_event_executed('precipitation', {'size': 5})
    assert not physics.has_pending_triggers()

    # 大尺寸触发
    physics.notify_event_executed('precipitation', {'size': 15})
    assert physics.has_pending_triggers()

    if rank == 0:
        print("[Test 3] Event trigger with condition: PASSED")

    return True


def test_event_trigger_with_solver():
    """测试事件触发调用求解器"""
    comm = MPI.COMM_WORLD
    rank = comm.rank

    config = {
        'box_size': [50.0, 50.0, 50.0],
        'verbose': False,
    }

    physics = UnifiedPhysics(comm, config)

    # 定义场
    physics.define_field('temperature', (10, 10, 10), initial_value=300.0)

    # 注册求解器
    solver = TestHeatSolver()
    physics.register_solver('heat', solver, fields=['temperature'])

    # 注册事件触发
    physics.register_event_trigger('cascade', ['temperature'], solver='heat')

    # 通知事件执行
    physics.notify_event_executed('cascade', {'defect_count': 100})

    # 处理待触发更新
    assert physics.has_pending_triggers()
    changed = physics.process_pending_triggers(step=1, time=0.1, object_data=None)

    # 验证求解器被调用
    assert solver.call_count == 1
    assert solver.last_trigger_source == 'cascade'
    assert changed  # 场应该更新了

    # 验证温度场已更新（从300变为~600因为有热spike）
    temp_field = physics.get_field('temperature')
    if rank == 0:
        assert temp_field[0, 0, 0] > 500.0  # 应该有热spike

    if rank == 0:
        print("[Test 4] Event trigger with solver: PASSED")

    return True


def test_event_trigger_clears_pending():
    """测试处理后清空待处理列表"""
    comm = MPI.COMM_WORLD
    rank = comm.rank

    config = {
        'box_size': [50.0, 50.0, 50.0],
        'verbose': False,
    }

    physics = UnifiedPhysics(comm, config)
    physics.define_field('temperature', (10, 10, 10))
    physics.register_event_trigger('cascade', ['temperature'])

    # 触发并处理
    physics.notify_event_executed('cascade', {})
    assert physics.has_pending_triggers()

    physics.process_pending_triggers(step=1, time=0.1, object_data=None)
    assert not physics.has_pending_triggers()  # 应该清空

    if rank == 0:
        print("[Test 5] Pending triggers cleared: PASSED")

    return True


def test_field_update_with_object_snapshots():
    """测试 field update_func 可以收到对象快照并更新场。"""
    comm = MPI.COMM_WORLD

    physics = UnifiedPhysics(comm, {
        'box_size': [20.0, 20.0, 20.0],
        'field_names': ['stress'],
        'field_shapes': [(4, 4, 4)],
        'verbose': False,
    })
    physics.bind_object_pool(DummyObjectPool(_make_objects(), {'A': 1, 'B': 2}))

    captured = {}

    def update_stress(physics_obj, snapshots, ctx):
        captured['keys'] = sorted(snapshots.keys())
        captured['defects_len'] = len(snapshots['defects'])
        captured['a_only_len'] = len(snapshots['a_only'])
        captured['trigger_reason'] = ctx.trigger_reason
        return np.full((4, 4, 4), float(len(snapshots['defects'])))

    physics.set_field_update(
        'stress',
        update_stress,
        queries={
            'defects': {'all': True},
            'a_only': {'types': ['A']},
        },
        update_policy={
            'trigger_events': ['cascade'],
            'trigger_on_object_change': True,
        }
    )

    physics.notify_event_executed('cascade', {
        'changed_summary': {
            'created_count': 0,
            'changed_count': 1,
            'deleted_count': 0,
            'created_types': [],
            'changed_types': ['A'],
            'deleted_types': [],
        }
    })
    changed = physics.process_pending_triggers(step=2, time=0.5, object_data=None)

    assert changed
    assert captured['keys'] == ['a_only', 'defects']
    assert captured['defects_len'] == 3
    assert captured['a_only_len'] == 2
    assert captured['trigger_reason'] == 'object_change'
    assert physics.get_field('stress')[0, 0, 0] == 3.0


def test_field_update_policy_step_interval():
    """测试按 step_interval 的强制更新逻辑。"""
    comm = MPI.COMM_WORLD

    physics = UnifiedPhysics(comm, {
        'box_size': [20.0, 20.0, 20.0],
        'field_names': ['temperature'],
        'field_shapes': [(3, 3, 3)],
        'verbose': False,
    })
    physics.bind_object_pool(DummyObjectPool(_make_objects(), {'A': 1, 'B': 2}))

    call_steps = []

    def update_temperature(physics_obj, snapshots, ctx):
        call_steps.append(ctx.step)
        return np.full((3, 3, 3), float(ctx.step))

    physics.set_field_update(
        'temperature',
        update_temperature,
        step_interval=2,
        queries={'all_objs': {'all': True}},
        update_policy={'step_interval': 2},
    )

    _, field_flags, _ = physics.check_updates(step=1, time=0.1)
    assert not np.any(field_flags)

    _, field_flags, _ = physics.check_updates(step=2, time=0.2)
    assert np.any(field_flags)

    changed = physics.apply_updates(
        step=2,
        time=0.2,
        param_flags=np.zeros(4, dtype=np.uint64),
        field_flags=field_flags,
        global_flag=False,
    )

    assert changed
    assert call_steps == [2]
    assert physics.get_field('temperature')[0, 0, 0] == 2.0


# ===== 主测试 =====

if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    rank = comm.rank

    if rank == 0:
        print("=" * 60)
        print("v1.6.8 Event Trigger Tests")
        print("=" * 60)

    tests = [
        test_event_trigger_registration,
        test_event_trigger_notification,
        test_event_trigger_with_condition,
        test_event_trigger_with_solver,
        test_event_trigger_clears_pending,
    ]

    all_passed = True
    for test in tests:
        try:
            test()
        except AssertionError as e:
            if rank == 0:
                print(f"FAILED: {test.__name__}")
                print(f"  Error: {e}")
            all_passed = False
        except Exception as e:
            if rank == 0:
                print(f"ERROR: {test.__name__}")
                print(f"  {type(e).__name__}: {e}")
            all_passed = False

    if rank == 0:
        print("=" * 60)
        if all_passed:
            print("All tests PASSED!")
        else:
            print("Some tests FAILED!")
        print("=" * 60)
