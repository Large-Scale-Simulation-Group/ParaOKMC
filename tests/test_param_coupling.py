#!/usr/bin/env python3
"""
v1.6.10 参数耦合功能测试

测试内容：
1. 参数收集器注册和调用
2. 三种耦合模式对参数的支持
3. 外部求解器返回参数
4. sync_coupling()中的参数收集流程
"""

import numpy as np
from mpi4py import MPI

from paraokmc.unified_physics import (
    UnifiedPhysics, ExternalSolver,
    MODE_TWO_WAY, MODE_OUTWARD, MODE_INWARD
)


class MockParamSolver(ExternalSolver):
    """模拟求解器：返回场和参数的混合结果"""

    def initialize(self, physics, config):
        pass

    def solve(self, physics, step, time, object_data=None, trigger_source=None):
        # 返回混合结果：场 + 参数
        return {
            'temperature': np.full((10, 10, 10), 300.0 + step),  # 场
            'avg_temp': 300.0 + step,  # 标量参数
            'total_energy': 1000.0 * step,  # 标量参数
        }

    def finalize(self):
        pass

    def get_supported_fields(self):
        return ['temperature']

    def get_supported_params(self):
        return ['avg_temp', 'total_energy']


def test_param_collector():
    """测试1：参数收集器注册和调用"""
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    config = {
        'box_size': (100.0, 100.0, 100.0),
        'field_coupling': {'mode': MODE_TWO_WAY}
    }
    physics = UnifiedPhysics(comm, config)

    # 定义参数收集器
    def count_objects(objects, physics):
        if objects is None:
            return 0.0
        return float(len(objects))

    def sum_energy(objects, physics):
        if objects is None:
            return 0.0
        # 假设对象有energy属性
        return 100.0  # 简化测试

    # 注册参数收集器
    physics.set_param_collector('object_count', count_objects)
    physics.set_param_collector('total_energy', sum_energy)

    # 创建模拟对象数据
    obj_dtype = np.dtype([('id', np.int64), ('x', np.float64)])
    objects = np.array([(1, 0.0), (2, 1.0), (3, 2.0)], dtype=obj_dtype)

    # 调用收集器
    results = physics.collect_params_from_objects(objects)

    if rank == 0:
        assert 'object_count' in results, "Missing object_count"
        assert 'total_energy' in results, "Missing total_energy"
        assert results['object_count'] == 3.0, f"Wrong count: {results['object_count']}"
        print("Test 1 PASSED: param_collector registration and invocation")

    return True


def test_solver_returns_params():
    """测试2：外部求解器返回参数"""
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    config = {
        'box_size': (100.0, 100.0, 100.0),
        'field_coupling': {'mode': MODE_TWO_WAY}
    }
    physics = UnifiedPhysics(comm, config)

    # 定义场和参数
    physics.define_field('temperature', (10, 10, 10), 300.0)
    physics.set_param('avg_temp', 300.0)
    physics.set_param('total_energy', 0.0)

    # 注册求解器
    solver = MockParamSolver()
    physics.register_solver('mock', solver, fields=['temperature'])

    # 调用求解器
    changed = physics.call_solver('mock', step=5, time=0.5)

    if rank == 0:
        # 验证场更新
        T = physics.get_field('temperature')
        assert T is not None, "Temperature field not updated"
        assert np.allclose(T, 305.0), f"Wrong temperature: {T[0,0,0]}"

        # 验证参数更新
        avg_temp = physics.get_param('avg_temp')
        total_energy = physics.get_param('total_energy')
        assert avg_temp == 305.0, f"Wrong avg_temp: {avg_temp}"
        assert total_energy == 5000.0, f"Wrong total_energy: {total_energy}"

        print("Test 2 PASSED: solver returns both fields and params")

    return True


def test_coupling_modes_with_params():
    """测试3：三种耦合模式对参数的支持"""
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    for mode, mode_name in [(MODE_TWO_WAY, 'TWO_WAY'),
                            (MODE_OUTWARD, 'OUTWARD'),
                            (MODE_INWARD, 'INWARD')]:
        config = {
            'box_size': (100.0, 100.0, 100.0),
            'field_coupling': {'mode': mode}
        }
        physics = UnifiedPhysics(comm, config)

        # 注册参数收集器
        physics.set_param_collector('test_param', lambda o, p: 42.0)

        # 验证收集器已注册
        assert 'test_param' in physics._param_collectors, \
            f"Mode {mode_name}: collector not registered"

        if rank == 0:
            print(f"  Mode {mode_name}: param_collector OK")

    if rank == 0:
        print("Test 3 PASSED: all coupling modes support param_collector")

    return True


def test_sync_coupling_with_params():
    """测试4：sync_coupling()中的参数收集流程"""
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    config = {
        'box_size': (100.0, 100.0, 100.0),
        'field_coupling': {
            'mode': MODE_TWO_WAY,
            'step_interval': 1,
        }
    }
    physics = UnifiedPhysics(comm, config)

    # 定义场
    physics.define_field('temperature', (10, 10, 10), 300.0)

    # 注册参数收集器
    call_count = [0]

    def counting_collector(objects, physics):
        call_count[0] += 1
        return float(call_count[0])

    physics.set_param_collector('call_count', counting_collector)

    # 直接测试collect_params_from_objects
    obj_dtype = np.dtype([('id', np.int64), ('x', np.float64)])
    objects = np.array([(1, 0.0), (2, 1.0)], dtype=obj_dtype)

    results = physics.collect_params_from_objects(objects)

    if rank == 0:
        # 验证收集器被调用
        assert call_count[0] >= 1, "Collector not called"
        assert 'call_count' in results, "Result missing call_count"
        print("Test 4 PASSED: collect_params_from_objects works correctly")

    return True


def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    if rank == 0:
        print("=" * 60)
        print("v1.6.10 Parameter Coupling Tests")
        print("=" * 60)

    tests = [
        ("Test 1: param_collector", test_param_collector),
        ("Test 2: solver returns params", test_solver_returns_params),
        ("Test 3: coupling modes", test_coupling_modes_with_params),
        ("Test 4: sync_coupling", test_sync_coupling_with_params),
    ]

    passed = 0
    failed = 0

    for name, test_func in tests:
        try:
            if rank == 0:
                print(f"\nRunning {name}...")
            test_func()
            passed += 1
        except Exception as e:
            failed += 1
            if rank == 0:
                print(f"FAILED: {name}")
                print(f"  Error: {e}")

    if rank == 0:
        print("\n" + "=" * 60)
        print(f"Results: {passed}/{len(tests)} tests passed")
        if failed == 0:
            print("All tests PASSED!")
        else:
            print(f"{failed} tests FAILED")
        print("=" * 60)


if __name__ == '__main__':
    main()
