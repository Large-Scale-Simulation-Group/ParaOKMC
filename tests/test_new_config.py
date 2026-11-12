"""
测试新的 7 块配置系统
"""

import pytest
import numpy as np
from numba import njit

from paraokmc import OKMCengine, ConfigValidationError
from paraokmc import SingleObjectEvent, PairObjectEvent, ZeroObjectEvent
from paraokmc import RateDeps, delete_obj, change_obj
from paraokmc.user_config import compile_user_config, validate_user_config


@njit
def simple_rate_kernel(temperature):
    return 1e13 * np.exp(-0.3 / (8.617e-5 * temperature))


class SimpleMigration(SingleObjectEvent):
    object_types = ['A']
    rate_deps = RateDeps(physics=['temperature'])
    rate_kernel = simple_rate_kernel

    @classmethod
    def execute(cls, event, objs, config):
        obj = objs[0]
        return [change_obj(obj, x=float(obj['x']) + 1.0)]


class SimpleReaction(PairObjectEvent):
    type_pairs = [('A', 'B')]
    max_distance = 2.0
    rate_deps = RateDeps()

    @classmethod
    def rate(cls, objs, params, config):
        return 1e10

    @classmethod
    def execute(cls, event, objs, config):
        return [delete_obj(objs[0]), delete_obj(objs[1])]


def test_minimal_config():
    """测试最小配置"""
    config = {
        "domain": {
            "lattice_constant": 1.0,
            "lattice_grid": [10, 10, 10],
            "periodic": [True, True, True],
            "max_interaction_distance": 3.0,
        },
        "simulation": {
            "max_steps": 10,
        },
        "objects": {
            "A": {"size": 1.0},
        },
        "ic": {
            "mode": "builtin",
            "initial_conditions": {
                "A": {"count": 5, "distribution": "random"},
            },
        },
        "events": {
            "migration": SimpleMigration,
        },
        "physics": {
            "params": {"temperature": 300.0},
        },
        "outputs": {},
    }

    engine = OKMCengine(config)
    assert engine is not None
    engine.destroy()


def test_validation_missing_block():
    """测试缺少必需块"""
    config = {
        "domain": {
            "lattice_constant": 1.0,
            "lattice_grid": [10, 10, 10],
            "periodic": [True, True, True],
            "max_interaction_distance": 3.0,
        },
        # 缺少 simulation
    }

    with pytest.raises(ConfigValidationError, match="simulation"):
        OKMCengine(config)


def test_validation_invalid_lattice_constant():
    """测试无效的 lattice_constant"""
    config = {
        "domain": {
            "lattice_constant": -1.0,  # 负数
            "lattice_grid": [10, 10, 10],
            "periodic": [True, True, True],
            "max_interaction_distance": 3.0,
        },
        "simulation": {"max_steps": 10},
        "objects": {"A": {}},
        "ic": {"mode": "builtin", "initial_conditions": {}},
        "events": {},
        "physics": {},
        "outputs": {},
    }

    with pytest.raises(ConfigValidationError, match="lattice_constant"):
        OKMCengine(config)


def test_validation_unknown_object_type_in_ic():
    """测试 IC 中引用未定义的对象类型"""
    config = {
        "domain": {
            "lattice_constant": 1.0,
            "lattice_grid": [10, 10, 10],
            "periodic": [True, True, True],
            "max_interaction_distance": 3.0,
        },
        "simulation": {"max_steps": 10},
        "objects": {"A": {}},
        "ic": {
            "mode": "builtin",
            "initial_conditions": {
                "B": {"count": 5},  # B 未定义
            },
        },
        "events": {},
        "physics": {},
        "outputs": {},
    }

    with pytest.raises(ConfigValidationError, match="unknown object type"):
        OKMCengine(config)


def test_validation_unknown_object_type_in_event():
    """测试事件中引用未定义的对象类型"""
    class BadEvent(SingleObjectEvent):
        object_types = ['C']  # C 未定义

        @classmethod
        def rate(cls, objs, params, config):
            return 1.0

        @classmethod
        def execute(cls, event, objs, config):
            return []

    config = {
        "domain": {
            "lattice_constant": 1.0,
            "lattice_grid": [10, 10, 10],
            "periodic": [True, True, True],
            "max_interaction_distance": 3.0,
        },
        "simulation": {"max_steps": 10},
        "objects": {"A": {}},
        "ic": {"mode": "builtin", "initial_conditions": {}},
        "events": {"bad": BadEvent},
        "physics": {},
        "outputs": {},
    }

    with pytest.raises(ConfigValidationError, match="not defined in config"):
        OKMCengine(config)


def test_validation_missing_physics_param():
    """测试事件依赖缺失的 physics 参数"""
    class EventNeedTemp(SingleObjectEvent):
        object_types = ['A']
        rate_deps = RateDeps(physics=['temperature'])

        @classmethod
        def rate(cls, objs, params, config):
            return 1.0

        @classmethod
        def execute(cls, event, objs, config):
            return []

    config = {
        "domain": {
            "lattice_constant": 1.0,
            "lattice_grid": [10, 10, 10],
            "periodic": [True, True, True],
            "max_interaction_distance": 3.0,
        },
        "simulation": {"max_steps": 10},
        "objects": {"A": {"properties": {}}},
        "ic": {"mode": "builtin", "initial_conditions": {}},
        "events": {"need_temp": EventNeedTemp},
        "physics": {"params": {}},  # 空 params，缺少 temperature
        "outputs": {},
    }

    with pytest.raises(ConfigValidationError, match="temperature.*params"):
        OKMCengine(config)


def test_validation_missing_object_property():
    """测试事件依赖缺失的对象属性"""
    class EventNeedEm(SingleObjectEvent):
        object_types = ['A']
        rate_deps = RateDeps(obj1=['E_m'])

        @classmethod
        def rate(cls, objs, params, config):
            return 1.0

        @classmethod
        def execute(cls, event, objs, config):
            return []

    config = {
        "domain": {
            "lattice_constant": 1.0,
            "lattice_grid": [10, 10, 10],
            "periodic": [True, True, True],
            "max_interaction_distance": 3.0,
        },
        "simulation": {"max_steps": 10},
        "objects": {"A": {"properties": {}}},  # 缺少 E_m
        "ic": {"mode": "builtin", "initial_conditions": {}},
        "events": {"need_em": EventNeedEm},
        "physics": {},
        "outputs": {},
    }

    with pytest.raises(ConfigValidationError, match="E_m.*properties"):
        OKMCengine(config)


def test_validation_missing_object_property_no_properties():
    """测试对象没有 properties 字段时的严格校验"""
    class EventNeedEm(SingleObjectEvent):
        object_types = ['A']
        rate_deps = RateDeps(obj1=['E_m'])

        @classmethod
        def rate(cls, objs, params, config):
            return 1.0

        @classmethod
        def execute(cls, event, objs, config):
            return []

    config = {
        "domain": {
            "lattice_constant": 1.0,
            "lattice_grid": [10, 10, 10],
            "periodic": [True, True, True],
            "max_interaction_distance": 3.0,
        },
        "simulation": {"max_steps": 10},
        "objects": {"A": {}},  # 完全没有 properties
        "ic": {"mode": "builtin", "initial_conditions": {}},
        "events": {"need_em": EventNeedEm},
        "physics": {},
        "outputs": {},
    }

    with pytest.raises(ConfigValidationError, match="E_m.*properties"):
        OKMCengine(config)


def test_validation_missing_physics_param_empty_physics():
    """测试 physics 为空时的严格校验"""
    class EventNeedTemp(SingleObjectEvent):
        object_types = ['A']
        rate_deps = RateDeps(physics=['temperature'])

        @classmethod
        def rate(cls, objs, params, config):
            return 1.0

        @classmethod
        def execute(cls, event, objs, config):
            return []

    config = {
        "domain": {
            "lattice_constant": 1.0,
            "lattice_grid": [10, 10, 10],
            "periodic": [True, True, True],
            "max_interaction_distance": 3.0,
        },
        "simulation": {"max_steps": 10},
        "objects": {"A": {"properties": {}}},
        "ic": {"mode": "builtin", "initial_conditions": {}},
        "events": {"need_temp": EventNeedTemp},
        "physics": {},  # 空 physics
        "outputs": {},
    }

    with pytest.raises(ConfigValidationError, match="physics.*params"):
        OKMCengine(config)


def test_pair_event_missing_max_distance():
    """测试 PairObjectEvent 缺少 max_distance"""
    class BadPairEvent(PairObjectEvent):
        type_pairs = [('A', 'B')]
        # 缺少 max_distance

        @classmethod
        def rate(cls, objs, params, config):
            return 1.0

        @classmethod
        def execute(cls, event, objs, config):
            return []

    config = {
        "domain": {
            "lattice_constant": 1.0,
            "lattice_grid": [10, 10, 10],
            "periodic": [True, True, True],
            "max_interaction_distance": 3.0,
        },
        "simulation": {"max_steps": 10},
        "objects": {"A": {}, "B": {}},
        "ic": {"mode": "builtin", "initial_conditions": {}},
        "events": {"bad_pair": BadPairEvent},
        "physics": {},
        "outputs": {},
    }

    with pytest.raises(ConfigValidationError, match="max_distance"):
        OKMCengine(config)


def test_validation_rate_deps_params_typo():
    """测试 rate_deps.params 拼写错误"""
    class EventWithTypo(SingleObjectEvent):
        object_types = ['A']
        params = ['distance']  # 正确拼写
        rate_deps = RateDeps(params=['distnace'])  # 拼写错误

        @classmethod
        def rate(cls, objs, params, config):
            return 1.0

        @classmethod
        def execute(cls, event, objs, config):
            return []

    config = {
        "domain": {
            "lattice_constant": 1.0,
            "lattice_grid": [10, 10, 10],
            "periodic": [True, True, True],
            "max_interaction_distance": 3.0,
        },
        "simulation": {"max_steps": 10},
        "objects": {"A": {}},
        "ic": {"mode": "builtin", "initial_conditions": {}},
        "events": {"typo": EventWithTypo},
        "physics": {},
        "outputs": {},
    }

    with pytest.raises(ConfigValidationError, match="distnace.*not defined"):
        OKMCengine(config)


def test_validation_physics_solver_invalid_field():
    """测试 physics.solvers 引用不存在的 field"""
    class DummySolver:
        def initialize(self, physics, config): pass
        def solve(self, physics, step, time, object_data=None, trigger_source=None): return {}

    config = {
        "domain": {
            "lattice_constant": 1.0,
            "lattice_grid": [10, 10, 10],
            "periodic": [True, True, True],
            "max_interaction_distance": 3.0,
        },
        "simulation": {"max_steps": 10},
        "objects": {"A": {}},
        "ic": {"mode": "builtin", "initial_conditions": {}},
        "events": {},
        "physics": {
            "field_names": ["temperature"],
            "field_shapes": [(10, 10, 10)],
            "solvers": [
                {
                    "name": "heat",
                    "solver": DummySolver(),
                    "fields": ["stress_xx"],  # 不存在
                }
            ],
        },
        "outputs": {},
    }

    with pytest.raises(ConfigValidationError, match="stress_xx.*not defined"):
        OKMCengine(config)


def test_validation_physics_trigger_invalid_event():
    """测试 physics.triggers 引用不存在的事件"""
    config = {
        "domain": {
            "lattice_constant": 1.0,
            "lattice_grid": [10, 10, 10],
            "periodic": [True, True, True],
            "max_interaction_distance": 3.0,
        },
        "simulation": {"max_steps": 10},
        "objects": {"A": {}},
        "ic": {"mode": "builtin", "initial_conditions": {}},
        "events": {},
        "physics": {
            "field_names": ["stress_xx"],
            "field_shapes": [(10, 10, 10)],
            "triggers": [
                {
                    "event": "cascade",  # 不存在
                    "fields": ["stress_xx"],
                }
            ],
        },
        "outputs": {},
    }

    with pytest.raises(ConfigValidationError, match="cascade.*not defined"):
        OKMCengine(config)


def test_validation_physics_solver_not_callable():
    """测试 physics.solvers 的 solver 缺少 ExternalSolver 接口"""
    config = {
        "domain": {
            "lattice_constant": 1.0,
            "lattice_grid": [10, 10, 10],
            "periodic": [True, True, True],
            "max_interaction_distance": 3.0,
        },
        "simulation": {"max_steps": 10},
        "objects": {"A": {}},
        "ic": {"mode": "builtin", "initial_conditions": {}},
        "events": {},
        "physics": {
            "field_names": ["temperature"],
            "field_shapes": [(10, 10, 10)],
            "solvers": [
                {
                    "name": "heat",
                    "solver": "not_a_solver",  # 字符串，没有 initialize/solve
                    "fields": ["temperature"],
                }
            ],
        },
        "outputs": {},
    }

    with pytest.raises(ConfigValidationError, match="must have 'initialize' method"):
        OKMCengine(config)


def test_validation_physics_fields_query_unknown_object_type():
    """测试 physics.fields.queries 引用不存在的对象类型。"""
    config = {
        "domain": {
            "lattice_constant": 1.0,
            "lattice_grid": [10, 10, 10],
            "periodic": [True, True, True],
            "max_interaction_distance": 3.0,
        },
        "simulation": {"max_steps": 10},
        "objects": {"A": {}},
        "ic": {"mode": "builtin", "initial_conditions": {}},
        "events": {},
        "physics": {
            "fields": {
                "stress": {
                    "shape": (10, 10, 10),
                    "queries": {
                        "bad": {"types": ["B"]},
                    },
                }
            }
        },
        "outputs": {},
    }

    with pytest.raises(ConfigValidationError, match="unknown object type 'B'"):
        OKMCengine(config)


def test_validation_rate_deps_fields_accepts_physics_fields_block():
    """测试 rate_deps.fields 可直接引用 physics.fields 中定义的字段。"""
    class FieldDrivenEvent(SingleObjectEvent):
        object_types = ['A']
        rate_deps = RateDeps(fields=['stress'])

        @classmethod
        def rate(cls, objs, params, config):
            return 1.0

        @classmethod
        def execute(cls, event, objs, config):
            return []

    config = {
        "domain": {
            "lattice_constant": 1.0,
            "lattice_grid": [10, 10, 10],
            "periodic": [True, True, True],
            "max_interaction_distance": 3.0,
        },
        "simulation": {"max_steps": 10},
        "objects": {"A": {}},
        "ic": {"mode": "builtin", "initial_conditions": {}},
        "events": {"field_evt": FieldDrivenEvent},
        "physics": {
            "fields": {
                "stress": {"shape": (10, 10, 10)}
            }
        },
        "outputs": {},
    }

    engine = OKMCengine(config)
    assert engine is not None
    engine.destroy()


def test_compile_user_config_chains_physics_ic_hooks_without_recursion():
    """physics.ic_func 的多层包装应按顺序执行且不递归。"""

    class Tick(ZeroObjectEvent):
        @classmethod
        def rate(cls, objs, params, config):
            return 1.0

        @classmethod
        def execute(cls, event, objs, config):
            return []

    calls = []

    def base_ic(physics, config):
        calls.append('base')

    def update_func(field, queries, physics, step, time):
        return field

    class DummySolver:
        def initialize(self, physics, config):
            pass

        def solve(self, physics, step, time, object_data=None, trigger_source=None):
            return {}

    config = {
        "domain": {
            "lattice_constant": 1.0,
            "lattice_grid": [4, 4, 4],
            "periodic": [True, True, True],
            "max_interaction_distance": 2.0,
        },
        "simulation": {"max_steps": 1},
        "objects": {"A": {}},
        "ic": {"mode": "builtin", "initial_conditions": {}},
        "events": {"tick": Tick},
        "physics": {
            "ic_func": base_ic,
            "params": {"temperature": 300.0},
            "fields": {
                "stress": {
                    "shape": [2, 2, 2],
                    "initial_value": 1.5,
                    "update_func": update_func,
                }
            },
            "solvers": [
                {
                    "name": "dummy",
                    "solver": DummySolver(),
                    "fields": ["stress"],
                }
            ],
            "triggers": [
                {
                    "event": "tick",
                    "fields": ["stress"],
                    "solver": "dummy",
                }
            ],
        },
        "outputs": {},
    }

    validate_user_config(config)
    flat = compile_user_config(config)
    physics_ic = flat["physics"]["ic_func"]

    class StubPhysics:
        def set_param(self, name, value):
            calls.append(("param", name, value))

        def set_field(self, name, field):
            calls.append(("field", name, field.shape, float(field[0, 0, 0])))

        def set_field_update(self, name, func, **kwargs):
            calls.append(("field_update", name, kwargs["enabled"]))

        def register_solver(self, name, solver, fields):
            calls.append(("solver", name, tuple(fields)))

        def register_event_trigger(self, event_name, fields, solver_name, condition):
            calls.append(("trigger", event_name, tuple(fields), solver_name))

    physics_ic(StubPhysics(), {})

    assert 'base' in calls
    assert ("param", "temperature", 300.0) in calls
    assert ("field", "stress", (2, 2, 2), 1.5) in calls
    assert ("field_update", "stress", True) in calls
    assert ("solver", "dummy", ("stress",)) in calls
    assert ("trigger", "tick", ("stress",), "dummy") in calls
    assert calls.count('base') == 1


# ---- outputs 校验测试 ----

def _make_config_with_outputs(outputs):
    """辅助：构造带指定 outputs 的最小配置"""
    return {
        "domain": {
            "lattice_constant": 1.0,
            "lattice_grid": [10, 10, 10],
            "periodic": [True, True, True],
            "max_interaction_distance": 3.0,
        },
        "simulation": {"max_steps": 10},
        "objects": {"A": {}},
        "ic": {"mode": "builtin", "initial_conditions": {}},
        "events": {},
        "physics": {},
        "outputs": outputs,
    }


def test_validation_ovito_dump_not_dict():
    """outputs.ovito_dump 必须是 dict"""
    config = _make_config_with_outputs({"ovito_dump": "bad"})
    with pytest.raises(ConfigValidationError, match="ovito_dump must be a dictionary"):
        OKMCengine(config)


def test_validation_ovito_dump_enabled_not_bool():
    """outputs.ovito_dump.enabled 必须是 bool"""
    config = _make_config_with_outputs({"ovito_dump": {"enabled": 1}})
    with pytest.raises(ConfigValidationError, match="enabled must be bool"):
        OKMCengine(config)


def test_validation_ovito_dump_properties_not_list():
    """outputs.ovito_dump.properties 必须是字符串列表"""
    config = _make_config_with_outputs({"ovito_dump": {"properties": "id,type"}})
    with pytest.raises(ConfigValidationError, match="properties must be a list"):
        OKMCengine(config)


def test_validation_ovito_dump_properties_item_not_str():
    """outputs.ovito_dump.properties 中每项必须是字符串"""
    config = _make_config_with_outputs({"ovito_dump": {"properties": ["id", 123]}})
    with pytest.raises(ConfigValidationError, match="properties\\[1\\] must be string"):
        OKMCengine(config)


def test_validation_csv_not_supported():
    """outputs.csv 尚未实现，应报错"""
    config = _make_config_with_outputs({"csv": {"enabled": True}})
    with pytest.raises(ConfigValidationError, match="csv is not yet supported"):
        OKMCengine(config)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
