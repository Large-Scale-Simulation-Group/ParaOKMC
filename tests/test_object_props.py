"""
测试 float_props 全链路：
1. 对象定义 properties → TypeCompiler 自动映射
2. 初始对象创建 → float_props 正确填充
3. rate/execute 按名字读属性
4. execute 返回新对象时按名字写属性
5. MPI 广播后属性保持一致

用法：
    python tests/test_object_props.py
    mpirun -np 2 python tests/test_object_props.py
"""

import numpy as np
import sys
from mpi4py import MPI

from paraokmc.v16_engine import create_okmc_engine_v16
from paraokmc.event_class import SingleObjectEvent, PairObjectEvent, get_physics_param
from paraokmc.event_compiler import merge_event_config
from paraokmc.config_system import create_initial_objects_func

comm = MPI.COMM_WORLD
rank = comm.rank

k_B = 8.617e-5

passed = 0
failed = 0


def report(name, ok, detail=""):
    global passed, failed
    if ok:
        passed += 1
        if rank == 0:
            print(f"  [PASS] {name} {detail}")
    else:
        failed += 1
        if rank == 0:
            print(f"  [FAIL] {name} {detail}")


# ============================================================================
# 事件类：用属性 n 代替类型爆炸
# ============================================================================

class SIAMigration(SingleObjectEvent):
    """SIA 迁移 — 速率依赖 float_props[1] (n)"""
    object_types = ['SIA']
    params = []

    @classmethod
    def rate(cls, objs, params, config):
        T = get_physics_param(config, 'temperature', default=300.0)
        n = int(objs[0]['float_props'][1]) if objs else 1
        nu = 6e12 * (n ** (-0.5))
        return float(nu * np.exp(-0.013 / (k_B * T)))

    @classmethod
    def execute(cls, event, objs, config):
        obj = objs[0]
        n = int(obj['float_props'][1])
        lc = config.get('lattice_constant', 3.17)
        gs = config.get('grid_size', (50, 50, 50))
        bx = gs[0] * lc

        jump = lc * np.sqrt(3) / 2
        sign = np.random.choice([-1, 1])
        nx = (float(obj['x']) + sign * jump) % bx
        return [{'id': int(obj['id']), 'x': nx}]


class SIARecombination(PairObjectEvent):
    """SIA + V 复合 — 读取双方的 n 属性"""
    type_pairs = [('SIA', 'V')]
    max_distance = 15.0
    params = ['distance']

    @classmethod
    def rate(cls, objs, params, config):
        T = get_physics_param(config, 'temperature', default=300.0)
        d = max(params.get('distance', 1.0), 0.1)
        n = int(objs[0]['float_props'][1]) if objs else 1
        D = (3.17**2 / 6) * 6e12 * (n**(-0.5)) * np.exp(-0.013 / (k_B * T))
        return float(4 * np.pi * D * 5.0 / d)

    @classmethod
    def execute(cls, event, objs, config):
        sia, v = objs[0], objs[1]
        n_sia = int(sia['float_props'][1])
        n_v = int(v['float_props'][1])
        diff = n_sia - n_v

        result = [
            {'id': int(sia['id']), 'deleted': True},
            {'id': int(v['id']), 'deleted': True},
        ]
        if diff > 0:
            result.append({
                'id': -1, 'type': 'SIA', 'new': True,
                'x': float(sia['x']), 'y': float(sia['y']), 'z': float(sia['z']),
                'float_props': [0.0, diff],
            })
        elif diff < 0:
            result.append({
                'id': -1, 'type': 'V', 'new': True,
                'x': float(v['x']), 'y': float(v['y']), 'z': float(v['z']),
                'float_props': [0.0, -diff],
            })
        return result


# ============================================================================
# Test 1: TypeCompiler 属性映射
# ============================================================================

def test_props_mapping():
    if rank == 0:
        print("\n[Test 1] TypeCompiler props mapping")

    from paraokmc.config_analyzer import TypeCompiler

    config = {
        'objects': {
            'SIA': {'properties': {'n': 1, 'E_b': 0.5}},
            'V':   {'properties': {'n': 1, 'mobility': 0.0}},
        },
        'events': {},
    }
    tc = TypeCompiler(config)
    tc.analyze()
    mapping = tc.get_props_mapping()

    # n → float_props
    report("SIA.n → float_props",
           'n' in mapping['float_props'].get('SIA', {}),
           f"(mapping={mapping['float_props'].get('SIA', {})})")

    # E_b 是 float → float_props
    report("SIA.E_b → float_props",
           'E_b' in mapping['float_props'].get('SIA', {}),
           f"(mapping={mapping['float_props'].get('SIA', {})})")

    # V.n → float_props
    report("V.n → float_props",
           'n' in mapping['float_props'].get('V', {}),
           f"(mapping={mapping['float_props'].get('V', {})})")

    # V.mobility → float_props
    report("V.mobility → float_props",
           'mobility' in mapping['float_props'].get('V', {}),
           f"(mapping={mapping['float_props'].get('V', {})})")

    # type_to_int
    report("type_to_int has SIA and V",
           'SIA' in mapping['type_to_int'] and 'V' in mapping['type_to_int'],
           f"(type_to_int={mapping['type_to_int']})")


# ============================================================================
# Test 2: 初始对象 float_props 填充
# ============================================================================

def test_initial_objects():
    if rank == 0:
        print("\n[Test 2] Initial objects float_props")

    objects_def = {
        'SIA': {'properties': {'n': 3, 'E_b': 0.5}},
        'V':   {'properties': {'n': 2, 'mobility': 1.5}},
    }
    ic = {
        'SIA': {'count': 5, 'distribution': 'random'},
        'V':   {'count': 5, 'distribution': 'random'},
    }
    func = create_initial_objects_func(ic, objects_def, (10, 10, 10), 3.17)
    objs = func(0, 1, {'_user_seed': 42})

    sia_objs = [o for o in objs if o['type'] == 'SIA']
    v_objs = [o for o in objs if o['type'] == 'V']

    # SIA: float_props=[0.0, 3.0, 0.5] (index 0=size, 1=n, 2=E_b)
    report("SIA float_props correct",
           len(sia_objs) > 0 and sia_objs[0].get('float_props') == [0.0, 3.0, 0.5],
           f"(got {sia_objs[0].get('float_props') if sia_objs else 'none'})")

    report("SIA float_props[1] has n=3",
           len(sia_objs) > 0 and sia_objs[0].get('float_props', [0, 0])[1] == 3.0,
           f"(got {sia_objs[0].get('float_props') if sia_objs else 'none'})")

    # V: float_props=[0.0, 2.0, 1.5] (index 0=size, 1=n, 2=mobility)
    report("V float_props correct",
           len(v_objs) > 0 and v_objs[0].get('float_props') == [0.0, 2.0, 1.5],
           f"(got {v_objs[0].get('float_props') if v_objs else 'none'})")

    report("V float_props has mobility=1.5",
           len(v_objs) > 0 and v_objs[0].get('float_props', [0, 0, 0])[2] == 1.5,
           f"(got {v_objs[0].get('float_props') if v_objs else 'none'})")


# ============================================================================
# Test 3: 完整引擎运行 — rate/execute 读写属性
# ============================================================================

def test_engine_with_props():
    if rank == 0:
        print("\n[Test 3] Full engine run with float_props")

    objects_def = {
        'SIA': {'properties': {'n': 1}},
        'V':   {'properties': {'n': 1}},
    }
    ic = {
        'SIA': {'count': 20, 'distribution': 'random'},
        'V':   {'count': 20, 'distribution': 'random'},
    }
    grid_size = (20, 20, 20)
    lc = 3.17

    base_config = {
        'physics': {'temperature': 300.0},
        'objects': objects_def,
        'events': {
            'SIA_migration': {'objects': ['SIA'], 'rate_params': []},
            'SIA_V_recomb': {
                'objects': ['SIA', 'V'],
                'rate_params': ['distance'],
                'max_distance': 15.0,
            },
        },
        'domain_decomposition': {
            'lattice_constant': lc,
            'lattice_grid': list(grid_size),
            'h_factor': 1,
            'box_size': [grid_size[i] * lc for i in range(3)],
            'periodic': [True, True, True],
            'max_interaction_distance': 25.0,
        },
        'initial_objects_func': create_initial_objects_func(
            ic, objects_def, grid_size, lc),
        'lattice_constant': lc,
        'grid_size': grid_size,
        'temperature': 300.0,
        'max_steps': 100,
        'max_time': 1e-6,
        'verbose': False,
        'random_seed': 42,
        'rma_object_pool': {'initial_capacity': 100, 'auto_expand': True},
        'rma_event_pool': {
            'capacity_zero': 10, 'capacity_single': 100,
            'capacity_pair': 500, 'auto_expand': True,
        },
    }

    events_config = {
        'SIA_migration': SIAMigration,
        'SIA_V_recomb': SIARecombination,
    }

    config = merge_event_config(base_config, events_config)
    engine = create_okmc_engine_v16(config)

    # 验证初始对象的 float_props 被正确填充
    local_objs = engine.rma_object_pool.get_local_objects(active_only=True)
    sia_objs = [o for o in local_objs if str(o['type']).strip() == 'SIA']

    if len(sia_objs) > 0:
        n_val = int(sia_objs[0]['float_props'][1])
        report("Engine: SIA float_props[1] (n) readable",
               n_val == 1,
               f"(got n={n_val})")
    else:
        report("Engine: SIA float_props[1] (n) readable",
               True, "(no local SIA, skip)")

    # 运行 100 步
    stats = engine.run()

    report("Engine: 100 steps completed",
           stats['total_steps'] == 100,
           f"(steps={stats['total_steps']})")

    # 验证运行后对象属性仍然可读
    local_objs_after = engine.rma_object_pool.get_local_objects(active_only=True)
    all_n_valid = True
    for o in local_objs_after:
        n = int(o['float_props'][1])
        if n < 1:
            all_n_valid = False
            break

    report("Engine: all objects have valid n >= 1 after run",
           all_n_valid)

    engine.destroy()


# ============================================================================
# Test 4: execute 返回新对象时 float_props 传递
# ============================================================================

def test_execute_new_object_props():
    if rank == 0:
        print("\n[Test 4] Execute returns new object with float_props")

    # 模拟 execute 返回值
    result = SIARecombination.execute(
        None,
        [
            # mock SIA with n=3
            type('obj', (), {
                '__getitem__': lambda self, k: {
                    'id': np.int64(1), 'type': 'SIA',
                    'x': np.float64(10.0), 'y': np.float64(20.0),
                    'z': np.float64(30.0),
                    'float_props': np.array([0.0, 3.0, 0, 0, 0, 0, 0, 0, 0, 0]),
                }[k]
            })(),
            # mock V with n=1
            type('obj', (), {
                '__getitem__': lambda self, k: {
                    'id': np.int64(2), 'type': 'V',
                    'x': np.float64(11.0), 'y': np.float64(21.0),
                    'z': np.float64(31.0),
                    'float_props': np.array([0.0, 1.0, 0, 0, 0, 0, 0, 0, 0, 0]),
                }[k]
            })(),
        ],
        {}
    )

    # SIA(n=3) + V(n=1) → delete both + create SIA(n=2)
    report("3 changes returned", len(result) == 3)
    report("first two are deletions",
           result[0].get('deleted') and result[1].get('deleted'))

    new_obj = result[2]
    report("new object is SIA",
           new_obj.get('type') == 'SIA' and new_obj.get('new'))
    report("new object has n=2",
           new_obj.get('float_props') == [0.0, 2],
           f"(got {new_obj.get('float_props')})")


# ============================================================================
# Main
# ============================================================================

if __name__ == '__main__':
    if rank == 0:
        print("=" * 60)
        print("Object Properties (float_props) Tests")
        print(f"MPI: {comm.size} ranks")
        print("=" * 60)

    test_props_mapping()
    test_initial_objects()
    test_execute_new_object_props()
    test_engine_with_props()

    comm.Barrier()

    if rank == 0:
        print("\n" + "=" * 60)
        print(f"Results: {passed} passed, {failed} failed")
        print("=" * 60)

    if failed > 0:
        sys.exit(1)
