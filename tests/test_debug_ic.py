"""Debug initial conditions"""
import numpy as np
from mpi4py import MPI
from numba import njit
from paraokmc.jit_rate_engine import RateDeps
from paraokmc.config_system import create_initial_objects_func
from paraokmc.event_compiler import merge_event_config
from paraokmc.event_class import SingleObjectEvent, get_physics_param

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# 简单配置
objects = {
    'H': {'properties': {'radius': 0.5}},
    'V': {'properties': {'radius': 0.8}},
}

initial_conditions = {
    'H': {'count': 10, 'distribution': 'random'},
    'V': {'count': 10, 'distribution': 'random'},
}

grid_size = (30, 30, 30)
lc = 1.0

# JIT kernel
@njit
def arrhenius_kernel(E_m, nu, temperature, k_B):
    return nu * np.exp(-E_m / (k_B * temperature))


# 事件类
class HMigration(SingleObjectEvent):
    object_types = ['H']
    params = ['E_m', 'nu']
    rate_deps = RateDeps(obj1=['E_m', 'nu'], physics=['temperature', 'k_B'])
    rate_kernel = arrhenius_kernel

    @classmethod
    def rate(cls, objs, params, config):
        T = get_physics_param(config, 'temperature', default=500.0)
        k_B = 8.617e-5
        E_m = params.get('E_m', 0.5)
        nu = params.get('nu', 1e13)
        return float(nu * np.exp(-E_m / (k_B * T)))

    @classmethod
    def execute(cls, event, objs, config):
        return [{'id': int(objs[0]['id']), 'x': float(objs[0]['x']) + 0.1}]

base_config = {
    'physics': {'temperature': 500.0},
    'objects': objects,
    'events': {
        'H_migration': {'objects': ['H'], 'rate_params': ['E_m', 'nu'],
                        'E_m': 0.3, 'nu': 1e13},
    },
    'domain_decomposition': {
        'lattice_constant': lc,
        'lattice_grid': list(grid_size),
        'h_factor': 1,
        'box_size': [grid_size[0] * lc, grid_size[1] * lc, grid_size[2] * lc],
        'periodic': [True, True, True],
        'max_interaction_distance': 5.0,
    },
    'initial_objects_func': create_initial_objects_func(
        initial_conditions, objects, grid_size, lc),
    'max_steps': 10,
}

config = merge_event_config(base_config, {'H_migration': HMigration})

# 模拟IDGenerator
class FakeIDGen:
    def __init__(self, rank):
        self.counter = rank * 1000000
    def generate_id(self, obj_type):
        self.counter += 1
        return self.counter

config['id_generator'] = FakeIDGen(rank)

# 调用initial_objects_func
initial_objects_func = config['initial_objects_func']
objs = initial_objects_func(rank, size, config)

print(f"Rank {rank}: Generated {len(objs)} objects")
for obj in objs[:3]:
    print(f"  Rank {rank}: {obj['type']} at ({obj['x']:.2f}, {obj['y']:.2f}, {obj['z']:.2f})")

# 检查domain_decomposition
dd = config['domain_decomposition']
box_size = dd['box_size']
print(f"Rank {rank}: box_size = {box_size}")

# 根据z坐标计算owner
z_per_rank = box_size[2] / size
for obj in objs:
    z = obj['z']
    expected_owner = int(z / z_per_rank)
    if expected_owner >= size:
        expected_owner = size - 1
    print(f"  Rank {rank}: obj z={z:.2f}, expected_owner={expected_owner}")
