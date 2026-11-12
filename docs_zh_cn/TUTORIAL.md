# ParaOKMC 教程

*English original: [`docs/TUTORIAL.md`](../docs/TUTORIAL.md)*

用八个短脚本带你浏览本库。每章介绍一个概念，给出体现该概念的代码，并说明运行时应关注什么。脚本位于 `tutorial/`，单核几秒即可跑完；每个脚本也都可以不加修改地在 `mpirun` 下运行。第一次请按顺序阅读各章；之后以 `docs_zh_cn/USER_GUIDE.md` 作为参考手册。

```bash
cd paraokmc_release
pip install -e .
export PYTHONPATH=$PWD NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1   # one kernel thread per MPI rank
cd tutorial
python 01_first_model.py
mpirun -np 2 python 01_first_model.py
```

| 章 | 脚本 | 学习内容 |
|---|---|---|
| 1 | `01_first_model.py` | 需要编写的两种类、七个配置块、一次运行返回什么 |
| 2 | `02_two_rate_styles.py` | Python `rate()` 与编译的 `rate_kernel` 对比；kernel 能看到什么、不能看到什么 |
| 3 | `03_objects_and_execute.py` | 对象属性；`change_obj`、`delete_obj`、`new_obj`；后处理器 |
| 4 | `04_initial_condition_and_random_numbers.py` | 编写自己的初始条件；随机数如何保持可复现 |
| 5 | `05_zero_object_events.py` | 不依附于任何对象的事件：缺陷源 |
| 6 | `06_physics_parameters_and_fields.py` | 随时间变化的参数；由速率读取的空间场 |
| 7 | `07_outputs_and_restart.py` | OVITO 帧、事件钩子、从保存的帧续算 |
| 8 | `08_parallel_runs_and_load_balancing.py` | rank 数会改变什么；三层负载均衡 |

以上的教程后，`example/` 中包含配套论文的全部验证算例和基准测试，每节一个目录；其计算结果随论文的程序包发布在 CPC Program Library。

---

## 1. 第一个模型

**概念。** 一个 OKMC 模型由一组*对象*（点缺陷、团簇、阱）和每个对象可能经历的一组*事件*组成。ParaOKMC 需要你提供两样东西：每种事件类型一个 Python 类，以及一个由七个块组成的字典，用来描述模拟盒子、对象、初始状态、物理参数和输出。其余工作由 `OKMCengine(config).run()` 完成。

`01_first_model.py` 模拟 Frenkel 对：间隙原子（`I`）和空位（`V`）通过随机跳跃迁移；距离小于 2 Å 的 `I` 与 `V` 发生复合。

**事件类**继承自 `SingleObjectEvent`（一个对象参与）或 `PairObjectEvent`（两个彼此距离在 `max_distance` 以内的对象参与），声明它适用于哪些类型，并提供速率和事件结果：

```python
class Migration(SingleObjectEvent):
    object_types = ['I', 'V']
    rate_deps = RateDeps(obj1=['E_m', 'nu'], physics=['temperature', 'k_B'])

    @staticmethod
    @njit
    def rate_kernel(E_m, nu, temperature, k_B):
        return nu * np.exp(-E_m / (k_B * temperature))

    @classmethod
    def execute(cls, event, objs, config):
        obj, rng = objs[0], config['rng']
        ...                                    # pick a random direction with rng
        return [change_obj(obj, x=new[0], y=new[1], z=new[2])]
```

* `rate_deps` 指定速率的输入：这里是两个对象属性和两个物理参数。`rate_kernel` 是一个 Numba 函数，其参数遵循该声明。引擎对所有候选事件一次性求值，因此这是快速路径。
* `execute` 是普通 Python 代码。它以字典形式接收事件涉及的对象（`x`、`y`、`z`、`size`、`type`，以及按名称给出的每个已声明属性），并返回一个变更列表。随机数来自 `config['rng']`（第 4 章解释原因）。

```python
class Recombination(PairObjectEvent):
    type_pairs = [('I', 'V')]
    max_distance = 2.0

    @classmethod
    def rate(cls, objs, params, config):
        return 1.0e12

    @classmethod
    def execute(cls, event, objs, config):
        return [delete_obj(objs[0]), delete_obj(objs[1])]
```

双对象事件列出允许的类型对和一个截断距离；这些类型的对象中，每一对距离在 `max_distance` 以内的对象都对应一个候选事件。这个事件用普通 Python 方法而不是 kernel 来写速率；第 2 章比较这两种形式。

**七个块。**

| 块 | 在本脚本中的作用 |
|---|---|
| `domain` | 模拟盒子每边长为 `lattice_grid × lattice_constant` = 20 × 2.87 Å，周期性边界；`max_interaction_distance` 必须不小于所有双对象事件中最大的 `max_distance`，因为它决定了一个 rank 能看到的相邻 rank 鬼域的宽度 |
| `simulation` | 2000 步或 1 s 模拟时间，以先到者为准；随机种子；每 500 步输出一行进度 |
| `objects` | 两种类型，各有一个 `size` 以及速率所读取的标量 `properties` |
| `ic` | 随机位置上的 40 + 40 个对象 |
| `events` | 已注册的类，使用输出中将采用的名称 |
| `physics` | 任何速率都可依赖的具名标量 |
| `outputs` | 后处理的频率（尚未注册任何后处理器） |

**动手试试。** 先在一个 rank 上运行脚本，再在两个 rank 上运行。两者都会打印类似下面的一行

```
steps = 2000, simulated time = 6.737e-09 s, objects left = 64
```

模拟时间是 2000 步停留时间之和；它只有几纳秒，因为在 600 K 下间隙原子的跳跃频率约为 10¹⁰ Hz。两次运行在时间和剩余对象数上略有差异：两条轨迹是同一随机过程的两个样本（第 8 章详述）。把单 rank 的算例运行两次，数字完全相同。

---

## 2. 编写速率的两种方式

**概念。** 速率可以是 Python 方法，也可以是编译核函数（kernel）。除非引擎能证明事件没有变化，否则两者都会在每一步对每个候选事件求值。Python 形式是通用形式；kernel 形式是快速形式，并对其可读取的内容有严格规定。

`02_two_rate_styles.py` 把 Frenkel 对模型写了两遍。`--style python` 使用

```python
class MigrationPy(SingleObjectEvent):
    object_types = ['I', 'V']

    @classmethod
    def rate(cls, objs, params, config):
        o = objs[0]
        T = config['physics'].get_param('temperature')
        return o['nu'] * np.exp(-o['E_m'] / (K_B * T))


class CapturePy(PairObjectEvent):
    type_pairs = [('I', 'V')]
    max_distance = 6.0
    params = ['distance']            # ask the engine to put the pair distance in params

    @classmethod
    def rate(cls, objs, params, config):
        r = max(params['distance'], 0.5)
        return 1.0e12 * (2.0 / r)
```

`--style kernel` 使用

```python
class MigrationKernel(SingleObjectEvent):
    object_types = ['I', 'V']
    rate_deps = RateDeps(obj1=['E_m', 'nu'], physics=['temperature'])

    @staticmethod
    @njit
    def rate_kernel(E_m, nu, temperature):
        return nu * np.exp(-E_m / (8.617333262e-5 * temperature))


class CaptureKernel(PairObjectEvent):
    type_pairs = [('I', 'V')]
    max_distance = 6.0
    rate_deps = RateDeps(kind='pair')

    @staticmethod
    @njit
    def rate_kernel(distance):
        r = max(distance, 0.5)
        return 1.0e12 * (2.0 / r)
```

**kernel 接收什么。** `rate_kernel` 的参数依次为：

1. `rate_deps.obj1` 中指定的第一个对象的属性，
2. 对于双对象事件，`rate_deps.obj2` 中指定的第二个对象的属性，
3. 对于双对象事件，`distance`：两个对象之间的最小镜像距离，
4. `rate_deps.fields` 中指定的场值，在对象位置处插值得到（第 6 章），
5. `rate_deps.physics` 中指定的物理参数。

由此有两个推论：

* **不传入坐标。** kernel 无法根据对象位置计算几何条件。如果速率需要位置（到某平面、某直线、某表面的距离），要么用 Python 编写，要么把所需坐标镜像到由 `execute()` 保持更新的对象属性中。`example/3.2_sink_strength/run_dislocation_sink_strength.py` 中的位错算例并列给出了两种形式：Python `rate()` 根据 `x, y, z` 判断是否落在捕获圆柱内；被注释掉的编译版本读取镜像的 `pos_y`、`pos_z` 属性。
* **不含对象属性的 kernel 必须声明其类别。** `RateDeps` 根据所声明的属性推断事件类别（只有 `obj1` → 单对象，`obj1` 和 `obj2` → 双对象，两者都没有 → 零对象）。`CaptureKernel` 一个属性也没有声明，所以它写成 `RateDeps(kind='pair')`，其 kernel 只接收 `distance`。

对于双对象事件的 kernel，无论这一对是以何种顺序找到的，第一个对象总是类型在 `type_pairs` 中排在前面的那个（`('I', 'V')` → `obj1` 是 `I`）。

**动手试试。** 运行两种写法：

```
style=python: steps=3000 time=3.868133e-08 s objects=8  rate-eval 0.640 ms/step
style=kernel: steps=3000 time=3.868133e-08 s objects=8  rate-eval 0.422 ms/step
```

两条轨迹直到最后一位都完全相同：计算出的数相同，只是计算方式不同。即使对这样很小的事件目录，kernel 每步也更快；当每步有 10⁴ 到 10⁵ 个候选事件时，差距可达一到两个数量级，因为 Python 形式对每个事件调用一次函数。

**何时用哪种。** 对求值最频繁的速率（可动组元的迁移、常见的双对象反应）使用 kernel。对需要坐标、字符串逻辑、查表或外部模型的速率，以及很少求值的速率，使用 Python。在同一模型中混用两者是常态。

---

## 3. 对象、属性与事件的结果

**概念。** 对象是一条记录，包含位置、`size`、类型以及为其类型声明的标量 `properties`。`execute()` 通过三个辅助函数描述事件对对象做了什么：

| 辅助函数 | 含义 |
|---|---|
| `change_obj(obj, **fields)` | 更新已有对象的坐标和/或属性 |
| `delete_obj(obj)` | 删除该对象 |
| `new_obj(type=..., x=, y=, z=, size=, float_props=[...])` | 创建一个对象 |

`03_objects_and_execute.py` 形成并分解团簇：

```
H + V  -> HV        capture (a new object)
HV     -> H + V     dissociation (the cluster disappears, two objects appear)
H                   migrates; V and HV do not
```

**修改属性。** 迁移事件用一个属性记录每个 `H` 的跳跃次数：

```python
return [change_obj(o, x=new[0], y=new[1], z=new[2], jumps=o['jumps'] + 1)]
```

任何已声明的属性都可以按名称传给 `change_obj`。

**创建对象。** 新对象必须完整：类型、位置、`size` 以及完整的属性向量。属性存放在定长数组（`float_props`）中，其槽位顺序由引擎在启动时分配；运行时可通过 `config['_props_mapping']['float_props'][type]` 获得该映射。脚本把这一点封装成一个辅助函数：

```python
def make_props(config, obj_type, **overrides):
    n = config['_object_dtype'].fields['float_props'][0].shape[0]
    slots = config['_props_mapping']['float_props'][obj_type]
    props = np.zeros(n)
    for name, value in config['objects'][obj_type].get('properties', {}).items():
        props[slots[name]] = value                     # declared defaults
    for name, value in overrides.items():
        props[slots[name]] = value                     # what this event decides
    return props.tolist()
```

于是捕获事件写成

```python
h, v = (objs[0], objs[1]) if objs[0]['type'] == 'H' else (objs[1], objs[0])
return [delete_obj(h), delete_obj(v),
        new_obj(type='HV', x=v['x'], y=v['y'], z=v['z'],
                size=config['objects']['HV']['size'],
                float_props=make_props(config, 'HV', h_jumps=h['jumps']))]
```

注意第一行：在双对象事件中，`objs` 的顺序对编译速率遵循 `type_pairs`，但 `execute()` 不应依赖这一点，所以要检查类型。新对象由引擎分配唯一 id（id 是 64 位整数，其中编码了创建它的 rank）。

**后处理器**是列在 `outputs.postprocessors` 下的任意可调用对象 `f(step, time, pool, comm)`；它每隔 `outputs.interval` 步在每个 rank 上运行。`pool.get_local_objects()` 以 NumPy 结构化数组返回本 rank 拥有的对象；要得到全局数值，需在 `comm` 上做归约：

```python
def count_types(step, time, pool, comm):
    local = {}
    for o in pool.get_local_objects():
        t = pool.get_type_name(o)
        local[t] = local.get(t, 0) + 1
    totals = comm.gather(local, root=0)
    ...
```

**动手试试。**

```
  step      0  t = 2.424e-14 s  H=49  HV=1  V=24
  step   1000  t = 6.165e-10 s  H=38  HV=12  V=13
  step   2000  t = 1.315e-09 s  H=34  HV=16  V=9
  step   4000  t = 2.876e-09 s  H=32  HV=18  V=7
```

各组元数目在捕获与分解之间达到稳定平衡。打开脚本，把 `HV` 的 `E_b` 降到 0.6 eV：分解变快，存活的团簇很少。

---

## 4. 自定义初始条件，以及随机数的工作方式

**由函数给出初始条件。** `ic.mode = "function"` 向引擎提供

```python
def initial_objects(rank, size, config) -> list[dict]
```

每个 rank 调用它一次，且必须**只返回位于自身子域内的对象**。最简单的正确做法是：在每个 rank 上用带随机种子的生成器以完全相同的方式生成整个构型，然后只保留本 rank 拥有的部分：

```python
def two_slabs_ic(n_per_slab):
    def initial_objects(rank, size, config):
        dd = config['_domain_decomp']
        box = np.asarray(config['grid_size'], dtype=float) * config['lattice_constant']
        gen = np.random.default_rng(config['_user_seed'])      # same stream on every rank
        mine = []
        for k in range(2 * n_per_slab):
            pos = ...                                           # drawn from gen
            if dd.get_owner_rank(pos) == rank:
                mine.append({'id': k + 1, 'type': 'A', 'x': pos[0], 'y': pos[1], 'z': pos[2],
                             'size': 1.0, 'float_props': []})
        return mine
    return initial_objects
```

每个字典需要 `id`（在所有 rank 间唯一）、`type`、`x`、`y`、`z`；若该类型有 `size` 和 `float_props`，也要给出（槽位顺序见第 3 章）。上面的生成器以 `config['_user_seed']`（即 `simulation.random_seed`）为种子，因此换一个随机种子，初始状态和轨迹都会不同。对于数百万个对象，应以数组形式生成坐标并用子域边界做掩码，而不是对每个对象调用 `get_owner_rank`（`example/4.1_benchmark_workloads/pair_scaling.py` 就是这样做的）。

**事件内部的随机数。** `config['rng']` 是事件应使用的唯一生成器。每次抽取都是 `(random_seed, step, draw index)` 的纯函数：该元组作为 NumPy `SeedSequence` 的种子，后者再为一个 PCG64 生成器提供种子，其第一个输出就是所需的数。步与步之间不携带任何状态，因此

* 一次运行可由其随机种子复现：在相同 rank 数上使用相同随机种子得到相同轨迹（rank 数不同则得到同一过程的不同样本，见第 8 章）；
* 每个请求第 *n* 步第 *k* 个数的 rank 都得到相同的值，因此从不需要通信任何随机状态；
* 在事件中使用 `np.random` 会破坏这一点：每个 rank 看到各自的随机流，各 rank 对发生了什么会产生分歧。

可用方法有 `rng.random()`、`rng.uniform(a, b)`、`rng.normal(mu, sigma)`、`rng.randint(lo, hi)`（半开区间）和 `rng.choice(seq)`。速率中绝不能抽取随机数：速率何时重算由引擎决定，随机的速率会使总速率依赖于它上次求值的时刻。

**动手试试。** 脚本在第 0、1000 和 2000 步打印一个指纹（所有坐标的加权和）：

```
python 04_initial_condition_and_random_numbers.py --seed 5
  step  2000  t = 5.053982e-12  fingerprint = 27968.301200
python 04_initial_condition_and_random_numbers.py --seed 5
  step  2000  t = 5.053982e-12  fingerprint = 27968.301200
python 04_initial_condition_and_random_numbers.py --seed 6
  step  2000  t = 5.070687e-12  fingerprint = 28106.869662
```

相同随机种子，数字直到最后一位都相同。在 `mpirun -np 2` 下，指纹与单 rank 的值不同，但每次运行之间同样完全一致。完整说明见 `docs_zh_cn/RANDOM_NUMBERS.md`。

---

## 5. 零对象事件

**概念。** 有些过程不属于任何对象：产生缺陷的辐照、热源、全局开关。`ZeroObjectEvent` 是一个独立存在的候选事件。默认情况下它恰好只有一份，位于 rank 0（`owner_rank_mode = 'rank0'`），其速率与其他事件一样计入全局总速率。当它被选中时，其 `execute()` 收到一个空的对象列表，并可以在模拟盒子内任意位置创建对象。

`05_zero_object_events.py` 从一个**空**模拟盒子开始，让一个源以每秒 `pair_production_rate` 的速率产生 Frenkel 对，同时 `I` 和 `V` 迁移并复合：

```python
class FrenkelSource(ZeroObjectEvent):
    rate_deps = RateDeps(physics=['pair_production_rate'])   # no objects: kind = zero

    @staticmethod
    @njit
    def rate_kernel(pair_production_rate):
        return pair_production_rate

    @classmethod
    def execute(cls, event, objs, config):
        rng, box = config['rng'], box_of(config)
        pos = np.array([rng.random(), rng.random(), rng.random()]) * box
        ...
        return [new_obj(type='V', x=..., size=1.0, float_props=default_props(config, 'V')),
                new_obj(type='I', x=..., size=1.0, float_props=default_props(config, 'I'))]
```

新对象被赋予其类型所声明的属性默认值（`default_props` 按第 3 章的方法构造该向量）。空的初始条件写作 `"ic": {"mode": "builtin", "initial_conditions": {}}`。

**动手试试。**

```
  step      0  t = 1.000e-10 s  objects = 2
  step   1000  t = 6.136e-09 s  objects = 210
  ...
  step   6000  t = 1.922e-08 s  objects = 386
```

对象数先增长，随后随着复合追上产生而趋于饱和。第一个停留时间平均为 1/(2·10¹⁰ Hz) = 5·10⁻¹¹ s：模拟盒子中别无他物时，源是唯一的事件。

对于依赖局部场的过程，还有第二种空间模式：设置 `spatial_mode = 'per_cell'` 时，分解的每个单元创建一个候选事件，`field_deps` 中指定的场通过 `params` 传入。参见 `ZeroObjectEvent` 的文档字符串（docstring）和 `docs_zh_cn/USER_GUIDE.md`。

---

## 6. 随时间变化的物理参数与空间场

**概念。** `physics.params` 是速率按名称读取的具名标量；`physics.fields` 是覆盖整个模拟盒子的三维网格，当速率需要时在对象位置处插值。`physics.ic_func(physics, config)` 在第一步之前于每个 rank 上运行一次，用于向场中填充数据并注册定时更新。

`06_physics_parameters_and_fields.py` 对一群间隙原子做退火，温度从 300 K 升至 900 K，并处于一个静态应力场中，该场使迁移势垒沿 +x 方向降低。

```python
def physics_ic(physics, config):
    nx, ny, nz = physics.get_field('stress').shape
    x = np.linspace(0.0, 1.0, nx)
    physics.set_field('stress', np.broadcast_to(x[:, None, None], (nx, ny, nz)).copy())

    def ramp(physics, step, time):
        return min(900.0, 300.0 + 0.1 * step)
    physics.set_param_update('temperature', ramp, step_interval=200)
```

```python
"physics": {
    "params": {"temperature": 300.0},
    "fields": {"stress": {"shape": [41, 11, 11]}},
    "ic_func": physics_ic,
},
```

形状为 `[nx, ny, nz]` 的场，其首尾网格点位于模拟盒子的边界面上。当定时更新改变某个参数或场时，引擎会使依赖它的所有速率失效，因此下一步在各处都能看到新值。

**在速率中读取场。** 编译形式声明该场，并以参数形式接收插值后的值，位置在对象属性之后、物理参数之前：

```python
class MigrationKernel(SingleObjectEvent):
    object_types = ['I']
    rate_deps = RateDeps(obj1=['E_m', 'nu'], fields=['stress'], physics=['temperature'])

    @staticmethod
    @njit
    def rate_kernel(E_m, nu, stress, temperature):
        return nu * np.exp(-(E_m - 0.05 * stress) / (8.617333262e-5 * temperature))
```

Python 形式则请求一个插值器：

```python
    @classmethod
    def rate(cls, objs, params, config):
        o = objs[0]
        T = get_physics_param(config, 'temperature')
        stress = float(get_physics_field(config, 'stress')(o['x'], o['y'], o['z']))
        return o['nu'] * np.exp(-(o['E_m'] - COUPLING * stress) / (K_B * T))
```

**事件钩子**是一个可调用对象 `f(event_type, changed, step, time, pool, comm, config)`，在每个事件执行后于每个 rank 上运行（第 7 章详述）。这里用一个钩子每 1000 步打印温度和平均位置；放在钩子里是合适的，因为与后处理器不同，它接收 `config`，从而可以读取物理参数。

**动手试试。**

```
  step      0  t = 0.000e+00 s  T =  300.0 K  <x>/L = 0.488
  step   1000  t = 3.655e-08 s  T =  400.0 K  <x>/L = 0.466
  step   2000  t = 3.927e-08 s  T =  500.0 K  <x>/L = 0.473
  ...
```

随着 T 升高，每 1000 步的模拟时间缩短几个数量级。`<x>/L` 漂移到 0.5 以下：跳跃*速率*沿 +x 增大，但跳跃方向仍是各向同性的，因此对象在移动快的地方停留时间较短，于是聚集在慢的一侧。依赖位置的跳跃速率改变的是扩散系数，而不是运动方向；要产生漂移，需要依赖方向的速率。

运行 `--style python` 并比较：输出完全相同，因为计算的数完全相同。

**进一步的耦合。** 场或参数也可以*由事件*写入（`physics.set_param_in_event`、`physics.add_field_delta_in_event`，随广播的状态更新包在每个 rank 上应用），可以按步数或时间间隔根据对象进行更新（`fields.<name>.update_func` 配合 `queries`），也可以由外部求解器产生（`physics.solvers`、`physics.triggers`）。`docs_zh_cn/USER_GUIDE.md` 列出了这些选项。

---

## 7. 输出、观测与从保存的帧续算

观察运行中模拟的三种方式，全部在 `outputs` 中声明：

```python
"outputs": {
    "interval": 500,
    "ovito_dump": {"enabled": True, "output_dir": DUMP_DIR, "prefix": "frenkel",
                   "mode": "distributed",
                   "properties": ["id", "type", "x", "y", "z", "size", "jumps"]},
    "postprocessors": [summary],
    "event_hooks": [count_recombinations],
},
```

**OVITO 帧。** 每隔 `interval` 步写出一个 LAMMPS 风格的文本帧，最后一步之后再写一次。默认（`mode: "distributed"`）每个 rank 把自己拥有的对象写入 `prefix.<step>.rank_<r>.dump`，不做汇集；OVITO 通过通配符 `frenkel.1500.rank_*.dump` 加载一帧。`mode: "collect"` 把所有对象汇集到 rank 0 写入一个文件。各列即 `properties` 中列出的内容：标准的 `id type x y z size owner_rank type_id`，外加任何已声明的对象属性。`type` 以类型名写出。

**后处理器**每隔 `interval` 步运行（第 3 章）。

**事件钩子**在每个事件执行后于每个 rank 上运行：

```python
def count_recombinations(event_type, changed, step, time, pool, comm, config):
    if event_type == 'recombination':
        STATS['recombinations'] += 1
```

`event_type` 是触发的类的注册名称；`changed` 是广播的事件变更记录，它是一个结构化数组，每个被创建、修改或删除的对象占一行，被删除的行仍带有完整数据。每个 rank 收到相同的记录，因此在钩子中维护的计数在每个 rank 上都相同，无需你自己编写任何 MPI 代码。论文第 3 章的算例就用一个钩子收集每个被吸收缺陷的跳跃次数。

**续算。** dump 文件就是一种初始条件：

```python
"ic": {"mode": "dump", "file": DUMP_DIR, "frame": -1, "id_mode": "keep"}
```

`file` 是单个 dump 文件或一个 dump 目录；`frame` 是步数，`-1` 表示现有的最后一帧；同一帧的各 rank 文件会被合并。位置、尺寸以及文件中存在的每个已声明属性都会被恢复，每个 rank 保留位于自身子域内的对象，因此续算可以使用**不同的 rank 数**。`id_mode: "keep"` 保留文件中的 id（真正的接续）；`"renumber"` 则分配新的 id。

**动手试试。**

```bash
python 07_outputs_and_restart.py                        # 1500 steps -> out/07_dump/
mpirun -np 2 python 07_outputs_and_restart.py --restart   # 1000 more steps on two ranks
```

```
  step   1500  t = 5.136e-10 s  objects = 258  recombinations so far = 21
done: 1500 steps, 258 objects; frames ['0', '500', '1000', '1500'] in .../out/07_dump
  step      0  t = 1.167e-13 s  objects = 258  recombinations so far = 0
  ...
done: 1000 steps, 222 objects; frames ['0', '500', '1000'] in .../out/07_dump_restart
```

续算以第一次运行结束时的 258 个对象开始。在 OVITO 中打开 `out/07_dump/frenkel.1500.rank_*.dump`，按 `jumps` 列着色。

---

## 8. 并行运行与负载均衡开关

**rank 数会改变什么。** 模型中没有任何内容涉及 rank。引擎把 rank 数分解为子域组成的长方体网格，把位于每个 rank 自身子域内的对象以及其周围宽度为 `max_interaction_distance` 的鬼域交给该 rank，并*精确地*执行停留时间算法（BKL）：每一步，每个 rank 计算自己候选事件的速率，一次 `allgather` 收集各部分和以及在 rank 0 上抽取的选择随机数，每个 rank 确定同一个胜出事件，由其所有者执行该事件并广播一个状态更新包，由每个 rank 应用。无论 rank 数多少，每步都是一个事件、一次时钟推进。

因此，在 1 个和 64 个 rank 上的统计结果相同。轨迹并非逐位相同，因为候选事件目录的枚举顺序不同，选择随机数落在了不同的事件上。如果一个模型在不同 rank 数上给出不同的*平均值*，那它有 bug（通常是初始条件对空间域分解不安全，或在事件中使用了 `np.random`）。

**负载均衡。** 缺陷会聚集；此时简单的空间域分解会让大多数 rank 空闲。有三层机制重新分配工作，全部在 `simulation` 块中开关：

| 层 | 键 | 作用 | 默认值 |
|---|---|---|---|
| 1 | `object_load_balancer` | 每隔 `balance_interval` 步，当对象数的 max/mean − 1 超过 `imbalance_threshold` 时，按对象密度重新切分模拟盒子（递归坐标二分（RCB）） | 关闭；`balance_interval` 500，`imbalance_threshold` 0.2 |
| 2 | `predictive_rebalance` | 每隔 `check_interval` 步，当待计算速率的事件按计算代价加权后的不均衡度超过 `threshold` 时，把这些事件记录从过载 rank 移到欠载 rank；当所有事件类都设置了 `incremental_rate = True` 时自动关闭 | 开启；`threshold` 1.5，`check_interval` 100 |
| 3 | `runtime_load_balancer` | 在 Python 速率循环中，空闲的 rank 从繁忙的 rank 取走待处理事件；繁忙的 rank 每计算 `check_interval` 个事件查找一次空闲 rank | 开启；`check_interval` 32 |

第 1 层移动对象。第 2 层和第 3 层移动*事件记录*，这些记录是自包含的（携带其对象的副本）；被移动的记录留在接收它的 rank 上，当其某个对象发生变化时，广播的状态更新包会在那里使它失效。各层都不会改变采样的统计结果，因为每个事件恰好计入一个部分和（`tests/test_lb_rate_conservation.py` 在每一步都检查这一点）。

两个性质决定了哪一层对给定模型有帮助：

* 第 2 层和第 3 层平衡的是*尚待计算的速率*。设置 `incremental_rate = True` 时几乎所有速率都被复用，剩下可平衡的很少；当每个类都复用其速率时，第 2 层会自动关闭。
* 第 3 层只窃取 Python 速率类的事件。编译核函数一次批量计算其全部事件，因此没有可窃取的内容。

`08_parallel_runs_and_load_balancing.py` 把粒子放在靠近模拟盒子一角、边长为盒子边长 10 % 的立方体中，形成一个由单个 rank 拥有的热点。`--levels` 选择启用的层，`--rate` 选择跳跃类：`kernel` 是带速率复用的编译速率，`python` 是每步重算的 Python 速率。

```bash
mpirun -np 4 python 08_parallel_runs_and_load_balancing.py --levels ""  --steps 600
mpirun -np 4 python 08_parallel_runs_and_load_balancing.py --levels 1   --steps 600
mpirun -np 4 python 08_parallel_runs_and_load_balancing.py --levels 23  --steps 600
mpirun -np 4 python 08_parallel_runs_and_load_balancing.py --rate python --n 5000 --steps 200 --levels 3
```

在一台 10 核工作站上的测量结果（每步墙钟时间）：

| rank 数 | 负载均衡层 | `--rate kernel`，20 000 个粒子 | `--rate python`，5 000 个粒子 |
|---|---|---|---|
| 1 | 无 | 25.8 ms | 91.3 ms |
| 4 | 无 | 27.6 ms | 99.0 ms |
| 4 | 1 | 17.7 ms | 41.2 ms |
| 4 | 3 | 未运行 | 84.9 ms |
| 4 | 2 + 3 | 28.0 ms | 88.0 ms |
| 4 | 1 + 3 | 未运行 | 68.7 ms |

kernel 情形没有单独运行第 3 层或第 1 + 3 层：第 3 层在那里无事件可窃取（见下文）。

这些数字说明：

* 不做负载均衡时，四个 rank 并不比一个快：拥有热点的 rank 承担全部工作，其他 rank 都在等待。
* 第 1 层把切分面移入热点，在两种情况下收益都最大。
* 对于编译且复用的速率，第 2 层和第 3 层没有任何作用：第 2 层自动关闭，第 3 层找不到可窃取的内容。
* 对于 Python 速率，单独使用第 3 层略有帮助。一旦第 1 层已经平衡了对象，再加上第 3 层反而使每步变慢：它消除的等待少于它增加的消息和阶段末尾的同步屏障开销。

对于持续存在的热点，先用第 1 层。对于昂贵速率为 Python 且每步重算的模型，可启用第 2 层和第 3 层，并在保留它们之前先做测量。

启用第 2 层时，脚本打印的各项每步平均时间之和小于墙钟时间：重新分配本身是单独计时的。

论文中测得的强扩展和弱扩展曲线以及负载均衡研究，连同分析脚本，位于 `example/4.*`。

**实用设置。**

* 每个核一个 MPI rank，kernel 单线程：`NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1`。
* 在事件类上设置 `incremental_rate = True`，可让引擎对其对象未发生变化的事件保留原速率而不重算。当速率只依赖于对象属性、两对象间距离和不随时间变化的参数时，这样做是安全的；对于随定时参数变化或依赖会被更新的场的速率，应保持关闭。
* 池容量（`rma_object_pool`、`rma_event_pool`）会自动扩展，但每次扩展都会重建一个 MPI 窗口；应按每个 rank 的预期对象数来设定容量，而不是按全局总数。
* `stats = engine.run()` 返回 `total_steps`、`total_time`、`total_events`、`object_count`，以及每步平均值 `avg_build_time`、`avg_rate_time`、`avg_select_time`、`avg_execute_time`（单位为秒，仅 rank 0）。当运行慢于预期时，首先应查看它们。

---

## 下一步

* `docs_zh_cn/USER_GUIDE.md`：七个块中的每个键、事件类的约定、你的函数在运行时可用的 `config` 条目、校验错误。
* `docs_zh_cn/RANDOM_NUMBERS.md`：以计数器为键的随机数生成器详解。
* `docs_zh_cn/ARCHITECTURE.md`：数据布局、每步协议以及各负载均衡层。
* `example/`：论文的验证算例（自由扩散、阱强度、1D→3D 转变、并行执行下的不变性）及其基准测试，每个都附有模型脚本和运行命令。
