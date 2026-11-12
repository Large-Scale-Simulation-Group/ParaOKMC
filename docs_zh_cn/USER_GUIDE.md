# ParaOKMC 用户指南

*English original: [`docs/USER_GUIDE.md`](../docs/USER_GUIDE.md)*

编写模型的参考手册。入门请先阅读 `docs_zh_cn/TUTORIAL.md`；
设计说明见 `docs_zh_cn/ARCHITECTURE.md`。

一个模型由以下部分组成：

1. 每种事件类型一个 Python 类（第 2 节），
2. 一个包含七个配置块的字典（第 3 节），
3. `engine = OKMCengine(config); stats = engine.run(); engine.destroy()`。

同一脚本既可在单个 rank 上运行，也可在 `mpirun -np N` 下运行。每个核心运行一个 rank，
kernel 设为单线程：`export NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1`。

```python
from paraokmc import (OKMCengine, ZeroObjectEvent, SingleObjectEvent, PairObjectEvent,
                      RateDeps, change_obj, delete_obj, new_obj,
                      get_physics_param, get_physics_field, ConfigValidationError)
```

---

## 1. 对象

对象是一条记录，包含以下字段：

| 字段 | 含义 |
|---|---|
| `id` | 64 位整数，在整个运行中唯一；由初始条件指定，新建对象则由引擎分配 |
| `type` | 类型名（用户代码中为字符串；内部为整数 `type_id`） |
| `x`, `y`, `z` | 在模拟盒子中的位置，长度单位与 `lattice_constant` 相同（示例中均为 Å） |
| `size` | 直径：对象的影响范围为 `size/2 + max_interaction_distance`；影响范围超出邻居鬼域容纳能力的对象称为*大对象*，在每个 rank 上都保存一份 |
| `float_props` | 为其类型声明的标量属性，按启动时分配的固定槽位顺序排列 |

在 `rate()` 和 `execute()` 中，对象以字典形式传入，包含上述字段以及按名称访问的每个已声明属性
（`obj['E_m']`）。在后处理器和事件钩子中，对象是含原始字段的 NumPy 结构化数组；
`pool.get_type_name(rec)` 给出类型名，`config['_props_mapping']['float_props'][type][prop]` 给出属性的槽位。

---

## 2. 事件类

### 2.1 选择基类

| 基类 | 涉及对象 | 候选事件 | 典型用途 |
|---|---|---|---|
| `ZeroObjectEvent` | 无 | 每次运行一个（默认）或每个单元一个 | 源项、全局过程 |
| `SingleObjectEvent` | 一个 | 所列类型的每个对象一个 | 迁移、发射、转变 |
| `PairObjectEvent` | 两个 | 所列类型中距离在 `max_distance` 以内的每对对象一个 | 复合、俘获、合并 |

### 2.2 类属性

| 属性 | 适用于 | 含义 | 默认值 |
|---|---|---|---|
| `object_types` | 单对象 | 事件适用的类型名列表 | 必需 |
| `type_pairs` | 双对象 | `(typeA, typeB)` 元组列表；对编译速率而言，每个元组的第一个成员为 `obj1` | 必需 |
| `max_distance` | 双对象 | 候选对的截断距离（最小镜像距离）；不得超过 `domain.max_interaction_distance` | 必需 |
| `params` | 全部 | 引擎放入 `rate()` 的 `params` 字典中的名称；在构建事件时解析，先查对象属性，再查 `physics.params`；对双对象事件，名称 `distance` 给出两对象间距离；最多 8 个名称 | `[]` |
| `rate_deps`, `rate_kernel` | 全部 | 编译速率（2.3） | 无 |
| `incremental_rate` | 全部 | `True`：对象未发生变化的事件保留其速率，不重新计算 | `False` |
| `weight` | 全部 | 负载均衡各层使用的相对代价；未设置时自动检测（编译核函数（kernel）为 1，Python 速率为 1000） | auto |
| `spatial_mode` | 零对象 | `'global'`（一个事件）或 `'per_cell'`（每个分解单元一个，`field_deps` 中的场通过 `params` 传入） | `'global'` |
| `owner_rank_mode` | 零对象 | `'rank0'`（事件位于 rank 0）或 `'round_robin'`（每个 rank 一个） | `'rank0'` |
| `field_deps` | 零对象（per_cell） | 在单元中心插值后放入 `params` 的场名 | `[]` |

### 2.3 速率

以下两种形式必须且只能提供一种。

**Python。** `rate(cls, objs, params, config) -> float`，每个候选事件调用一次。
`objs` 为对象字典列表（零对象事件为空），`params` 为根据 `params` 属性构建的字典，
`config` 为运行时配置（第 5 节）。它可以读取坐标，调用 `get_physics_param(config, name)` 和
`get_physics_field(config, name)`（返回插值函数 `f(x, y, z)`），或进行其他任何操作。它必须是确定性的，
且不得抽取随机数。

**编译。** `rate_deps = RateDeps(obj1=[...], obj2=[...], fields=[...], physics=[...],
kind=None)` 声明输入；`rate_kernel` 是一个 Numba `@njit` 标量函数（也接受普通 Python 函数，
但较慢），其位置参数依次为：

1. `obj1` 的属性，
2. `obj2` 的属性（双对象事件），
3. `distance`（双对象事件），
4. 在 `obj1` 位置插值得到的 `fields` 值，
5. `physics` 标量。

引擎对整个事件数组一次调用完成求值。坐标不会传入；几何条件要么用 Python 编写，要么读取由
`execute()` 镜像保存在属性中的坐标（`docs_zh_cn/TUTORIAL.md` 第 2 章）。`kind` 由声明推断
（仅 `obj1`：单对象；`obj1` 和 `obj2`：双对象；均无：零对象）；当双对象或单对象事件的 kernel 未声明任何对象属性时，
必须显式给出（`RateDeps(kind='pair')`）。
`obj1`/`obj2` 中的每个名称都必须是事件所适用的每种类型的已声明属性；`physics` 中的每个名称都必须是
`physics.params` 的键；`fields` 中的每个名称都必须是已声明的场。
在类内定义的 kernel 写法如下：

```python
    @staticmethod
    @njit
    def rate_kernel(E_m, nu, temperature):
        ...
```

负的或非有限的速率会被截为零。

### 2.4 事件结果

`execute(cls, event, objs, config) -> list[dict]` 是普通 Python 函数。它以列表形式返回变更，
列表由以下辅助函数构建：

| 辅助函数 | 作用 |
|---|---|
| `change_obj(obj, **fields)` | 更新已有对象的 `x`、`y`、`z`、`size` 和/或已声明属性（按名称） |
| `delete_obj(obj)` | 删除对象 |
| `new_obj(type=, x=, y=, z=, size=, float_props=[...])` | 新建对象；id 由引擎分配；`float_props` 必须是按槽位顺序排列的完整属性向量（辅助函数见 `docs_zh_cn/TUTORIAL.md` 第 3 章） |

随机数只能来自 `config['rng']`（`docs_zh_cn/RANDOM_NUMBERS.md`）。返回的列表作为一个状态更新包
广播，并在每个 rank 上应用；在同一列表中新建又删除的对象不会存在。事件也可以修改物理量
（`config['physics'].set_param_in_event(name, value)`、
`config['physics'].add_field_delta_in_event(field, position, delta, radius)`）；该修改随状态更新包传递，
并在每个 rank 上应用。事件可能设置的每个参数都须事先在 `physics.params` 中声明：物理模块未知的名称
只会在执行该事件的 rank 上创建，其他 rank 只收到其索引而收不到名称。

### 2.5 筛选候选事件

可以重写 `filter()`，将某些对象排除在单对象事件之外。其调用形式为
`filter(cls, obj_types, obj_props, config)`，传入候选对象的类型数组；
`obj_props` 目前传入的是空字典，因此筛选只能依据类型和 `config` 做判断。它返回候选对象上的布尔掩码，
或返回 `False` 以完全屏蔽该事件。若条件涉及位置或属性，应改为在速率中返回零。

---

## 3. 七个配置块

每个配置块都必须存在（注明之处允许为空字典）。未知的顶层键会被拒绝。

### 3.1 `domain`

| 键 | 类型 | 含义 |
|---|---|---|
| `lattice_constant` | float > 0 | 网格的长度单位；模拟盒子为 `lattice_grid × lattice_constant` |
| `lattice_grid` | `[nx, ny, nz]` ints > 0 | 每个轴上的单元数 |
| `periodic` | `[bool, bool, bool]` | 每个轴是否为周期性边界 |
| `max_interaction_distance` | float > 0 | 每个 rank 保留的邻居鬼域宽度；必须 ≥ 每个 `max_distance` |
| `h_factor` | int ≥ 1，可选 | 将每个晶格单元细分为 `h³` 个分解单元（更细的单元列表） |

rank 数被分解为覆盖该模拟盒子的长方体子域网格；区域分解没有其他可配置项。`box_size` 为派生量，
运行时可通过 `config['grid_size'] * config['lattice_constant']` 获得。

### 3.2 `simulation`

| 键 | 默认值 | 含义 |
|---|---|---|
| `max_steps` | 必需 | 运行到该步数后停止 |
| `max_time` | `inf` | 模拟时间超过该值（s）时停止 |
| `random_seed` | 42 | 本次运行中所有随机数的随机种子 |
| `verbose` | `False` | 引擎诊断信息及运行结束时的物理量汇总 |
| `progress_interval` | 500 | 进度行之间的步数（0：不输出） |
| `predictive_rebalance` | `{"enabled": True, "threshold": 1.5, "check_interval": 100}` | 第 2 层：每 `check_interval` 步，当待求值事件的不均衡度超过 `threshold` 时，在速率求值前迁移加权事件记录；当每个事件类都设置了 `incremental_rate = True` 时自动关闭 |
| `runtime_load_balancer` | `{"enabled": True, "check_interval": 32}` | 第 3 层：在 Python 速率循环中进行工作窃取（带编译 kernel 的事件一次批量求值，从不被窃取）；被窃取的事件与第 2 层迁移的事件一样留在窃取方；当其涉及的某个对象发生变化时，广播的状态更新包会在窃取方使其失效 |
| `object_load_balancer` | `{"enabled": False, "balance_interval": 500, "imbalance_threshold": 0.2, "min_improvement": 0.05}` | 第 1 层：每 `balance_interval` 步，当各 rank 对象数的 max/mean − 1 超过阈值时，用分布式递归坐标二分（RCB）重新计算切分，若不均衡度至少降低 `min_improvement` 则采用 |
| `adaptive_weight` | `{"enabled": False, "update_interval": 1000}` | 根据墙钟计时重新测量事件权重 |
| `rma_object_pool` | `{"initial_capacity": 1000, "auto_expand": True}` | 每个 rank 的对象池 |
| `rma_event_pool` | `{"capacity_zero": 10, "capacity_single": 5000, "capacity_pair": 5000, "auto_expand": True}` | 每个 rank 的事件池 |

事件目录的结构性增量更新始终开启。
池扩容会重建 MPI 窗口；应按预期的每 rank 对象数量设置池大小。

### 3.3 `objects`

```python
"objects": {
    "I": {"size": 1.0, "properties": {"E_m": 0.34, "nu": 6.0e12}},
    "S": {"size": 20.0},
}
```

每个类型名一个条目。`size`（float ≥ 0，默认 0）和 `properties`（标量字典）均为可选。
属性是编译速率唯一能读取的逐对象数据；其声明值既是内置初始条件赋予的默认值，也是 dump 文件
缺少该列时恢复使用的值。

### 3.4 `ic`

| `mode` | 键 | 行为 |
|---|---|---|
| `"builtin"` | `initial_conditions: {type: {"count": n, "distribution": "random" \| "lattice"}}` | 每个 rank 由随机种子生成相同的构型并保留属于自己的部分；`"random"` 均匀放置，`"lattice"` 按索引顺序填充单元中心；空字典表示初始无对象 |
| `"function"` | `func(rank, size, config) -> list[dict]` | 每个 rank 调用一次；只能返回位于本 rank 子域内的对象（`config['_domain_decomp'].get_owner_rank([x, y, z]) == rank`），每个对象形如 `{'id', 'type', 'x', 'y', 'z', 'size', 'float_props'}`，且 id 在所有 rank 间唯一 |
| `"dump"` | `file`（dump 文件或目录）、`frame`（步数；−1 = 最后一帧）、`id_mode`（`"keep"` \| `"renumber"`） | 恢复已保存的帧；各 rank 的文件会被合并；rank 数可以与写出时的运行不同 |

### 3.5 `events`

`{"name": EventClass, ...}`。该名称即事件钩子中 `event_type` 所携带的值，也是输出中报告的名称。
一个类可以以多个名称注册；未注册的类不会运行。校验会检查：类中引用的每个类型都已声明，pair 类具有
正的 `max_distance`，且每个 `rate_deps` 名称都存在。

### 3.6 `physics`

可以为空。键：

| 键 | 含义 |
|---|---|
| `params` | `{name: scalar}`；可通过 `rate_deps.physics`、`get_physics_param`、`config['physics'].get_param(name)` 读取 |
| `fields` | `{name: {"shape": [nx, ny, nz], "initial_value": v, "update_func": f, "update_policy": {...}, "queries": {...}}}`；覆盖模拟盒子的均匀网格，网格点位于边界面上，采用三线性插值 |
| `ic_func` | `f(physics, config)`：在第 0 步之前于每个 rank 上运行一次；用于填充场（`physics.set_field(name, array)`）和注册定时参数更新（`physics.set_param_update(name, f(physics, step, time) -> value, step_interval=…, time_interval=…)`） |
| `update_func` | `f(physics, step, time, config) -> bool`：每步调用的全局更新；有变化时返回 `True` |
| `solvers` | `[{"name": str, "solver": ExternalSolver, "fields": [...]}]`；求解器是一个具有 `initialize(physics, config)`、`solve(physics, step, time, object_data=None, trigger_source=None) -> {name: array \| scalar}` 和 `finalize()` 的对象 |
| `triggers` | `[{"event": name, "fields": [...], "solver": name (optional), "condition": f (optional)}]`：在指定事件发生后请求更新场 |
| `field_coupling` | `{"mode": "inward" \| "outward" \| "two_way", "enable_cache": bool}` |

场的 `update_policy`：`step_interval`、`time_interval`、`enabled`、`trigger_events`
（事件名列表）、`trigger_on_object_create/change/delete`（bool）以及 `object_types`
（列表）。`queries` 指定更新函数接收的对象快照：
`{"sinks": {"types": ["S"]}, "all": {"all": True}}`。

场的 `update_func` 按其位置参数识别：

| 签名 | 调用方式 |
|---|---|
| `f(physics, snapshots, ctx)`（第二个参数名为 `snapshots`、`snapshot`、`objects` 或 `object_snapshots`） | 传入查询快照和一个包含 `step`、`time`、`field_name`、`trigger_source`、`trigger_reason` 的上下文 |
| `f(physics, step, time, config)` | 其他情况下，四个或更多位置参数 |
| `f(physics, step, time)` | 三个 |
| `f(physics, step)` / `f(physics)` | 两个 / 一个 |

它可返回数组（新的场）、由数组和标量组成的字典（要设置的场和参数）、`True`/`False`
（自行设置场之后，表示是否有变化）或 `None`。

**物理量变化会使什么失效。** 变化从不重建事件结构；它把依赖于被修改名称的那些类的速率标记为待重新计算（状态 TODO）：

* 带 `rate_deps` 的类：`rate_deps.physics` 和 `rate_deps.fields` 中的名称；
* Python 速率类：其 `params` 中解析为物理参数的名称（事件中缓存的值也会被重写）；
* 设置了 `incremental_rate = True` 且没有 `params` 的 Python 速率类：所有参数和场，
  因为其 `rate()` 只能直接读取它们。

未设置 `incremental_rate` 的 Python 速率类本来就每步重新计算所有速率。全局 `update_func`
会使所有类失效。

### 3.7 `outputs`

可以为空。键：

| 键 | 含义 |
|---|---|
| `interval` | 两次后处理调用之间的步数（默认 100） |
| `ovito_dump` | `{"enabled": bool, "output_dir": str, "prefix": str, "mode": "distributed" \| "collect", "properties": [...]}`；每 `interval` 步及最后一步之后写出一帧 LAMMPS 风格的文本；`distributed`（默认）每个 rank 写出 `prefix.<step>.rank_<r>.dump`，`collect` 由 rank 0 写出一个文件；`properties` 可列出 `id type x y z size owner_rank type_id` 及任何已声明属性 |
| `postprocessors` | 可调用对象 `f(step, time, pool, comm)`，每 `interval` 步在每个 rank 上运行 |
| `event_hooks` | 可调用对象 `f(event_type, changed, step, time, pool, comm, config)`，每应用一个事件后在每个 rank 上运行；`changed` 是广播的结构化数组，包含新建/修改/删除的记录 |

`pool.get_local_objects()` 返回本 rank 拥有的活跃对象；`pool.get_local_count()`
返回其数量；`pool.get_type_name(rec)` 返回记录的类型名。

---

## 4. 运行

```python
engine = OKMCengine(config)      # validate, compile, decompose, build the pools (collective)
stats = engine.run()             # the step loop (collective)
engine.destroy()                 # free the MPI windows (collective)
```

`run()` 在达到 `max_steps`、`max_time` 或总速率为零时结束。它返回一个字典
（rank 0 上为实际值；其他 rank 上为零）：

| 键 | 含义 |
|---|---|
| `total_steps` | 已执行的步数 |
| `total_time` | 模拟时间（s） |
| `total_events` | 已执行的事件数（等于步数） |
| `object_count` | 结束时的活跃对象数 |
| `avg_build_time`, `avg_rate_time`, `avg_select_time`, `avg_execute_time` | 四个阶段在各 rank 上的每步平均耗时（s） |

环境变量：`NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1`（每个 rank 一个 kernel 线程）；
`PARAOKMC_VERBOSE_DUMP=1` 打印每个 dump 帧的类型映射。

---

## 5. 用户代码在运行时可从 `config` 中获取的内容

传给 `rate()`、`execute()`、初始条件函数、物理函数和事件钩子的字典是编译后的运行时配置，
而不是七个配置块组成的字典。常用条目：

| 条目 | 内容 |
|---|---|
| `config['rng']` | 可复现的随机数生成器（`random`、`uniform`、`normal`、`randint`、`choice`） |
| `config['lattice_constant']`, `config['grid_size']` | 二者之积即模拟盒子 |
| `config['objects']` | 原样的 `objects` 配置块（声明的尺寸和属性默认值） |
| `config['physics']` | `UnifiedPhysics` 实例：`get_param`、`set_param_in_event`、`get_field`、`set_field`、`get_field_interpolator`、`add_field_delta_in_event`、`set_param_update` |
| `config['_domain_decomp']` | 区域分解：`get_owner_rank([x, y, z])`、`subdomain_bounds`、`box_size`、`periodic` |
| `config['_props_mapping']['float_props'][type]` | `{property: slot}` |
| `config['_object_dtype']` | 对象记录的 NumPy dtype（`fields['float_props'][0].shape[0]` 为槽位数） |
| `config['_type_to_int']`, `config['_int_to_type']` | 类型名 ↔ 整数 id |
| `config['_user_seed']` | `simulation.random_seed` |
| `config['_current_time']` | 模拟时间 |

以下划线开头的条目由引擎注入，只读。

---

## 6. 校验错误

当配置不一致时，`OKMCengine(config)` 会在分配任何资源之前抛出 `ConfigValidationError`。
错误信息会指明配置块和键。常见情况：

| 信息（节选） | 原因 |
|---|---|
| `Invalid top-level blocks: {...}` | 七个配置块之外的键 |
| `domain.max_interaction_distance is required` | 缺少必需的键 |
| `ic.initial_conditions references unknown object type 'B'` | 类型未在 `objects` 中声明 |
| `Event 'm': SingleObjectEvent must define non-empty object_types` / `PairObjectEvent must define non-empty type_pairs` | 类缺少其类型列表 |
| `Event 'c': PairObjectEvent must define max_distance` / `max_distance must be positive number` | 截断距离缺失或非正 |
| `Event 'm': missing object property 'E_m' in objects['I']['properties'] for rate_deps.obj1` | kernel 请求了未声明的属性（坐标不是属性） |
| `Event 'm': physics quantity 'temperature' requested by rate_deps.physics is not defined in config['physics']['params']` | 缺少参数 |
| `Event 'm': field 's' requested by rate_deps.fields is not defined in config['physics']` | 缺少场 |
| `physics.fields['s'].shape must be [nx, ny, nz]` | 场定义格式错误 |
| `outputs.ovito_dump.mode must be 'distributed' ... or 'collect'` | 输出模式错误 |
| `ic.mode must be 'builtin', 'function' or 'dump'` | 初始条件模式错误 |

---

## 7. 并行安全模型检查清单

* 初始条件在每个 rank 上恰好返回位于该 rank 子域内的对象，id 全局唯一，并且在任意 rank 数下
  都给出相同的全局构型。
* 事件只从 `config['rng']` 抽取随机数；速率不抽取随机数。
* `domain.max_interaction_distance` ≥ 每个 pair 的 `max_distance`。
* 仅当类的速率只依赖于对象属性、两对象间距离和常量参数时，才设置 `incremental_rate = True`。
* 需要整个体系信息的观测量在事件钩子中累积（在每个 rank 上相同），或在后处理器中通过 `comm`
  归约。
* 池容量与预期的每 rank 对象数量相匹配。

各要点的完整算例见 `docs_zh_cn/TUTORIAL.md`，完整且经过验证的模型见 `example/`。
