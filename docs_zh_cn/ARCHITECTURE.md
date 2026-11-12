# ParaOKMC 架构

*English original: [`docs/ARCHITECTURE.md`](../docs/ARCHITECTURE.md)*

本文说明引擎的组织方式以及一步之内发生的过程。库的使用方法见
`docs_zh_cn/TUTORIAL.md` 和 `docs_zh_cn/USER_GUIDE.md`。

ParaOKMC 在分布式内存上*严格*执行停留时间算法（BKL）：无论 rank 数是多少，每步都从全局速率分布中抽取一个事件，并推进一个时钟。并行工作是围绕这次抽样展开的全部其余工作，来源有三：

1. **空间域分解。** 每个 rank 拥有一个子域内的对象，并负责构建和计算这些对象的候选事件。
2. **两次集合通信的步协议。** 部分速率和由一次 `Allgather` 汇总，该次通信同时携带选择用的随机数；被选中事件的结果通过一条广播的状态更新包传播，所有 rank 都应用这条状态更新包。每个 rank 根据这条状态更新包推导出其余工作（哪些事件已失效、哪些速率需要重算、哪些物理量发生了变化）。
3. **三层负载均衡**，作用于对象（第 1 层）或自包含的事件记录（第 2 层和第 3 层）。

---

## 1. 模块

| 模块 | 作用 |
|---|---|
| `user_config.py` | 校验七块配置，并将其编译为内部扁平配置；`OKMCengine()` |
| `event_class.py`、`event_compiler.py` | 事件基类、注册表、辅助函数（`change_obj`、`delete_obj`、`new_obj`、`get_physics_param` 等） |
| `v16_engine.py` | 引擎本体：初始化、步循环、选择、执行、状态更新包广播 |
| `domain_decomposition.py` | 长方体 rank 网格、网格单元划分（`lattice_grid × h_factor`）、归属映射、鬼域 rank |
| `rma_object_pool.py` | 三类对象存储、边界 RMA 窗口、鬼域缓存、状态更新的应用 |
| `rma_event_pool.py` | 三个事件数组（零对象、单对象、双对象）、状态、速率和镜像 |
| `parallel_event_builder.py`、`vectorized_event_builder.py`、`jit_event_builder.py`、`jit_cell_list.py` | 事件构建：全量与增量构建、Numba kernel、网格单元链表 |
| `jit_rate_engine.py` | `RateDeps`，批量速率启动器的代码生成 |
| `incremental_update_v16.py` | 记录两步之间发生变化、新建和删除的对象 |
| `load_balancer.py` | 第 1 层：基于递归坐标二分（RCB）的对象重新划分 |
| `predictive_rebalance_v16.py` | 第 2 层：速率计算前的事件重分配 |
| `runtime_load_balancer_v16.py` | 第 3 层：速率计算期间的工作窃取 |
| `unified_physics.py` | 参数、场、插值、定时更新、求解器、触发器 |
| `random_generator.py` | 计数器式随机数生成器（`docs_zh_cn/RANDOM_NUMBERS.md`） |
| `post_processor.py` | 后处理器、事件钩子、OVITO dump 写出器和 dump 读取器 |

---

## 2. 数据布局

### 2.1 对象

对象是一条 NumPy 结构化 dtype 记录，该 dtype 在启动时根据 `objects` 块生成，字段包括：`id`（int64）、`type_id`、`x y z`、`size`、`owner_rank`、标志位 `active new changed`，以及 `float_props`——一个定长浮点数组，按槽位顺序（`config['_props_mapping']`）存放所有类型声明的属性。

每个 rank 的对象池分三类保存对象：

| 类别 | 内容 | 邻居通过何种方式可见 |
|---|---|---|
| 中心对象 | 本 rank 拥有、到子域各面的距离都大于截断半径的对象 | 不可见（无需交换） |
| 边界对象 | 本 rank 拥有、距某个面在截断半径以内的对象 | 一个 MPI RMA 窗口，邻居通过单边 `Get` 读取 |
| 大对象 | 影响范围（size/2 + 截断半径）超过某一阈值的对象，该阈值由鬼域容量导出 | 每个 rank 上各保存一份副本 |

每个 rank 还持有一个**鬼域缓存**：邻居边界对象的副本。它通过单边读取构建一次，之后根据广播的状态更新包增量保持最新，因此每一步无需重新读取窗口。

### 2.2 事件

事件记录是自包含的：它包含 128 位 id、类型 id、`status`、`rate`、`weight`、一个 `params` 数组，以及**其所涉及对象的副本**。因此一个 rank 可以计算不归它所有的事件的速率；正是这一点使第 2 层和第 3 层可以只迁移工作而不迁移对象。

事件池对每个类别（零对象、单对象、双对象）各维护一个数组。状态如下：

| 状态 | 含义 |
|---|---|
| TODO (0) | 速率需要（重新）计算 |
| BUSY (1) | 正在计算（只在速率阶段内短暂出现） |
| DONE (2) | 速率有效，并已计入本地速率和 |
| SENT (3) | 本步内已通过工作窃取交给另一个 rank；速率为 0；在下次构建前移除 |

事件池为每个类别维护一个本地速率和的镜像，在写入速率或移除事件时增量更新。

---

## 3. 一步的过程

```
build events  →  (level 2)  →  evaluate rates (level 3)  →  select  →  execute + broadcast  →  apply
     ↑                                                                                         │
     └──────────── changed / new / deleted objects, pair zone, physics changes ←───────────────┘
```

### 3.1 构建

三个类别的事件目录都**增量**更新；只有在第一步、重新划分（第 1 层）之后或用户请求时才进行全量重建。

* 构建之前，先丢弃上一轮速率阶段留下的 SENT 空壳；上一步中被删除的对象（以及在开启第 2 层或第 3 层时，被修改的对象）的事件会在所有 rank 上移除，因为这类事件可能位于任意 rank。
* **单对象事件**原地更新：重建本 rank 拥有的已修改或新建对象的事件，其余保留。
* **双对象事件**：位于网格单元级失效区之外的予以保留；失效区由每个 rank 根据上一次广播中的旧位置和新位置推导出来。失效区内的双对象事件用 Numba 网格单元链表重建。
* **零对象事件**保留，除非某个全局变化要求重建。

被保留的事件只有在其事件类设置了 `incremental_rate = True` 时才保留 DONE 速率；否则回到 TODO。由此，结构与速率是分开的：结构增量始终开启，速率复用则由每个事件类自行选择。

### 3.2 计算

带有 `rate_deps` + `rate_kernel` 的事件类的事件按类型分组，由生成的 Numba 启动器按类型一次调用完成计算（先把属性、距离、插值得到的场值和物理标量收集到连续数组中）。其余 TODO 事件逐个通过 Python `rate()` 计算。速率被截断为有限的非负值，并累加到速率和镜像中。

### 3.3 选择

Rank 0 抽取选择用的随机数。一次 `Allgather` 从每个 rank 收集 `(local rate sum, number)`；每个 rank 用相同的数据计算全局速率和与胜出 rank。胜出 rank 通过累积和在自己的数组中定位该事件。

### 3.4 执行、广播、应用

胜出 rank 运行 `execute()`，把返回的变更补全为完整的对象记录，抽取时间增量，并确定物理更新标志。随后它广播一条状态更新包：一个定长头部（事件类型、时间步长、类别、物理标志、计数），后接其声明的各数组（带旧位置的已修改对象、参数变更、场增量）。之后每个 rank

1. 把对象变更应用到自己的对象池和鬼域缓存，并迁移跨过子域面的对象；
2. 记录已修改、新建和删除的 id 供下次构建使用，并保留该状态更新包用于双对象事件的失效区；
3. 应用物理变更，运行所有定时或触发的物理更新，并丢弃依赖于已变化内容的速率（第 5 节）；
4. 以该状态更新包调用事件钩子；
5. 按广播的时间步长推进时钟。

由于每个 rank 应用的是同一条状态更新包，任何 rank 都无需询问其他 rank 发生了什么。

---

## 4. 负载均衡

三层都保持抽样统计不变：每个存活事件恰好计入一个部分速率和。`tests/test_lb_rate_conservation.py` 在每一步、对各层的每种组合检验这一恒等式。

### 第 1 层：对象重新划分（`object_load_balancer`，默认关闭）

每隔 `balance_interval` 步比较各 rank 的对象数。当 max/mean − 1 超过 `imbalance_threshold` 时，在网格单元上通过分布式递归坐标二分重新计算切分：各网格单元的对象数用 `Allreduce` 求和，每一层二分都集体决定，因此没有 rank 承担特殊角色。子域始终是与网格单元对齐的长方体。只有当新划分使不均衡度至少降低 `min_improvement`（默认 0.05）时才采用；随后对象通过一次 `Alltoallv` 迁移，并在所有 rank 上进行一次事件全量重建。

### 第 2 层：事件重分配（`predictive_rebalance`，默认开启）

每隔 `check_interval` 步（以及每次重新划分之后），各 rank 通过一次小规模 `Allgather` 比较各自 TODO 事件的总权重。超过 `threshold` 时，rank 0 根据收集到的 `(id, weight)` 列表规划一次贪心重分配，计划中的事件记录通过一次 `Alltoallv` 迁移。被迁移的事件留在接收方 rank；当其涉及的某个对象发生变化时，广播的状态更新包会在接收方使其失效，并由所有者重建。

权重来自速率计算路径（编译核函数（kernel）为 1，Python 速率为 1000，或显式给出的 `weight`），启用 `adaptive_weight` 时还来自实测计时。当所有已注册事件类都设置了 `incremental_rate = True` 时，该层自动关闭：此时几乎没有 TODO 事件，没有值得迁移的工作。

### 第 3 层：工作窃取（`runtime_load_balancer`，默认开启）

在 Python 速率循环中，繁忙的 rank 每计算 N 个事件检查一次由空闲标志组成的单边（RMA）窗口。N 取配置项 `runtime_load_balancer.check_interval`（默认 32），并自动提高到本地事件目录规模的八分之一，上限 2048。如果有 rank 空闲，它就用非阻塞点对点消息把一部分 TODO 事件发送给它们，将这些事件标记为 SENT，然后继续计算。已完成自身事件的 rank 会持续接收并计算，直到它发出的每一批都得到确认，且一次覆盖所有 rank 的非阻塞屏障完成为止，因此为选择而收集的部分速率和中不会遗漏任何被窃取的事件。与第 2 层的迁移一样，被窃取的事件随后留在窃取方。

kernel 事件类的事件在该循环之前一次性批量计算，从不被窃取。

---

## 5. 物理耦合

`UnifiedPhysics` 保存具名标量参数和定义在覆盖整个盒子的直线网格上的三维场，并采用三线性插值。变更有四种来源：参数或场的定时更新（按步数或时间间隔）、全局 `update_func`、由触发器调用的外部求解器，以及在 `execute()` 内部所做的变更（`set_param_in_event`、`add_field_delta_in_event`），后者随状态更新包传播。更新决策由胜出 rank 作出并广播，因此每个 rank 在同一步更新相同的内容。

物理变更从不触及事件结构。引擎维护一张静态表，记录哪些事件类依赖哪些名称：对 kernel 是 `rate_deps.physics` 和 `rate_deps.fields`；对 Python 速率是解析为物理参数的 `params` 条目（其缓存列被原地改写）；对设置了 `incremental_rate = True` 且未声明 `params` 的 Python 事件类，则是所有名称。只有这些事件类的速率会回到 TODO。

---

## 6. 通信汇总

| 位置 | 模式 | 通信量 |
|---|---|---|
| 选择 | 每个 rank 两个 double 的一次 `Allgather` | 每步 |
| 状态更新 | 广播一个头部及其声明的数组 | 每步；大小取决于事件改变的内容 |
| 鬼域缓存 | 从邻居的边界窗口单边 `Get` | 启动时及重新划分之后 |
| 第 1 层 | 每层二分一次各网格单元计数的 `Allreduce`，对象的 `Alltoallv` | 触发时 |
| 第 2 层 | 总量的 `Allgather`；计划的 gather/broadcast；事件的 `Alltoallv` | 每 `check_interval` 步，仅在超过阈值时迁移 |
| 第 3 层 | 空闲标志的单边读取，事件批的 `Isend`/`Recv`，ACK，`Ibarrier` | Python 速率循环期间 |

所有数组都以其结构化 dtype 的原始字节（`MPI.BYTE`）传输；步路径上不进行任何 pickle 序列化。
