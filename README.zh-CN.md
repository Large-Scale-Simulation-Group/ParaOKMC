# ParaOKMC

**并行对象动力学蒙特卡罗（Parallel Object Kinetic Monte Carlo）**，用于模拟辐照材料中缺陷的长时间演化：迁移、反应、团簇化以及在阱处的吸收；在分布式内存上运行，且不削弱停留时间算法。

ParaOKMC 严格保持串行统计——无论 rank 数为多少，每步都从全局速率分布中抽取一个事件，并推进一个时钟——并围绕这一点做并行化：空间域分解；每步两次集合通信的协议（一次部分速率的 allgather，同时携带选择用随机数；一次广播所产生的状态变化）；以及三层热点感知的负载均衡，作用对象是自包含的事件记录而不是对象本身。

[English README](README.md) · 版本 2.0.0 · Apache-2.0

## 安装

需要 Python ≥ 3.9，NumPy ≥ 1.19、SciPy ≥ 1.5、Numba ≥ 0.55，以及基于 **MPI-3** 库的 mpi4py ≥ 3.0（对象池和事件池使用单边 RMA 窗口）。

```bash
conda env create -f environment.yml   # one step, from the release root: dependencies, MPI and ParaOKMC
conda activate paraokmc
```

在发布目录根下运行这一条命令，即可从 conda-forge 装好全部依赖、MPI 库（MPICH）和 ParaOKMC 本身。也可以在已有 MPI 库的机器上用 pip 安装：

```bash
pip install -e .                 # from the release root
pip install -e ".[dedalus]"      # optional field-solver coupling (Dedalus, GPL-3.0; see Licensing)
```

pip 安装需要的 MPI 库和集群相关说明见 `INSTALL.md`。每个核运行一个 MPI rank，并保持核函数（kernel）单线程：

```bash
export NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1
mpirun -np 8 python my_model.py
```

## 单文件完整模型

一个模型就是一个由七个块组成的普通 Python 字典，加上它注册的事件类；`OKMCengine(config).run()` 就是全部驱动程序，无需修改即可在任意 rank 数下运行。

```python
import numpy as np
from numba import njit
from paraokmc import OKMCengine, SingleObjectEvent, PairObjectEvent, RateDeps, change_obj, delete_obj

@njit
def arrhenius(E_m, nu, temperature, k_B):
    return nu * np.exp(-E_m / (k_B * temperature))

class Migration(SingleObjectEvent):
    object_types = ['I', 'V']
    rate_deps = RateDeps(obj1=['E_m', 'nu'], physics=['temperature', 'k_B'])  # compiled batch path
    rate_kernel = arrhenius
    incremental_rate = True        # rates depend only on object properties and constants: reuse them

    @classmethod
    def execute(cls, event, objs, config):
        o, u, lc = objs[0], config['rng'], config['lattice_constant']
        th, ph = 2 * np.pi * u.random(), np.arccos(2 * u.random() - 1)
        return [change_obj(o, x=o['x'] + lc * np.sin(ph) * np.cos(th),
                              y=o['y'] + lc * np.sin(ph) * np.sin(th),
                              z=o['z'] + lc * np.cos(ph))]

class Recombination(PairObjectEvent):
    type_pairs = [('I', 'V')]
    max_distance = 2.0

    @classmethod
    def rate(cls, objs, params, config):   # plain-Python rate: anything goes, including an ML model
        return 1.0e10

    @classmethod
    def execute(cls, event, objs, config):
        return [delete_obj(objs[0]), delete_obj(objs[1])]

def on_event(event_type, changed, step, time, pool, comm, config):
    if event_type == 'recombination' and comm.rank == 0:
        print(f"step {step}: recombination at t = {time:.3e} s")

config = {
    "domain":     {"lattice_constant": 1.0, "lattice_grid": [30, 30, 30],
                   "periodic": [True, True, True], "max_interaction_distance": 3.0},
    "simulation": {"max_steps": 1000, "random_seed": 42},
    "objects":    {"I": {"size": 1.0, "properties": {"E_m": 0.3, "nu": 1e13}},
                   "V": {"size": 1.0, "properties": {"E_m": 1.0, "nu": 1e13}}},
    "ic":         {"mode": "builtin", "initial_conditions": {"I": {"count": 20, "distribution": "random"},
                                                             "V": {"count": 20, "distribution": "random"}}},
    "events":     {"migration": Migration, "recombination": Recombination},
    "physics":    {"params": {"temperature": 500.0, "k_B": 8.617e-5}},
    "outputs":    {"interval": 200,
                   "ovito_dump": {"enabled": True, "output_dir": "dump", "prefix": "state",
                                  "properties": ["id", "type", "x", "y", "z", "E_m"]},
                   "event_hooks": [on_event]},
}

engine = OKMCengine(config)
stats = engine.run()        # {'total_steps', 'total_time', 'total_events', 'object_count', 'avg_*_time'}
engine.destroy()
```

### 七个块

| 块 | 内容 |
|---|---|
| `domain` | `lattice_constant`、`lattice_grid` `[nx, ny, nz]`、各轴的 `periodic`、`max_interaction_distance`（决定 鬼域宽度）。rank 数会被分解为覆盖该盒子的长方体网格。 |
| `simulation` | `max_steps`、`max_time`、`random_seed`；下文所述的负载均衡开关；池容量（`rma_object_pool`、`rma_event_pool`，自动扩容）。 |
| `objects` | 每种类型一项：`size`（计入影响范围的半径）和 `properties`，即存储在每个对象上、任何速率函数都可读取的标量用户属性。 |
| `ic` | `builtin`（随机放置，每种类型给定 `count`）、`function`（`func(rank, size, config) -> list[dict]`）或 `dump`（从保存的帧续算，见下文）。 |
| `events` | `{name: EventClass}`。 |
| `physics` | `params`（具名标量，如温度）；定义在三维网格上的 `fields`，带 `update_func`/`update_policy`/`queries`；可选的外部 `solvers` 和事件 `triggers`。速率通过 `rate_deps` 或 `get_physics_param` 读取它们；其变化会使依赖它的速率失效。 |
| `outputs` | `interval`、`ovito_dump`、`postprocessors`（可调用对象 `f(step, time, pool, comm)`）、`event_hooks`。 |

完整参考：`docs_zh_cn/USER_GUIDE.md`。入门引导：`docs_zh_cn/TUTORIAL.md`。

## 事件

事件类派生自 `ZeroObjectEvent`（全局或按格胞的过程，例如源）、`SingleObjectEvent`（迁移、发射、属性变化）或 `PairObjectEvent`（复合、反应、合并、解离），指明其适用的类型（`object_types` / `type_pairs` + `max_distance`），并提供：

- **速率**：可以是纯 Python 的 `rate(cls, objs, params, config)`，也可以是 `rate_deps` + Numba `rate_kernel`，在一次编译调用中对整个事件数组求值；
- **结果**：`execute(cls, event, objs, config)`，返回由 `new_obj`、`change_obj` 和 `delete_obj` 构成的列表。引擎将该列表作为一个状态更新包在每个 rank 上应用。

并行引擎依赖的规则：速率必须是确定性的，且不得抽取随机数；`execute` 内的随机数来自 `config['rng']`，即下文所述的计数器式随机数生成器。在类上设置 `incremental_rate = True` 后，引擎会保留其对象未发生变化的事件的速率（当速率只依赖对象属性和常数参数时是安全的；对于随时变场变化的速率，请不要开启）。

## 引擎每步做什么

1. **构建**——候选事件目录对所有三个类别都做结构化的增量更新（对偶事件通过格胞级失效区域更新，该区域由每个 rank 根据广播的新旧位置推导得到）；只有在第一步、重新划分之后或发生全局变化时才完全重建。
2. **速率**——速率在本地求值（JIT 批量或 Python），在 `incremental_rate` 允许的地方进行速率复用。
3. **选择**——一次 `allgather` 收集部分速率以及在 rank 0 上抽取的选择用随机数；每个 rank 根据累积和确定同一个获选事件。
4. **执行**——拥有该事件的 rank 执行事件、抽取时间增量并广播一个状态更新包；每个 rank 应用它、据此刷新自己的 鬼域缓存，并运行事件钩子。

**随机数。** 每次抽取都是 `(user_seed, step, state)` 的纯函数，以熵元组的形式传给 NumPy 的 `SeedSequence` → PCG64。因此一次运行就是一个可复现的序列，任何 rank 都不会偏离；权威数值仍在集合通信中交换，因此 rank 之间的浮点差异不会改变决策。

**负载均衡**（`simulation` 块，均为可选）：

| 层 | 机制 | 键 | 默认值 |
|---|---|---|---|
| 1 | 对象级重新划分：当测得的不平衡度 `max/mean − 1` 超过阈值时，递归坐标二分（RCB）按对象密度重新切分盒子 | `object_load_balancer: {enabled, balance_interval, imbalance_threshold}` | 关闭（间隔 500，阈值 0.2） |
| 2 | 事件级重分配：在速率求值前，将带权重的事件记录迁移到负载较低的 rank；当所有事件类都复用速率时自动关闭 | `predictive_rebalance: {enabled, threshold, check_interval}` | 开启（阈值 1.5，每 100 步） |
| 3 | 运行时工作窃取：空闲 rank 通过单边窗口从繁忙 rank 取走待处理的 Python 速率事件 | `runtime_load_balancer: {enabled, check_interval}` | 开启（每求值 32 个事件检查一次） |

三层在构造上都保持采样统计不变（每个事件恰好计入一个部分速率）。第 2 层和第 3 层根据测得的耗时作出响应，因此重复运行之间逐位一致的轨迹可复现性仅在固定划分下成立。

## 输出与续算

`outputs.ovito_dump` 写出 LAMMPS 风格的 dump 帧，可由 OVITO 直接打开。默认情况下（`mode: "distributed"`），每个 rank 各自写出 `prefix.<step>.rank_<r>.dump`，无需汇集；在 OVITO 中用通配符 `prefix.<step>.rank_*.dump` 加载。`mode: "collect"` 将全部数据汇集到 rank 0 写入一个文件。`properties` 列表可以包含任何已声明的对象属性——每个属性成为一列。每隔 `interval` 步写出一帧，并在最后一步之后再写一帧，无论运行因何结束。

保存的帧可作为初始条件：

```python
"ic": {"mode": "dump", "file": "run1/dump", "frame": -1, "id_mode": "keep"}
```

读取单个文件或整个 dump 目录（合并各 rank 的文件，`-1` 表示最后一帧），恢复位置、尺寸和所有已声明的属性，并将对象分配到新运行的各个 rank 上——续算可以使用不同的 rank 数。

`outputs.event_hooks` 在每个状态更新包应用之后于每个 rank 上调用，参数为 `(event_type, changed_objects, step, time, pool, comm, config)`；被删除的记录携带完整数据，因此诸如被吸收缺陷的跳跃次数之类的观测量无需用户编写任何 MPI 代码即可累积。

## 示例与基准测试

`example/` 包含配套论文的全部验证算例和性能测试，每节一个目录（`3.1_free_diffusion` … `4.5_step_time_budget`），每个目录都有模型脚本和确切的运行命令。计算结果（日志、CSV 表格和 OVITO 帧）不在本仓库中，随论文的程序包一起发布在 CPC Program Library；有了这些数据，`example/figures/make_pub_figures.py` 即可重新生成图 4–11。详见 `example/README.md`。

`tutorial/` 包含八个短脚本，与 `docs_zh_cn/TUTORIAL.md` 的各章一一对应（`01_first_model.py` … `08_parallel_runs_and_load_balancing.py`）。`benchmark/` 包含强/弱扩展测试的驱动脚本。

## 测试

```bash
python -m pytest -q tests
mpirun -np 2 python tests/test_dump_restart.py     # multi-rank tests are plain scripts; run them under mpirun as well
```

## 文档

英文原版文档位于 `docs/`。

- `docs_zh_cn/TUTORIAL.md`——八章，每章配有 `tutorial/` 中的一个脚本
- `docs_zh_cn/USER_GUIDE.md`——参考手册：七个块的逐键说明、事件类约定、运行时 `config` 条目、校验错误
- `docs_zh_cn/ARCHITECTURE.md`——数据设计、每步协议、三层负载均衡
- `docs_zh_cn/RANDOM_NUMBERS.md`——计数器式随机数生成器
- `CHANGELOG.md`

## 许可

ParaOKMC 以 Apache License 2.0 发布（`LICENSE`），版权归北京航空航天大学所有（见 `NOTICE`），作者名单见 `AUTHORS`。核心依赖（NumPy、SciPy、Numba、mpi4py）采用宽松的 BSD/Apache 许可证。Dedalus 仅用于可选的 `[dedalus]` 扩展以实现场求解器耦合，其许可证为 GPL-3.0：默认不安装，捆绑它的构建须遵守 GPL。
