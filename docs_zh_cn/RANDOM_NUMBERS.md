# ParaOKMC 中的随机数

*English original: [`docs/RANDOM_NUMBERS.md`](../docs/RANDOM_NUMBERS.md)*

ParaOKMC 的所有随机数都取自一个计数器式随机数生成器，因此在 rank 数固定时，一次运行完全由其随机种子决定，而且从不需要传递任何随机数状态。

## 构造方式

一次抽样由三个整数标识：用户随机种子（`simulation.random_seed`）、当前步数，以及用于区分同一步内各次抽样的*状态*索引。引擎用状态 1 进行事件选择（在 rank 0 上抽样），用状态 2 进行时间推进（由胜出 rank 抽样）；`execute()` 内的用户抽样按发生顺序依次使用状态 3、4、5、…。取值为

```python
np.random.default_rng((user_seed, step, state)).random()
```

即把该元组作为熵序列传给 NumPy 的 `SeedSequence`，由它为 PCG64 生成器设置种子，该生成器的第一个输出就是所取的数。`SeedSequence` 保证不同的整数元组给出统计独立的随机数流；这是 NumPy 推荐的独立并行随机数流构造方式。

由此可得：

* **可复现。** 相同的随机种子在每次运行中都给出相同的决策序列和相同的轨迹。
* **与 rank 无关。** 该构造不涉及 rank。第 n 步的第 k 次用户抽样在每个 rank 上取值相同，因此任何消息中都不传递随机数状态。决定一步的数值（选择用的随机数、时间推进量）仍在每步的集合通信中交换，因此各 rank 之间的浮点差异不会改变决策。
* **无状态。** 步与步之间不携带任何状态；从第 n 步的 dump 帧以相同随机种子续算时，只要其余状态完全相同，就能复现原运行从第 n 步起的各次抽样。

参考文献：M.E. O'Neill, *PCG: A family of simple fast space-efficient statistically good
algorithms for random number generation*, HMC-CS-2014-0905 (2014); C.R. Harris et al.,
*Array programming with NumPy*, Nature 585, 357–362 (2020).

## 使用方法

在 `execute()` 内，生成器为 `config['rng']`：

```python
@classmethod
def execute(cls, event, objs, config):
    rng = config['rng']
    r = rng.random()                 # float in [0, 1)
    u = rng.uniform(-1.0, 1.0)       # float in [low, high)
    g = rng.normal(0.0, 1.0)         # normal deviate
    k = rng.randint(0, 4)            # integer in [0, 4)
    s = rng.choice([-1.0, 1.0])      # one element of a sequence
    ...
```

规则：

1. **切勿在事件内使用 `np.random` 或 Python 的 `random`。** 每个 rank 会看到各自的随机数流，各 rank 对发生了什么将得出不一致的结论。
2. **速率中不得抽取随机数。** 速率何时重算由引擎决定（在 `incremental_rate` 允许时还会被复用）；随机的速率会使总速率依赖于其上次计算的时间。
3. 以函数形式编写的**初始条件**可以使用任意生成器，但每个 rank 必须先生成相同的全局构型，再保留属于自己的部分；请用 `config['_user_seed']` 为其设置随机种子（见 `docs_zh_cn/TUTORIAL.md` 第 4 章）。

## 不同 rank 数下的确定性

改变 rank 数会改变空间域分解以及候选事件的枚举顺序，因此选择用的随机数会落在不同的事件上，轨迹也随之不同。但抽样统计不变：停留时间算法在任意 rank 数下都被严格执行（见 `docs_zh_cn/TUTORIAL.md` 第 8 章及配套 CPC 论文第 3.4 节）。在 rank 数和划分都固定时，重复运行的结果逐位一致；第 2 层和第 3 层负载均衡会根据实测计时做出反应，因此可能在不同运行之间改变划分，需要逐位可复现时应将其关闭。

## API

`paraokmc.RandomGenerator(user_seed, rank=0)`：引擎实例化并作为 `config['rng']` 注入的类。`rank` 参数仅为向后兼容而保留，会被忽略。

| 方法 | 返回值 |
|---|---|
| `random()` | [0, 1) 内的浮点数 |
| `uniform(low, high)` | [low, high) 内的浮点数 |
| `normal(mu, sigma, size=None)` | 正态分布随机数 |
| `randint(low, high)` | [low, high) 内的整数 |
| `choice(seq)` | `seq` 中的一个元素 |
| `set_step(step)` | 选定步数并重置用户状态计数器（由引擎每步调用） |

测试：`tests/test_random_generator.py`（确定性、与 rank 无关、对随机种子和步数的敏感性）。
