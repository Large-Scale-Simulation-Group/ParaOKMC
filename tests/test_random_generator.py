#!/usr/bin/env python3
"""
测试 RandomGenerator 的确定性和全局一致性
"""

from paraokmc import RandomGenerator

def test_deterministic():
    """测试确定性：相同的 (seed, step, 调用序号) 产生相同的随机数"""
    rng1 = RandomGenerator(user_seed=42, rank=0)
    rng2 = RandomGenerator(user_seed=42, rank=0)

    # 生成10个随机数
    seq1 = [rng1.random() for _ in range(10)]
    seq2 = [rng2.random() for _ in range(10)]

    assert seq1 == seq2, "相同的 seed 应该产生相同的序列"
    print("✓ 确定性测试通过")

def test_global_consistency():
    """测试全局一致性：不同 rank 产生相同的随机数流（rank 参数不参与种子生成）"""
    rng_rank0 = RandomGenerator(user_seed=42, rank=0)
    rng_rank1 = RandomGenerator(user_seed=42, rank=1)

    seq0 = [rng_rank0.random() for _ in range(10)]
    seq1 = [rng_rank1.random() for _ in range(10)]

    assert seq0 == seq1, "不同 rank 应该产生相同的序列（全局一致设计）"
    print("✓ 全局一致性测试通过")

def test_seed_sensitivity():
    """测试种子敏感性：不同 seed 产生不同的随机数"""
    rng_seed42 = RandomGenerator(user_seed=42, rank=0)
    rng_seed43 = RandomGenerator(user_seed=43, rank=0)

    seq42 = [rng_seed42.random() for _ in range(10)]
    seq43 = [rng_seed43.random() for _ in range(10)]

    assert seq42 != seq43, "不同 seed 应该产生不同的序列"
    print("✓ 种子敏感性测试通过")

def test_step_sensitivity():
    """测试步数敏感性：不同 step 产生不同的序列，set_step 重置调用计数器"""
    rng = RandomGenerator(user_seed=42, rank=0)

    rng.set_step(0)
    seq_step0 = [rng.random() for _ in range(10)]

    rng.set_step(1)
    seq_step1 = [rng.random() for _ in range(10)]

    assert seq_step0 != seq_step1, "不同 step 应该产生不同的序列"

    # 回到 step 0，计数器重置后应重现相同序列
    rng.set_step(0)
    seq_step0_again = [rng.random() for _ in range(10)]
    assert seq_step0 == seq_step0_again, "相同 step 应该重现相同的序列"
    print("✓ 步数敏感性测试通过")

def test_methods():
    """测试不同的方法"""
    rng = RandomGenerator(user_seed=42, rank=0)

    # 测试 random()
    r = rng.random()
    assert 0 <= r < 1, "random() 应该返回 [0, 1) 范围内的值"

    # 测试 randint()
    idx = rng.randint(0, 10)
    assert 0 <= idx < 10, "randint(0, 10) 应该返回 [0, 10) 范围内的整数"

    # 测试 choice()
    choices = [-1, 1]
    val = rng.choice(choices)
    assert val in choices, "choice() 应该返回列表中的元素"

    # 测试 uniform()
    u = rng.uniform(-2.0, 3.0)
    assert -2.0 <= u < 3.0, "uniform(-2, 3) 应该返回 [-2, 3) 范围内的值"

    # 测试 normal()
    g = rng.normal(0.0, 1.0)
    assert isinstance(g, float), "normal() 无 size 时应返回标量"
    g3 = rng.normal(0.0, 2.0, 3)
    assert g3.shape == (3,), "normal(size=3) 应返回 (3,) 数组"

    print("✓ 方法测试通过")

def test_construction_anchored_to_numpy_streams():
    """锚定构造：输出必须等于 PCG64(SeedSequence((seed, step, state))) 的首输出

    这是论文中声明的构造（counter-based 独立流：NumPy SeedSequence
    元组熵 + PCG64），此测试保证实现与声明逐位一致。
    """
    import numpy as np

    rng = RandomGenerator(user_seed=42, rank=0)
    rng.set_step(7)

    # 引擎内部流：state=1（事件选择）、state=2（时间推进）
    assert rng.random_for_event_selection() == \
        np.random.default_rng((42, 7, 1)).random()
    assert rng.random_for_time_step() == \
        np.random.default_rng((42, 7, 2)).random()

    # 用户调用流：state 从 3 递增
    assert rng.random() == np.random.default_rng((42, 7, 3)).random()
    assert rng.uniform(-1.0, 1.0) == \
        np.random.default_rng((42, 7, 4)).uniform(-1.0, 1.0)
    assert rng.randint(0, 100) == \
        int(np.random.default_rng((42, 7, 5)).integers(0, 100))

    # 大种子 / 大步数不产生位域别名（旧 XOR 拼接方案的缺陷）
    big = RandomGenerator(user_seed=42 + 2**40, rank=0)
    big.set_step(7)
    small = RandomGenerator(user_seed=42, rank=0)
    small.set_step(7 + 2**8)
    assert big.random_for_event_selection() != small.random_for_event_selection(), \
        "结构化种子/步数组合不得复用同一随机流"

    print("✓ 构造锚定测试通过（SeedSequence 元组熵 + PCG64）")

if __name__ == "__main__":
    test_deterministic()
    test_global_consistency()
    test_seed_sensitivity()
    test_step_sensitivity()
    test_methods()
    test_construction_anchored_to_numpy_streams()
    print("\n所有测试通过！")
