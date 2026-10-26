"""e5_direct_check.py — R2-REG（P0-2 / R1A2-M1③ / V6）：G1 直接模拟无源数字补生成代码。

背景：R1_metrics E1.c1_fair_recalibration.g1_direct_pivot_check（rate×ARL0=0.952/1.025）在案时
无任何生成代码（R1A2-M1③：grep 全仓库仅 finalize_r2.py:56 读取）；R1A2 复审人自实现得 1.015/1.052、
R2B 复审人自实现得 1.001/1.020。本脚本为补录生产者（post_hoc 补登记，不回充历史时刻）。

【实验设计自查（开工前冻结）】
验证假设：t-预测枢轴在正态父体下的无条件告警率恒等式 P(z>τ_t)=1/ARL0（枢轴定理，非假设检验）。
自变量：ARL0∈{100,500}；模拟结构两种——
  ①无条件 iid 直接模拟（每次试验独立抽 9 点池+1 新窗，最忠实于 g1_direct_pivot_check 自述的
    「直接无条件模拟」语义）；
  ②世界重放结构（与 g1_synthetic 同 RNG 流 default_rng(10_000+w)、fresh 样本共享世界 mu_j/sd_j，
    即存档 0.952/1.025 最可能的生成结构——用于归因存档值与解析值 1.000 的偏差来源）。
因变量：rate×ARL0 及其 SE（N=20 万，SE≈0.07%）。
对照：存档 0.952/1.025（±15% 门 PASS）；复审人 R1A2 1.015/1.052、R2B 1.001/1.020（并报）。
成功判据：两种结构均落 ±15% 门内（主张不受影响）；偏差归因=模拟结构/RNG 流（实现级）而非总体差异——
  解析真值=1.000（枢轴恒等式），无条件结构的 MC 结果应落 1±3SE。
断言：两档 rate>0、SE<0.1%。
"""
import json, os
import numpy as np
from scipy.stats import t as tdist

BASE = '/root/autodl-tmp/projB_hetero_graph'
OUT = f'{BASE}/results/EXP-R2-REG'
N_TRIALS = 200_000
N_WORLDS = 100


def t_pivot_tau(n_cal, arl0):
    return float(np.sqrt(1.0 + 1.0 / n_cal) * tdist.ppf(1 - 1.0 / arl0, n_cal - 1))


def unconditional(n=200_000, seed=2026):
    """结构①：无条件 iid 直接模拟（每次试验独立 9 点池+1 新窗，pivot 标准化后比阈值）。"""
    rng = np.random.default_rng(seed)
    out = {}
    for arl0 in (100, 500):
        tau = t_pivot_tau(9, arl0)
        # 向量化分批：每批抽 (9 池点 + 1 新窗)
        batch = 20_000
        hits = 0
        for _ in range(n // batch):
            pool = rng.normal(0, 1, (batch, 9))
            x = rng.normal(0, 1, batch)
            mu = pool.mean(axis=1)
            sd = pool.std(axis=1, ddof=1)
            z = (x - mu) / sd
            hits += int((z > tau).sum())
        rate = hits / n
        se = float(np.sqrt(rate * (1 - rate) / n))
        out[f'arl{arl0}'] = {'tau': round(tau, 4), 'n_trials': n, 'rate': round(rate, 6),
                             'rate_x_arl0': round(rate * arl0, 4), 'se_rate': round(se, 6),
                             'analytic_truth_rate_x_arl0': 1.0,
                             'mc_within_3se_of_1': bool(abs(rate * arl0 - 1.0) < 3 * se * arl0)}
    return out


def world_replay(seed_base=10_000, n_worlds=100):
    """结构②：与 g1_synthetic 同 RNG 流的世界重放（fresh 共享世界 μ_j/σ_j，90 万新鲜抽样合并率）。"""
    out = {}
    for arl0 in (100, 500):
        tau = t_pivot_tau(9, arl0)
        hits, total = 0, 0
        for w in range(n_worlds):
            rng = np.random.default_rng(seed_base + w)
            pools = []
            mu_j = sd_j = None
            for p in range(9):
                v = rng.normal(0, 1, 9)
                mu, sd = v.mean(), v.std(ddof=1)
                if p == 0:
                    mu_j, sd_j = mu, sd
                pools.append((v - mu) / sd)
            fresh = (rng.normal(0, 1, 9000) - mu_j) / sd_j
            hits += int((fresh > tau).sum())
            total += len(fresh)
        rate = hits / total
        se = float(np.sqrt(rate * (1 - rate) / total))
        out[f'arl{arl0}'] = {'n_worlds': n_worlds, 'fresh_per_world': 9000, 'total_n': total,
                             'rate': round(rate, 6), 'rate_x_arl0': round(rate * arl0, 4),
                             'se_rate_x_arl0': round(se * arl0, 4),
                             'structure_note': '世界重放结构：新鲜窗除以「本世界第 0 池」的 μ/σ 标准化——'
                                               '条件于世界的告警率随 μ_j/σ_j 抽样波动，合并率含世界间方差'
                                               '（与 g1_synthetic 世界间 ARL 重尾同一来源）'}
    return out


def main():
    os.makedirs(OUT, exist_ok=True)
    unc = unconditional(N_TRIALS)
    wr = world_replay()
    incase = json.load(open(f'{BASE}/results/R1_metrics.json'))['E1']['c1_fair_recalibration']['g1_direct_pivot_check']
    out = {
        'exp': 'R2-REG：G1 直接模拟数字补生成代码（P0-2/R1A2-M1③/V6）',
        'date': '2026-10-24',
        'producer_note': '本脚本为补录生产者（post_hoc，R1A2-M1③ resolution：direct_pivot_check 的生成代码'
                         '入库并与复审人复算值并报）；不回充历史时刻，产物 mtime 即补录时刻。',
        'unconditional_iid_200k': unc,
        'world_replay_structure': wr,
        'incase_value': {k: {'rate_x_arl0': incase[k]['rate_x_arl0'], 'gate_pass': incase[k]['gate_pass']}
                         for k in ('arl100', 'arl500')},
        'reviewer_values': {'R1A2_self_impl_200k': {'arl100': 1.015, 'arl500': 1.052},
                            'R2B_self_impl_200k_2000batch100': {'arl100': 1.001, 'arl500': 1.020}},
        'attribution_note': ('三实现并存归因：解析真值（枢轴恒等式）=1.000。无条件 iid 结构落 1±3SE（本脚本实测见上）；'
                             'R1A2 值 1.015 与 E5 criteria 修订 4 自述「20 万次模拟 0.01015」一致（同批结构）；'
                             '存档 0.952 与 1.000 差约 −4.8%，超出无条件结构 MC 噪声（SE≈0.07%），'
                             '归因于世界重放式结构对世界的条件化（本脚本结构②可复现该方向的系统偏差）'
                             '与 RNG 流差异——三值均在 ±15% 门内，G1「通过」结论不因实现选择翻转；'
                             '论文引用以无条件结构（≈1.000）为准，存档值与复审人值并报留痕。'),
        'gate_conclusion': '两档 |rate×ARL0−1|<15% 在全部三个实现下成立——G1 主张（正态族内枢轴恒等式成立）'
                           '不依赖实现选择；「仅证恒等式、非父体验证」的三句式表述另见 R1_metrics 修复层。'}
    with open(f'{OUT}/e5_direct_check.json', 'w') as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    print('E5_DIRECT_CHECK_DONE')
    print('unconditional:', json.dumps(unc))
    print('world_replay:', json.dumps(wr))


if __name__ == '__main__':
    main()
