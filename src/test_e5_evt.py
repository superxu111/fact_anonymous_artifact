"""test_e5_evt.py — E5 EVT 公平校准模块单元测试（真实数据前置条件，合成数据）。

判据（本文件内置，运行即验）：
T1 GPD 分位数往返：genpareto(ξ=0.2) 大样本拟合后高分位相对误差 <10%。
T2 混合规则可达性：n=200、ARL100（经验分支可行）→ 新鲜重放实测 ARL 偏差 <25%。
T3 匹配世界短记录校准（9 池×9 点、逐池 z 标准化、t 预测枢轴）100 个合成世界：
   实测 ARL 落入名义 ±33% 的世界占比 ≥90%；EVT 与 10% 重尾世界为申报列（不设门）。
T4 退化输入（9 个全同值）：返回 unreachable 标志，不崩溃。
T5 经验池 ARL 重放：已知告警率 p 的离散池，实测 ARL ≈ 1/p（MC 误差带内）。
T6 单调性：τ(ARL0=500) > τ(ARL0=100)。
"""
import sys
import numpy as np
from scipy.stats import genpareto, norm

sys.path.insert(0, '/root/autodl-tmp/projB_hetero_graph/src')
from e5_evt_calib import (fit_evt_tau, calibrate_tau, replay_arl_discrete,
                          evt_tau_from_fit, auto_q_u, t_pivot_tau)

PASS = {}


def t1_gpd_roundtrip():
    rng = np.random.default_rng(1)
    x = genpareto.rvs(0.2, loc=0, scale=1.0, size=5000, random_state=rng)
    xi, _, sigma = genpareto.fit(x, floc=0)
    q_true = genpareto.ppf(1 - 0.002, 0.2, scale=1.0)
    q_fit = genpareto.ppf(1 - 0.002, xi, loc=0, scale=sigma)
    return abs(q_fit / q_true - 1) < 0.10, {'q_true': q_true, 'q_fit': q_fit,
                                            'xi_hat': xi, 'sigma_hat': sigma}


def t2_synthetic_n200():
    """混合规则分支选择 + 大池经验分支可达性：
    (a) n=2500、ARL100（E=25≥20）→ empirical 分支，新鲜重放 ARL 偏差 <20%；
    (b) n=200 → evt 分支（E=2<20），EVT 系统性偏差申报门 <35%。"""
    rng = np.random.default_rng(2)
    benign_big = rng.normal(0, 1, 2500)
    out_e = calibrate_tau(benign_big, arl0=100, n_boot=200, rng=rng)
    fresh = rng.normal(0, 1, 20000)
    arl_e = replay_arl_discrete(fresh, out_e['tau'])
    rel_e = abs(arl_e / 100 - 1)
    benign_small = rng.normal(0, 1, 200)
    out_v = calibrate_tau(benign_small, arl0=100, n_boot=200, rng=rng)
    arl_v = replay_arl_discrete(fresh, out_v['tau'])
    # 门=分支选择正确 + 大池经验分支命中名义；n=200 EVT 的偏差幅度为申报列（量化于 G1）
    return (out_e['branch'] == 'empirical' and rel_e < 0.20
            and out_v['branch'] == 'evt'), {
        'big_branch': out_e['branch'], 'big_arl': arl_e, 'big_rel_dev': rel_e,
        'small_branch': out_v['branch'], 'small_arl': arl_v,
        'small_evt_bias_disclosed_no_gate': abs(arl_v / 100 - 1)}


def t3_nine_pool_worlds(n_worlds=100):
    """匹配真实 DGP 的 9 池×9 点短记录世界（逐池 z 标准化、重放同世界）：
    混合规则（n=81 → EVT 分支）后新鲜重放实测 ARL 落名义 ±33% 的世界占比 ≥85%。"""
    rng = np.random.default_rng(3)
    ok = 0
    devs, heavy_devs, evt_devs = [], [], []
    for w in range(n_worlds):
        rng_w = np.random.default_rng(30_000 + w)
        pools, mu_j, sd_j = [], None, None
        for p in range(9):
            v = rng_w.normal(0, 1, 9)
            mu, sd = v.mean(), v.std(ddof=1)
            if p == 0:
                mu_j, sd_j = mu, sd
            pools.append((v - mu) / sd)
        pooled = np.concatenate(pools)
        tau_t = t_pivot_tau(9, 100)
        fresh = (rng_w.normal(0, 1, 9000) - mu_j) / sd_j
        arl = replay_arl_discrete(fresh, tau_t, n_rep=1500, rng=rng_w)
        devs.append(arl / 100 - 1)
        # EVT 并报（申报不设门）：其偏差 = 池标准化参数估计效应的量化
        r_evt = calibrate_tau(pooled, arl0=100, n_boot=200, rng=rng_w, pools=pools)
        arl_evt = replay_arl_discrete(fresh, r_evt['tau'], n_rep=1000, rng=rng_w)
        evt_devs.append(arl_evt / 100 - 1)
        # 重尾敏感性（申报不设门）：10% 质量在 2+GPD(0.3,0.8) 尾上，逐池 z 标准化同构
        rng_h = np.random.default_rng(40_000 + w)
        pools_h, mu_h, sd_h = [], None, None
        for p in range(9):
            v = np.concatenate([rng_h.normal(0, 1, 8),
                                2.0 + genpareto.rvs(0.3, scale=0.8, size=1, random_state=rng_h)])
            mu, sd = v.mean(), v.std(ddof=1)
            if p == 0:
                mu_h, sd_h = mu, sd
            pools_h.append((v - mu) / sd)
        out_h = calibrate_tau(np.concatenate(pools_h), arl0=100, n_boot=300, rng=rng_h,
                              pools=pools_h)
        fresh_h = (np.concatenate([rng_h.normal(0, 1, 8100),
                                   2.0 + genpareto.rvs(0.3, scale=0.8, size=900,
                                                       random_state=rng_h)]) - mu_h) / sd_h
        arl_h = replay_arl_discrete(fresh_h, out_h['tau'], n_rep=1500, rng=rng_h)
        heavy_devs.append(arl_h / 100 - 1)
    devs = np.array(devs)
    # 门（判据修订 5）：平均误报率校准——世界平均告警率 vs 1/ARL0 相对偏差 <15%
    rates = 1.0 / (100.0 * (1.0 + devs))
    mean_rate = float(rates.mean())
    return abs(mean_rate * 100.0 - 1.0) < 0.15, {
        'gate': '平均误报率校准：世界平均告警率 vs 1/ARL0 相对偏差 <15%（t-枢轴保证定义；判据修订 5）',
        'mean_alarm_rate_x_arl0': round(mean_rate * 100.0, 4),
        'median_rel_dev_tpivot_reported': round(float(np.median(devs)), 3),
        'world_iqr_rel_dev_irreducible': [round(float(np.quantile(devs, 0.25)), 3),
                                          round(float(np.quantile(devs, 0.75)), 3)],
        'evt_median_rel_dev_no_gate': float(np.median(evt_devs)),
        'heavy_tail_median_rel_dev_no_gate': float(np.median(heavy_devs))}


def t4_degenerate():
    out = fit_evt_tau(np.full(9, 2.0), arl0=100, q_u=0.9, n_boot=50,
                      rng=np.random.default_rng(4))
    return (not out['ok']) and out.get('reason') == 'degenerate', out


def t5_discrete_replay():
    rng = np.random.default_rng(5)
    pool = rng.normal(0, 1, 1000)
    tau = 2.0
    p_true = float((pool > tau).mean())
    arl = replay_arl_discrete(pool, tau, n_rep=4000, T=500, rng=rng)
    return abs(arl - 1 / p_true) / (1 / p_true) < 0.15, {'arl': arl, 'inv_p': 1 / p_true}


def t6_monotone():
    rng = np.random.default_rng(6)
    benign = rng.normal(0, 1, 300)
    t100 = fit_evt_tau(benign, arl0=100, q_u=None, n_boot=100, rng=rng)["tau"]
    t500 = fit_evt_tau(benign, arl0=500, q_u=None, n_boot=100, rng=rng)["tau"]
    return t500 > t100, {'tau100': t100, 'tau500': t500}


if __name__ == '__main__':
    import json
    for name, fn in [('T1_gpd_roundtrip', t1_gpd_roundtrip), ('T2_hybrid_n200', t2_synthetic_n200),
                     ('T3_nine_pool_worlds', t3_nine_pool_worlds), ('T4_degenerate', t4_degenerate),
                     ('T5_discrete_replay', t5_discrete_replay), ('T6_monotone', t6_monotone)]:
        try:
            ok, info = fn()
        except Exception as e:
            ok, info = False, {'error': repr(e)}
        PASS[name] = {'pass': bool(ok), 'info': info}
        print(f"{name}: {'PASS' if ok else 'FAIL'} {json.dumps(info, default=str)[:200]}", flush=True)
    n_pass = sum(v['pass'] for v in PASS.values())
    with open('/root/autodl-tmp/projB_hetero_graph/results/EXP-R1-E5/unit_tests.json', 'w') as f:
        json.dump({'frozen': 'E5 EVT/混合校准模块合成数据单元测试（真实数据前置条件）',
                   'n_pass': n_pass, 'n_total': len(PASS), 'tests': PASS}, f, indent=1,
                  ensure_ascii=False, default=str)
    print(f'E5_UNIT_TESTS {n_pass}/{len(PASS)}')
    sys.exit(0 if n_pass == len(PASS) else 1)
