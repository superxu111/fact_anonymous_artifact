"""test_seq_readout.py — R1-E1 单元测试（合成数据，全部先于真实数据运行；铁律：时序/递推模块
接入真实数据前先过本测试）。运行：python src/test_seq_readout.py；全 assert 通过打印 ALL_PASS。"""
import sys
import numpy as np
sys.path.insert(0, '/root/autodl-tmp/projB_hetero_graph/src')
from seq_readout import (cusum_stream, ewma_stream, fixed_stream, batch_runlen,
                         calibrate, measure_arl_stream, goel_wu_arl0)

RNG = np.random.RandomState(2026)


def t01_cusum_recursion():
    """递推语义：S_t=max(0,S_{t-1}+z_t−ν)、超 h 告警、告警后重置。手算对照。"""
    z = np.array([0.0, 1.0, 0.6, -2.0, 1.5, 1.2])
    nu, h = 0.5, 1.0
    S, al = cusum_stream(z, nu, h, reset_on_alarm=True)
    # 手算：S0=max(0,0-0.5)=0; S1=max(0,0+1-0.5)=0.5; S2=max(0,0.5+0.6-0.5)=0.6
    #      S3=max(0,0.6-2-0.5)=0; S4=max(0,0+1.5-0.5)=1.0（不> h=1.0，不告警）
    #      S5=max(0,1.0+1.2-0.5)=1.7>1 → 告警并重置，S5 记重置后 0（记录语义=告警后值）
    exp_S = [0.0, 0.5, 0.6, 0.0, 1.0, 0.0]
    assert np.allclose(S, exp_S), f'CUSUM 递推 {S} != {exp_S}'
    assert list(al) == [False, False, False, False, False, True], al
    # 不重置语义对照
    S2, al2 = cusum_stream(z, nu, h, reset_on_alarm=False)
    assert np.isclose(S2[-1], 1.7) and S2[4] == 1.0  # 不重置版保留累计值
    # 重置语义：首窗告警后，次窗从 0 起算
    z3 = np.array([1.6, 0.1])
    S3, al3 = cusum_stream(z3, nu, h, reset_on_alarm=True)
    assert al3[0], 'S0=1.1>1.0 应告警'
    assert np.isclose(S3[0], 0.0) and np.isclose(S3[1], 0.0), S3  # S1=max(0,0+0.1−0.5)=0


def t02_ewma_recursion():
    """EWMA 递推 Z_t=λz+(1−λ)Z_{t-1}、Z_0=0、阈值 L·sqrt(λ/(2−λ))。手算对照。"""
    z = np.array([1.0, 1.0, 1.0])
    lam, L = 0.2, 3.0
    Z, al = ewma_stream(z, lam, L)
    exp = [0.2, 0.2 * 1 + 0.8 * 0.2, 0.2 + 0.8 * 0.36]
    assert np.allclose(Z, exp), f'{Z} != {exp}'
    thr = L * np.sqrt(0.2 / 1.8)
    assert not al.any(), 'Z 未过阈值不应告警'
    Z2, al2 = ewma_stream(np.array([20.0]), lam, L)
    assert al2[0] and np.isclose(Z2[0], 4.0) and 4.0 > thr


def t03_arl0_iid_synthetic():
    """W1：iid 良性流 ARL0 校准到 100/500 两档，高精度实测落在 ±15% 带。"""
    out = {}
    for target in (100, 500):
        cal = calibrate('cusum', target, RNG, n_rep=400)
        m, p10, p90 = measure_arl_stream('cusum', cal['p1'], cal['p2'], RNG, target, n_rep=1000)
        rel = abs(m - target) / target
        assert rel < 0.15, f'CUSUM ARL0={target}: 实测 {m:.0f} 偏差 {rel:.1%}'
        out[f'cusum_arl{target}'] = {'nu': cal['p1'], 'h': round(cal['p2'], 3),
                                     'cal_arl': round(cal['measured_arl'], 1),
                                     'recheck_arl': round(m, 1), 'p10_p90': [round(p10), round(p90)]}
    for target in (100, 500):
        cal = calibrate('ewma', target, RNG, lam=0.2, n_rep=400)
        m, _, _ = measure_arl_stream('ewma', cal['p1'], cal['p2'], RNG, target, n_rep=1000)
        rel = abs(m - target) / target
        assert rel < 0.15, f'EWMA ARL0={target}: 实测 {m:.0f} 偏差 {rel:.1%}'
        out[f'ewma02_arl{target}'] = {'lam': cal['p1'], 'L': round(cal['p2'], 3),
                                      'recheck_arl': round(m, 1)}
    for target in (100, 500):
        cal = calibrate('fixed', target, RNG, n_rep=1000)
        m, _, _ = measure_arl_stream('fixed', None, cal['p2'], RNG, target, n_rep=2000)
        rel = abs(m - target) / target
        assert rel < 0.15, f'fixed ARL0={target}: 实测 {m:.0f} 偏差 {rel:.1%}'
        out[f'fixed_arl{target}'] = {'tau': round(cal['p2'], 3), 'recheck_arl': round(m, 1)}
    return out


def t04_drift_detection():
    """漂移注入：+2σ 漂移自窗 100 起，3 读出在 100 窗内检出率 >95%（ARL0=500 配置下）。"""
    n_rep = 400
    T_pre, T_drift = 100, 100
    cal_c = calibrate('cusum', 500, RNG, n_rep=300)
    cal_e = calibrate('ewma', 500, RNG, lam=0.3, n_rep=300)
    cal_f = calibrate('fixed', 500, RNG, n_rep=300)
    res = {}
    for name, (kind, p1, p2) in {'cusum': ('cusum', cal_c['p1'], cal_c['p2']),
                                 'ewma03': ('ewma', cal_e['p1'], cal_e['p2']),
                                 'fixed': ('fixed', None, cal_f['p2'])}.items():
        Z = RNG.standard_normal((n_rep, T_pre + T_drift))
        Z[:, T_pre:] += 2.0
        rl = batch_runlen(Z, kind, p1, p2)
        det = (rl <= T_pre + T_drift).mean()
        med_delay = np.median(np.minimum(rl, T_pre + T_drift) - T_pre)
        res[name] = (round(float(det), 3), float(med_delay))
        assert det > 0.95, f'{name} +2σ 漂移 100 窗内检出率 {det:.1%} < 95%'
    return res


def t05_one_sided_semantics():
    """单侧上侧：下漂（负漂移）不应触发告警率显著高于名义水平。"""
    cal = calibrate('cusum', 500, RNG, n_rep=300)
    n_rep, T = 300, 2000
    Z = RNG.standard_normal((n_rep, T)) - 2.0  # 全程下漂
    rl = batch_runlen(Z, 'cusum', cal['p1'], cal['p2'])
    early = (rl < T).mean()
    assert early < 0.05, f'下漂误触发率 {early:.1%} 过高（单侧语义破坏）'
    return {'downshift_alarm_rate': round(float(early), 4)}


def t06_no_nan():
    """全零/常数流不产生 NaN，且常数高于均值时正确告警。"""
    for stream in (np.zeros(100), np.ones(100) * 3.0):
        S, al = cusum_stream(stream, 0.5, 2.0)
        assert np.isfinite(S).all()
        Z, al2 = ewma_stream(stream, 0.2, 3.0)
        assert np.isfinite(Z).all()
    _, al = cusum_stream(np.ones(50) * 3.0, 0.5, 2.0)
    assert al.all(), '常数 +3σ 流应窗窗告警'


def t07_arl_monotone():
    """单调性：h 越大实测 ARL0 越长（校准正确性的方向哨兵）。"""
    r1 = batch_runlen(RNG.standard_normal((300, 5000)), 'cusum', 0.5, 3.0).mean()
    r2 = batch_runlen(RNG.standard_normal((300, 5000)), 'cusum', 0.5, 6.0).mean()
    assert r2 > r1 * 2, f'ARL 单调性破坏: {r1} vs {r2}'


def t08_goelwu_deviation():
    """W1：①MC 实现对照文献锚点——NIST/标准表 h=4→ARL0=168、h=5→465 为双侧方案口径
    （上下两单侧并联），本项目用单侧上侧 → 推算单侧 ARL0≈2×表值（336/930），±15% 带；
    ②Wald/Goel-Wu 零漂解析式偏差实测申报（偏差数字即 W1 答案，不作校准依据）。"""
    dev = {}
    anchors = {(0.5, 4.0): 336, (0.5, 5.0): 930}
    for k, h in [(0.5, 4.0), (0.5, 5.0), (0.25, 8.0)]:
        mc = batch_runlen(RNG.standard_normal((1500, 60000)), 'cusum', k, h).mean()
        wald = goel_wu_arl0(k, h)
        e = {'mc': round(float(mc), 1), 'wald': round(float(wald), 1),
             'wald_rel_dev': round(float((wald - mc) / mc), 3)}
        if (k, h) in anchors:
            rel = (mc - anchors[(k, h)]) / anchors[(k, h)]
            e['literature_anchor_onesided'] = anchors[(k, h)]
            e['anchor_rel_dev'] = round(float(rel), 3)
            assert abs(rel) < 0.15, f'MC 偏离文献锚点(单侧口径) k={k} h={h}: {rel:.0%}'
        assert abs((wald - mc) / mc) < 1.5, f'Wald 量级失控 k={k} h={h}'
        dev[f'k{k}_h{h}'] = e
    return dev


def main():
    results = {}
    for fn, name in [(t01_cusum_recursion, '01_cusum_recursion'),
                     (t02_ewma_recursion, '02_ewma_recursion'),
                     (t03_arl0_iid_synthetic, '03_arl0_iid_synthetic'),
                     (t04_drift_detection, '04_drift_detection'),
                     (t05_one_sided_semantics, '05_one_sided_semantics'),
                     (t06_no_nan, '06_no_nan'),
                     (t07_arl_monotone, '07_arl_monotone'),
                     (t08_goelwu_deviation, '08_goelwu_deviation')]:
        r = fn()
        results[name] = r
        print(f'PASS {name}', flush=True)
    print('ALL_PASS')
    return results


if __name__ == '__main__':
    import json
    res = main()
    with open('/root/autodl-tmp/projB_hetero_graph/results/EXP-R1-E1/unit_tests.json', 'w') as f:
        json.dump({'frozen': '合成流单元测试结果（真实数据前置条件）', 'tests': res}, f, indent=1)
