"""test_e6_inference.py — E6 统计模块单元测试（合成数据，真实数据前置条件）。"""
import sys
import numpy as np

sys.path.insert(0, '/root/autodl-tmp/projB_hetero_graph/src')
from e6_inference import (per_day_metric, dayblock_bootstrap_ci, perm_test_dayblock_indep,
                          holm_adjust)

PASS = {}


def _mk_pairs(rng, n_days=11, rows=400, effect=0.0, seed_shift=0.0):
    """构造同池两臂：两臂对同一行各有独立打分噪声，effect 使 B 臂正例排序更弱（排序差异非平移）。"""
    day = np.repeat(np.arange(n_days), rows)
    y = np.zeros(n_days * rows, dtype=int)
    y[::rows // 20] = 1                       # ~5% 正例
    base = rng.normal(0, 1, len(day))
    nA = rng.normal(0, 0.6, len(day))
    nB = rng.normal(0, 0.6, len(day))
    sA = base + y * 1.5 + nA + seed_shift
    sB = base + y * (1.5 - abs(effect)) + nB
    return day, sA, sB, y


def u1_bootstrap_ci():
    rng = np.random.default_rng(1)
    day, sA, sB, y = _mk_pairs(rng, effect=0.5)
    ci, est = dayblock_bootstrap_ci(day, sA, sB, y, B=2000, rng=rng)
    lo, hi = ci
    c1 = lo > 0                              # 有效应 → CI 不含 0
    day2, sA2, sB2, y2 = _mk_pairs(rng, effect=0.0)
    ci2, _ = dayblock_bootstrap_ci(day2, sA2, sB2, y2, B=2000, rng=rng)
    c2 = ci2[0] <= 0 <= ci2[1]               # 无效应 → CI 含 0
    return c1 and c2, {'ci_effect': [round(x, 3) for x in ci], 'ci_null': [round(x, 3) for x in ci2]}


def u2_perm_indep():
    rng = np.random.default_rng(2)
    day, sA, sB, y = _mk_pairs(rng, effect=0.8)
    obs, p1 = perm_test_dayblock_indep(day, sA, sB, y, B=500, rng=rng)
    sig = p1 < 0.05
    # 零假设：两臂同分布（effect=0, seed_shift=0）20 次中显著次数 ≤2
    n_sig = 0
    for i in range(20):
        r = np.random.default_rng(100 + i)
        d, a, b, yy = _mk_pairs(r, effect=0.0)
        _, p = perm_test_dayblock_indep(d, a, b, yy, B=300, rng=r)
        n_sig += p < 0.05
    return sig and n_sig <= 2, {'p_effect': p1, 'null_sig_20': int(n_sig)}


def u3_holm():
    ps = [0.001, 0.004, 0.02, 0.6]
    adj = holm_adjust(ps)
    exp = [0.004, 0.012, 0.04, 0.6]          # 手算：×4, ×3, ×2(但<前序累积), ×1.25→cap 1... 0.6×1.25=0.75→cap1? 无更小则不缩
    ok = abs(adj[0] - 0.004) < 1e-9 and abs(adj[1] - 0.012) < 1e-9 and abs(adj[2] - 0.04) < 1e-9
    return ok, {'adj': [round(x, 4) for x in adj]}


def u4_per_day():
    rng = np.random.default_rng(4)
    day, sA, sB, y = _mk_pairs(rng)
    m = per_day_metric(day, sA, y)
    from sklearn.metrics import roc_auc_score
    d0 = m[0.0]
    ref = roc_auc_score(y[day == 0], sA[day == 0])
    return abs(d0 - ref) < 1e-9, {'per_day_auc_day0': d0, 'sklearn': float(ref)}


if __name__ == '__main__':
    import json, os
    os.makedirs('/root/autodl-tmp/projB_hetero_graph/results/EXP-R1-E6', exist_ok=True)
    for name, fn in [('U1_bootstrap_ci', u1_bootstrap_ci), ('U2_perm_indep', u2_perm_indep),
                     ('U3_holm', u3_holm), ('U4_per_day_auc', u4_per_day)]:
        try:
            ok, info = fn()
        except Exception as e:
            ok, info = False, {'error': repr(e)}
        PASS[name] = {'pass': bool(ok), 'info': info}
        print(f"{name}: {'PASS' if ok else 'FAIL'} {json.dumps(info, default=str)[:200]}", flush=True)
    n_pass = sum(v['pass'] for v in PASS.values())
    with open('/root/autodl-tmp/projB_hetero_graph/results/EXP-R1-E6/unit_tests.json', 'w') as f:
        json.dump({'frozen': 'E6 统计模块合成数据单元测试', 'n_pass': n_pass,
                   'n_total': len(PASS), 'tests': PASS}, f, indent=1, ensure_ascii=False, default=str)
    print(f'E6_UNIT_TESTS {n_pass}/{len(PASS)}')
    sys.exit(0 if n_pass == len(PASS) else 1)
