"""e6_inference.py — R1-E6：E2/E3 统计推断补强（R1A-M3/M4、R1B-M1/M3 修复）。

① 日块 bootstrap CI（协议承诺 B=2000，补 A-M3）；② seed 层聚合置换检验（B-M1）；
③ 独立置换种子（修 A-M4 附带 shared-seed 机械相关）；④ Holm 族校正（A-M4）；
⑤ E3 verdict 由代码计算重写 + LOGON 差距缩小全量申报（B-M3）；⑥ E4 边级/事件级 AP 日块 CI。
"""
import json, os, sys
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from sklearn.metrics import roc_auc_score, average_precision_score

BASE = '/root/autodl-tmp/projB_hetero_graph'
OUT = f'{BASE}/results/EXP-R1-E6'
E2D = f'{BASE}/results/EXP-R1-E2'
E3D = f'{BASE}/results/EXP-R1-E3'
E4D = f'{BASE}/results/EXP-R1-E4'
E47B = '/root/autodl-tmp/results/E47B_fairpool_ci'
E47BR = '/root/autodl-tmp/results/E47B_fairpool_ci_rerun'
POINTS = ['uniform_1', 'uniform_4', 'uniform_16', 'temporal_1', 'temporal_4', 'temporal_16']


# ---------------- 共享统计核（可单元测试） ----------------
def fast_auc(y, s):
    order = np.argsort(s, kind='mergesort')
    ranks = np.empty(len(s), dtype=np.float64)
    ranks[order] = np.arange(1, len(s) + 1, dtype=np.float64)
    npos = float(y.sum())
    nneg = float(len(y) - npos)
    return float((ranks[y == 1].sum() - npos * (npos + 1) / 2.0) / (npos * nneg))


def per_day_metric(day, s, y, metric='auc'):
    """逐日指标（AUC 或 AP）。返回 {day: value}。"""
    out = {}
    for d in np.unique(day):
        m = day == d
        yy, ss = y[m], s[m]
        if yy.sum() == 0 or yy.sum() == len(yy):
            continue
        is_auc = metric.startswith('auc')
        out[float(d)] = (fast_auc(yy, ss) if is_auc
                         else float(average_precision_score(yy, ss)))
    return out


def dayblock_bootstrap_ci(day, sA, sB, y, B=2000, rng=None, metric='auc'):
    """两臂同池差值的日块 bootstrap CI。返回 (ci95, obs_diff)。"""
    rng = rng or np.random.default_rng(2026)
    dA = per_day_metric(day, sA, y, metric)
    dB = per_day_metric(day, sB, y, metric)
    days = sorted(set(dA) & set(dB))
    diffs = np.array([dA[d] - dB[d] for d in days])
    obs = float(diffs.mean())
    n = len(diffs)
    boots = diffs[rng.integers(0, n, (B, n))].mean(axis=1)
    return [float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))], obs


def perm_pooled(day, sA, sB, y, flip, metric='auc'):
    """给定日翻转向量下的 AUC/AP 差（整池重算，非逐日均值——与在案口径一致）。"""
    sa, sb = sA.copy(), sB.copy()
    for f, d in zip(flip, np.unique(day)):
        if f:
            m = day == d
            sa[m], sb[m] = sB[m], sA[m]
    if metric.startswith('auc'):
        return fast_auc(y, sa) - fast_auc(y, sb)
    return float(average_precision_score(y, sa)) - float(average_precision_score(y, sb))


def perm_test_dayblock_indep(day, sA, sB, y, B=500, rng=None, metric='auc'):
    """日块配对置换（独立种子版）。返回 (obs, p)。"""
    rng = rng or np.random.default_rng(2026)
    nd = len(np.unique(day))
    if metric.startswith('auc'):
        obs = fast_auc(y, sA) - fast_auc(y, sB)
    else:
        obs = float(average_precision_score(y, sA)) - float(average_precision_score(y, sB))
    cnt = 0
    for _ in range(B):
        flip = rng.random(nd) < 0.5
        d = perm_pooled(day, sA, sB, y, flip, metric)
        if abs(d) >= abs(obs) - 1e-12:
            cnt += 1
    return float(obs), (cnt + 1) / (B + 1)


def perm_test_seedagg(day, sA, sB, y, seeds_arr, B=2000, rng=None, metric='auc'):
    """seed 层聚合置换：stat = 跨 seed 平均池 AUC/AP 差；日块翻转逐 seed 独立。
    seeds_arr: 与行同长的 seed 标签。"""
    rng = rng or np.random.default_rng(2026)
    days = np.unique(day)
    obs, nperm_days = 0.0, len(days)
    sA_all = np.empty(0)
    per_seed_rows = [np.where(seeds_arr == s)[0] for s in np.unique(seeds_arr)]
    # 观测统计量
    stats_obs = []
    for rows in per_seed_rows:
        d, a, b, yy = day[rows], sA[rows], sB[rows], y[rows]
        if metric.startswith('auc'):
            stats_obs.append(fast_auc(yy, a) - fast_auc(yy, b))
        else:
            stats_obs.append(float(average_precision_score(yy, a)) - float(average_precision_score(yy, b)))
    obs = float(np.mean(stats_obs))
    cnt = 0
    for _ in range(B):
        acc = []
        for rows in per_seed_rows:
            d, a, b, yy = day[rows], sA[rows], sB[rows], y[rows]
            flip = rng.random(nperm_days) < 0.5
            acc.append(perm_pooled(d, a, b, yy, flip, metric))
        if abs(np.mean(acc)) >= abs(obs) - 1e-12:
            cnt += 1
    return obs, (cnt + 1) / (B + 1)


def holm_adjust(ps):
    """Holm 逐步下降校正，返回校正后 p（单调 cap）。"""
    ps = np.asarray(ps, float)
    order = np.argsort(ps)
    m = len(ps)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        val = (m - rank) * ps[i]
        running = max(running, val)
        adj[i] = min(1.0, running)
    return adj


# ---------------- 数据装载 ----------------
def load_npz(path):
    z = np.load(path, allow_pickle=True)
    return z['day'], z['rel'], z['score'].astype(np.float64), z['y'].astype(int)


def e2_task(args):
    """单协议点×指标：per-seed CI + per-seed p（独立种子）+ seed 聚合 p（独立种子）。"""
    point, metric = args
    strat, ratio = point.rsplit('_', 1)
    df = f'{E2D}/{strat}{ratio}_full_30ep'
    dh = f'{E2D}/{strat}{ratio}_homo_30ep'
    import zlib
    rng_master = np.random.default_rng(zlib.crc32(f'{point}|{metric}'.encode()))
    res = {'point': point, 'metric': metric, 'per_seed': []}
    days_l, sA_l, sB_l, y_l = [], [], [], []
    for seed in range(3):
        dA, relA, scA, yA = load_npz(f'{df}/scores_seed{seed}.npz')
        dB, relB, scB, yB = load_npz(f'{dh}/scores_seed{seed}.npz')
        assert np.array_equal(yA, yB)
        if metric.startswith('auc'):
            day, sA, sB, y = dA, scA, scB, yA
        else:
            lm = relA == 'LOGON'
            day, sA, sB, y = dA[lm], scA[lm], scB[lm], yA[lm]
        ci, obs = dayblock_bootstrap_ci(day, sA, sB, y, B=2000, rng=rng_master, metric=metric)
        p_obs, p = perm_test_dayblock_indep(day, sA, sB, y, B=500, rng=rng_master, metric=metric)
        res['per_seed'].append({'seed': seed, 'obs_diff': round(p_obs, 5), 'p_indep': round(p, 5),
                                'ci95_dayblock_b2000': [round(x, 5) for x in ci]})
        days_l.append(day)
        sA_l.append(sA)
        sB_l.append(sB)
        y_l.append(y)
    # seed 层聚合（拼接 3 seed，翻转逐 seed 独立）
    day_c = np.concatenate(days_l)
    sA_c = np.concatenate(sA_l)
    sB_c = np.concatenate(sB_l)
    y_c = np.concatenate(y_l)
    seeds_c = np.concatenate([np.full(len(d), s) for s, d in enumerate(days_l)])
    agg_obs, agg_p = perm_test_seedagg(day_c, sA_c, sB_c, y_c, seeds_c, B=2000,
                                       rng=rng_master, metric=metric)
    res['seed_aggregate'] = {'obs_mean_diff': round(float(agg_obs), 5), 'p_b2000': round(agg_p, 5)}
    return res


def main():
    os.makedirs(OUT, exist_ok=True)
    out = {'exp': 'R1-E6 统计推断补强', 'date': '2026-10-24',
           'fixes': ['R1A-M3 日块 bootstrap CI B=2000', 'R1A-M4 Holm 族校正+检验族申报',
                     'R1A-M4附 独立置换种子（原 e2_closure 全部 mixed 检验共享 seed=2026）',
                     'R1B-M1 seed 层聚合检验', 'R1B-M3 E3 verdict 代码重算+LOGON 差距申报'],
           'E2': {'per_point': [], 'families': {}}, 'E3': {}, 'E4_edge_event_ci': {}}

    # ---- E2 主计算（并行）----
    tasks = [(p, m) for p in POINTS for m in ['auc_mixed', 'ap_logon']]
    with ProcessPoolExecutor(max_workers=12) as ex:
        for r in ex.map(e2_task, tasks):
            out['E2']['per_point'].append(r)
            print(f"  E2 {r['point']}/{r['metric']} agg_p={r['seed_aggregate']['p_b2000']}", flush=True)

    # ---- E2 Holm 族校正（族定义预申报）----
    fam = {}
    for metric, fname in [('auc_mixed', 'family_half2_mixed'), ('ap_logon', 'family_half1_logonAP')]:
        per_seed_ps, agg_ps = [], []
        for r in sorted([x for x in out['E2']['per_point'] if x['metric'] == metric],
                        key=lambda x: x['point']):
            for s in r['per_seed']:
                per_seed_ps.append(s['p_indep'])
            agg_ps.append(r['seed_aggregate']['p_b2000'])
        fam[fname] = {
            'family_def': f'{fname}: 6 协议点×3 seeds 逐 seed 检验（n={len(per_seed_ps)}）+ 6 点 seed 层聚合检验',
            'holm_per_seed': [round(x, 5) for x in holm_adjust(per_seed_ps)],
            'holm_aggregate': [round(x, 5) for x in holm_adjust(agg_ps)],
            'agg_p_raw': agg_ps}
    out['E2']['families'] = fam
    # 头条主张复核：temporal 训练混合池可分（聚合检验 + Holm 后）——按协议点名映射（勿用字典序切片）
    hm = dict(zip(sorted(POINTS), fam['family_half2_mixed']['holm_aggregate']))
    out['E2']['headline_recheck'] = {
        'claim': 'temporal 训练 3/3 seeds 混合池可分（full 更优）——seed 层聚合检验（B=2000，日块翻转逐 seed 独立）',
        'agg_p_raw': [r['seed_aggregate']['p_b2000'] for r in sorted(
            [x for x in out['E2']['per_point'] if x['metric'] == 'auc_mixed' and x['point'].startswith('temporal')],
            key=lambda x: x['point'])],
        'agg_p_holm_within_family': [hm['temporal_1'], hm['temporal_4'], hm['temporal_16']],
        'survives_holm': [hm['temporal_1'] < 0.05, hm['temporal_4'] < 0.05, hm['temporal_16'] < 0.05]}

    # ---- E3（串行，重算 verdict）----
    e3_res = {}
    for metric in ['auc_mixed', 'auc_logon', 'ap_logon']:
        rows32, rows33 = [], []
        for seed in range(3):
            dA, relA, scA, yA = load_npz(f'{E2D}/uniform1_full_60ep/scores_seed{seed}.npz')
            _, relH, scH, _ = load_npz(f'{E2D}/uniform1_homo_60ep/scores_seed{seed}.npz')
            _, relW, scW, _ = load_npz(f'{E3D}/homo_w33_60ep/scores_seed{seed}.npz')
            if metric == 'auc_mixed':
                day, s0 = dA, scA
                s1, s2, yy = scH, scW, yA
                met = 'auc'
            else:
                lm = relA == 'LOGON'
                day, s0 = dA[lm], scA[lm]
                s1, s2, yy = scH[lm], scW[lm], yA[lm]
                met = 'auc' if metric == 'auc_logon' else 'ap'
            rng = np.random.default_rng(5000 + seed)
            ci32, obs32 = dayblock_bootstrap_ci(day, s0, s1, yy, B=2000, rng=rng, metric=met)
            ci33, obs33 = dayblock_bootstrap_ci(day, s0, s2, yy, B=2000, rng=rng, metric=met)
            rows32.append({'seed': seed, 'obs': round(float(obs32), 6), 'ci95': [round(x, 6) for x in ci32]})
            rows33.append({'seed': seed, 'obs': round(float(obs33), 6), 'ci95': [round(x, 6) for x in ci33]})
        e3_res[metric] = {'w32': rows32, 'w33': rows33,
                          'band_w32': [round(min(r['obs'] for r in rows32), 6),
                                       round(max(r['obs'] for r in rows32), 6)],
                          'inband_w33_computed': [bool(min(r['obs'] for r in rows32) - 0.02 <= r['obs']
                                                        <= max(r['obs'] for r in rows32) + 0.02)
                                                  for r in rows33]}
    # LOGON 差距缩小（signed diff 逐 seed 移动方向，B-M3）
    d32_auc = [r['obs'] for r in e3_res['auc_logon']['w32']]
    d33_auc = [r['obs'] for r in e3_res['auc_logon']['w33']]
    d32_ap = [r['obs'] for r in e3_res['ap_logon']['w32']]
    d33_ap = [r['obs'] for r in e3_res['ap_logon']['w33']]
    e3_res['logon_gap_shift_disclosure'] = {
        'signed_diff_decreased_auc': [bool(b < a) for a, b in zip(d32_auc, d33_auc)],
        'signed_diff_decreased_ap': [bool(b < a) for a, b in zip(d32_ap, d33_ap)],
        'mean_auc_diff': [round(float(np.mean(d32_auc)), 6), round(float(np.mean(d33_auc)), 6)],
        'mean_ap_diff': [round(float(np.mean(d32_ap)), 6), round(float(np.mean(d33_ap)), 6)],
        'note': 'signed diff = full−homo，负值=homo 更优；w33 相对 w32 的 signed diff 一致下移=补偿使 LOGON 差距向 homo 更优方向移动（容量部分贡献的模式证据），与 mixed 侧反向证据并存——两方向均申报'}

    # ---- E4 边级/事件级 AP 日块 CI（借 E6 机器；E7 复用）----
    arms4 = {'full': f'{E47B}/full', 'homo_rerun': f'{E47BR}/homo', 'homoflows_rerun': f'{E47BR}/homoflows'}
    for a, b in [('full', 'homo_rerun'), ('full', 'homoflows_rerun')]:
        rows_ci = []
        for seed in range(3):
            dA, relA, scA, yA = load_npz(f'{arms4[a]}/scores_seed{seed}.npz')
            dB, relB, scB, yB = load_npz(f'{arms4[b]}/scores_seed{seed}.npz')
            lm = relA == 'LOGON'
            rng = np.random.default_rng(7000 + seed)
            ci, obs = dayblock_bootstrap_ci(dA[lm], scA[lm], scB[lm], yA[lm], B=2000,
                                            rng=rng, metric='ap')
            rows_ci.append({'seed': seed, 'obs_ap_diff': round(obs, 6),
                            'ci95': [round(x, 6) for x in ci]})
        out['E4_edge_event_ci'][f'{a}_vs_{b}_edgeAP_logon'] = rows_ci

    out['E3'] = e3_res
    with open(f'{OUT}/e6_inference.json', 'w') as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    print('E2 headline:', json.dumps(out['E2']['headline_recheck']))
    print('E6_DONE')


if __name__ == '__main__':
    main()
