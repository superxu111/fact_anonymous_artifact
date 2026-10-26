"""e9_auc_logon_agg.py — R2-E9（P0-4 / R1A2-M3）：auc_logon 六协议点独立种子重跑 + seed 层聚合 + Holm 族。

【实验设计自查（开工前冻结）】
验证假设：LOGON 同池「full 更差」（auc_logon 口径，R1 头条 p=0.002/0.002 出处指标）在独立种子重跑
  与 seed 层聚合下是否保持显著。
自变量：协议点{uniform,temporal}×{1,4,16} × seed{0,1,2}（30ep 网格，与 E6 主机器同数据同口径）。
因变量：池化 AUC 差 obs、逐 seed 日块置换 p（B=500，独立种子 rng_master=zlib.crc32(point|auc_logon)）、
  seed 层聚合置换 p（B=2000）、以及聚合 p 的三口径交叉（聚合置换主口径 / Stouffer 单侧 / Fisher / ACAT）。
对照与 baseline：在案 e2_closure.json flip_boundary 的 p_logon（共享 seed=2026 机器，本实验替换该机器）。
族口径（R1A2-m2 修复）：分层主口径=每族 18 逐 seed + 6 聚合共 24 检验一次性 Holm（family_half1_logonAP
  与 family_half1_logonAUC、family_half2_mixed 各自 m=24 重算）；附 48 检验大族（half1 两指标合并）Holm
  敏感性交叉，证明结论对族定义不敏感。
规模/成本：6 点×3 seeds，纯 CPU（置换向量化），预计 <15 分钟。
分析口径：与 src/e6_inference.py 完全相同的统计机器（import 复用，仅 metric=auc_logon=LOGON 行掩码+AUC）。
成功判据：逐 seed p 与在案邻域一致（置换种子不同允许档位级漂移）、方向计数申报如实；聚合 p 无论显著
  与否均申报（红线：聚合不显著≠等效）。
断言：池行数>0、正例数>0、score 无 NaN、obs 与在案池化差同向。
"""
import json, os, sys, zlib
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from scipy.stats import norm as snorm

sys.path.insert(0, '/root/autodl-tmp/projB_hetero_graph/src')
from e6_inference import (BASE, E2D, POINTS, load_npz, fast_auc, perm_pooled,
                          perm_test_dayblock_indep, perm_test_seedagg, holm_adjust)

OUT = f'{BASE}/results/EXP-R2-E9'


def acat(ps):
    """Liu & Xie 2020 (JASA) AGENT/ACAT：任意相关 p 值合并（此处作交叉申报口径）。"""
    ps = np.clip(np.asarray(ps, float), 1e-16, 1 - 1e-16)
    t = np.tan((0.5 - ps) * np.pi)
    return float(np.clip(0.5 - np.arctan(t.mean()) / np.pi, 1e-16, 1 - 1e-16))


def fisher(ps):
    ps = np.clip(np.asarray(ps, float), 1e-16, 1.0)
    chi2 = -2.0 * np.log(ps).sum()
    from scipy.stats import chi2 as chi2d
    return float(chi2d.sf(chi2, 2 * len(ps)))


def stouffer_one_sided(ps, obs_signs):
    """Stouffer 单侧（方向已对齐：obs<0 = full−homo<0 = homo 更优 = 「full 更差」方向）。"""
    ps = np.asarray(ps, float)
    # 双侧 p → 单侧：obs<0 取 p/2，obs>0 取 1-p/2
    p_one = np.where(np.asarray(obs_signs) < 0, ps / 2.0, 1.0 - ps / 2.0)
    p_one = np.clip(p_one, 1e-16, 1 - 1e-16)
    z = snorm.ppf(1 - p_one)
    Z = z.sum() / np.sqrt(len(z))
    return float(snorm.sf(Z))


def e2_task_auc_logon(point):
    """与 e6_inference.e2_task 同构，metric=auc_logon（LOGON 行 + AUC）。"""
    strat, ratio = point.rsplit('_', 1)
    df = f'{E2D}/{strat}{ratio}_full_30ep'
    dh = f'{E2D}/{strat}{ratio}_homo_30ep'
    rng_master = np.random.default_rng(zlib.crc32(f'{point}|auc_logon'.encode()))
    res = {'point': point, 'metric': 'auc_logon', 'per_seed': []}
    days_l, sA_l, sB_l, y_l = [], [], [], []
    for seed in range(3):
        dA, relA, scA, yA = load_npz(f'{df}/scores_seed{seed}.npz')
        dB, relB, scB, yB = load_npz(f'{dh}/scores_seed{seed}.npz')
        assert np.array_equal(yA, yB)
        lm = relA == 'LOGON'
        day, sA, sB, y = dA[lm], scA[lm], scB[lm], yA[lm]
        assert len(y) > 0 and y.sum() > 0, f'{point} s{seed}: LOGON 池空'
        assert np.isfinite(sA).all() and np.isfinite(sB).all(), 'score NaN'
        obs_check = fast_auc(y, sA) - fast_auc(y, sB)
        p_obs, p = perm_test_dayblock_indep(day, sA, sB, y, B=500, rng=rng_master, metric='auc')
        assert abs(p_obs - obs_check) < 1e-9, 'obs 断言失败'
        res['per_seed'].append({'seed': seed, 'obs_diff': round(p_obs, 5), 'p_indep': round(p, 5),
                                'auc_mixed_pool_n': int(len(y)), 'n_pos': int(y.sum())})
        days_l.append(day); sA_l.append(sA); sB_l.append(sB); y_l.append(y)
    day_c = np.concatenate(days_l); sA_c = np.concatenate(sA_l)
    sB_c = np.concatenate(sB_l); y_c = np.concatenate(y_l)
    seeds_c = np.concatenate([np.full(len(d), s) for s, d in enumerate(days_l)])
    agg_obs, agg_p = perm_test_seedagg(day_c, sA_c, sB_c, y_c, seeds_c, B=2000,
                                       rng=rng_master, metric='auc')
    res['seed_aggregate'] = {'obs_mean_diff': round(float(agg_obs), 5), 'p_b2000': round(agg_p, 5)}
    return res


def main(n_workers=12):
    os.makedirs(OUT, exist_ok=True)
    incase = json.load(open(f'{BASE}/results/EXP-R1-E2/e2_closure.json'))['flip_boundary']
    out = {'exp': 'R2-E9 auc_logon 六协议点独立种子重跑+聚合（P0-4/R1A2-M3）',
           'date': '2026-10-24',
           'design': '见模块 docstring（开工前冻结）',
           'machine': '与 src/e6_inference.py 完全同统计机器（import 复用）；独立置换种子 rng=zlib.crc32(point|auc_logon)',
           'per_point': [], 'families': {}, 'cross_aggregate_calibers': {}, 'incase_crossref': {}}

    tasks = list(POINTS)
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        for r in ex.map(e2_task_auc_logon, tasks):
            out['per_point'].append(r)
            print(f"  {r['point']} agg_p={r['seed_aggregate']['p_b2000']} "
                  f"per_seed_p={[x['p_indep'] for x in r['per_seed']]}", flush=True)

    # ---- 在案对照（共享 seed=2026 机器 → 独立种子机器）----
    for r in sorted(out['per_point'], key=lambda x: x['point']):
        p = r['point']
        fb = incase[p]
        out['incase_crossref'][p] = {
            'incase_p_logon_sharedseed2026': [fb[f's{i}']['p_logon'] for i in range(3)],
            'incase_obs_auc_logon': [fb[f's{i}']['auc_logon_diff_full_minus_homo'] for i in range(3)],
            'indep_p': [x['p_indep'] for x in r['per_seed']],
            'indep_obs': [x['obs_diff'] for x in r['per_seed']],
            'agg_p_b2000': r['seed_aggregate']['p_b2000'],
            'direction_count_full_worse': sum(1 for x in r['per_seed'] if x['obs_diff'] < 0)}

    # ---- 聚合三口径交叉（V-R2-2）----
    agg_ps, obss = {}, {}
    for r in out['per_point']:
        agg_ps[r['point']] = r['seed_aggregate']['p_b2000']
        obss[r['point']] = r['seed_aggregate']['obs_mean_diff']
    cross = {}
    for grp, pts in [('uniform', [p for p in POINTS if p.startswith('uniform')]),
                     ('temporal', [p for p in POINTS if p.startswith('temporal')])]:
        ps = [agg_ps[p] for p in sorted(pts)]
        sg = [obss[p] for p in sorted(pts)]
        cross[grp] = {'points': sorted(pts), 'agg_p_b2000': ps, 'obs_mean': sg,
                      'stouffer_one_sided': stouffer_one_sided(ps, sg),
                      'fisher': fisher(ps), 'acat': acat(ps)}
    out['cross_aggregate_calibers'] = cross

    # ---- Holm 族：分层主口径（m=24 一次性）+ half1 大族（m=48）敏感性（R1A2-m2 修复）----
    def collect(metric_pts):
        per, agg = [], []
        for p in sorted(metric_pts):
            r = next(x for x in out['per_point'] if x['point'] == p)
            per += [x['p_indep'] for x in r['per_seed']]
            agg.append(r['seed_aggregate']['p_b2000'])
        return per, agg

    fams = {}
    # 旧交付重算：family_half1_logonAP / family_half2_mixed 统一 m=24 一次性 Holm（原交付按 m=18/6 两套）
    e6 = json.load(open(f'{BASE}/results/EXP-R1-E6/e6_inference.json'))
    for fname, metric in [('family_half1_logonAP', 'ap_logon'), ('family_half2_mixed', 'auc_mixed')]:
        per, agg = [], []
        for p in sorted(POINTS):
            r = next(x for x in e6['E2']['per_point'] if x['point'] == p and x['metric'] == metric)
            per += [x['p_indep'] for x in r['per_seed']]
            agg.append(r['seed_aggregate']['p_b2000'])
        adj = holm_adjust(per + agg)
        fams[fname] = {'family_def': f'{fname}: 18 逐 seed + 6 聚合 = m=24 一次性 Holm（R1A2-m2 修复：'
                                     f'原交付按 m=18/m=6 分层两套，本表统一为全族一次校正）',
                       'holm_per_seed_m24': [round(x, 5) for x in adj[:18]],
                       'holm_aggregate_m24': [round(x, 5) for x in adj[18:]],
                       'agg_p_raw': agg,
                       'point_order': sorted(POINTS),
                       'supersedes': 'e6_inference.json E2.families.%s（原 holm_per_seed 按 m=18、'
                                     'holm_aggregate 按 m=6 分层执行）' % fname}
    # 新族：family_half1_logonAUC（本实验）
    per, agg = collect(POINTS)
    adj = holm_adjust(per + agg)
    fams['family_half1_logonAUC'] = {
        'family_def': 'family_half1_logonAUC: 6 协议点×3 seeds 逐 seed（18）+ 6 聚合 = m=24 一次性 Holm',
        'holm_per_seed_m24': [round(x, 5) for x in adj[:18]],
        'holm_aggregate_m24': [round(x, 5) for x in adj[18:]],
        'agg_p_raw': agg, 'point_order': sorted(POINTS),
        'note': 'P0-4 新增：auc_logon 口径（R1 头条 p=0.002 出处指标）首次进入 Holm 族'}
    # half1 大族敏感性：ap_logon + auc_logon 两指标合并 m=48
    per_ap, agg_ap = [], []
    for p in sorted(POINTS):
        r = next(x for x in e6['E2']['per_point'] if x['point'] == p and x['metric'] == 'ap_logon')
        per_ap += [x['p_indep'] for x in r['per_seed']]
        agg_ap.append(r['seed_aggregate']['p_b2000'])
    adj48 = holm_adjust(per_ap + agg_ap + per + agg)
    fams['family_half1_combined_m48_sensitivity'] = {
        'family_def': '敏感性：half1 家族合并（ap_logon 24 + auc_logon 24 = m=48 一次性 Holm）',
        'holm_aggregate_m48': {'ap_logon': {p: round(v, 5) for p, v in zip(sorted(POINTS), adj48[18:24])},
                               'auc_logon': {p: round(v, 5) for p, v in zip(sorted(POINTS), adj48[42:48])}},
        'note': '证明 half1 结论对族定义（分层 m=24 ×2 vs 合并 m=48）不敏感'}
    out['families'] = fams

    with open(f'{OUT}/e9_auc_logon_agg.json', 'w') as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    print('E9_DONE', json.dumps(out['cross_aggregate_calibers'], ensure_ascii=False)[:600])


if __name__ == '__main__':
    main()
