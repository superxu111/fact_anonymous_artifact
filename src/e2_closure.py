"""e2_closure.py — R1-E2 收口：复现校验判定 + 协议敏感性主表 + 翻转边界判定 + 评估侧 sweep 汇总。

【口径（冻结于 criteria.json）】
复现校验：uniform 1:1 60ep 在 projB 内重产，对照在案——计数逐位；
  指标 |Δ|<0.02（full 混合池 AUC {0.8958,0.8994,0.9210}；homo {0.9314,0.8873,0.8867}；
  homo LOGON AP {0.0084,0.0168,0.0056}）。
置换检验：日块配对置换（块=评估日，11 块，随机翻转日块归属，B=500，双侧 p）——与在案口径一致。
翻转边界判定：H2 主命题 =「LOGON 同池 full 更差（2/3 seed 显著）+ 混合池统计不可分」；
  各协议点（训练负采样策略×比例）重查两半命题，任一半翻转即入翻转边界表。
"""
import json, glob, os
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

BASE = '/root/autodl-tmp/projB_hetero_graph/results/EXP-R1-E2'
OUTF = f'{BASE}/e2_closure.json'
EVAL_DAYS = [12, 13, 14, 15, 20, 21, 22, 26, 27, 28, 29]

INCASE = {
    'full': {'auc_mixed': [0.8958, 0.8994, 0.9210], 'ap_logon': None, 'auc_logon': [0.6478, 0.6831, 0.7402]},
    'homo': {'auc_mixed': [0.9314, 0.8873, 0.8867], 'ap_logon': [0.0084, 0.0168, 0.0056],
             'auc_logon': [0.8342, 0.8291, 0.7393]},
}


def fast_auc(y, s):
    """秩统计 AUC（Mann-Whitney；连续分数并列可忽略，口径=sklearn roc_auc_score）。"""
    order = np.argsort(s, kind='mergesort')
    ranks = np.empty(len(s), dtype=np.float64)
    ranks[order] = np.arange(1, len(s) + 1, dtype=np.float64)
    npos = float(y.sum())
    nneg = float(len(y) - npos)
    return float((ranks[y == 1].sum() - npos * (npos + 1) / 2.0) / (npos * nneg))


def load_scores(d, seed):
    z = np.load(f'{d}/scores_seed{seed}.npz', allow_pickle=True)
    return z['day'], z['rel'], z['score'], z['y']


def metrics_from_npz(d, seed):
    day, rel, score, y = load_scores(d, seed)
    lm = rel == 'LOGON'
    return {'auc_mixed': float(roc_auc_score(y, score)),
            'ap_logon': float(average_precision_score(y[lm], score[lm])),
            'auc_logon': float(roc_auc_score(y[lm], score[lm])),
            'n_pool': int(len(y)), 'n_pos': int(y.sum()),
            'n_pool_logon': int(lm.sum()), 'n_pos_logon': int(y[lm].sum()),
            'day': day, 'score': score, 'y': y, 'rel': rel}


def perm_test_dayblock(a, b, B=500, seed=2026):
    """a/b: (day, score, y) 三元组（同池同 y 同行序）。日块配对置换 AUC 差双侧 p。"""
    day_a, sc_a, y = a
    day_b, sc_b, y2 = b
    assert np.array_equal(y, y2), '两臂标签不一致——非同池'
    rng = np.random.default_rng(seed)
    days = np.unique(day_a)
    obs = fast_auc(y, sc_a) - fast_auc(y, sc_b)
    idx_by_day = [np.where(day_a == d)[0] for d in days]
    count = 0
    for _ in range(B):
        flip = rng.random(len(days)) < 0.5
        sa = sc_a.copy()
        sb = sc_b.copy()
        for f, idx in zip(flip, idx_by_day):
            if f:
                sa[idx], sb[idx] = sc_b[idx], sc_a[idx]
        diff = fast_auc(y, sa) - fast_auc(y, sb)
        if abs(diff) >= abs(obs) - 1e-12:
            count += 1
    return obs, (count + 1) / (B + 1)


def main():
    out = {'exp': 'R1-E2 closure', 'date': '2026-10-24', 'criteria': 'criteria.json（冻结）',
           'regression_check': {}, 'grid_30ep': {}, 'flip_boundary': {}, 'eval_sweep_table': {}}

    # ---- 1. 复现校验（60ep uniform 1:1 vs 在案） ----
    for arm, incase in INCASE.items():
        d = f'{BASE}/uniform1_{arm}_60ep'
        rows = []
        for seed in range(3):
            m = metrics_from_npz(d, seed)
            cnt_ok = (m['n_pool'], m['n_pos'], m['n_pool_logon'], m['n_pos_logon']) == (445229, 277, 131235, 277)
            d_auc = abs(m['auc_mixed'] - incase['auc_mixed'][seed])
            d_alog = abs(m['auc_logon'] - incase['auc_logon'][seed])
            d_ap = abs(m['ap_logon'] - incase['ap_logon'][seed]) if incase['ap_logon'] else None
            rows.append({'seed': seed, 're_auc_mixed': round(m['auc_mixed'], 4),
                         'in_auc_mixed': incase['auc_mixed'][seed], 'abs_d_auc': round(d_auc, 4),
                         're_auc_logon': round(m['auc_logon'], 4),
                         'in_auc_logon': incase['auc_logon'][seed],
                         're_ap_logon': round(m['ap_logon'], 4),
                         'in_ap_logon': incase['ap_logon'][seed] if incase['ap_logon'] else None,
                         'counts_exact': cnt_ok,
                         'within_band': bool(d_auc < 0.02 and d_alog < 0.02 and (d_ap is None or d_ap < 0.02))})
        out['regression_check'][arm] = rows
    rc = out['regression_check']
    rc_pass = all(r['within_band'] and r['counts_exact'] for arm in rc for r in rc[arm])
    out['regression_check_verdict'] = ('PASS：uniform 1:1 60ep 双臂在 projB 内重产，混合池/LOGON AUC/AP '
                                       '全部 |Δ|<0.02、计数逐位一致' if rc_pass else 'FAIL：见 rows')

    # ---- 2. 30ep 网格主表 ----
    grid = {}
    for strat in ['uniform', 'temporal']:
        for ratio in [1, 4, 16]:
            for arm in ['full', 'homo']:
                d = f'{BASE}/{strat}{ratio}_{arm}_30ep'
                if not os.path.exists(f'{d}/DONE.sentinel'):
                    continue
                rj = json.load(open(f'{d}/results.json'))
                for seed in range(3):
                    s = rj['seeds'][str(seed)]
                    grid[f'{strat}_{ratio}_{arm}_s{seed}'] = {
                        'ap_logon': round(s['attack_ap_logon'], 4),
                        'auc_logon': round(s['attack_auc_logon'], 4),
                        'auc_mixed': round(s['attack_auc'], 4),
                        'n_params': s['n_params']}
    out['grid_30ep'] = grid

    # ---- 3. 翻转边界：各协议点 full vs homo 置换检验 ----
    flip = {}
    for strat in ['uniform', 'temporal']:
        for ratio in [1, 4, 16]:
            df = f'{BASE}/{strat}{ratio}_full_30ep'
            dh = f'{BASE}/{strat}{ratio}_homo_30ep'
            if not (os.path.exists(f'{df}/DONE.sentinel') and os.path.exists(f'{dh}/DONE.sentinel')):
                continue
            flip[f'{strat}_{ratio}'] = {}
            for seed in range(3):
                mf = metrics_from_npz(df, seed)
                mh = metrics_from_npz(dh, seed)
                obs, p = perm_test_dayblock((mf['day'], mf['score'], mf['y']),
                                            (mh['day'], mh['score'], mh['y']), B=500)
                obs_l, p_l = None, None
                lmf, lmh = mf['rel'] == 'LOGON', mh['rel'] == 'LOGON'
                scf, sch, yl = mf['score'][lmf], mh['score'][lmh], mf['y'][lmf]
                dayl = mf['day'][lmf]
                obs_l = fast_auc(yl, scf) - fast_auc(yl, sch)
                rng = np.random.default_rng(2026 + seed)
                days = np.unique(dayl)
                idx_by_day = [np.where(dayl == d)[0] for d in days]
                cnt = 0
                for _ in range(500):
                    fl = rng.random(len(days)) < 0.5
                    sa, sb = scf.copy(), sch.copy()
                    for f, idx in zip(fl, idx_by_day):
                        if f:
                            sa[idx], sb[idx] = sch[idx], scf[idx]
                    if abs(fast_auc(yl, sa) - fast_auc(yl, sb)) >= abs(obs_l) - 1e-12:
                        cnt += 1
                p_l = (cnt + 1) / 501
                apf = float(average_precision_score(yl, scf))
                aph = float(average_precision_score(yl, sch))
                flip[f'{strat}_{ratio}'][f's{seed}'] = {
                    'auc_mixed_diff_full_minus_homo': round(obs, 4), 'p_mixed': round(p, 4),
                    'auc_logon_diff_full_minus_homo': round(obs_l, 4), 'p_logon': round(p_l, 4),
                    'ap_logon_full': round(apf, 4), 'ap_logon_homo': round(aph, 4),
                    'ap_logon_full_worse': bool(apf < aph)}
            # H2 两半命题在该协议点的状态
            ss = flip[f'{strat}_{ratio}']
            n_full_worse_ap = sum(1 for k in ss if ss[k]['ap_logon_full_worse'])
            n_mixed_insep = sum(1 for k in ss if ss[k]['p_mixed'] > 0.05)
            flip[f'{strat}_{ratio}']['half1_LOGON_full_worse'] = f'{n_full_worse_ap}/3 seeds'
            flip[f'{strat}_{ratio}']['half2_mixed_inseparable'] = f'{n_mixed_insep}/3 seeds (p>0.05)'
    out['flip_boundary'] = flip

    # ---- 4. 评估侧 sweep 汇总（uniform 训练臂为主对照） ----
    sw = {}
    for strat in ['uniform', 'temporal']:
        for ratio in [1, 4, 16]:
            for arm in ['full', 'homo']:
                d = f'{BASE}/{strat}{ratio}_{arm}_30ep'
                if not os.path.exists(f'{d}/results.json'):
                    continue
                rj = json.load(open(f'{d}/results.json'))
                acc = {}
                for seed in range(3):
                    s = rj['seeds'][str(seed)]['eval_sweep']
                    for dayk, dayv in s.items():
                        for pk, pv in dayv.items():
                            acc.setdefault(pk, {'ap': [], 'auc': []})
                            acc[pk]['ap'].append(pv['ap'])
                            acc[pk]['auc'].append(pv['auc'])
                sw[f'{strat}{ratio}_{arm}'] = {
                    pk: {'ap_mean': round(float(np.mean(v['ap'])), 6),
                         'auc_mean': round(float(np.mean(v['auc'])), 6)}
                    for pk, v in sorted(acc.items())}
    out['eval_sweep_table'] = sw

    with open(OUTF, 'w') as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    print(out['regression_check_verdict'])
    for k, v in flip.items():
        if isinstance(v, dict) and 'half1_LOGON_full_worse' in v:
            print(f"{k}: half1(LOGON full更差)={v['half1_LOGON_full_worse']} "
                  f"half2(混合池不可分)={v['half2_mixed_inseparable']}")
    print('E2_CLOSURE_DONE')


if __name__ == '__main__':
    main()
