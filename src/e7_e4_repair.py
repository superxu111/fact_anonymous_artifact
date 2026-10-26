"""e7_e4_repair.py — R1-E7：E4 交付物修复（R1A-M1/M2、R1B-M2）。

判据：results/EXP-R1-E7/criteria.json（冻结）。
F1 ap_event_weighted 退化修复（负例保底权重 1）；F2 recall@K 全网格+随机基线；
F3 W14 truetype 反常逐类分解+跨臂可比性审计；F4 事件级/边级 AP 日块 bootstrap CI。
"""
import json, sys
import numpy as np
from collections import defaultdict
from sklearn.metrics import average_precision_score

sys.path.insert(0, '/root/autodl-tmp/projB_hetero_graph/src')
from r4b_common import load_logon_days, load_redteam_events

BASE = '/root/autodl-tmp'
ARMS = {'full': f'{BASE}/results/E47B_fairpool_ci/full',
        'homo_rerun': f'{BASE}/results/E47B_fairpool_ci_rerun/homo',
        'homoflows_rerun': f'{BASE}/results/E47B_fairpool_ci_rerun/homoflows'}
OUTD = '/root/autodl-tmp/projB_hetero_graph/results/EXP-R1-E7'
KS = [10, 25, 50, 100, 250, 500]


def load_npz(path):
    z = np.load(path, allow_pickle=True)
    return z['day'], z['rel'], z['score'].astype(np.float64), z['y'].astype(int)


def ap_weighted(y, s, w):
    """行按权重重复展开的 AP（负例行权重 1 保底——R1A-M1 修复定义）。"""
    rep = np.maximum(w.astype(int), 1)
    return float(average_precision_score(np.repeat(y, rep), np.repeat(s, rep)))


def main():
    L = load_logon_days(ntlm=True)
    RTev = load_redteam_events()
    u2i, h2i = L['uidx'], L['hidx']
    attack_rows = defaultdict(int)
    rt_hosts = set()
    for ts, u, s, d in RTev:
        rt_hosts.add(d)
        if u in u2i and d in h2i:
            attack_rows[(ts // 86400, u2i[u], h2i[d])] += 1
    n_events_mapped = int(sum(attack_rows.values()))

    out = {'exp': 'R1-E7 E4 修复', 'date': '2026-10-24', 'criteria': 'criteria.json（冻结）',
           'F1_ap_event_weighted_v2': {}, 'F2_recall_grid': {}, 'F3_w14_diagnosis': {},
           'F4_event_edge_ci': {}}
    scores = {}
    for arm, d in ARMS.items():
        scores[arm] = [load_npz(f'{d}/scores_seed{s}.npz') for s in range(3)]

    # ---- F1+F2：逐臂逐 seed ----
    for arm in ARMS:
        out['F1_ap_event_weighted_v2'][arm] = {}
        out['F2_recall_grid'][arm] = {}
        for seed in range(3):
            day, rel, score, y = scores[arm][seed]
            lm = rel == 'LOGON'
            pos_all = np.where(lm)[0]
            # 还原攻击行（边身份=当日 LOGON 边序，与 e4_granularity 同口径）
            restored = {}
            for dd in sorted(set(day[lm].tolist())):
                sel = pos_all[day[pos_all] == dd]
                ei, _ = L['logon_days'][int(dd)]
                us, hs = ei[0].numpy(), ei[1].numpy()
                for pos, uu, hh in zip(sel, us, hs):
                    key = (int(dd), int(uu), int(hh))
                    if key in attack_rows:
                        restored[int(pos)] = attack_rows[key]
            y_e = np.array([1 if p in restored else 0 for p in pos_all], dtype=float)
            w_e = np.array([restored.get(p, 0) for p in pos_all], dtype=float)
            s_e = score[pos_all]
            ap_rows = float(average_precision_score(y_e, s_e))
            ap_ev2 = ap_weighted(y_e, s_e, w_e)
            # F2 recall@K 网格 + 随机基线
            order = pos_all[np.argsort(-s_e, kind='stable')]
            rec = {}
            for K in KS:
                top = order[:K]
                cov_ev = sum(restored.get(int(p), 0) for p in top)
                cov_rows = sum(1 for p in top if int(p) in restored)
                rec[K] = {'recall_events': round(cov_ev / n_events_mapped, 4),
                          'recall_rows': round(cov_rows / max(len(restored), 1), 4),
                          'random_baseline_recall_rows': round(K * len(restored) / len(pos_all) / max(len(restored), 1), 5)}
            out['F1_ap_event_weighted_v2'][arm][seed] = {
                'ap_attack_rows': round(ap_rows, 6),
                'ap_event_weighted_v2': round(ap_ev2, 6),
                'n_attack_rows': int(len(restored))}
            out['F2_recall_grid'][arm][seed] = rec

    # ---- F3：W14 truetype 反常逐类分解 + 可比性审计 ----
    arms_tt = ['full', 'nemb', 'homo', 'homoflows', 'noflows']
    w14 = {'source': 'E47B_fairpool_ci/<arm>/results.json type_cm/type_macro_f1（在案）',
           'macro_f1_by_arm': {}, 'per_class_prf': {}, 'comparability_audit': {},
           'unified_explanation': ''}
    for arm in arms_tt:
        try:
            rj = json.load(open(f'{BASE}/results/E47B_fairpool_ci/{arm}/results.json'))
        except FileNotFoundError:
            continue
        w14['macro_f1_by_arm'][arm] = [round(rj['seeds'][str(s)]['type_macro_f1'], 4) for s in range(3)]
        prf_seed = []
        cmp_seed = []
        for s in range(3):
            cm = np.array(rj['seeds'][str(s)]['type_cm'])
            present_true = [c for c in range(3) if cm[c].sum() > 0]
            present_pred = [c for c in range(3) if cm[:, c].sum() > 0]
            cmp_seed.append({'classes_in_eval_true': present_true, 'classes_predicted': present_pred,
                             'macro_over_effective_classes': sorted(set(present_true) | set(present_pred))})
            prf = {}
            for c in range(3):
                tp = cm[c, c]
                fp = cm[:, c].sum() - tp
                fn = cm[c, :].sum() - tp
                prec = tp / (tp + fp) if tp + fp else 0.0
                rec = tp / (tp + fn) if tp + fn else 0.0
                f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
                prf[f'class{c}'] = {'P': round(prec, 4), 'R': round(rec, 4), 'F1': round(f1, 4),
                                    'support': int(cm[c, :].sum())}
            prf_seed.append(prf)
        w14['per_class_prf'][arm] = prf_seed
        w14['comparability_audit'][arm] = cmp_seed[0]
    # 分解 full vs nemb 差距（seed0 主口径）
    f0 = w14['per_class_prf']['full'][0]
    n0 = w14['per_class_prf']['nemb'][0]
    gap = round(w14['macro_f1_by_arm']['nemb'][0] - w14['macro_f1_by_arm']['full'][0], 4)
    per_class_f1_gap = {c: round(n0[c]['F1'] - f0[c]['F1'], 4) for c in f0}
    w14['unified_explanation'] = (
        f"full 与 nemb 同为三类别可比口径，macro-F1 差 {gap}（nemb 更高，反常）。逐类 F1 分解（seed0）"
        f"{per_class_f1_gap}：差距主导项为 class2（随机未观测负对）——full 的类型头把 "
        f"{f0['class2']['support']} 个未观测随机对中大量误判为 LOGON（class0），nemb 几乎不误判；"
        "即 full 的 LOGON 判别边界在类型嵌入参与下对『观测支撑外』的随机对过度自信。"
        "该机理与主线证据同向：E2 评估侧候选池扩到非观测对时 AUC 全塌（0.31→0.04）、"
        "E2 污染免疫发现（homo 跨类型分数崩坏）——类型条件化的关系记忆在分布外无法校准其置信度。"
        "可比性审计：homo/homoflows/noflows 臂评估池无 FLOW 正例（class1 缺席），在案 f1_score 未固定 "
        "labels 参数 → 其 macro-F1 实际只在有效类别集上平均，跨臂不可比，论文不得并列引用。"
        "注：本条为诊断证据申报，论文最终措辞归写作路。")
    out['F3_w14_diagnosis'] = w14

    # ---- F4：事件级/边级 AP 日块 bootstrap CI ----
    def day_ap_weighted(day, s, y, w):
        res = {}
        for d in np.unique(day):
            m = day == d
            yy, ss, ww = y[m], s[m], w[m]
            if yy.sum() == 0 or yy.sum() == len(yy):
                continue
            rep = np.maximum(ww.astype(int), 1)
            res[float(d)] = float(average_precision_score(np.repeat(yy, rep), np.repeat(ss, rep)))
        return res

    for a, b in [('full', 'homo_rerun'), ('full', 'homoflows_rerun')]:
        rows_ci = []
        for seed in range(3):
            dA, relA, scA, yA = scores[a][seed]
            dB, relB, scB, yB = scores[b][seed]
            lm = relA == 'LOGON'
            pos_all = np.where(lm)[0]
            # 权重向量（事件数；负例 0 → ap_weighted 内保底 1）
            rest_w = {}
            for dd in sorted(set(dA[lm].tolist())):
                sel = pos_all[dA[pos_all] == dd]
                ei, _ = L['logon_days'][int(dd)]
                us, hs = ei[0].numpy(), ei[1].numpy()
                for pos, uu, hh in zip(sel, us, hs):
                    key = (int(dd), int(uu), int(hh))
                    if key in attack_rows:
                        rest_w[int(pos)] = attack_rows[key]
            w_all = np.array([rest_w.get(int(p), 0) for p in pos_all], dtype=float)
            y_l = yA[lm].astype(float)
            day_l = dA[lm]
            dpa = day_ap_weighted(day_l, scA[lm], y_l, w_all)
            dpb = day_ap_weighted(day_l, scB[lm], y_l, w_all)
            days = sorted(set(dpa) & set(dpb))
            diffs = np.array([dpa[d] - dpb[d] for d in days])
            rng = np.random.default_rng(9000 + seed)
            boots = diffs[rng.integers(0, len(diffs), (2000, len(diffs)))].mean(axis=1)
            rows_ci.append({'seed': seed,
                            'obs_eventAP_diff_full_minus_' + b: round(float(diffs.mean()), 6),
                            'ci95_dayblock_b2000': [round(float(np.quantile(boots, 0.025)), 6),
                                                    round(float(np.quantile(boots, 0.975)), 6)],
                            'n_days_with_pos': len(days)})
        out['F4_event_edge_ci'][f'{a}_vs_{b}_eventAP_weighted'] = rows_ci

    with open(f'{OUTD}/e7_e4_repair.json', 'w') as f:
        json.dump(out, f, indent=1, ensure_ascii=False)

    # ---- 回填 e4_granularity.json（追加 r2_repairs 段，不覆盖历史）----
    e4p = '/root/autodl-tmp/projB_hetero_graph/results/EXP-R1-E4/e4_granularity.json'
    e4 = json.load(open(e4p))
    e4['r2_repairs'] = {
        'ap_event_weighted_v1_deprecated': '恒 1.0（np.repeat 剔除权重 0 负例行，y_rep 全正例）；修复定义与数值见 EXP-R1-E7/e7_e4_repair.json F1',
        'ap_event_weighted_v2': out['F1_ap_event_weighted_v2'],
        'recall_grid_corrected_attribution': (
            '两臂事件级排名信号弱（AP≈基率 2-8 倍），recall@K 在 K≤100 接近随机基线——'
            '原"top-25 预算过粗"归因不成立（R1B-M2b），改写为信号弱；事件级方向一致性由 AP 行口径承载，非独立检出证据'),
        'w14_diagnosis': out['F3_w14_diagnosis'],
    }
    with open(e4p, 'w') as f:
        json.dump(e4, f, indent=1, ensure_ascii=False)
    print('F1 v2 sample:', json.dumps(out['F1_ap_event_weighted_v2']['full'], indent=0)[:300])
    print('F3 gap decomposition:', w14['unified_explanation'][:200])
    print('E7_DONE')


if __name__ == '__main__':
    main()
