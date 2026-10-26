"""e4_granularity.py — R1-E4 边→事件→窗三粒度证据链收口 + 每类型样本量申报。

【实验设计自查（开工前冻结）】
H4：边级结论（full vs homo_rerun LOGON 同池 2/3 seed 显著更差，p=0.002/0.002/0.972）聚合到
  事件级（redteam 749 事件）与窗级（E1 输出）后仍一致，或如实报不一致处。
自变量：聚合粒度（边→事件→窗）× 臂（full/homo_rerun/homoflows_rerun，在案 npz 已核验）。
因变量：①每类型样本量+基率计数表（W15）；②749 事件丢样表（W12）；③事件级 Recall@K 与
  top-25 host footprint；④三粒度一致性矩阵（窗级行由 E1 输出回填）。
边身份还原口径：npz LOGON 行块序=当日 L['logon_days'][d] 边序（在案管线 rows.append 顺序），
还原后以「还原攻击行标签 ≡ npz y」强断言验证。
"""
import json, sys
import numpy as np
from collections import defaultdict
sys.path.insert(0, '/root/autodl-tmp/projB_hetero_graph/src')
from r4b_common import load_logon_days, load_flow_days, load_redteam_events, attack_truth_by_day

BASE = '/root/autodl-tmp'
ARMS = {
    'full': f'{BASE}/results/E47B_fairpool_ci/full',
    'homo_rerun': f'{BASE}/results/E47B_fairpool_ci_rerun/homo',
    'homoflows_rerun': f'{BASE}/results/E47B_fairpool_ci_rerun/homoflows',
}
OUT = '/root/autodl-tmp/projB_hetero_graph/results/EXP-R1-E4'
KS = [10, 25, 50, 100]

L = load_logon_days(ntlm=True)
F = load_flow_days()
RTev = load_redteam_events()
u2i, h2i = L['uidx'], L['hidx']
N = L['n_users'] + len(L['hosts']) + len([h for h in F['hosts'] if h not in L['hidx']])

# redteam 事件按 (day,(u,h)) 聚合出攻击行集合与每行承载事件数
attack_rows = defaultdict(int)          # (day,u_idx,h_idx) -> n_events
rt_hosts = set()                         # redteam 涉及 dst host（footprint 分母）
rt_user_drop = defaultdict(int)
for ts, u, s, d in RTev:
    rt_hosts.add(d)
    if u in u2i and d in h2i:
        attack_rows[(ts // 86400, u2i[u], h2i[d])] += 1
    else:
        rt_user_drop[u] += 1

# ---------- ② W12 丢样表 ----------
match = {
    'n_raw_rows_redteam_file': 749,
    'n_after_machine_anon_filter': len(RTev),
    'dropped_events_user_not_in_U_pool': int(sum(rt_user_drop.values())),
    'dropped_users': dict(rt_user_drop),
    'unique_attack_rows_LOGON_pool': len(attack_rows),
    'n_events_mapped_to_rows': int(sum(attack_rows.values())),
    'n_redteam_dst_hosts_total': len(rt_hosts),
}
match['dropped_events_no_row'] = match['n_after_machine_anon_filter'] - match['n_events_mapped_to_rows']
match['row_coverage_ratio'] = round(match['n_events_mapped_to_rows'] / match['n_after_machine_anon_filter'], 4)

# ---------- ① 样本量表 + ③ 事件级指标（逐臂逐 seed） ----------
sample_table = {}
event_results = {}
align_checks = {}
for arm, d in ARMS.items():
    event_results[arm] = {}
    align_checks[arm] = []
    for seed in range(3):
        z = np.load(f'{d}/scores_seed{seed}.npz', allow_pickle=True)
        day_, rel_, score_, y_ = z['day'], z['rel'], z['score'], z['y']
        if seed == 0:
            lmask = rel_ == 'LOGON'
            fmask = rel_ == 'FLOW'
            sample_table[arm] = {
                'n_pool_mixed': int(len(y_)), 'n_pos_mixed': int(y_.sum()),
                'n_pool_LOGON': int(lmask.sum()), 'n_pos_LOGON': int(y_[lmask].sum()),
                'base_rate_LOGON': round(float(y_[lmask].mean()), 8),
                'n_pool_FLOW': int(fmask.sum()), 'n_pos_FLOW': int(y_[fmask].sum()),
                'base_rate_mixed': round(float(y_.mean()), 8)}
        # 还原 LOGON 行身份
        logon_pos = np.where(rel_ == 'LOGON')[0]
        rows_by_day = defaultdict(list)  # day -> [(pos, u, h)]
        it = iter(logon_pos)
        cur_day = None
        # npz 按评估日序（EVAL_DAYS）分块，块内 LOGON 连续；逐块对照 logon_days 边序
        day_blocks = []
        for dd in sorted(set(day_[logon_pos].tolist())):
            sel = logon_pos[day_[logon_pos] == dd]
            day_blocks.append((int(dd), sel))
        restored_attack = {}
        for dd, sel in day_blocks:
            ei, _ = L['logon_days'][dd]
            us, hs = ei[0].numpy(), ei[1].numpy()
            assert len(sel) == len(us), f'{arm} s{seed} day{dd}: 行数 {len(sel)} != 边数 {len(us)}'
            for pos, uu, hh in zip(sel, us, hs):
                key = (dd, int(uu), int(hh))
                if key in attack_rows:
                    restored_attack[int(pos)] = (attack_rows[key], key[2])
            # 对齐断言：还原攻击行标签 ≡ npz y
        y_at_restored = np.array([y_[p] for p in sorted(restored_attack)])
        n_mismatch = int((y_at_restored != 1).sum())
        align_checks[arm].append({'n_restored_attack_rows': len(restored_attack),
                                  'y_mismatch': n_mismatch,
                                  'n_pos_npz': int(y_[rel_ == 'LOGON'].sum())})
        assert n_mismatch == 0, f'{arm} seed{seed}: 还原攻击行与 npz 标签不一致 {n_mismatch}'
        # 事件级：告警行覆盖事件（score 降序 top-K）
        pos_all = logon_pos
        order = pos_all[np.argsort(-score_[pos_all], kind='stable')]
        ev_covered = 0
        rows_hit = {}
        for K in KS:
            top = order[:K]
            cov_events = sum(restored_attack.get(int(p), (0, 0))[0] for p in top)
            cov_rows = sum(1 for p in top if int(p) in restored_attack)
            hosts_hit = set(restored_attack[int(p)][1] for p in top if int(p) in restored_attack)
            event_results[arm].setdefault(seed, {})[K] = {
                'recall_events@K': round(cov_events / match['n_events_mapped_to_rows'], 4),
                'recall_rows@K': round(cov_rows / len(restored_attack), 4),
                'n_attack_rows_in_topK': cov_rows,
                'host_footprint_cover@K': round(len(hosts_hit) / match['n_redteam_dst_hosts_total'], 4)}
        # 事件级 AP（行=样本、权重=事件数；未匹配事件不入分母，丢样表申报）
        y_e = np.array([restored_attack.get(int(p), (0,))[0] > 0 for p in pos_all], dtype=float)
        w_e = np.array([restored_attack.get(int(p), (0,))[0] for p in pos_all], dtype=float)
        s_e = score_[pos_all]
        from sklearn.metrics import average_precision_score
        ap_rows = float(average_precision_score(y_e, s_e))
        # 事件加权 AP：按事件数重复展开
        rep_w = w_e.astype(int)
        s_rep = np.repeat(s_e, rep_w)
        y_rep = np.repeat(y_e, rep_w)
        ap_ev = float(average_precision_score(y_rep, s_rep)) if y_rep.sum() > 0 else None
        event_results[arm][seed]['ap_attack_rows'] = round(ap_rows, 6)
        event_results[arm][seed]['ap_event_weighted'] = round(ap_ev, 6) if ap_ev else None
        event_results[arm][seed]['n_attack_rows'] = int(len(restored_attack))

# ---------- ④ 三粒度一致性矩阵（边级/事件级；窗级回填） ----------
def direction(a, b):
    if a is None or b is None:
        return 'NA'
    return 'full_better' if a > b else ('homo_better' if b > a else 'tie')

matrix = {}
for arm_a, arm_b in [('full', 'homo_rerun'), ('full', 'homoflows_rerun'),
                     ('homo_rerun', 'homoflows_rerun')]:
    col = f'{arm_a}_vs_{arm_b}'
    matrix[col] = {}
    # 边级（在案口径：npz 复算 LOGON AP）
    for arm, d in [(arm_a, ARMS[arm_a]), (arm_b, ARMS[arm_b])]:
        pass
    def per_seed_ap(arm):
        out = []
        for seed in range(3):
            z = np.load(f'{ARMS[arm]}/scores_seed{seed}.npz', allow_pickle=True)
            lm = z['rel'] == 'LOGON'
            from sklearn.metrics import average_precision_score
            out.append(float(average_precision_score(z['y'][lm], z['score'][lm])))
        return out
    aps_a, aps_b = per_seed_ap(arm_a), per_seed_ap(arm_b)
    matrix[col]['edge_AP_logon'] = {
        arm_a: [round(x, 4) for x in aps_a], arm_b: [round(x, 4) for x in aps_b],
        'direction_per_seed': [direction(a, b) for a, b in zip(aps_a, aps_b)]}
    # 事件级 AP（行口径）
    ev_a = [event_results[arm_a][s]['ap_attack_rows'] for s in range(3)]
    ev_b = [event_results[arm_b][s]['ap_attack_rows'] for s in range(3)]
    matrix[col]['event_AP_rows'] = {
        arm_a: ev_a, arm_b: ev_b,
        'direction_per_seed': [direction(a, b) for a, b in zip(ev_a, ev_b)]}
    # 事件级 Recall@25
    r_a = [event_results[arm_a][s][25]['recall_events@K'] for s in range(3)]
    r_b = [event_results[arm_b][s][25]['recall_events@K'] for s in range(3)]
    matrix[col]['event_recall@25'] = {
        arm_a: r_a, arm_b: r_b,
        'direction_per_seed': [direction(a, b) for a, b in zip(r_a, r_b)]}

# W14 truetype 反常复核（读取在案 truetype 结果，申报数字）
try:
    tt = json.load(open(f'{BASE}/results/E47B_fairpool_ci/truetype/results_truetype.json'))
    w14 = {'truetype_incase': tt}
except Exception as e:
    w14 = {'error': str(e)}

out = {
    'exp': 'R1-E4 三粒度证据链收口', 'date': '2026-10-24',
    'design_frozen': 'results/EXP-R1-E4/criteria.json + 本文件 docstring',
    'sample_table_per_type': sample_table,
    'w12_dropped_events_table': match,
    'align_check_edge_restore': align_checks,
    'event_level': event_results,
    'granularity_matrix_edge_event': matrix,
    'granularity_matrix_window_row': 'PENDING_E1（由 e1_window_alert.py 输出回填）',
    'w14_truetype_recheck': w14,
}
with open(f'{OUT}/e4_granularity.json', 'w') as f:
    json.dump(out, f, indent=1, ensure_ascii=False)
print(json.dumps({'sample_table': sample_table, 'match': match}, indent=1, ensure_ascii=False))
print('matrix:')
for k, v in matrix.items():
    for m, vv in v.items():
        print(' ', k, m, vv['direction_per_seed'])
print('E4_DONE')
