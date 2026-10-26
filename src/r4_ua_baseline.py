"""r4_ua_baseline.py — R4/E18：UA 记忆规则+频率基线臂（用户 2026-10-25 批准，ICEICT §5 附表行）。

判据：results/EXP-R4-REG/criteria_r4.json（先冻后跑，P1-4 规则首次执行；ledger 冻结行先于本产物）。
设计（冻结口径）：LOGON 同池=EVAL_DAYS 全部边、事件级 (user,dst) 真值、训练对=TRAIN_DAYS(0-9)
名称空间对——与管线协议逐项同构（同一 loader/同一真值映射/同一日切），仅分数源换为启发式：
  UA   分数 = (u,h)∈训练日对集合 ? 0.0 : 1.0（Euler §VI-C UA 规则同型）
  FREQ 分数 = 1/(1+训练日计数)（未见对=1.0 最异常）
指标=池化 plain AP/AUC + 随机参照 AP=n_pos/n_pool。描述性附表行：不做与模型臂的优劣判语
（模型主张本就是 not-detectably-improve）；训练窗含 redteam 事件与管线同协议（可比利害一致）如实申报。
输出：results/EXP-R4-E18/ua_baseline.json。纯 CPU 分钟级。
"""
import sys, os, json, datetime, gzip
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

BASE = '/root/autodl-tmp/projB_hetero_graph'
sys.path.insert(0, f'{BASE}/src')
from r4b_common import (load_logon_days, load_redteam_events, attack_truth_by_day,
                        TRAIN_DAYS, EVAL_DAYS)

OUT = f'{BASE}/results/EXP-R4-E18/ua_baseline.json'
E40B = '/root/autodl-tmp/results/E40B_hetero_graph'

L = load_logon_days(ntlm=True)
truth = attack_truth_by_day(load_redteam_events())
inv_u = {i: u_ for u_, i in L['uidx'].items()}
inv_h = {i: h_ for h_, i in L['hidx'].items()}

# 训练对（名称空间，TRAIN_DAYS=0-9 与管线同协议）
train_pairs, train_cnt = set(), {}
tr_days_used = [d for d in TRAIN_DAYS if d in L['logon_days']]
for d in tr_days_used:
    ei = L['logon_days'][d][0]
    for a, b in zip(ei[0].tolist(), ei[1].tolist()):
        p = (inv_u[a], inv_h[b])
        train_pairs.add(p)
        train_cnt[p] = train_cnt.get(p, 0) + 1

# 评估池（EVAL_DAYS 全边，事件级真值）
ev_days = [d for d in EVAL_DAYS if d in L['logon_days']]
ys, s_ua, s_freq, days_r = [], [], [], []
for d in ev_days:
    ei = L['logon_days'][d][0]
    truth_d = truth.get(d, {'logon_pairs': {}})
    for a, b in zip(ei[0].tolist(), ei[1].tolist()):
        p = (inv_u[a], inv_h[b])
        ys.append(1 if p in truth_d['logon_pairs'] else 0)
        s_ua.append(0.0 if p in train_pairs else 1.0)
        s_freq.append(1.0 / (1.0 + train_cnt.get(p, 0)))
        days_r.append(d)
ys, s_ua, s_freq = np.array(ys), np.array(s_ua), np.array(s_freq)

n_pool, n_pos = int(len(ys)), int(ys.sum())
# 冻结判据：池计数与在案逐位一致，不一致即 FAIL 退出（绝不带病落盘）
assert n_pool == 131235 and n_pos == 277, f"pool mismatch: {n_pool}/{n_pos}（判据 errata R3fix-E2：LOGON 同池=131,235；445,229 为混合池）"

res = {
 'exp': 'R4/E18 UA 记忆规则+频率基线臂（LOGON 同池，判据先冻 results/EXP-R4-REG/criteria_r4.json）',
 'date': datetime.datetime.now().astimezone().isoformat(timespec='seconds'),
 'producer': 'src/r4_ua_baseline.py（R4；loader/真值/日切与管线 r4b_common 逐项同构）',
 'design': {
  'train_days': tr_days_used, 'eval_days': ev_days,
  'ua_rule': 'score=0.0 if (user,host) in train_pairs else 1.0（Euler §VI-C UA 规则同型）',
  'freq_rule': 'score=1/(1+train_count)，未见对=1.0',
  'metrics': '池化 plain AP/AUC（与管线 in_ap_logon 同池同真值）'},
 'pool': {'n_pool': n_pool, 'n_pos': n_pos, 'random_ref_ap': round(n_pos / n_pool, 8),
          'train_pair_count': len(train_pairs),
          'eval_seen_in_train_frac': round(float(np.mean([1.0 if s == 0.0 else 0.0 for s in s_ua])), 6)},
 'ua_rule': {'ap': float(average_precision_score(ys, s_ua)),
             'auc': float(roc_auc_score(ys, s_ua))},
 'freq_baseline': {'ap': float(average_precision_score(ys, s_freq)),
                   'auc': float(roc_auc_score(ys, s_freq))},
 'sanity': {'pool_bitwise_match': True,
            'note': '训练窗(0-9)可能含 redteam 事件——与管线训练协议一致（可比利害一致），如实申报；'
                    'UA/频率为弱启发式基线，本行仅作 §5 附表上下文，不做与模型臂优劣判语'},
 'reading': '描述性基线行：LANL auth 流上记忆型规则的表现背景（对照 Euler §VI-C「28% 异常为曾见连接」'
            '的 UA 局限观察）；模型臂主张维持 not-detectably-improve 口径不变。'
}
json.dump(res, open(OUT, 'w'), indent=1, ensure_ascii=False)
print(f"E18 DONE n_pool={n_pool} n_pos={n_pos} UA AP={res['ua_rule']['ap']:.5f} AUC={res['ua_rule']['auc']:.4f} "
      f"FREQ AP={res['freq_baseline']['ap']:.5f} AUC={res['freq_baseline']['auc']:.4f} "
      f"seen_frac={res['pool']['eval_seen_in_train_frac']}")
