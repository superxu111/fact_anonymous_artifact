"""e16_e12_seed_aggregate.py — R2-E16（R3B-m4 响应 + R2A-m5 标注）：E12 60ep seed 层聚合补齐。

【实验设计自查（开工前冻结，判据见 results/EXP-R2-E16/criteria_e16.json）】
缺口：E12 60ep 抽查仅有逐 seed p，无 seed 层聚合（R3B-m4：LOGON AUC 60ep obs 均值 −0.0074 vs
  30ep −0.0208，方向保持幅度减半——该句此前无产物支撑）。
方法：与 E9 完全同统计机器（import 复用 e6_inference），数据=results/EXP-R2-E12/temporal1_{full,homo}_60ep
  在案 npz（纯 CPU 秒级，不重训练）；三指标 {auc_mixed, auc_logon, ap_logon}；逐 seed 日块置换 B=500 +
  seed 层聚合置换 B=2000；rng=zlib.crc32('temporal_1|60ep|<metric>')。
小样本自测：逐 seed obs 须与 E12 交付值逐位一致（同 npz 同机器 → mixed_diff [0.25983,0.27623,0.26953]、
  logon_auc_diff [−0.01255,0.00785,−0.01752]）。
成功判据：聚合 p 无论显著与否均申报；30ep/60ep 聚合并报（R3B-m4 的幅度减半句以本产物为准落账）。
族申报标注（R2A-m5 同款）：本实验为稳健性抽查（temporal_1 单点 60ep），全部 p 值不入 Holm 申报族、
  不进摘要；红线：未检出≠等效。
规模/成本：纯 CPU，3 指标×（逐 seed+聚合），预计 <3 分钟（若超时 setsid nohup 后台化）。
"""
import json, sys, zlib

import numpy as np

sys.path.insert(0, '/root/autodl-tmp/projB_hetero_graph/src')
from e6_inference import (BASE, load_npz, fast_auc, perm_test_dayblock_indep,
                          perm_test_seedagg)
from sklearn.metrics import average_precision_score

D60F = f'{BASE}/results/EXP-R2-E12/temporal1_full_60ep'
D60H = f'{BASE}/results/EXP-R2-E12/temporal1_homo_60ep'
OUT = f'{BASE}/results/EXP-R2-E16'
METRICS = ['auc_mixed', 'auc_logon', 'ap_logon']

EXPECT_E12 = {'mixed_diff': [0.25983, 0.27623, 0.26953],
              'logon_auc_diff': [-0.01255, 0.00785, -0.01752],
              'logon_ap_diff': [-0.00409, -0.0005, 0.00072]}


def load_all():
    rows = []
    for seed in range(3):
        dA, relA, scA, yA = load_npz(f'{D60F}/scores_seed{seed}.npz')
        dB, relB, scB, yB = load_npz(f'{D60H}/scores_seed{seed}.npz')
        assert np.array_equal(yA, yB)
        rows.append((dA, relA, scA, yA, scB))
    return rows


def main():
    import os
    os.makedirs(OUT, exist_ok=True)
    rows = load_all()
    out = {'exp': 'R2-E16：E12 temporal_1 60ep seed 层聚合（R3B-m4）',
           'date': '2026-10-24',
           'machine': 'import 复用 e6_inference 统计机器；在案 60ep npz，不重训练；'
                      'rng=zlib.crc32(temporal_1|60ep|metric)',
           'family_declaration_r2am5': '稳健性抽查（temporal_1 单点 60ep 口径）：全部 p 值不入 Holm 申报族、'
                                       '不进摘要（逐 seed p 抽查定位可接受——mixed 三 p Bonferroni×3 后 '
                                       '0.024/0.006/0.018 仍全过）；红线：未检出≠等效',
           'per_metric': {}}
    cache = {}
    for seed, (d, rel, scA, y, scB) in enumerate(rows):
        lm = rel == 'LOGON'
        cache[seed] = {'d_mix': d, 'a_mix': scA, 'b_mix': scB, 'y_mix': y,
                       'd_l': d[lm], 'a_l': scA[lm], 'b_l': scB[lm], 'y_l': y[lm]}

    for metric in METRICS:
        rng_master = np.random.default_rng(zlib.crc32(f'temporal_1|60ep|{metric}'.encode()))
        per_seed, days_l, a_l, b_l, y_l = [], [], [], [], []
        for seed in range(3):
            c = cache[seed]
            if metric == 'auc_mixed':
                d, a, b, y = c['d_mix'], c['a_mix'], c['b_mix'], c['y_mix']
            else:
                d, a, b, y = c['d_l'], c['a_l'], c['b_l'], c['y_l']
            if metric == 'auc_mixed':
                obs = fast_auc(y, a) - fast_auc(y, b)
            else:
                f = fast_auc if metric == 'auc_logon' else lambda yy, s: float(average_precision_score(yy, s))
                obs = f(y, a) - f(y, b)
            _, p = perm_test_dayblock_indep(d, a, b, y, B=500, rng=rng_master,
                                            metric='auc' if metric.startswith('auc') else 'ap')
            per_seed.append({'seed': seed, 'obs_diff': round(float(obs), 5), 'p_indep': round(float(p), 5)})
            days_l.append(d); a_l.append(a); b_l.append(b); y_l.append(y)
        d_c = np.concatenate(days_l); a_c = np.concatenate(a_l)
        b_c = np.concatenate(b_l); y_c = np.concatenate(y_l)
        seeds_c = np.concatenate([np.full(len(x), s) for s, x in enumerate(days_l)])
        agg_obs, agg_p = perm_test_seedagg(d_c, a_c, b_c, y_c, seeds_c, B=2000, rng=rng_master,
                                           metric='auc' if metric.startswith('auc') else 'ap')
        out['per_metric'][metric] = {'per_seed': per_seed,
                                     'seed_aggregate': {'obs_mean_diff': round(float(agg_obs), 5),
                                                        'p_b2000': round(float(agg_p), 5)},
                                     'direction_count_full_worse': sum(1 for x in per_seed if x['obs_diff'] < 0)}
        print(f"  {metric}: per_seed_obs={[x['obs_diff'] for x in per_seed]} "
              f"agg_obs={agg_obs:.5f} agg_p={agg_p:.5f}", flush=True)

    # ---- 小样本自测：逐 seed obs 须复现 E12 交付值 ----
    got = {'mixed_diff': [x['obs_diff'] for x in out['per_metric']['auc_mixed']['per_seed']],
           'logon_auc_diff': [x['obs_diff'] for x in out['per_metric']['auc_logon']['per_seed']],
           'logon_ap_diff': [x['obs_diff'] for x in out['per_metric']['ap_logon']['per_seed']]}
    for k, v in EXPECT_E12.items():
        assert got[k] == v, f'机器复算与 E12 交付不一致 {k}: {got[k]} != {v}'
    out['e12_replay_check'] = {'status': '逐位一致（同 npz 同机器，管线零漂移）', 'mixed_diff': got['mixed_diff'],
                               'logon_auc_diff': got['logon_auc_diff'], 'logon_ap_diff': got['logon_ap_diff']}

    # ---- 30ep/60ep 聚合并报（R3B-m4 幅度对照以本表为准）----
    e9 = json.load(open(f'{BASE}/results/EXP-R2-E9/e9_auc_logon_agg.json'))
    e6 = json.load(open(f'{BASE}/results/EXP-R1-E6/e6_inference.json'))
    t1_auc = next(r for r in e9['per_point'] if r['point'] == 'temporal_1')['seed_aggregate']
    t1_ap = next(r for r in e6['E2']['per_point'] if r['point'] == 'temporal_1' and r['metric'] == 'ap_logon')['seed_aggregate']
    t1_mix = next(r for r in e6['E2']['per_point'] if r['point'] == 'temporal_1' and r['metric'] == 'auc_mixed')['seed_aggregate']
    out['ep30_vs_ep60_aggregate'] = {
        'auc_mixed': {'ep30_incase': t1_mix, 'ep60_this': out['per_metric']['auc_mixed']['seed_aggregate']},
        'auc_logon': {'ep30_incase': t1_auc, 'ep60_this': out['per_metric']['auc_logon']['seed_aggregate']},
        'ap_logon': {'ep30_incase': t1_ap, 'ep60_this': out['per_metric']['ap_logon']['seed_aggregate']},
        'reading': ('temporal_1 单点：mixed 可分 30ep/60ep 方向一致；LOGON AUC 聚合 obs 幅度 30ep→60ep 变化'
                    '如实并报（R3B-m4「方向保持」句以本表为准）；AP 口径未检出≠等效。')}
    with open(f'{OUT}/e12_60ep_seed_aggregate.json', 'w') as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    print('E16_DONE', json.dumps(out['ep30_vs_ep60_aggregate']['auc_logon'], ensure_ascii=False))


if __name__ == '__main__':
    main()
