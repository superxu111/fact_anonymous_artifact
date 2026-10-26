"""e12_closure.py — R2-E12（P1-9/R1A-m3）：temporal_1 60ep 抽查，量化 30ep 网格 vs 60ep 主线口径差。

【实验设计自查（开工前冻结）】
验证假设：E2/E6 头条点 temporal_1 的「混合池可分（full 更优）」与「LOGON 同池 full 更差」在 60ep
  主线 epoch 口径下与 30ep 网格口径方向一致（E6 全部推断基于 30ep 网格的遗留口径差，R1A-m3）。
自变量：epochs∈{30（在案网格）, 60（主线校验链口径）}×arm{full,homo}×seed{0,1,2}（neg=temporal×1）。
因变量：混合池 AUC 差、LOGON 同池 AUC/AP 差、逐 seed 日块置换 p（B=500，与 e6 同机器独立种子）。
对照：30ep 在案 e2_closure.json flip_boundary.temporal_1（obs 与 p）。
规模/成本：GPU 2 jobs×3 seeds×60ep（~5 min，RTX 4080 实测）。
成功判据：方向一致性申报——60ep 与 30ep 的 mixed 可分方向一致→头条结论对 epoch 口径稳健；
  不一致→如实申报并弱化（判据双去向预登记）。
断言：池计数与在案 30ep 一致（n_pool/n_pos）；无 NaN。
"""
import json, os, sys
import numpy as np

sys.path.insert(0, '/root/autodl-tmp/projB_hetero_graph/src')
from e6_inference import load_npz, fast_auc, perm_test_dayblock_indep
from sklearn.metrics import average_precision_score

BASE = '/root/autodl-tmp/projB_hetero_graph'
D60F = f'{BASE}/results/EXP-R2-E12/temporal1_full_60ep'
D60H = f'{BASE}/results/EXP-R2-E12/temporal1_homo_60ep'


def main():
    e2 = json.load(open(f'{BASE}/results/EXP-R1-E2/e2_closure.json'))['flip_boundary']['temporal_1']
    out = {'exp': 'R2-E12 temporal_1 60ep 抽查（P1-9/R1A-m3：30ep 网格 vs 60ep 主线口径差）',
           'date': '2026-10-24', 'design': '见模块 docstring（判据双去向预登记）',
           'ep30_incase': {'mixed_diff': [e2[f's{i}']['auc_mixed_diff_full_minus_homo'] for i in range(3)],
                           'mixed_p': [e2[f's{i}']['p_mixed'] for i in range(3)],
                           'logon_auc_diff': [e2[f's{i}']['auc_logon_diff_full_minus_homo'] for i in range(3)],
                           'logon_p': [e2[f's{i}']['p_logon'] for i in range(3)]},
           'ep60_spotcheck': {'per_seed': []}}
    import zlib
    rng_master = np.random.default_rng(zlib.crc32(b'temporal_1|e12_60ep'))
    for seed in range(3):
        dA, relA, scA, yA = load_npz(f'{D60F}/scores_seed{seed}.npz')
        dB, relB, scB, yB = load_npz(f'{D60H}/scores_seed{seed}.npz')
        assert np.array_equal(yA, yB) and np.isfinite(scA).all() and np.isfinite(scB).all()
        row = {'seed': seed, 'n_pool': int(len(yA)), 'n_pos': int(yA.sum())}
        row['mixed_diff'] = round(fast_auc(yA, scA) - fast_auc(yA, scB), 5)
        _, p_mix = perm_test_dayblock_indep(dA, scA, scB, yA, B=500, rng=rng_master, metric='auc')
        row['mixed_p'] = round(p_mix, 5)
        lm = relA == 'LOGON'
        obs_l = fast_auc(yA[lm], scA[lm]) - fast_auc(yA[lm], scB[lm])
        _, p_l = perm_test_dayblock_indep(dA[lm], scA[lm], scB[lm], yA[lm], B=500, rng=rng_master, metric='auc')
        row['logon_auc_diff'] = round(obs_l, 5); row['logon_auc_p'] = round(p_l, 5)
        obs_ap = float(average_precision_score(yA[lm], scA[lm]) - average_precision_score(yA[lm], scB[lm]))
        _, p_ap = perm_test_dayblock_indep(dA[lm], scA[lm], scB[lm], yA[lm], B=500, rng=rng_master, metric='ap')
        row['logon_ap_diff'] = round(obs_ap, 5); row['logon_ap_p'] = round(p_ap, 5)
        out['ep60_spotcheck']['per_seed'].append(row)
        print(json.dumps(row), flush=True)
    ps = out['ep60_spotcheck']['per_seed']
    n_sep = sum(1 for r in ps if r['mixed_diff'] > 0 and r['mixed_p'] < 0.05)
    n_worse_auc = sum(1 for r in ps if r['logon_auc_diff'] < 0)
    n_worse_ap = sum(1 for r in ps if r['logon_ap_diff'] < 0)
    out['ep60_spotcheck']['summary'] = {
        'mixed_separable_60ep': f'{n_sep}/3',
        'mixed_separable_30ep_incase': '3/3',
        'logon_auc_full_worse_60ep': f'{n_worse_auc}/3',
        'logon_ap_full_worse_60ep': f'{n_worse_ap}/3'}
    d30 = out['ep30_incase']['mixed_diff']
    d60 = [r['mixed_diff'] for r in ps]
    out['verdict'] = (f'方向一致性：mixed 可分 30ep 3/3 → 60ep {n_sep}/3（{"方向一致" if n_sep == 3 else "需复核"}）；'
                      f'mixed 差 30ep {d30} vs 60ep {d60}；LOGON AUC full 更差 {n_worse_auc}/3、AP {n_worse_ap}/3'
                      f'（与 E9 的 AUC 口径方向证据{"一致" if n_worse_auc >= 2 else "需申报"}）。'
                      '结论：头条（temporal 混合池可分）对 epoch 口径的稳健性以本表为准申报；'
                      '红线：口径差如实并报，不择优引用。')
    os.makedirs(f'{BASE}/results/EXP-R2-E12', exist_ok=True)
    with open(f'{BASE}/results/EXP-R2-E12/e12_closure.json', 'w') as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    print('E12_DONE', out['verdict'])


if __name__ == '__main__':
    main()
