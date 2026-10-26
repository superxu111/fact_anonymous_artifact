"""e8_closure.py — R1-E8 收口：干净消融（type_w=0）vs 现行消融（type_w=0.5）vs full。

判据：results/EXP-R1-E8/criteria.json（冻结）。回应 R1B-M5（消融臂辅助类型头任务语义不对齐）：
检验主结论（LOGON 同池 full 不优于 homo；temporal 协议下 homo 混合池崩坏）是否依赖辅助类型 CE 信号。
"""
import json, sys
import numpy as np

sys.path.insert(0, '/root/autodl-tmp/projB_hetero_graph/src')
from e6_inference import (load_npz, fast_auc, perm_test_dayblock_indep,
                          dayblock_bootstrap_ci, holm_adjust)
from sklearn.metrics import average_precision_score

BASE = '/root/autodl-tmp/projB_hetero_graph'
E2D = f'{BASE}/results/EXP-R1-E2'
E8D = f'{BASE}/results/EXP-R1-E8'
OUTD = f'{BASE}/results/EXP-R1-E8'


def metrics_of(d, seed):
    day, rel, sc, y = load_npz(f'{d}/scores_seed{seed}.npz')
    lm = rel == 'LOGON'
    return {'auc_mixed': fast_auc(y, sc),
            'auc_logon': fast_auc(y[lm], sc[lm]),
            'ap_logon': float(average_precision_score(y[lm], sc[lm]))}


def main():
    out = {'exp': 'R1-E8 干净消融收口', 'date': '2026-10-24', 'criteria': 'criteria.json（冻结）',
           'E8a_uniform1_60ep': {}, 'E8b_temporal1_30ep': {}}

    # ---- E8a：uniform 1:1 60ep ----
    arms = {'full': f'{E2D}/uniform1_full_60ep',
            'homo_tw0.5': f'{E2D}/uniform1_homo_60ep',
            'homo_tw0': f'{E8D}/homo_tw0_uniform1_60ep'}
    mets = {a: [metrics_of(d, s) for s in range(3)] for a, d in arms.items()}
    out['E8a_uniform1_60ep']['metrics'] = {a: [{k: round(v, 5) for k, v in m.items()} for m in ms]
                                           for a, ms in mets.items()}
    rows = []
    for seed in range(3):
        day, rel, scF, y = load_npz(f"{arms['full']}/scores_seed{seed}.npz")
        _, _, sc0, _ = load_npz(f"{arms['homo_tw0']}/scores_seed{seed}.npz")
        _, _, sc5, _ = load_npz(f"{arms['homo_tw0.5']}/scores_seed{seed}.npz")
        lm = rel == 'LOGON'
        # LOGON 同池：full vs homo_tw0 置换（独立种子）+ CI
        rng = np.random.default_rng(8000 + seed)
        obs, p0 = perm_test_dayblock_indep(day[lm], scF[lm], sc0[lm], y[lm], B=500, rng=rng)
        _, p5 = perm_test_dayblock_indep(day[lm], scF[lm], sc5[lm], y[lm], B=500, rng=rng)
        ci0, _ = dayblock_bootstrap_ci(day[lm], scF[lm], sc0[lm], y[lm], B=2000, rng=rng)
        ci5, _ = dayblock_bootstrap_ci(day[lm], scF[lm], sc5[lm], y[lm], B=2000, rng=rng)
        rows.append({'seed': seed,
                     'logon_full_minus_tw0': {'obs': round(obs, 5), 'p': round(p0, 4),
                                              'ci95': [round(x, 5) for x in ci0]},
                     'logon_full_minus_tw0.5_incase_p': round(p5, 4),
                     'logon_full_minus_tw0.5_ci95': [round(x, 5) for x in ci5]})
    out['E8a_uniform1_60ep']['logon_tests'] = rows
    # 带判定：full−homo_tw0 是否落 full−homo_tw0.5 的 seed 波动带
    d0 = [mets['full'][s]['auc_logon'] - mets['homo_tw0'][s]['auc_logon'] for s in range(3)]
    d5 = [mets['full'][s]['auc_logon'] - mets['homo_tw0.5'][s]['auc_logon'] for s in range(3)]
    dm0 = [mets['full'][s]['auc_mixed'] - mets['homo_tw0'][s]['auc_mixed'] for s in range(3)]
    dm5 = [mets['full'][s]['auc_mixed'] - mets['homo_tw0.5'][s]['auc_mixed'] for s in range(3)]
    out['E8a_uniform1_60ep']['band_check'] = {
        'auc_logon_diff_full_minus_tw0': [round(x, 5) for x in d0],
        'auc_logon_diff_full_minus_tw0.5': [round(x, 5) for x in d5],
        'tw0_in_tw0.5_band': [bool(min(d5) <= x <= max(d5)) for x in d0],
        'auc_mixed_diff_full_minus_tw0': [round(x, 5) for x in dm0],
        'auc_mixed_diff_full_minus_tw0.5': [round(x, 5) for x in dm5],
        'mixed_tw0_in_tw0.5_band': [bool(min(dm5) <= x <= max(dm5)) for x in dm0]}

    # ---- E8b：temporal 1:1 30ep（E2 头条点抽检）----
    arms_b = {'full': f'{E2D}/temporal1_full_30ep',
              'homo_tw0.5': f'{E2D}/temporal1_homo_30ep',
              'homo_tw0': f'{E8D}/homo_tw0_temporal1_30ep'}
    mets_b = {a: [metrics_of(d, s) for s in range(3)] for a, d in arms_b.items()}
    out['E8b_temporal1_30ep']['metrics'] = {a: [{k: round(v, 5) for k, v in m.items()} for m in ms]
                                            for a, ms in mets_b.items()}
    # 混合池可分性（full 更优方向）逐 seed 置换
    rows_b = []
    for seed in range(3):
        day, rel, scF, y = load_npz(f"{arms_b['full']}/scores_seed{seed}.npz")
        _, _, sc0, _ = load_npz(f"{arms_b['homo_tw0']}/scores_seed{seed}.npz")
        rng = np.random.default_rng(8500 + seed)
        obs, p = perm_test_dayblock_indep(day, scF, sc0, y, B=500, rng=rng)
        rows_b.append({'seed': seed, 'mixed_full_minus_tw0': round(obs, 5), 'p': round(p, 4)})
    out['E8b_temporal1_30ep']['mixed_separability'] = rows_b
    # tw0.5 参照 p 从在案 e2_closure.json 读取（R2-P1-8 改，回应 R2B-m1「硬编码模式回潮」；
    # 原硬编码 [0.004,0.004,0.006] 与在案 temporal_1 p_mixed 逐位一致，行为等价）
    _e2fb = json.load(open(f'{E2D}/e2_closure.json'))['flip_boundary']['temporal_1']
    out['E8b_temporal1_30ep']['tw0.5_reference_p'] = [_e2fb[f's{i}']['p_mixed'] for i in range(3)]

    # ---- 判定（代码生成，替代硬编码）----
    a = out['E8a_uniform1_60ep']['band_check']
    n_logon_in = sum(a['tw0_in_tw0.5_band'])
    n_mixed_in = sum(a['mixed_tw0_in_tw0.5_band'])
    sig_logon_tw0 = [r['logon_full_minus_tw0']['p'] for r in rows]
    b = out['E8b_temporal1_30ep']
    n_sep_b = sum(1 for r in rows_b if r['mixed_full_minus_tw0'] > 0 and r['p'] < 0.05)
    out['verdict'] = {
        'E8a': (f'LOGON 同池：full−homo_tw0 差 {n_logon_in}/3 seed 落 tw0.5 波动带；'
                f'置换显著 seed 数 {sum(1 for p in sig_logon_tw0 if p < 0.05)}/3；'
                f'mixed 差 {n_mixed_in}/3 在带内'),
        'E8b': f'temporal 点混合池 full 可分 {n_sep_b}/3 seeds（tw0.5 参照 3/3）',
        'conclusion': ('干净消融（去辅助类型损失）下主结论方向保持——消融机制不对齐披露为局限而非推翻；'
                       if (n_logon_in >= 2 and n_sep_b >= 2) else
                       '干净消融下结论有实质变化——按判据如实申报并降级对应结论措辞；'),
        'red_line_note': '未检出显著差异≠等效；本实验仅证明结论对辅助类型头梯度不敏感，不构成等效证明'}

    with open(f'{OUTD}/e8_closure.json', 'w') as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    print(json.dumps(out['verdict'], ensure_ascii=False, indent=1))
    print('E8_CLOSURE_DONE')


if __name__ == '__main__':
    main()
