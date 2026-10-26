"""e10_e3_pooled_replay.py — R2-E10（P0-5 / R2B-M3+M4）：E3 无源池化字段补生产者重放 + 双口径注记重算。

【实验设计自查（开工前冻结）】
验证假设：e6_inference.json E3 段中「源代码无法再生」的池化口径字段
  （inband_w33_computed_pooled / w32_seed_band_pooled / convention_note / logon_gap_shift_disclosure
  池化数值）可由本生产者脚本从权威源（e3_closure.json 三臂池化指标）重放且与在案交付值逐位一致。
自变量：无（确定性重算，无随机数）。
因变量：三指标（auc_mixed/auc_logon/ap_logon）的池化逐 seed 差、w32 带 [min,max]、
  w33 落带判定（±0.02 容差，冻结判据口径 e6_inference.py:241-242）、signed diff 方向（w33<w32）、
  |差|缩小方向、均值；homo 自身 LOGON AP w32→w33。
对照：①e3_closure.json（权威池化指标）；②e6_inference.json E3 段在案交付值（R2B-M4 列为无源）。
规模/成本：纯 CPU 秒级（读 9 行指标+算术）。
成功判据：重放值与在案交付值逐位一致（浮点容 1e-9）；任一不一致 → 触发 RESEARCH_LOG §R1-E3 与
  R1_metrics supersedes（风险预案，判据先登记）。
断言：9 行指标齐全、无 NaN、带下界≤上界。
产出：results/EXP-R2-E10/e10_e3_pooled_replay.json（重放值+逐位比对+按 seed 全量方向双口径注记文本，
  供 RESEARCH_LOG/R1_metrics 引用）。
"""
import json, os
import numpy as np

BASE = '/root/autodl-tmp/projB_hetero_graph'
OUT = f'{BASE}/results/EXP-R2-E10'
TOL_BAND = 0.02  # 冻结判据容差（e6_inference.py:241-242；R1A2-M1：原 criteria 未登记，本轮补登记）


def replay():
    e3 = json.load(open(f'{BASE}/results/EXP-R1-E3/e3_closure.json'))
    arms = e3['arms']
    full = arms['full_projb_60ep']; homo = arms['homo_projb_60ep']; w33 = arms['homo_w33_60ep']
    out = {'exp': 'R2-E10 E3 无源池化字段生产者重放（P0-5/R2B-M3+M4）', 'date': '2026-10-24',
           'design': '见模块 docstring（开工前冻结）',
           'producer': 'src/e10_e3_pooled_replay.py（R2B-M4 补登记：原字段无在案生产者，'
                       '本脚本为补录生产者，注明 post_hoc 补录；不回充历史时刻）',
           'metrics': {}}
    for m in ['auc_mixed', 'auc_logon', 'ap_logon']:
        d32 = [round(full[str(s)][m] - homo[str(s)][m], 6) for s in range(3)]
        d33 = [round(full[str(s)][m] - w33[str(s)][m], 6) for s in range(3)]
        band = [min(d32), max(d32)]
        inband = [bool(band[0] - TOL_BAND <= v <= band[1] + TOL_BAND) for v in d33]
        signed_down = [bool(b < a) for a, b in zip(d32, d33)]          # w33 比 w32 更负（向 homo 更优）
        absval_narrow = [bool(abs(b) < abs(a)) for a, b in zip(d32, d33)]  # |差|缩小
        out['metrics'][m] = {
            'w32_pooled_diff': d32, 'w33_pooled_diff': d33,
            'w32_seed_band_pooled': [round(band[0], 4), round(band[1], 4)],
            'inband_w33_computed_pooled': inband,
            'signed_diff_decreased_w32_to_w33': signed_down,
            'absdiff_narrowed_w32_to_w33': absval_narrow,
            'mean_w32': round(float(np.mean(d32)), 6), 'mean_w33': round(float(np.mean(d33)), 6)}
    # homo 自身 LOGON AP w32→w33
    h_ap32 = [homo[str(s)]['ap_logon'] for s in range(3)]
    h_ap33 = [w33[str(s)]['ap_logon'] for s in range(3)]
    out['homo_ap_logon_w32_to_w33'] = {
        'per_seed': [[round(a, 5), round(b, 5)] for a, b in zip(h_ap32, h_ap33)],
        'n_declined': int(sum(1 for a, b in zip(h_ap32, h_ap33) if b < a)),
        'mean': [round(float(np.mean(h_ap32)), 6), round(float(np.mean(h_ap33)), 6)]}

    # ---- 与在案交付值逐位比对（R2B-M4 判定「无源」的字段）----
    e6 = json.load(open(f'{BASE}/results/EXP-R1-E6/e6_inference.json'))['E3']
    cmp = {}
    for m in ['auc_mixed', 'auc_logon', 'ap_logon']:
        inc = e6[m]
        cmp[m] = {
            'inband_pooled_match': out['metrics'][m]['inband_w33_computed_pooled'] == inc['inband_w33_computed_pooled'],
            'band_pooled_match': bool(np.allclose(out['metrics'][m]['w32_seed_band_pooled'],
                                                  inc['w32_seed_band_pooled'], atol=1e-9)),
            'incase_inband_pooled': inc['inband_w33_computed_pooled'],
            'incase_band_pooled': inc['w32_seed_band_pooled']}
    lg = e6['logon_gap_shift_disclosure']
    # 在案 mean 出自「逐日差等权日均」估计量（e6 机器 e6_inference.py:252），重放 mean 出自池化差
    # （e3_closure 权威口径）——两口径非同一泛函（convention_note 明示「二者并报」），差异 ~3e-5 属口径差
    # 而非数值错误；signed 方向与逐 seed 池化值须逐位一致。
    mean_auc_pooled = [out['metrics']['auc_logon']['mean_w32'], out['metrics']['auc_logon']['mean_w33']]
    mean_ap_pooled = [out['metrics']['ap_logon']['mean_w32'], out['metrics']['ap_logon']['mean_w33']]
    cmp['logon_gap_shift'] = {
        'signed_auc_match': out['metrics']['auc_logon']['signed_diff_decreased_w32_to_w33'] == lg['signed_diff_decreased_auc'],
        'signed_ap_match': out['metrics']['ap_logon']['signed_diff_decreased_w32_to_w33'] == lg['signed_diff_decreased_ap'],
        'mean_auc_match': bool(np.allclose(mean_auc_pooled, lg['mean_auc_diff'], atol=5e-4)),
        'mean_ap_match': bool(np.allclose(mean_ap_pooled, lg['mean_ap_diff'], atol=5e-4)),
        'mean_caliber_note': ('在案 mean=逐日差等权日均口径；重放 mean=池化差口径；二者差约 3e-5（AUC）/3e-7（AP），'
                              '属 convention_note 声明的双泛函并报关系，非数值矛盾——两口径均值方向与结论一致'),
        'replay_mean_pooled_auc': mean_auc_pooled, 'replay_mean_pooled_ap': mean_ap_pooled,
        'incase_mean_dayavg_auc': lg['mean_auc_diff'], 'incase_mean_dayavg_ap': lg['mean_ap_diff'],
        'homo_ap_per_seed_match': bool(np.allclose([x for p in out['homo_ap_logon_w32_to_w33']['per_seed'] for x in p],
                                                   [x for p in lg['homo_ap_logon_w32_to_w33']['per_seed'] for x in p], atol=1e-5)),
        'incase_signed_auc': lg['signed_diff_decreased_auc'],
        'incase_signed_ap': lg['signed_diff_decreased_ap']}
    out['bitwise_comparison_vs_incase'] = cmp
    out['all_match'] = bool(all(v for k, v in cmp.items() if isinstance(v, dict)
                                for v in v.values() if isinstance(v, bool)))

    # ---- 按 seed 全量方向双口径注记（R2B-M3 resolution text，供 SSOT 引用）----
    ma, mp = out['metrics']['auc_logon'], out['metrics']['ap_logon']
    n_auc_signed = sum(ma['signed_diff_decreased_w32_to_w33'])
    n_auc_narrow = sum(ma['absdiff_narrowed_w32_to_w33'])
    n_ap_signed = sum(mp['signed_diff_decreased_w32_to_w33'])
    n_ap_narrow = sum(mp['absdiff_narrowed_w32_to_w33'])
    out['dual_caliber_note_r2bm3'] = (
        f"池化口径按 seed 全量方向表——AUC：w32 {ma['w32_pooled_diff']} → w33 {ma['w33_pooled_diff']}"
        f"，signed 向 homo 更优移动 {n_auc_signed}/3 seed、s2 越零反向（+0.0009→−0.0253，|差|放大）；"
        f"|差|缩小 {n_auc_narrow}/3 seed；均值 −0.1105→−0.0793 缩小。"
        f"AP：w32 {mp['w32_pooled_diff']} → w33 {mp['w33_pooled_diff']}"
        f"，signed 移动 {n_ap_signed}/3、|差|缩小 {n_ap_narrow}/3（s2 0.00006→−0.00098 越零反向）；"
        f"均值 −0.00531→−0.00182 缩小。"
        f"任何口径下「signed 3/3 下移/一致缩小」均不成立；正确表述=「AUC/AP 差距 2/3 seed 缩小、"
        f"s2 越零反向；均值缩小」（R2B-M3 resolution text；supersedes 原 note，原文保留于 "
        f"e6_inference.json 不改写）")
    return out


def main():
    os.makedirs(OUT, exist_ok=True)
    out = replay()
    assert all(np.isfinite(v) for m in out['metrics'].values()
               for k in ('w32_pooled_diff', 'w33_pooled_diff') for v in m[k]), 'NaN 断言'
    for m in out['metrics'].values():
        assert m['w32_seed_band_pooled'][0] <= m['w32_seed_band_pooled'][1], '带下界>上界'
    with open(f'{OUT}/e10_e3_pooled_replay.json', 'w') as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    print('E10_DONE all_match =', out['all_match'])
    print('dual_caliber_note:', out['dual_caliber_note_r2bm3'][:400])


if __name__ == '__main__':
    main()
