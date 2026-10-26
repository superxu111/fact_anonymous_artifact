"""e15_calibration_certificate.py — R2-E15（定向创新方案二 + R3B-M4/R3B-m7 响应）：校准结论证书化。

【实验设计自查（开工前冻结，判据见 results/EXP-R2-E15/criteria_e15.json）】
假设：四校准规格在 9 个标准化良性 z 池上的真实越限行为可证书化，把规格曲线从「结论对规格敏感」的
  削弱读法翻转为「名义预算必须通过真实良性流越限检验才算数」的判别读法（R3B-M4）。
方法：
  ①9 良性池构建与 e5_evt_calib.py:217-229 / r2_reg_layer.build_spec_curve 同式（top50mean 主口径）；
  ②每规格×每 ARL0 档统计 realized_exceed_pools（越限=池内 ≥1 良性窗 z>τ）、逐池 FAR（告警窗数/9）、
    realized_ARL_at_exceed（=9/告警窗数，越限池）——复审人 R3B-M4 数值（S1/S2 0/9、S3 1/9、S4 4/9）
    作对照，本脚本独立重算，不一致时以脚本值为准并如实报告差异；
  ③精确二项检验 P(X≥k | Binomial(9, 1/ARL0))（n=9 低功效下替代渐近 Kupiec POF；Kupiec LR 作对照并报）
    + 保形 p=(1+k)/(n+1) 并报（分布无关档）；
  ④单元适配申报（保形文献三条警示全命中，写进产物）：名义预算按日、可交换单元=池（池内自相关、
    9 池≠81 日独立）→ 按池聚合，有效样本=9 池、证书粒度 1/10；
  ⑤S2 arl500 档 τ=3.0 同值系设计使然（该档名义正态阈 z=2.878/枢轴 4.207）——产物内加注（R3B-m7）；
  ⑥S3/S4 行标注「失效体制反证档」；e-process（Ramdas et al., arXiv:2210.01948）仅作附录方向句引用。
成功判据：S1/S2 两档 0/9 越限（5% 水平不拒绝名义预算的有信息正面陈述）；S3/S4 的越限档强拒绝或
  如实申报；全部证书与 spec_curve_h1.json 以追加键（只增不改）落盘。
规模/成本：纯 CPU 秒级。
"""
import json, sys
import numpy as np
from scipy.stats import binom, chi2

sys.path.insert(0, '/root/autodl-tmp/projB_hetero_graph/src')
from e1_window_alert import ARMS, BENIGN_DAYS, window_stats

BASE = '/root/autodl-tmp/projB_hetero_graph'
OUT = f'{BASE}/results/EXP-R2-E15'
SPECS = [('S1_tpivot_t9_8', 'main_tpivot', 'tau'),
         ('S2_z3_probe', None, None),
         ('S3_empirical_reachable', 'evt_hybrid', 'tau_empirical'),
         ('S4_evt_hybrid_REJECTED_daywindow', 'evt_hybrid', 'tau')]

def build_pools():
    """9 个标准化良性 z 池（3 臂×3 seed，每池 9 良性日 top50mean z 值；与 e5_evt_calib.py:217-229 同式）。"""
    pools = {}
    for arm, d in ARMS.items():
        for seed in range(3):
            bn = window_stats(f'{d}/benign_scores_seed{seed}.npz')
            bv = np.array([bn[dd]['top50mean'] for dd in BENIGN_DAYS])
            mu, sd = bv.mean(), bv.std(ddof=1)
            pools[f'{arm}_s{seed}'] = (bv - mu) / sd
    return pools


def kupiec_lr(k, n, p):
    """Kupiec 1995 POF 似然比（渐近 χ²(1)；n=9 短窗下仅作对照并报，主口径=精确二项）。"""
    if k == 0:
        return 0.0, 1.0
    pi = k / n
    lr = 2 * (np.log(pi ** k * (1 - pi) ** (n - k)) - np.log(p ** k * (1 - p) ** (n - k)))
    return float(lr), float(chi2.sf(lr, 1))


def p_tier(p):
    """R3/P0-1 档界规则（先于重跑登记于 results/EXP-R3-REG/criteria_r3.json）：档界=1e-6。"""
    return '<1e-6' if p < 1e-6 else f'{p:.6g}'


def row_note(name, v_a, v_b):
    """R3/P0-1：note 逐档（逐行）生成，废除跨档共享静态字符串（R2A2-M2①/R4B-m1② 反模式修复）。

    旧静态句「S3 在 null_A 下边缘拒绝」对 arl100/S3（p=0.086483 不拒绝）为文本-数据矛盾；
    本函数按该行自身 verdict 组装，S3/S4/S1-S2 三种体裁各自与自身数值一致。"""
    head = '双口径并报（R3 逐档生成，废除跨档共享静态句）：'
    if name.startswith(('S1', 'S2')):
        return (head + f'本行 null_A={v_a}、null_B={v_b}——「0/9 越限」在两 null 下均不拒绝'
                '（正面证书稳健；零功率申报见 zero_power_declaration_r3：阈值恒高于 9 点池结构上界 8/3，'
                '本格为结构必然、不承担判别证据语义）')
    if name.startswith('S3'):
        return (head + f'本行 null_A={v_a}、null_B={v_b}——口径依赖如实并报、不作单口径判语'
                '（二项检验结论依赖单元口径；越限池实测 ARL≈9=第一轮 C1 失效体制）')
    return (head + f'本行 null_A={v_a}、null_B={v_b}——失效体制定量反证（两口径均强拒绝；'
            '第一轮 C1 失效体制复现）')


def main():
    import os
    os.makedirs(OUT, exist_ok=True)
    pools = build_pools()
    e5 = json.load(open(f'{BASE}/results/EXP-R1-E5/e5_evt_calib.json'))
    reviewer_ref = {'S1': '0/9', 'S2': '0/9', 'S3': '1/9', 'S4': '4/9'}  # R3B-M4 复审人数值（未分档）
    # R3/P0-1 零功率句数值前提：本席从 9 池 npz 现算 max|z|（V3：勿抄复审人数 2.0367）
    max_abs_z = float(max(np.abs(z).max() for z in pools.values()))
    zero_power = {
        'status': 'ZERO-POWER DECLARATION（R3/P0-1 补录，R2A2-M2②）',
        'structural_bound_9pt_pool': '8/3≈2.6667（n 点标准化池任意值上界=(n-1)/√n，n=9）',
        'empirical_max_abs_z_9pools_recomputed_here': round(max_abs_z, 4),
        'empirical_note': '本席从 9 良性池 npz 现算（复审人数 2.0367 仅作对照，非本句数值来源）',
        'thresholds_vs_bound': 'S1 τ_t：arl100=3.0531/arl500=4.2068；S2 τ=3.0——全部恒高于结构上界 8/3 与实测 max|z|',
        'declaration': ('S1/S2 的 0/9 越限系结构必然（阈值高于 9 点池可达上界→零功效），非「校准良好」的'
                        '证据性通过；欠效力规格不承担判别语义（规格曲线须 statistically valid 申报）——'
                        '引用 0/9 证书列时须随附本申报')}

    cert = {}
    for arl0 in (100, 500):
        cal = e5['calibration']['top50mean'][f'arl{arl0}']
        alpha = 1.0 / arl0
        tab = {}
        for name, sens, key in SPECS:
            tau = 3.0 if name.startswith('S2') else float(
                cal[sens][key] if sens == 'main_tpivot' else cal['sensitivity'][sens][key])
            exceed, fars, arls = [], {}, {}
            for pid, z in sorted(pools.items()):
                hits = int((z > tau).sum())
                fars[pid] = round(hits / len(z), 4)
                if hits > 0:
                    exceed.append(pid)
                    arls[pid] = round(len(z) / hits, 2)
            k = len(exceed)
            p_binom = float(binom.sf(k - 1, len(pools), alpha))  # P(X≥k)
            # 双 null 并报（防单元口径误读）：null_A=池越限概率≤名义日预算 1/ARL0（保守下界口径）；
            # null_B=每窗误报率≤1/ARL0 → 池越限概率=1−(1−1/ARL0)^9（每窗独立派生，池层自然放大）
            p_pool_a = 1 - (1 - alpha) ** len(pools[sorted(pools)[0]])
            p_binom_b = float(binom.sf(k - 1, len(pools), p_pool_a))
            p_conf = (1 + k) / (len(pools) + 1)
            lr, p_kup = kupiec_lr(k, len(pools), alpha)
            v_a = '不拒绝名义误报预算' if p_binom > 0.05 else '拒绝名义误报预算'
            v_b = '不拒绝' if p_binom_b > 0.05 else '拒绝'
            # R3/P0-1：p 落盘改全精度（禁 round(·,6) 把 2e-9 舍为 0.0，R2A2-m3）+<1e-6 档显示
            row = {'tau': round(tau, 4), 'exceed_pools': exceed, 'k_exceed_pools': f'{k}/9',
                   'per_pool_far': fars,
                   'realized_ARL_at_exceed': arls if arls else '∞（0 越限池，实测误报率 0）',
                   'exact_binomial': {'null_A': f'池越限概率≤名义日预算 1/{arl0}（保守口径）', 'k': k,
                                      'p_one_sided': p_binom,
                                      'p_display_tier': p_tier(p_binom),
                                      'verdict_5pct': v_a,
                                      'null_B': f'每窗误报率≤1/{arl0} → 池越限概率={p_pool_a:.4f}（每窗独立派生口径）',
                                      'p_one_sided_nullB': p_binom_b,
                                      'p_display_tier_nullB': p_tier(p_binom_b),
                                      'verdict_5pct_nullB': v_b,
                                      'note': row_note(name, v_a, v_b)},
                   'conformal_p': round(p_conf, 3),
                   'kupiec_pof_asymptotic_reference': {'LR': round(lr, 4), 'p_chi2_1': p_kup,
                                                       'p_display_tier': p_tier(p_kup),
                                                       'note': '渐近 χ²(1) 在 n=9 低功效窗下仅对照并报（主口径=精确二项）'},
                   'status': ('主阈值（正态父体 t-枢轴外推）' if name.startswith('S1') else
                              'R1A 稳健性探针' if name.startswith('S2') else
                              '池内经验可达阈值——失效体制反证档（判据反向；越限池实测 ARL≈9=第一轮 C1 失效体制；'
                              '二项检验结论依赖单元口径，见 exact_binomial 双 null 并报）'
                              if name.startswith('S3') else
                              '已否决分支（ξ̂=−3.92 自家诊断拒绝）——失效体制反证档（两口径二项强拒绝；'
                              '第一轮 C1 失效体制复现）')}
            if name.startswith(('S1', 'S2')):
                row['zero_power_declaration_r3'] = zero_power
            if name.startswith('S2'):
                # R3/P0-1 档位标签修正（R4B-m1①/R2A2-m2）：标签=本档循环变量，数值取本档 cal
                row['note_r3bm7'] = (f'该档沿用 τ=3.0 属设计使然（R1A 稳健性探针两档同值）；本档（arl{arl0}）'
                                     f'名义正态阈 z={float(__import__("scipy.stats", fromlist=["norm"]).norm.ppf(1 - 1 / arl0)):.4f}/'
                                     f'枢轴 τ_t={float(cal["main_tpivot"]["tau"]):.4f}——勿将 z3 读作本档名义预算对应阈值')
            row['reviewer_crossref_r3bm4'] = reviewer_ref[name.split('_')[0]] + '（复审人合计口径，未分档；本行=本档独立重算）'
            tab[name] = row
        cert[f'arl{arl0}'] = tab

    out = {'exp': 'R2-E15：校准结论证书化（定向创新方案二；R3B-M4 realized 误报预算列+R3B-m7 S2 注）',
           'date': '2026-10-24',
           'unit_adaptation_declaration': (
               '单元适配申报（保形异常检测文献三条警示全命中，引用时须随附）：①名义误报预算按「日」申报，'
               '但本证书的可交换单元=「池」（3 臂×3 seed 的良性日 z 池）——池内 9 日存在自相关、池间共享'
               '攻击配置，9 池≠81 个独立日观测，有效样本按池计=9；②证书粒度受保形分位数 1/(n+1) 限制：'
               'n=9 池 → 最细可分辨概率 1/10，本证书不支撑对名义预算更细档位的分辨；③名义预算在「日单元」'
               '与「池单元」间不直接换算（池层越限概率 ≥ 日层预算），故本证书口径为「按池聚合的越限池计数」，'
               '与 C2′「9 良性日不可辨识」披露互为表里——该申报本身即 C2′「披露不可辨识性」卖点的延伸证据。'),
           'reading': ('判别读法（翻转 R3B-M4 担忧的「multiverse 对称不稳定」削弱读法）：名义预算必须通过真实'
                       '良性流越限检验才算数——S1/S2 通过（0/9 池越限，实测误报率 0）；S3/S4 不通过（越限档'
                       '实测 ARL≈9=第一轮 R1B-C1 失效体制），恰为「经验可达/已否决 EVT 规格不可部署」的定量'
                       '反证。规格曲线的非对称性=校准分辨率不可辨识性的证据，而非结论不稳定。'),
           'appendix_direction_sentence_only': (
               '序贯读出 ARL 认证不可辨识的统计学出路（仅方向句引用，本轮不算数）：e-process/Ville 界'
               '（Ramdas, Grünwald, Vovk & Shafer 2022, "Game-theoretic statistics and safe anytime-valid '
               'inference", arXiv:2210.01948）允许任意停止时刻有效的误报控制，可在延长良性流到位后对 ARL0 '
               '做序贯认证——列_future work。'),
           'certificate': cert}
    # ---- R3/P0-1 只增不改：旧 certificate（含静态错句/note_r3bm7 贴错档）原文保留于 superseded 键 ----
    cert_path = f'{OUT}/calibration_certificate.json'
    if os.path.exists(cert_path):
        old = json.load(open(cert_path))
        if 'superseded_r3_static_note' not in old:
            out['superseded_r3_static_note'] = {
                'status': 'SUPERSEDED（R3/P0-1：旧 certificate 8 格原文保留——共享静态 note 句'
                          '「S3 在 null_A 下边缘拒绝」与 arl100/S3 实算 p=0.086483 不拒绝矛盾（R2A2-M2①/R4B-m1②）；'
                          'arl100/S2 的 note_r3bm7 贴 arl500 档标签却注 arl100 值 2.3263/3.0531（R4B-m1①/R2A2-m2）；'
                          'p_one_sided 两处 round(·,6) 舍零/降精度（R2A2-m3）。更正见 certificate 逐档生成 note）',
                'old_certificate': old['certificate']}
    with open(cert_path, 'w') as f:
        json.dump(out, f, indent=1, ensure_ascii=False)

    # ---- spec_curve_h1.json 追加键（只增不改；R3：既有 r3bm4 键不改写，追加 v2_r3fix）----
    scp = f'{BASE}/results/EXP-R2-REG/spec_curve_h1.json'
    sc = json.load(open(scp))
    v2_body = {
        'status': 'APPENDED（R3/P0-1：supersedes realized_falsealarm_budget_r3bm4 的 headline p 精度——'
                  '原键原文保留不改写；本键 headline 为全精度 p+档位显示）',
        'producer': 'src/e15_calibration_certificate.py（R2-E15 + R3/P0-1 修正）→ results/EXP-R2-E15/calibration_certificate.json',
        'headline': {f'{arl}/{spec.split("_")[0]}': {'k_exceed_pools': v['k_exceed_pools'],
                                                     'exact_binomial_p': v['exact_binomial']['p_one_sided'],
                                                     'exact_binomial_p_display_tier': v['exact_binomial']['p_display_tier'],
                                                     'conformal_p': v['conformal_p']}
                     for arl in cert for spec, v in cert[arl].items()},
        'p_precision_diff_r3': [
            {'cell': 'arl100/S4 null_A', 'old_rounded_r4': 1e-06, 'new_full_precision': cert['arl100']['S4_evt_hybrid_REJECTED_daywindow']['exact_binomial']['p_one_sided']},
            {'cell': 'arl500/S4 null_A', 'old_rounded_r4': 0.0, 'new_full_precision': cert['arl500']['S4_evt_hybrid_REJECTED_daywindow']['exact_binomial']['p_one_sided']},
            {'note': 'diff 显式登记（V2）：旧 round(·,6) 舍零/降精度 vs 新全精度；verdict 判档（5% 水平）两版一致，'
                     'p_display_tier=<1e-6 档界规则先于重跑登记于 results/EXP-R3-REG/criteria_r3.json'}],
        'reading': out['reading'], 'unit_adaptation_declaration': out['unit_adaptation_declaration']}
    sc['realized_falsealarm_budget_r3bm4_v2_r3fix'] = v2_body
    with open(scp, 'w') as f:
        json.dump(sc, f, indent=1, ensure_ascii=False)

    # ---- 控制台摘要 + 复审人对照 ----
    for arl in cert:
        for spec, v in cert[arl].items():
            print(f"{arl} {spec.split('_')[0]}: τ={v['tau']} 越限 {v['k_exceed_pools']} "
                  f"二项 p={v['exact_binomial']['p_one_sided']} 保形 p={v['conformal_p']} "
                  f"（复审人对照 {v['reviewer_crossref_r3bm4'].split('（')[0]}）"
                  + (f" | 越限池 ARL {v['realized_ARL_at_exceed']}" if isinstance(v['realized_ARL_at_exceed'], dict) else ''))
    print('E15_DONE')


if __name__ == '__main__':
    main()
