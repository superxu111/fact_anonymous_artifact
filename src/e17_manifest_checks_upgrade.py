"""e17_manifest_checks_upgrade.py — R2-E17 闸2 前置：number_manifest 机器可读化升级（R3B-m2/R2A-M1）。

每行新增 checks=[{path(精确点路径),expect(现值固化)}]；原 path/expect 自然语言式字段原文保留（格式债登记）。
expect 取自本脚本运行时刻 JSON 现值——闸2 只做「manifest expect vs JSON 现读」diff，不重算统计量。
幂等：重复运行以相同现值重写 checks（安全）。
"""
import json

B = '/root/autodl-tmp/projB_hetero_graph'
mp = f'{B}/results/EXP-R2-REG/number_manifest.json'
mf = json.load(open(mp))


def jp(f, p):
    cur = json.load(open(f'{B}/results/{f}'))
    for part in p.split('.'):
        if '[' in part:
            k, idx = part[:-1].split('[')
            cur = cur[k][int(idx)]
        else:
            cur = cur[part]
    return cur


e6 = 'EXP-R1-E6/e6_inference.json'
e9 = 'EXP-R2-E9/e9_auc_logon_agg.json'
e5 = 'EXP-R1-E5/e5_evt_calib.json'
e11 = 'EXP-R2-E11/e11_g3_sensitivity.json'
rows = mf['mappings']
by_n = lambda s: next(r for r in rows if r['n'].startswith(s))

by_n('temporal 聚合')['checks'] = (
    [{'path': f'{e6}:E2.families.family_half2_mixed.agg_p_raw[{i}]', 'expect': v}
     for i, v in enumerate(jp(e6, 'E2.families.family_half2_mixed.agg_p_raw'))]
    + [{'path': f'{e9}:families.family_half2_mixed.holm_aggregate_m24[{i}]', 'expect': v}
       for i, v in enumerate(jp(e9, 'families.family_half2_mixed.holm_aggregate_m24'))])

by_n('half1 LOGON AP')['checks'] = [
    {'path': f'{e6}:E2.families.family_half1_logonAP.agg_p_raw[{i}]', 'expect': v}
    for i, v in enumerate(jp(e6, 'E2.families.family_half1_logonAP.agg_p_raw'))]

pp = jp(e9, 'per_point')
by_n('auc_logon 聚合')['checks'] = [
    {'path': f'{e9}:per_point[{i}].seed_aggregate.p_b2000', 'expect': x['seed_aggregate']['p_b2000']}
    for i, x in enumerate(pp)]

by_n('fixed@τ_t arl500')['checks'] = [
    {'path': f'{e5}:g3_h1_reverdict.arl500.rows[7].fixed_evt.det_rate', 'expect': 0.0,
     'note': 'homo_s1 全区间下限端点（delay=None）'},
    {'path': f'{e5}:g3_h1_reverdict.arl500.rows[0].fixed_evt.det_rate', 'expect': 0.182,
     'note': 'full_s0=全区间最大值 0.182（禁单格代表值，R1A2-m1）'},
    {'path': f'{e5}:g3_h1_reverdict.arl500.noninferior_counts.cusum', 'expect': '9/9'}]

by_n('τ_t=3.0531')['checks'] = [
    {'path': f'{e5}:calibration.top50mean.arl100.main_tpivot.tau', 'expect': 3.0531},
    {'path': f'{e5}:calibration.top50mean.arl500.main_tpivot.tau', 'expect': 4.2068}]

by_n('G1 直接模拟')['checks'] = [
    {'path': 'EXP-R2-REG/e5_direct_check.json:unconditional_iid_200k.arl100.rate_x_arl0',
     'expect': jp('EXP-R2-REG/e5_direct_check.json', 'unconditional_iid_200k.arl100.rate_x_arl0')},
    {'path': 'EXP-R2-REG/e5_direct_check.json:unconditional_iid_200k.arl500.rate_x_arl0',
     'expect': jp('EXP-R2-REG/e5_direct_check.json', 'unconditional_iid_200k.arl500.rate_x_arl0')}]

by_n('G1 原门 fail')['checks'] = [
    {'path': f'{e5}:g1_synthetic.arl100.pass_frac_lt33pct_reported', 'expect': 0.08},
    {'path': f'{e5}:g1_synthetic.arl500.pass_frac_lt33pct_reported', 'expect': 0.02},
    {'path': f'{e5}:g1_synthetic.arl100.mean_alarm_rate_x_arl0',
     'expect': jp(e5, 'g1_synthetic.arl100.mean_alarm_rate_x_arl0')},
    {'path': f'{e5}:g1_synthetic.arl500.mean_alarm_rate_x_arl0',
     'expect': jp(e5, 'g1_synthetic.arl500.mean_alarm_rate_x_arl0')}]

by_n('ξ̂')['checks'] = [
    {'path': f'{e5}:calibration.top50mean.gpd_xi_main', 'expect': jp(e5, 'calibration.top50mean.gpd_xi_main')},
    {'path': f'{e5}:calibration.top50mean.qq_corr_main', 'expect': jp(e5, 'calibration.top50mean.qq_corr_main')}]

by_n('E3 signed')['checks'] = [
    {'path': 'EXP-R2-E10/e10_e3_pooled_replay.json:metrics.auc_mixed.inband_w33_computed_pooled',
     'expect': [False, True, True]},
    {'path': 'EXP-R2-E10/e10_e3_pooled_replay.json:metrics.auc_logon.inband_w33_computed_pooled',
     'expect': [True, True, True]},
    {'path': 'EXP-R2-E10/e10_e3_pooled_replay.json:metrics.ap_logon.inband_w33_computed_pooled',
     'expect': [True, True, True]},
    {'path': 'EXP-R2-E10/e10_e3_pooled_replay.json:metrics.auc_mixed.signed_diff_decreased_w32_to_w33',
     'expect': jp('EXP-R2-E10/e10_e3_pooled_replay.json', 'metrics.auc_mixed.signed_diff_decreased_w32_to_w33')},
    {'path': 'EXP-R2-E10/e10_e3_pooled_replay.json:metrics.ap_logon.signed_diff_decreased_w32_to_w33',
     'expect': jp('EXP-R2-E10/e10_e3_pooled_replay.json', 'metrics.ap_logon.signed_diff_decreased_w32_to_w33')}]

by_n('E8a tw0')['checks'] = [
    {'path': 'EXP-R1-E8/e8_closure.json', 'note': 'presence 检查；E8a/E8b 逐 p 写作期取表（表结构无稳定点路径）',
     'expect': None}]

by_n('E7-F1')['checks'] = [
    {'path': 'EXP-R1-E7/e7_e4_repair.json:F1_ap_event_weighted_v2.full[2]', 'expect': 0.0081,
     'note': 'R1A2-m3/R2B-m3 滑笔更正值：第 3 seed=0.0081'},
    {'path': 'EXP-R1-E7/e7_e4_repair.json:F1_ap_event_weighted_v2.homo_rerun[2]', 'expect': 0.0081}]

by_n('class1 support')['checks'] = [
    {'path': 'EXP-R1-E7/e7_e4_repair.json:F3_w14_diagnosis.per_class_prf.full[0].class1.support',
     'expect': 313994}]

by_n('fixed 实测 ARL')['checks'] = [
    {'path': 'EXP-R1-E1/e1_window_alert.json',
     'note': 'presence 检查（54 点实测 ARL 8.7-9.4 聚合自 arms.* 窗级校准行，写作期取表）', 'expect': None}]

by_n('族级误报预算')['checks'] = [
    {'path': f'{e5}:g3_h1_reverdict.arl100.familywise_falsealarm_11win',
     'expect': jp(e5, 'g3_h1_reverdict.arl100.familywise_falsealarm_11win')},
    {'path': f'{e5}:g3_h1_reverdict.arl500.familywise_falsealarm_11win',
     'expect': jp(e5, 'g3_h1_reverdict.arl500.familywise_falsealarm_11win')}]

sens = jp(e11, 'sensitivity')
by_n('E9/E11 敏感性')['checks'] = [
    {'path': f'{e11}:sensitivity.{k}.noninferior_kappa_tightened', 'expect': v['noninferior_kappa_tightened']}
    for k, v in sens.items()]

by_n('E11 κ 收紧因子')['checks'] = [
    {'path': f'{e11}:kappa_t_pivot_inflation.arl100', 'expect': 1.3124},
    {'path': f'{e11}:kappa_t_pivot_inflation.arl500', 'expect': 1.4616}]

by_n('D1 三臂')['checks'] = [{'path': 'EXP-R1-VERIFY/incase_check.json:tolerance', 'expect': 0.0005}]

by_n('D12')['checks'] = [
    {'path': 'EXP-R1-E1/e1_hourly_readout.json:readouts.ewma0.2@arl100.attack_coverage', 'expect': 0.544},
    {'path': 'EXP-R1-E1/e1_hourly_readout.json:readouts.fixed@arl100.attack_coverage', 'expect': 0.044},
    {'path': 'EXP-R1-E1/e1_hourly_readout.json:readouts.fixed@arl100.delay_h', 'expect': 49},
    {'path': 'EXP-R1-E1/e1_hourly_readout.json:readouts.fixed@arl500.attack_coverage', 'expect': 0.0},
    {'path': 'EXP-R1-E1/e1_hourly_readout.json:readouts.fixed@arl500.delay_h', 'expect': 50},
    {'path': 'EXP-R1-E1/e1_hourly_readout.json:readouts.cusum0.5@arl500.delay_h', 'expect': 7}]

by_n('三臂参数数')['checks'] = [
    {'path': 'EXP-R1-E3/w9_param_recount.json:full_w32.total', 'expect': 843631},
    {'path': 'EXP-R1-E3/w9_param_recount.json:homo_w32.total', 'expect': 834653},
    {'path': 'EXP-R1-E3/w9_param_recount.json:homo_w33.total', 'expect': 860998}]

mf['checks_upgrade_note'] = (
    'R2 第四轮 E17 闸2 规范化：每行新增 checks=[{path(精确点路径),expect(现值固化)}]；原 path/expect 自然语言式'
    '字段原文保留（格式债登记）。expect 取自本脚本运行时刻 JSON 现值——闸2 只做「manifest expect vs JSON 现读」'
    'diff，不重算任何统计量（statcheck 教训适配：Holm/置换校正数字不可重算）。')
json.dump(mf, open(mp, 'w'), ensure_ascii=False, indent=1)
print('checks upgraded:', sum(1 for r in rows if 'checks' in r), '/', len(rows))
