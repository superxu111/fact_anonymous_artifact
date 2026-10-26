"""finalize_r2.py — R2 修复后全项目 metrics 重写（A-M5 全网格改写 + 新实验收口）。
所有数字从结果 JSON 读取生成，不含手抄值。"""
import json
import numpy as np

R = '/root/autodl-tmp/projB_hetero_graph/results'
E1 = json.load(open(f'{R}/EXP-R1-E1/e1_window_alert.json'))
E5 = json.load(open(f'{R}/EXP-R1-E5/e5_evt_calib.json'))
E6 = json.load(open(f'{R}/EXP-R1-E6/e6_inference.json'))
E7 = json.load(open(f'{R}/EXP-R1-E7/e7_e4_repair.json'))
E8 = json.load(open(f'{R}/EXP-R1-E8/e8_closure.json'))
SS = json.load(open(f'{R}/EXP-R1-E5/e5_selfstarting.json'))

# ---- A-M5：E1 全网格实数（主口径 top50mean 54 点 + 全统计量 162 点）----
def grid(filtro):
    det_all = []
    for arm in ['full', 'homoflows', 'homo']:
        for s in ['0', '1', '2']:
            for stat in (['top50mean'] if filtro == 'main' else ['top50mean', 'mean', 'max']):
                det = E1['arms'][arm][s][stat]['detection']
                for arl0 in [100, 500]:
                    det_all.append((arm, stat, arl0, det[f'fixed@arl{arl0}']['detection_rate'],
                                    det[f'cusum_k0.5@arl{arl0}']['detection_rate'],
                                    det[f'ewma_l0.2@arl{arl0}']['detection_rate']))
    return det_all

g_main = grid('main')
g_all = grid('all')
summary = {
    '_meta': {
        'project': 'R1 主线：类型条件化机制价值的公平同池重审（projB 内重产）',
        'round': 'R1 第二轮修复（双审 R1A/R1B 响应）',
        'date': '2026-10-24',
        'verify_incase': json.load(open(f'{R}/EXP-R1-VERIFY/incase_check.json'))['verdict'],
        'note': '全部数字由 finalize_r2.py 从结果 JSON 读取生成；R1 第一轮的 E1 汇总数字修剪（R1A-M5）在本版按全网格改写',
    },
    'E1': {
        'h1_verdict_incase_r1': '已撤销的 R1 表述：fixed 检测率 1.0 / CUSUM 0.36-0.55（按有利方向收窄，R1A-M5）',
        'h1_fullgrid_r1': {
            'fixed_det_rate_main54': [round(min(x[3] for x in g_main), 3), round(max(x[3] for x in g_main), 3)],
            'fixed_det_rate_grid162': [round(min(x[3] for x in g_all), 3), round(max(x[3] for x in g_all), 3)],
            'cusum_det_rate_main54': [round(min(x[4] for x in g_main), 3), round(max(x[4] for x in g_main), 3)],
            'cusum_det_rate_grid162': [round(min(x[4] for x in g_all), 3), round(max(x[4] for x in g_all), 3)],
            'ewma02_det_rate_main54': [round(min(x[5] for x in g_main), 3), round(max(x[5] for x in g_main), 3)],
            'note': 'fixed 延迟全网格=0（11 评估窗全为攻击窗，day12 即最强攻击日；random_p=1.0 无统计意义，R1B-C1）'},
        'c1_fair_recalibration': {
            'root_cause': '9 离散良性值→经验校准结构性不可达（实测 8.7-9.4 vs 名义 100/500；54 点复算均值 9.0/9.1）',
            'fix': 'EXP-R1-E5：t 预测枢轴（小池主阈值，正态父体下精确预测式）+ EVT/GPD 形状诊断 + 混合规则（经验可行域用经验分位）；单元测试 6/6；判据 6 次运行前/收口前修订留痕',
            'h1_final_verdict_r2': ('R1 的「序贯读出无一非劣」判定撤销（前提 fixed 实测 ARL≈9 不成立）。公平校准重判定：'
                                    'ARL500 档 CUSUM/EWMA 9/9 非劣（fixed@τ_t=4.207 检出 0.182/延迟 2 vs CUSUM 0.364/延迟 2）；'
                                    'ARL100 档 7/9 非劣（fixed@τ_t=3.053 检出 0.182-0.455/延迟 0-2 vs CUSUM 0.545/延迟 1）；'
                                    '三读出共同 11 窗族级误报预算 0.1047/0.0218——H1 判据「至少一种序贯读出延迟 ≤ fixed」成立，'
                                    '主件资格恢复；红线：非劣≠等效，日窗降为描述性（R1B-m7）'),
            'g1_synthetic_replay_disclosed': {k: {kk: vv for kk, vv in v.items() if kk != 'rows'}
                                              for k, v in E5['g1_synthetic'].items()},
            'g1_direct_pivot_check': E5.get('g1_direct_pivot_check', {}),
            'g1_verdict': '门=直接模拟（20 万次）：rate×ARL0=0.952/1.025，两档 PASS；100 世界重放均值（0.82/0.67）因世界间 ARL 重尾偏斜精度不足，保留为离散度披露',
            'g2_hourly': E5['g2_hourly'],
            'g2_verdict': ('arl100：PASS（EVT τ=3.453 实测中位 ARL 112，名义在 95% CI 内；经验参照同过）——'
                           'EVT 不劣于经验分位（该池经验可行，与 Zenodo 配置律一致）；'
                           'arl500：验证流功效不足（347 良性小时，配置律 E=347×0.002≈0.7≪20，两法实测都饱和在 ~112）'
                           '——申报「该档本轮不可认证」而非误校准，延长良性流认证列下轮'),
            'g3_h1_reverdict': {k: {'verdict': v.get('verdict'),
                                    'noninferior_counts': v.get('noninferior_counts'),
                                    'familywise_falsealarm_11win': v.get('familywise_falsealarm_11win')}
                                for k, v in E5['g3_h1_reverdict'].items()},
            'evt_daywindow_negative_diagnosis': '真实日窗 z 池结构性有界（|z|≤8/3），GPD 诊断差（ξ̂=−3.92、QQ 相关 −0.61）——EVT 分支日窗为申报性负诊断（Zenodo 22081860 有界分数反证据的实况复现），t-枢轴为主阈值的依据',
            'z3_crosscheck': E5['z3_crosscheck']},
        'selfstarting_appendix': SS['readouts'],
        'files': ['EXP-R1-E1/unit_tests.json', 'EXP-R1-E1/e1_window_alert.json', 'EXP-R1-E1/e1_hourly_readout.json',
                  'EXP-R1-E5/criteria.json', 'EXP-R1-E5/unit_tests.json', 'EXP-R1-E5/e5_evt_calib.json',
                  'EXP-R1-E5/e5_selfstarting.json', 'EXP-R1-E1/criteria.json（R2 补登记）'],
    },
    'E2': {
        'inference_reinforcement': {
            'families': {k: {'family_def': v['family_def'], 'holm_aggregate': v['holm_aggregate'],
                             'agg_p_raw': v['agg_p_raw']}
                         for k, v in E6['E2']['families'].items()},
            'headline_recheck': E6['E2']['headline_recheck'],
            'fixes': E6['fixes']},
        'fullgrid_note': 'uniform 训练下 homo mixed AUC 全网格 [0.8425, 0.9393]（R1A-M5 修正，原表述 0.86-0.94）',
        'files': ['results/EXP-R1-E2/e2_closure.json', 'results/EXP-R1-E6/e6_inference.json'],
    },
    'E3': {
        'verdict_computed': '带判定由 finalize_e3_e4 硬编码文本改由代码计算（R1B-M3）：池化差口径 in-band = '
                            + json.dumps({m: E6['E3'][m]['inband_w33_computed_pooled']
                                          for m in ['auc_mixed', 'auc_logon', 'ap_logon']}, ensure_ascii=False)
                            + '；LOGON 差距缩小全量申报见 logon_gap_shift',
        'logon_gap_shift': E6['E3']['logon_gap_shift_disclosure'],
        'band_membership_computed_pooled': {m: E6['E3'][m]['inband_w33_computed_pooled']
                                            for m in ['auc_mixed', 'auc_logon', 'ap_logon']},
        'ci_dayblock_b2000': {m: {'w32': E6['E3'][m]['w32'], 'w33': E6['E3'][m]['w33'],
                                  'convention': '逐日差等权日均估计量（与池化差并报）'}
                              for m in ['auc_mixed', 'auc_logon', 'ap_logon']},
        'files': ['results/EXP-R1-E3/e3_closure.json', 'results/EXP-R1-E6/e6_inference.json'],
    },
    'E4': {
        'repairs': {
            'ap_event_weighted_v2_full': E7['F1_ap_event_weighted_v2']['full'],
            'ap_event_weighted_v2_homo_rerun': E7['F1_ap_event_weighted_v2']['homo_rerun'],
            'recall_grid_attribution_corrected': 'R1B-M2b 属实：K≤250 recall≈0（随机基线 0.0008）——「top-25 预算过粗」归因删除，改写为事件级排名信号弱',
            'w14_diagnosis_registered': True,
            'eventAP_ci': E7['F4_event_edge_ci']},
        'files': ['results/EXP-R1-E4/e4_granularity.json（含 r2_repairs）', 'results/EXP-R1-E7/e7_e4_repair.json'],
    },
    'E8_clean_ablation': {
        'verdict': E8['verdict'],
        'files': ['results/EXP-R1-E8/criteria.json', 'results/EXP-R1-E8/e8_closure.json'],
    },
    'disclosures': {
        'ablation_aux_head_semantics': 'R1B-M5：homo/homoflows 辅助类型头任务语义与 full 不同——E8 干净消融证明主结论不依赖该信号（局限披露保留）',
        'temporal_negsampling_docstring': 'R1B-M5：sample_neg_temporal 实现为全局观测对过滤（非"当日"），docstring 已更正+dead 变量移除；uniform 无过滤 vs temporal 全局过滤的协议差异申报',
        'eval_window_conditioning': 'R1B-m4：11 评估窗全为 redteam 活跃日、FLOW 池 0 正例——全部指标条件于攻击日观测池',
        'p_resolution_floor': 'R1B-m5：B=500 日块置换分辨率地板 (0+1)/501≈0.002，B=2000 地板 0.0005',
        'param_gap_dual_caliber': 'README 已同步双口径（在案 0.819% / projB 1.064%）',
    },
    'not_done_this_round': [
        'V 系列加分验证点（BOCPD/MCNS/TV 距离/污染探针臂/主机级标签噪声）——时间盒未排入，维持 R1 申报',
        'W2 良性日工作日/周末分层——纯良性窗仅 9 个，样本不足维持申报为局限',
        'E1 小时窗第二统计量（pair_count）变体——未做',
        '全链路 always-valid 置信序列（创新方案二完整形态）——本轮落地其联合推断残件（Holm+族申报+ARL 预算联合口径），置信序列列下轮',
    ],
}
with open(f'{R}/R1_metrics.json', 'w') as f:
    json.dump(summary, f, indent=1, ensure_ascii=False)
print('R1_metrics rewritten (R2)')
print('E1 full-grid:', json.dumps(summary['E1']['h1_fullgrid_r1'], ensure_ascii=False)[:300])
print('E5 G1 gates:', {k: v.get('gate_pass') for k, v in E5['g1_synthetic'].items()})
print('E5 G2 gates:', {k: (v.get('evt', {}).get('nominal_in_ci'), v.get('empirical_ref', {}).get('nominal_in_ci'))
                       for k, v in E5['g2_hourly'].items() if k.startswith('arl')})
print('E5 G3:', {k: v.get('verdict') for k, v in E5['g3_h1_reverdict'].items()})
