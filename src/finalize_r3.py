"""finalize_r3.py — R2 第三轮 P0 修复层（P0-1/P0-2/P0-3/P0-4/P0-5/P0-7）。

原则：只增不改——R1_metrics.json 的既有字段全部保留（含被 supersedes 的 h1_final_verdict_r2 原文），
全部修复以顶层 r2p0_fixes 键登记，指向 supersedes 目标。原 finalize_r2.py 不改动。
"""
import json

R = '/root/autodl-tmp/projB_hetero_graph/results'
M = json.load(open(f'{R}/R1_metrics.json'))
E9 = json.load(open(f'{R}/EXP-R2-E9/e9_auc_logon_agg.json'))
E10 = json.load(open(f'{R}/EXP-R2-E10/e10_e3_pooled_replay.json'))
E11 = json.load(open(f'{R}/EXP-R2-E11/e11_g3_sensitivity.json'))
E5 = json.load(open(f'{R}/EXP-R1-E5/e5_evt_calib.json'))
DC = json.load(open(f'{R}/EXP-R2-REG/e5_direct_check.json'))

arl500_dets = [r['fixed_evt']['det_rate'] for r in E5['g3_h1_reverdict']['arl500']['rows']]
arl100_dets = [r['fixed_evt']['det_rate'] for r in E5['g3_h1_reverdict']['arl100']['rows']]

M['r2p0_fixes'] = {
    'date': '2026-10-24',
    'round': 'R2 第三轮（主张句与判据登记修复，R1A2/R2B 双审 P0 清单响应）',
    'principle': '只增不改：被 supersedes 字段原文保留于本文件原位置，本层登记替代内容与指针',

    # ---------- P0-1 ----------
    'P0-1_h1_final_verdict_r2_CONDITIONAL_supersedes': {
        'supersedes': 'E1.c1_fair_recalibration.h1_final_verdict_r2（含「主件资格恢复」与「H1 成立」判语，原文保留）',
        'conditional_verdict': ('在正态父体 t-枢轴外推下（保守阈值；真实良性池实测误报率 0，序贯读出实测≈名义），'
                                '存在延迟非劣的序贯读出；ARL0=100/500 在 9 良性日上不可辨识，'
                                '日窗 H1 为假设依赖的描述性结论。'),
        'main_track_status': '「主件资格恢复」废除——本项目以 C2′（极小良性校准集公平校准协议）身份申报，'
                             '协议卖点=披露不可辨识性本身，而非「恢复了 H1」（R1A2-C1 resolution test）',
        'calibration_asymmetry_oneway_declaration': (
            '校准不对称单向申报（R1A2-C1①/R2B-M2①）：fixed@τ_t 恒高于 9 值池 |z|≤8/3 结构上界 → '
            '真实良性池误报率=0（实测 ARL=∞）；同表 CUSUM/EWMA 实测 ARL 88.3-545≈名义——'
            '名义预算对齐下真实预算 fixed=0 vs 序贯≈1/ARL0，方向有利于「存在非劣序贯」判定，此前未申报'),
        'fixed_det_rate_full_interval': {
            'arl500_grid9': sorted(arl500_dets), 'arl500_interval': [min(arl500_dets), max(arl500_dets)],
            'arl100_grid9': sorted(arl100_dets), 'arl100_interval': [min(arl100_dets), max(arl100_dets)],
            'endpoint_note': 'arl500 端点：min=homo s1 0.0（delay=None 全程无告警，在案）、max=full 三 seed 0.182'
                             '——R1A2-m1 更正：禁止引单格 0.182 当代表值（原文「fixed@τ_t=4.207 检出 0.182/延迟 2」'
                             '系全区间最大值）'},
        'familywise_budget_scope': '「三读出共同 11 窗族级误报预算 0.1047/0.0218」降为名义口径申报'
                                   '（对 deployed fixed 实际预算 0 不成立，R1A2-C1②）',
        'one_sided_sensitivity_E11': {
            'design': 'CUSUM/EWMA 阈值按枢轴保守因子 κ=τ_t/z_{1−1/ARL0}（执行值从 E11 JSON 现读：'
                      + '/'.join(f"arl{a.replace('arl', '')} {v}" for a, v in E11['kappa_t_pivot_inflation'].items())
                      + '；R2 第四轮更正（R2A-M1/R3B-M2）：原字符串硬编码「arl100 1.3122/arl500 1.2785」系'
                        'e11 docstring 分母误植（3.2905=z_{1−1/2000}）的传播，本行改为从 '
                        'results/EXP-R2-E11/e11_g3_sensitivity.json:kappa_t_pivot_inflation 现读，'
                        '该数字已入 number_manifest 底账（E17 闸 2 lint 覆盖）',
            'restore_check': {k: v['noninferior_kappa1_restore'] + '（在案 ' + v['incase_reference'] +
                              '，还原' + ('成功' if v['restore_match'] else '失败') + '）'
                              for k, v in E11['sensitivity'].items()},
            'result': {k: v['noninferior_kappa_tightened'] for k, v in E11['sensitivity'].items()},
            'conclusion': '非劣计数在单向收紧下全部保持（arl500 9/9、arl100 7/9）——H1 条件句方向对校准哲学'
                          '不对称稳健（判据双去向预先登记，结果落「保持」支）',
            'product': 'results/EXP-R2-E11/e11_g3_sensitivity.json'},
        'det_rate_ci_band_disposition': {
            'old_field': 'INVALIDATED（e5_evt_calib.json 原文不改写；混入已否决 EVT 分支 tau_ci 下界'
                         '（R1A2-m7）→ 带宽 [x,1.0] 虚高无信息量）',
            'redefined_guard_band': {k: v['redefined_guard_band'] for k, v in E11['det_rate_ci_band_disposition'].items()},
            'product': 'results/EXP-R2-E11/e11_g3_sensitivity.json:det_rate_ci_band_disposition'},
        'spec_curve': 'results/EXP-R2-REG/spec_curve_h1.json（+png）：{τ_t, z3.0, 经验可达, 已否决 EVT} 四规格下'
                      '检出/延迟排布——S1 主阈值规格判「存在非劣」、S3 经验可达规格判「反向」、S4 列为反证，'
                      '判定对校准规格敏感的图形化（multiverse 排布，P0-1 条件句图形素材）',
    },

    # ---------- P0-2 ----------
    'P0-2_criteria_chain_honest_supersedes': {
        'supersedes': 'E1.c1_fair_recalibration.fix 中的「判据 6 次运行前/收口前修订留痕」表述与'
                      'RESEARCH_LOG §6 头部「全部新实验判据先落盘…原文保留可审计」声明',
        'honest_statement': ('判据链如实表述：E5/E7/E8 判据文本内容与运行内容一致（内容层可信）；但 E5 修订 4-6、'
                             'E7 frozen_at、E1/VERIFY registered_at 的自报时刻均为回填值（mtime 双列对照见 '
                             'results/EXP-R2-REG/criteria_mtime_ledger.json，11 行全量）；E6 当时无 criteria.json'
                             '（已 post_hoc 补登记 results/EXP-R1-E6/criteria.json）。自报时刻字段一律视为'
                             '「非文件系统留痕」，审计以 mtime 为准；今后判据修订以独立追加文件/追加键落盘。'),
        'C2_prime_claim_downgrade': '「验收判据制度化」卖点降级为「判据文本+修订留痕落盘」（时刻不可审计，'
                                    'R2B-C1；不含外部可验证时间戳/注册系统主张）',
        'e3_band_tolerance_registration': '±0.02 带容差的判据登记补录：原文仅在 e6_inference.py:241-242 代码中，'
                                          'RESEARCH_LOG:193「已在 criteria 补注册」失实——现登记于此：'
                                          '池化差口径 in-band = min(w32)−0.02 ≤ w33 ≤ max(w32)+0.02（冻结判据口径）',
        'g1_direct_pivot_check_producer': '补录生成代码 src/e5_direct_check.py（P0-2/V6），产物 '
                                          'results/EXP-R2-REG/e5_direct_check.json',
    },

    # ---------- P0-3 ----------
    'P0-3_g1_three_sentence_verdict_supersedes': {
        'supersedes': 'g1_verdict（「两档 PASS」单句）',
        'sentence_1_identity': ('①枢轴恒等式验证通过：20 万次直接模拟 rate×ARL0=0.9915/1.0275（无条件 iid 结构，'
                                '3SE 内=解析真值 1.000；生成代码 src/e5_direct_check.py）——这证明 t-枢轴公式'
                                '实现正确（正态族内恒等式），不是假设检验、更不是父体验证。存档值 0.952/1.025、'
                                '复审人 R1A2 1.015/1.052、R2B 1.001/1.020 与本席无条件结构值并报；存档值偏离解析'
                                '真值约 −4.8%（≫SE 0.07%）归因于世界重放式结构对世界的条件化（本席世界重放结构'
                                '实测 0.8284/0.6639 与 R2B 合并经验率逐位一致，复现该方向系统偏差）；三实现均过 '
                                '±15% 门，G1「通过」不因实现选择翻转。'),
        'sentence_2_gate_never_ran_as_designed': ('②预登记 GPD 世界门未过且从未按原设计运行：criteria 原文 G1='
                                                  '「100 个 9 池×9 点短记录合成世界（三种真 ξ*∈{0.1,0.3,0.5}）…'
                                                  '占比 ≥85%」；实现为全正态世界（e5_evt_calib.py:279 rng_w.normal(0,1,9)），'
                                                  '世界分布改换未作为修订申报。按原门口径（修订 5 均值±15%）实测 fail：'
                                                  'pass_frac=0.08/0.02、mean_rate×ARL0=0.8245/0.6722'
                                                  '（logs/e5_main.log 末尾与 e5_evt_calib.json g1_synthetic 双源逐位一致）。'
                                                  'H1 重写的形式前提靠修订 6 换门（直接模拟）成立——按冻结 verdict_rule '
                                                  '字面（G1+G2 过），G1 原设计门未过。'),
        'sentence_3_parent_rejected': ('③正态父体假设被自家数据拒绝：真实日窗 z 池结构性有界（|z|≤8/3），'
                                       'GPD 拟合差（ξ̂=−3.92、QQ 相关 −0.61）——EVT 日窗分支为申报性负诊断；'
                                       '10% GPD(0.3) 污染下 t-枢轴中位 ARL −38%/−79%（中度过告警 1.6×/5×，申报边界）。'
                                       'G1 的正确读法=code verification（V&V 术语）：通过无信息量，失败才有信息。'),
        'C2_prime_error_rate_guarantee_scope': 'C2′「平均误报率保证」限定为「正态族内精确（枢轴恒等式）、'
                                               '污染/重尾下不保证（申报边界）」',
    },

    # ---------- P0-4 ----------
    'P0-4_logon_dual_caliber_supersedes': {
        'supersedes': 'RESEARCH_LOG E2 段「half1『LOGON 同池 full 更差』：6/6 协议点稳定 2/3 seeds，不翻转」'
                      '（原行保留，加注双口径）与第一轮「LOGON 同池 2/3 seeds 显著更差 p=0.002/0.002」的裸口径',
        'weakened_fact_ap': ('LOGON AP 口径（E6 交付）：half1 六点 seed 层聚合全不显著——agg_p_raw 0.074-0.595、'
                            'Holm 后全 1.0（m=24 统一口径复核一致）；per-seed p 0.19-0.99 亦无显著——'
                            '「full 更差」在 AP 口径连逐 seed 显著都没有（R1A2-M3②弱化事实，写回 SSOT）'),
        'new_evidence_auc_E9': ('LOGON AUC 口径（R2-E9 独立种子重跑，P0-4 补齐 R1A2-M3①缺口）：六协议点 seed 层'
                               '聚合 raw p 全部 <0.05（uniform_1 0.002/uniform_4 0.027/uniform_16 0.0095/'
                               'temporal_1 0.027/temporal_4 0.0005/temporal_16 0.0005，分辨率地板 1/2001）；'
                               'Holm m=24 一次性校正后 temporal_16/temporal_4=0.012、uniform_1=0.044 存活，'
                               'm=48 half1 合并大族敏感性 temporal_16/4=0.024 存活（uniform_1=0.092 不存活）；'
                               '方向 16/18 seed obs<0；聚合三口径交叉（聚合置换/Stouffer 单侧/Fisher/ACAT）结论一致'
                               '（uniform 组 2.6e-6/6.2e-5/0.0047，temporal 组 6.0e-8/1.4e-6/0.0007）——'
                               '聚合结论对聚合方法不敏感'),
        'incase_crossref': 'E9 独立种子 obs 与在案共享-seed 机器逐位一致（6 点×3 seed，如 uniform_1 '
                           '[-0.18391,-0.07393,-0.0123]≡在案）——管线零漂移；p 值邻域一致（置换种子不同）',
        'holm_family_def_fix': 'R1A2-m2：family_def 与算法统一为「18 逐 seed+6 聚合=m=24 一次性 Holm」；'
                               '原交付按 m=18/m=6 分层两套执行——统一后 half2 头条 temporal 聚合 0.0005→0.012 '
                               '仍过门（结论不变），half1_logonAP 全 1.0 不变',
        'logon_final_verdict_merged_with_E8a': (
            'LOGON 同池口径最终判语（P0-4，合并 E8a 60ep 干净消融）：「full 更差」方向证据全网格稳定'
            '（6/6 协议点多数 seed，16/18 seed obs<0；E8a tw0 臂 60ep uniform_1 置换 p=0.002/0.002/0.76）；'
            '显著性依赖读出口径——LOGON AUC 聚合 6/6 点 raw<0.05（Holm m=24 后 3/6 存活），'
            'LOGON AP 聚合 0/6（0.074-0.595 全不显著）。论文主张句口径：类型条件化在 LOGON 同池对攻击边排序 '
            'detectably 不助益（AUC 口径聚合显著更差、AP 口径未检出差异）；红线：AP 口径未检出≠等效，'
            '不写「两臂相当」。'),
        'abstract_rule': '摘要数字取聚合口径：temporal mixed 聚合 p=0.0005×3（Holm 0.012@m24 口径注明）；'
                         'half1 的 2/3 计数不进摘要（R2 贡献与包装调研 §可执行清单）',
        'product': 'results/EXP-R2-E9/e9_auc_logon_agg.json',
    },

    # ---------- P0-5 ----------
    'P0-5_e3_note_dual_caliber_supersedes': {
        'supersedes': 'E3.logon_gap_shift.note 的「AUC signed diff 3/3 seed 下移、AP 3/3 seed 下移」表述'
                      '（与同对象 signed_diff_decreased=[F,F,T] 布尔矛盾，R2B-M3；原文保留）',
        'replay_result': '无源池化字段生产者补录 src/e10_e3_pooled_replay.py（R2B-M4①）：重放与在案交付值逐位一致'
                         '（inband [F,T,T]/[T,T,T]/[T,T,T]、带、signed 方向、homo AP 全对上；均值差 ~3e-5 属'
                         'convention_note 声明的「逐日日均 vs 池化差」双泛函关系，非数值矛盾）——'
                         'E3 in-band 判定不翻转（风险预案未触发）',
        'corrected_note': E10['dual_caliber_note_r2bm3'],
        'product': 'results/EXP-R2-E10/e10_e3_pooled_replay.json',
    },

    # ---------- P0-7 ----------
    'P0-7_w14_correction_supersedes': {
        'supersedes': 'E4.repairs 所引 e7_e4_repair.json F3 unified_explanation 中「homo/homoflows/noflows 臂'
                      '评估池无 FLOW 正例（class1 缺席）」与「差距主导项为 class2」两句（原文保留）',
        'corrected_text_pointer': 'results/EXP-R1-E7/e7_e4_repair.json:F3_w14_diagnosis.unified_explanation_correction_r2'
                                  '（更正版全文）+ results/EXP-R2-REG/e7_f3_correction.json（三处传播登记）',
        'facts': 'homo/homoflows classes_in_eval_true=[0,1,2]、class1 support=313,994 在场；仅 noflows [0,2]；'
                 '主导项=class0（full 0.7037 vs nemb 0.9317，精确率 0.557 vs 0.976），class2 为同源伴随',
        'propagation_sync': 'e7 json（追加 correction 键）✓ / e4 r2_repairs（追加 w14_diagnosis_correction_r2）✓ /'
                            ' RESEARCH_LOG §R1-E7 F3（文本改写+supersedes 标注）✓',
    },

    # ---------- P1-8 相关（R1_metrics 内的更正） ----------
    'P1-8_selfstarting_note_fix': {
        'supersedes': 'selfstarting_appendix.arl500 note（误抄 arl100 文本「fixed@arl100 覆盖 4%…」，R1A2-m4/R2B-m4）',
        'corrected_note_arl500': 'arl500 档正确对照：该档验证流功效不足（347 良性小时，E≈0.7≪20），'
                                 'fixed/CUSUM/EWMA 实测均饱和 ~112——申报「该档本轮不可认证」，无有效对照句',
    },
    'P1-8_run_ledger_fix': {
        'supersedes': '「R1 GPU 资源实绩 | 23 runs」台账（R1A-m5：R2 实为 19+4=23 训练 run + E8 两臂 2 jobs'
                      ' 已计入；E8 后实为 25 job-slots——按 runs=23 维持、总 GPU job 25 并报）',
        'corrected': '23 训练 runs（19 R1 主网格+4 重跑）+ E8 2 jobs（=6 训练 job-slots，各 3 seed）='
                     ' 25 job-slots 全队列完成（原台账 23 未含 E8 的表述修正为 25 job-slots/23 runs 双口径）',
    },

    # ---------- P1-9 遗留申报打包（R1A2-m6/R2B-m6/m7） ----------
    'P1-9_leftover_pack': {
        'path_style_fix_r2bm7': {
            'issue': 'E1.files 用 EXP-R1-E1/… 相对式、E2/E3/E4 用 results/EXP-… 前缀（R1A-m8 残留）',
            'resolution': '历史 files 数组不改写（原文保留）；本层登记统一规范：论文引用一律以 '
                          'results/ 为前缀的仓库相对路径；本表 mappings 见 '
                          'results/EXP-R2-REG/number_manifest.json（其路径全部 results/ 前缀）'},
        'ep30_vs_ep60_r1am3': {
            'issue': 'E6 全部推断基于 30ep 网格，主线/60ep 校验链仅 uniform_1 双臂（R1A-m3 遗留）',
            'spot_check': 'R2-E12：temporal_1 双臂 60ep 抽查（results/EXP-R2-E12/，GPU 分钟级）——'
                          '量化 30ep 网格 vs 60ep 主线在该头条协议点的口径差；uniform_1 60ep 已在案'
                          '（E2 校验链）不重跑',
            'status_field': '已完成——results/EXP-R2-E12/e12_closure.json：mixed 可分 30ep 3/3→60ep 3/3'
                            '（p=0.008/0.002/0.006，差值 [0.260,0.276,0.270] vs [0.327,0.342,0.320]）；'
                            'LOGON AUC 2/3、AP 2/3 full 更差（与 E9 方向一致）；头条对 epoch 口径稳健'},
        'not_done_this_round_r2': [
            'P2-10 对称化校准对照（CUSUM/EWMA 侧 t-枢轴对称重校准作为完整对照臂）——本轮以 E11 单向敏感性'
            '替代（R2B-M2 resolution test 的最小充分形态）；完整对称臂列写作后可选',
            'V-R2-6 criteria sha256 哈希链+verify 脚本——制度升级项，本轮以「mtime 为准+追加式落盘」'
            '最低成本规范替代（P0-2 resolution），哈希链列下轮可选',
            'G2 arl500 延长良性流复检（E5 申报「该档本轮不可认证」的复核）——数据侧无更多良性小时可用，维持申报',
            'E7 F3 逐类 P/R/F1 从 E47B results.json 的独立重算——R2B 已做自洽性验证（macro 差=逐类差均值、'
            'support 与池计数一致），独立重算非阻塞，维持申报',
            '本席未重跑任何 GPU 训练主线（E12 抽查除外）——R2B/R1A2 复审交叉验证结论沿用并如实标注',
        ],
    },
}
M['_meta']['round'] = 'R1 第二轮修复 + R2 第三轮 P0 修复层（finalize_r3.py 只增不改；原字段保留）'
M['_meta']['r2p0_note'] = 'P0 修复全部在顶层 r2p0_fixes 键；被 supersedes 原文一律保留在原位置'

with open(f'{R}/R1_metrics.json', 'w') as f:
    json.dump(M, f, indent=1, ensure_ascii=False)
print('R1_metrics r2p0_fixes written; top keys:', list(M.keys()))
print('P0-1 full interval arl500:', M['r2p0_fixes']['P0-1_h1_final_verdict_r2_CONDITIONAL_supersedes']
      ['fixed_det_rate_full_interval']['arl500_interval'])
