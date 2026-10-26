"""r2_reg_layer.py — R2-REG（P0-2/P0-6/P0-7 + V-R2-4）：登记层产物一次生成。

产出（全部追加式登记，不覆盖任何历史产物；新文件以落盘 mtime 为唯一留痕时刻）：
 ① results/EXP-R2-REG/criteria_mtime_ledger.json — P0-2/V4：全量 criteria mtime vs 自报时刻双列对照
 ② results/EXP-R2-REG/e4_window_row_fair_replay.json — P0-6/R2B-M4②：e4 窗级公平格补录生产者
   （从 e5 G3 rows 重放，与在案值逐位比对；R2B-m5 补判定规则声明=检出率，附延迟规则对照）
 ③ results/EXP-R2-REG/e7_f3_correction.json — P0-7/R1A2-M2：E7-F3 审计矛盾更正层
   （同时向 e7_e4_repair.json 与 e4_granularity.json 追加 correction 键——只增不改）
 ④ results/EXP-R2-REG/spec_curve_h1.json + spec_curve_h1.png — V-R2-4：H1 规格曲线
   （{τ_t, z3.0, τ_EVT(已否决), τ_经验可达} 四规格 × 三读出 × 2 ARL 档的检出率/延迟排布）
 ⑤ results/EXP-R2-REG/number_manifest.json — V-R2-3/V15：SSOT 叙述数字→JSON 字段路径映射
"""
import json, os, subprocess, sys
import numpy as np

sys.path.insert(0, '/root/autodl-tmp/projB_hetero_graph/src')
BASE = '/root/autodl-tmp/projB_hetero_graph'
OUT = f'{BASE}/results/EXP-R2-REG'

# ---------- ① mtime 双列 ledger（V4 采集命令冻结：stat -c '%y|%n'） ----------
FILES = [
    ('results/EXP-R1-E1/criteria.json', 'registered_at=2026-10-24T17:00:00+08:00', 'registered_at 晚于 mtime 100 分钟 → 回填值'),
    ('results/EXP-R1-E2/criteria.json', '无自报时刻字段（第一轮原文）', '一致（mtime 即落盘时刻）'),
    ('results/EXP-R1-E3/criteria.json', '无 0.02/补录字样（R1A2-M1②：LOG 声称「±0.02 已在 criteria 补注册」失实）', 'RESEARCH_LOG 声明与文件不符'),
    ('results/EXP-R1-E4/criteria.json', '无自报时刻字段', '一致'),
    ('results/EXP-R1-E5/criteria.json', 'frozen_at=14:40:00；amendments 4/5/6 自报 16:25/16:40/16:55', 'amendments 4-6 全部晚于 mtime 16:20:33 → 回填值；修订 4「主件真实数据运行前」与 logs/e5_main.log 结束时刻 16:16:53 矛盾；修订 6 自述「G1 首轮运行后」属实但时刻不实'),
    ('results/EXP-R1-E6/e6_inference.json', '（补登记前无 criteria.json——R1A2-M1①/R2B-C1①）', '缺口已于 2026-10-24 P0-2 补登记 post_hoc'),
    ('results/EXP-R1-E7/criteria.json', 'frozen_at=15:35:00', 'frozen_at 晚于 mtime 15:01:16 与产物 15:07:20 → 回填值'),
    ('results/EXP-R1-E8/criteria.json', 'frozen_at=14:33（先于训练运行 14:46/14:47）', '一致（R2B 确认本轮唯一全链时间自洽的新实验判据）'),
    ('results/EXP-R1-VERIFY/criteria.json', 'registered_at=2026-10-24T17:00:00+08:00', 'registered_at 晚于 mtime → 回填值'),
    ('src/e5_evt_calib.py', '—', 'mtime 16:32:02 晚于产物 e5_evt_calib.json 16:20:33（R2B-M4③时序倒挂；复算证明当前源码可精确再生 G1/校准/G3，编辑大概率无害，倒挂如实登记）'),
    ('src/r1_train_arm.py', '—', 'mtime 15:22:16 晚于 E8 训练运行 14:46-14:47（申报为 docstring 更正+死变量删除，改动范围无法独立验证，如实登记）'),
]


def build_mtime_ledger():
    rows = []
    for rel, self_reported, verdict in FILES:
        p = f'{BASE}/{rel}'
        mtime = subprocess.run(['stat', '-c', '%y', p], capture_output=True, text=True).stdout.strip()
        rows.append({'file': rel, 'mtime_filesystem': mtime,
                     'self_reported_time': self_reported, 'verdict': verdict})
    ledger = {
        'exp': 'R2-REG：criteria mtime 双列对照表（P0-2/R2B-C1②/V4）',
        'date': '2026-10-24',
        'collection_command_frozen': "stat -c '%y' <file>（本表生成前冻结，精度到秒+纳秒，时区 +0800）",
        'audit_rule': ('自报时刻字段（frozen_at/registered_at/amendments.at）一律视为「非文件系统留痕的回填值」，'
                       '不作为事前性证据；审计以 mtime 为准（无版本库条件下的最低成本可验证留痕）。'
                       '今后判据修订以独立追加文件/追加键落盘，恢复可验证事前性（R2B-C1 resolution test ④）。'),
        'rows': rows,
        'honest_statement_supersede': ('RESEARCH_LOG §6 头部原文「全部新实验判据先落盘 results/EXP-R1-E{5,6,7,8}/criteria.json'
                                       '（含 E5 的 5 次运行前修订记录——每次修订均由合成数据单元测试驱动、先于真实数据运行，'
                                       '原文保留可审计）」整体作废（supersedes）：E6 当时无 criteria.json；E5 修订 4-6 '
                                       '自报时刻为回填值且修订 4/6 的「先于真实数据运行/运行前」声明与文件系统证据矛盾'
                                       '（e5_main.log 16:16:53 结束 vs 修订 4 自称 16:25 运行前）。'
                                       '如实表述改为：E5/E7/E8 判据文本内容与运行内容一致（内容层可信，R2B-C1「内容层面'
                                       '多数仍可信」），但时间链不可审计；E1/VERIFY/E6 三份为运行后补登记（post_hoc 标注）。'
                                       '本表述同步写入 R1_metrics 修复层与 RESEARCH_LOG §6。'),
    }
    with open(f'{OUT}/criteria_mtime_ledger.json', 'w') as f:
        json.dump(ledger, f, indent=1, ensure_ascii=False)
    print('① mtime ledger done:', len(rows), 'rows')


# ---------- ② e4 窗级公平格补录生产者（P0-6/R2B-M4②/R2B-m5） ----------
def build_e4_fair_row_replay():
    e5 = json.load(open(f'{BASE}/results/EXP-R1-E5/e5_evt_calib.json'))
    e4 = json.load(open(f'{BASE}/results/EXP-R1-E4/e4_granularity.json'))
    incase = e4['r2_repairs']['granularity_matrix_window_row_fair']['fair_window_row']
    replay = {}
    for arl in ('arl100', 'arl500'):
        rows = e5['g3_h1_reverdict'][arl]['rows']
        pas = {f"{r['arm']}_s{r['seed']}": {'det_rate': r['fixed_evt']['det_rate'],
                                            'delay': r['fixed_evt']['delay']} for r in rows}
        # 方向规则（R2B-m5 补声明；R2 第四轮 R2A-M2/R3B-M1 方向修正，判据 results/EXP-R2-E13/criteria.json）：
        # 主规则=检出率（det 高者 better，tie 同值）；对照=延迟（先告警者 better=延迟小者 better；
        # None=永不告警为最差 → 映射 +1e9。原版误复用 det 的 va>vb 比较符且 None→−1e9，把更晚告警判为更好）
        def dirs(rule):
            out_d = {}
            for a, b in [('full', 'homoflows'), ('full', 'homo'), ('homoflows', 'homo')]:
                dd = []
                for s in range(3):
                    ra, rb = pas[f'{a}_s{s}'], pas[f'{b}_s{s}']
                    if rule == 'det_rate':
                        va, vb = ra['det_rate'], rb['det_rate']
                        dd.append('tie' if va == vb else ('first_better' if va > vb else 'second_better'))
                    else:  # delay：延迟小者 better；None=永不告警=最差
                        va = ra['delay'] if ra['delay'] is not None else 10 ** 9
                        vb = rb['delay'] if rb['delay'] is not None else 10 ** 9
                        dd.append('tie' if va == vb else ('first_better' if va < vb else 'second_better'))
                out_d[f'{a}_vs_{b}'] = dd
            return out_d
        replay[arl] = {'tau_t': e5['g3_h1_reverdict'][arl]['rows'][0]['fixed_evt']['tau'],
                       'per_arm_seed': pas,
                       'pair_directions_det_rate_rule_PRIMARY': dirs('det_rate'),
                       'pair_directions_delay_rule_sensitivity': dirs('delay')}
    match = all(replay[arl]['per_arm_seed'] == incase[arl]['per_arm_seed'] for arl in replay)
    det_match = all(replay[arl]['pair_directions_det_rate_rule_PRIMARY'] == incase[arl]['pair_directions']
                    for arl in replay)
    # 真实分歧（声明规则下）：修正版 delay 行 vs 在案 det 主规则行，逐对逐格枚举
    delay_diff = {arl: {k: replay[arl]['pair_directions_delay_rule_sensitivity'][k]
                        for k in replay[arl]['pair_directions_delay_rule_sensitivity']
                        if replay[arl]['pair_directions_delay_rule_sensitivity'][k] != incase[arl]['pair_directions'][k]}
                  for arl in replay}
    n_diff_pairs = sum(1 for arl in delay_diff for k in delay_diff[arl])
    n_diff_cells = sum(1 for arl in delay_diff for k in delay_diff[arl]
                       for i, v in enumerate(replay[arl]['pair_directions_delay_rule_sensitivity'][k])
                       if v != incase[arl]['pair_directions'][k][i])
    out = {'exp': 'R2-REG：e4 窗级公平格补录生产者（P0-6/R2B-M4②；R2 第四轮 E13 方向修正版）', 'date': '2026-10-24',
           'producer': 'src/r2_reg_layer.py::build_e4_fair_row_replay（修正版由 src/e13_delay_rule_fix.py 驱动重生成；'
                       'R2B-M4②：原键 granularity_matrix_window_row_fair 无在案生成代码，本脚本为补录生产者并注明 '
                       'post_hoc；在案 e4 对象不改写）',
           'replay': replay,
           'bitwise_match_per_arm_seed': match,
           'pair_directions_match_under_det_rate_rule': det_match,
           'direction_rule_declaration_r2bm5': ('主判定规则=检出率比较（det 高者 better）——原键未声明规则，此处补声明；'
                                                '延迟规则（先告警者 better=延迟小者 better，None=永不告警为最差）为敏感性对照；'
                                                'delay 分支方向修正见 superseded 键与 R2A-M2/R3B-M1'),
           'delay_rule_divergence': delay_diff,
           'delay_rule_divergence_count': {'pairs': n_diff_pairs, 'cells': n_diff_cells,
                                           'note': '声明规则（先告警者 better）下 delay 行与 det 主规则行的真实分歧：'
                                                   '完整双档口径。R2A-M2「4 对 8 格」即此口径；R3B-M1「2 对 3 格」'
                                                   '为其 arl100 子集（full_vs_homoflows s0/s1 + homoflows_vs_homo s0）'},
           'delay_rule_note': ('两规则真实分歧=4 对 ' + str(n_diff_cells) + ' 格（arl100：full_vs_homoflows s0/s1、'
                               'homoflows_vs_homo s0；arl500：full_vs_homoflows s0/s1/s2、full_vs_homo s0/s2）——'
                               '原文「仅 arl100 full_vs_homoflows 两格分歧」与原复审转述「仅 arl100 full_vs_homoflows '
                               '前两 seed 不同」均失实（supersedes，原文保留于 superseded 键）；论文引用公平格时须带规则声明。')}
    # 只增不改：原（方向错误）版本的方向行/枚举/括注全部保留
    prev = f'{OUT}/e4_window_row_fair_replay.json'
    if os.path.exists(prev):
        old = json.load(open(prev))
        out['superseded_r2am2_r3bm1'] = {
            'status': 'SUPERSEDES：原 pair_directions_delay_rule_sensitivity 由方向写反的实现生成'
                      '（r2_reg_layer.py 旧版对 delay 复用 det 的 va>vb 比较符且 None→−1e9，把更晚告警判为更好）——'
                      'arl100 三对 5 格 first/second 互换（full_vs_homoflows s1/s2、full_vs_homo s1/s2、'
                      'homoflows_vs_homo s1），arl500 全 tie/None 格侥幸不变；本键原文保留。',
            'old_pair_directions_delay_rule_sensitivity': {arl: old['replay'][arl]['pair_directions_delay_rule_sensitivity']
                                                           for arl in old['replay']},
            'old_delay_rule_divergence': old.get('delay_rule_divergence'),
            'old_delay_rule_note': old.get('delay_rule_note'),
            'review_refs': 'R2A-M2 / R3B-M1 / R2 第四轮 E13'}
    with open(f'{OUT}/e4_window_row_fair_replay.json', 'w') as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    print('② e4 fair-row replay done: match=', match, det_match)


# ---------- ③ E7-F3 更正层（P0-7/R1A2-M2/R2B-m2） ----------
F3_CORRECTED = (
    'full 与 nemb 同为三类别可比口径，macro-F1 差 0.1136（nemb 更高，反常）。逐类 F1 分解（seed0）'
    '{class0: +0.228, class1: −0.0066, class2: +0.1193}：差距主导项为 class0（full 0.7037 vs nemb 0.9317，'
    '驱动=class0 精确率 0.557 vs 0.976——full 的类型头把 445,229 个未观测随机对中 22.4% 误判为 LOGON，'
    '压低的是 class0 精确率；class2 recall 0.776 为同源伴随信号，非主导项）。'
    '可比性审计（R1A2-M2 更正版）：仅 noflows 臂评估池无 FLOW 正例（classes_in_eval_true=[0,2]、class1 support=0），'
    '其 macro-F1 在 {0,2} 上平均、与其余臂不可比；homo/homoflows 臂三类齐全（class1 support=313,994 在场、'
    'classes_in_eval_true=[0,1,2]），其 macro 低（≈0.57）是 class0 F1=0（从不预测 LOGON）的真实表现，'
    '属可比差异而非口径不可比——「homo/homoflows 无 FLOW 正例」的原表述错误，予以更正。'
    '跨臂并列申报规则不变：noflows 的 macro 不得与三类别臂直接并列；full vs nemb 逐类分解为 W14 主证据。'
    '该机理与主线证据同向：E2 评估侧候选池扩到非观测对时 AUC 全塌（0.31→0.04）、E2 污染免疫发现'
    '（homo 跨类型分数崩坏）——类型条件化的关系记忆在分布外无法校准其置信度。'
    '本段为 R2 第三轮更正版（P0-7/R1A2-M2/R2B-m2）；被更正原文保留于 F3_w14_diagnosis.unified_explanation 不删改。')


def build_e7_f3_correction():
    e7p = f'{BASE}/results/EXP-R1-E7/e7_e4_repair.json'
    e4p = f'{BASE}/results/EXP-R1-E4/e4_granularity.json'
    e7 = json.load(open(e7p))
    e4 = json.load(open(e4p))
    # 只增不改：追加 correction 键
    e7['F3_w14_diagnosis']['unified_explanation_correction_r2'] = {
        'status': 'SUPERSEDES 上方 unified_explanation 中两处错误表述（原文保留不删改）',
        'errors_corrected': [
            '原「homo/homoflows/noflows 臂评估池无 FLOW 正例（class1 缺席）」——与同文件 comparability_audit 矛盾：'
            'homo/homoflows classes_in_eval_true=[0,1,2]、class1 support=313,994；仅 noflows 为 [0,2]/support=0',
            '原「差距主导项为 class2」——逐类差最大项为 class0（+0.228 > class2 +0.1193），'
            '机理（class2 行被误判为 class0）与归因主项（class0 F1）混写'],
        'corrected_text': F3_CORRECTED,
        'review_refs': 'R1A2-M2 / R2B-m2 / 本轮 P0-7'}
    with open(e7p, 'w') as f:
        json.dump(e7, f, indent=1, ensure_ascii=False)
    # e4 r2_repairs.w14_diagnosis 同步（追加键，不改写原对象）
    if 'w14_diagnosis_correction_r2' not in e4['r2_repairs']:
        e4['r2_repairs']['w14_diagnosis_correction_r2'] = {
            'status': 'SUPERSEDES 同段 w14_diagnosis.unified_explanation（若含原文案；原文保留）',
            'corrected_text': F3_CORRECTED,
            'review_refs': 'R1A2-M2 / R2B-m2 / 本轮 P0-7；三处传播文本（e7 json / RESEARCH_LOG:156 / e4 r2_repairs）同步'}
        with open(e4p, 'w') as f:
            json.dump(e4, f, indent=1, ensure_ascii=False)
    reg = {'exp': 'R2-REG：E7-F3 审计矛盾更正层（P0-7/R1A2-M2）', 'date': '2026-10-24',
           'propagation_sites': [
               {'file': 'results/EXP-R1-E7/e7_e4_repair.json',
                'action': '追加键 F3_w14_diagnosis.unified_explanation_correction_r2（supersedes，原文保留）'},
               {'file': 'results/EXP-R1-E4/e4_granularity.json',
                'action': '追加键 r2_repairs.w14_diagnosis_correction_r2（supersedes，原文保留）'},
               {'file': 'RESEARCH_LOG.md §R1-E7 F3 段（:156 区）',
                'action': '由 P0-7 文档修复改写并标注 supersedes（见 RESEARCH_LOG 当轮追加行）'}],
           'facts_used': {
               'classes_in_eval_true': {'homo': [0, 1, 2], 'homoflows': [0, 1, 2], 'noflows': [0, 2]},
               'class1_support': 313994,
               'per_class_f1_gap_seed0': {'class0': 0.228, 'class1': -0.0066, 'class2': 0.1193},
               'dominant_term': 'class0（full 0.7037 vs nemb 0.9317；精确率 0.557 vs 0.976）'},
           'corrected_text': F3_CORRECTED}
    with open(f'{OUT}/e7_f3_correction.json', 'w') as f:
        json.dump(reg, f, indent=1, ensure_ascii=False)
    print('③ e7 f3 correction done（e7/e4 追加键+登记层三处同步）')


# ---------- ④ 规格曲线（V-R2-4） ----------
def build_spec_curve():
    from e1_window_alert import ARMS, EVAL_DAYS, BENIGN_DAYS, window_stats
    from e5_evt_calib import t_pivot_tau
    e5 = json.load(open(f'{BASE}/results/EXP-R1-E5/e5_evt_calib.json'))
    # 重建 z 池（与 e5_evt_calib.py:217-229 同式）
    evz = {}
    for arm, d in ARMS.items():
        for seed in range(3):
            ev = window_stats(f'{d}/scores_seed{seed}.npz')
            bn = window_stats(f'{d}/benign_scores_seed{seed}.npz')
            bv = np.array([bn[dd]['top50mean'] for dd in BENIGN_DAYS])
            mu, sd = bv.mean(), bv.std(ddof=1)
            evz[(arm, seed)] = np.array([(ev[dd]['top50mean'] - mu) / sd for dd in EVAL_DAYS])
    specs = {}
    for arl0 in (100, 500):
        tau_t = t_pivot_tau(9, arl0)
        cal = e5['calibration']['top50mean'][f'arl{arl0}']
        tau_evt = cal['sensitivity']['evt_hybrid'].get('tau')          # 已否决 EVT 分支点估计
        tau_emp = cal['sensitivity']['evt_hybrid'].get('tau_empirical')  # 池内经验可达
        z3 = 3.0
        table = {}
        for spec, tau in [('S1_tpivot_t9_8', tau_t), ('S2_z3_probe', z3),
                          ('S3_empirical_reachable', tau_emp), ('S4_evt_hybrid_REJECTED_daywindow', tau_evt)]:
            if tau is None:
                continue
            dets, delays = [], []
            for (arm, seed), z_ev in sorted(evz.items()):
                al = z_ev > tau
                dets.append(round(float(al.mean()), 3))
                delays.append(int(np.argmax(al)) if al.any() else None)
            table[spec] = {'tau': round(float(tau), 4), 'det_rate_grid9': dets,
                           'delay_grid9': delays,
                           'det_range': [min(dets), max(dets)],
                           'status': ('主阈值（正态父体 t-枢轴外推）' if spec.startswith('S1') else
                                      'R1A 稳健性探针' if spec.startswith('S2') else
                                      '池内经验可达阈值（配置律不可行域内，判据反向档）' if spec.startswith('S3') else
                                      '已否决分支（ξ̂=−3.92 自家诊断拒绝；仅列作反证）')}
        table['_noninferior_counts_by_spec'] = {}
        for spec, v in table.items():
            if spec.startswith('_'):
                continue
            tau = v['tau']
            cnt = 0
            g3rows = e5['g3_h1_reverdict'][f'arl{arl0}']['rows']
            for r in g3rows:
                z_ev = evz[(r['arm'], r['seed'])]
                al = z_ev > tau
                fd = int(np.argmax(al)) if al.any() else None
                for seq_d in (r['cusum_incase']['delay'], r['ewma02_incase']['delay']):
                    if seq_d is not None and (fd is None or seq_d <= fd):
                        cnt += 1
                        break
            table['_noninferior_counts_by_spec'][spec] = f'{cnt}/9'
        specs[f'arl{arl0}'] = table
    out = {'exp': 'V-R2-4：H1 校准规格曲线（multiverse 排布；P0-1 图形化素材）', 'date': '2026-10-24',
           'design': '四校准规格 {τ_t=3.0531/4.2068、z3.0 探针、池内经验可达 τ、已否决 EVT τ} × '
                     'fixed 读出在 9 格（3 臂×3 seed）的检出率/延迟；CUSUM/EWMA 在案 bootstrap 校准为跨规格恒定参照行。'
                     '读法：主阈值规格（S1）与经验可达规格（S3）给出相反判定 → H1 为假设依赖的描述性结论'
                     '（P0-1 条件句的图形化；规格间离散度即校准分辨率不可辨识性的定量证据）。',
           'spec_curve': specs}
    #CUSUM/EWMA 参照行（在案，跨规格恒定）
    for arl0 in (100, 500):
        rows = e5['g3_h1_reverdict'][f'arl{arl0}']['rows']
        out['spec_curve'][f'arl{arl0}']['_seq_reference_incase_bootstrap'] = {
            'cusum': {'det_range': [min(r['cusum_incase']['det_rate'] for r in rows),
                                    max(r['cusum_incase']['det_rate'] for r in rows)],
                      'noninferior': e5['g3_h1_reverdict'][f'arl{arl0}']['noninferior_counts']['cusum'],
                      'measured_arl_range': [min(r['cusum_incase']['measured_arl'] for r in rows),
                                             max(r['cusum_incase']['measured_arl'] for r in rows)]},
            'ewma02': {'det_range': [min(r['ewma02_incase']['det_rate'] for r in rows),
                                     max(r['ewma02_incase']['det_rate'] for r in rows)],
                       'noninferior': e5['g3_h1_reverdict'][f'arl{arl0}']['noninferior_counts']['ewma02'],
                       'measured_arl_range': [min(r['ewma02_incase']['measured_arl'] for r in rows),
                                              max(r['ewma02_incase']['measured_arl'] for r in rows)]}}
    with open(f'{OUT}/spec_curve_h1.json', 'w') as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    # PNG
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
        for ax, arl0 in zip(axes, (100, 500)):
            tb = out['spec_curve'][f'arl{arl0}']
            names = [s for s in tb if not s.startswith('_')][::1]
            mus, los, his, cols = [], [], [], []
            palette = {'S1': '#1f77b4', 'S2': '#2ca02c', 'S3': '#d62728', 'S4': '#7f7f7f'}
            for i, s in enumerate(names):
                d = tb[s]['det_rate_grid9']
                mus.append(np.mean(d)); los.append(np.min(d)); his.append(np.max(d))
                cols.append(palette[s.split('_')[0]])
            x = np.arange(len(names))
            ax.vlines(x, los, his, color=cols, lw=6, alpha=0.55)
            ax.scatter(x, mus, color=cols, zorder=3, s=42)
            ref = tb['_seq_reference_incase_bootstrap']
            ax.axhline(np.mean(ref['cusum']['det_range']), color='#9467bd', ls='--', lw=1.2,
                       label=f"CUSUM@bootstrap校准 (NI {ref['cusum']['noninferior']})")
            ax.axhline(np.mean(ref['ewma02']['det_range']), color='#ff7f0e', ls=':', lw=1.2,
                       label=f"EWMA02@bootstrap校准 (NI {ref['ewma02']['noninferior']})")
            ax.set_xticks(x)
            ax.set_xticklabels([s.replace('_', '\n', 1) for s in names], fontsize=7.5)
            ax.set_ylim(0, 1.02)
            ax.set_ylabel('fixed 检出率（9 格 min–max，点=均值）')
            ax.set_title(f'ARL0={arl0}（τ_t={tb[names[0]]["tau"]}）')
            ax.legend(fontsize=7, loc='lower right')
        fig.suptitle('H1 校准规格曲线：判定对校准规格敏感（9 良性日不可辨识性的图形证据）', fontsize=11)
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        fig.savefig(f'{OUT}/spec_curve_h1.png', dpi=160)
        print('④ spec curve done（json+png）')
    except Exception as e:
        print('④ png failed:', e)


# ---------- ⑤ number manifest（V-R2-3/V15） ----------
def build_number_manifest():
    mf = {
        'exp': 'V-R2-3：number manifest（SSOT 叙述数字→JSON 字段路径映射；防转写滑笔 lint 底账）',
        'date': '2026-10-24',
        'usage': '论文正文每个关键数字须能映射到本表某行的 JSON 路径；写作路收口时跑 grep 式交叉核对'
                 '（V15）。历史教训：R1A2-m3（0.008057→「0.0071」）、R1A2-m1（全区间 0.0-0.182 取最大值当代表值）。',
        'mappings': [
            {'n': 'temporal 聚合 p=0.0005×3 / Holm 0.003',
             'path': 'results/EXP-R1-E6/e6_inference.json:E2.families.family_half2_mixed.agg_p_raw[0:3] / holm_aggregate[0:3]',
             'note': 'm=24 统一口径 Holm=0.012 见 results/EXP-R2-E9/e9_auc_logon_agg.json:families.family_half2_mixed.holm_aggregate_m24；摘要引用两口径其一并注明'},
            {'n': 'half1 LOGON AP 聚合不显著 agg_p 0.074-0.595',
             'path': 'results/EXP-R1-E6/e6_inference.json:E2.families.family_half1_logonAP.agg_p_raw',
             'note': 'P0-4 弱化事实；双口径写回 RESEARCH_LOG E2 段'},
            {'n': 'auc_logon 聚合（E9 新证据）',
             'path': 'results/EXP-R2-E9/e9_auc_logon_agg.json:per_point[*].seed_aggregate + cross_aggregate_calibers',
             'note': '本轮 P0-4 新增，引用以该文件为准'},
            {'n': 'fixed@τ_t arl500 检出全区间 0.0-0.182（含 homo s1=0.0/delay=None）',
             'path': 'results/EXP-R1-E5/e5_evt_calib.json:g3_h1_reverdict.arl500.rows[*].fixed_evt.det_rate',
             'note': 'R1A2-m1：禁止引单格 0.182 当代表值'},
            {'n': 'τ_t=3.0531/4.2068',
             'path': 'results/EXP-R1-E5/e5_evt_calib.json:calibration.top50mean.arl*.main_tpivot.tau + src/e5_evt_calib.py:160-164',
             'note': 'scipy 逐位复算一致（R1A2/R2B 双复审+本轮环境自测三重）'},
            {'n': 'G1 直接模拟 rate×ARL0',
             'path': 'results/EXP-R2-REG/e5_direct_check.json:unconditional_iid_200k.arl*.rate_x_arl0（论文引用口径）+ incase_value（存档 0.952/1.025）+ reviewer_values',
             'note': 'P0-2/V6：三实现并报，论文用无条件结构值'},
            {'n': 'G1 原门 fail 值 pass_frac=0.08/0.02、mean_rate×ARL0=0.8245/0.6722',
             'path': 'logs/e5_main.log 末尾 G1 行 == results/EXP-R1-E5/e5_evt_calib.json:g1_synthetic.arl*（双源逐位一致，本轮 V7 核对）',
             'note': 'P0-3 三句式第二句证据'},
            {'n': 'ξ̂=−3.92、QQ 相关 −0.61',
             'path': 'results/EXP-R1-E5/e5_evt_calib.json:calibration.top50mean.gpd_xi_main / qq_corr_main',
             'note': 'P0-3 三句式第三句证据'},
            {'n': 'E3 signed_diff_decreased=[F,F,T]/[F,F,T]、inband=[F,T,T]/[T,T,T]/[T,T,T]',
             'path': 'results/EXP-R2-E10/e10_e3_pooled_replay.json:metrics.*（重放逐位一致）+ results/EXP-R1-E6/e6_inference.json:E3',
             'note': 'P0-5：注记改「2/3 seed 缩小、s2 越零反向、均值缩小」双口径'},
            {'n': 'E8a tw0 置换显著 2/3、E8b 3/3 可分 p=0.004/0.002/0.004',
             'path': 'results/EXP-R1-E8/e8_closure.json',
             'note': 'LOGON 口径合并判语见 P0-4 写回文本'},
            {'n': 'E7-F1 v2 full [0.0061,0.0065,0.0081] / homo_rerun [0.0119,0.0208,0.0081]',
             'path': 'results/EXP-R1-E7/e7_e4_repair.json:F1_ap_event_weighted_v2',
             'note': 'R1A2-m3/R2B-m3 滑笔更正：第 3 seed=0.0081（非 0.0071/~0.007）'},
            {'n': 'class1 support=313,994、逐类差 class0 +0.228 主导',
             'path': 'results/EXP-R1-E7/e7_e4_repair.json:F3_w14_diagnosis.per_class_prf + unified_explanation_correction_r2',
             'note': 'P0-7 更正版文本以 correction 键为准'},
            {'n': 'fixed 实测 ARL 8.7-9.4（54 点）',
             'path': 'results/EXP-R1-E1/e1_window_alert.json + results/R1_metrics.json:E1.c1_fair_recalibration.root_cause',
             'note': '双审复算一致'},
            {'n': '族级误报预算 0.1047/0.0218',
             'path': 'results/EXP-R1-E5/e5_evt_calib.json:g3_h1_reverdict.arl*.familywise_falsealarm_11win',
             'note': 'P0-1：仅名义口径申报，对 deployed fixed（真实误报率 0）不成立'},
            {'n': 'E9/E11 敏感性：非劣计数收紧后 9/9、7/9 保持',
             'path': 'results/EXP-R2-E11/e11_g3_sensitivity.json:sensitivity.*.noninferior_kappa_tightened',
             'note': 'P0-1 单向敏感性（本轮新增）'},
        ],
    }
    with open(f'{OUT}/number_manifest.json', 'w') as f:
        json.dump(mf, f, indent=1, ensure_ascii=False)
    print('⑤ number manifest done:', len(mf['mappings']), 'mappings')


if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True)
    build_mtime_ledger()
    build_e4_fair_row_replay()
    build_e7_f3_correction()
    build_spec_curve()
    build_number_manifest()
    print('R2-REG ALL DONE')
