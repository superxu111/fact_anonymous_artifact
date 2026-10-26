"""r4_p0_registry_fixes.py — R4 收官修复（P0-1+P1-2+P1-3 JSON/登记层，主会话直接修·用户裁定 2026-10-25）。

判据：results/EXP-R4-REG/criteria_r4.json（先冻后修）。全部追加键 supersedes 风格只增不改；幂等（重跑零插入）。
P0-1  E15 证书 family_multiplicity_stance_r3 追加 corrected_r4 子键：全 16 格 Bonferroni×16 枚举，
      scipy binom 现算与在案 p_one_sided 逐位断言一致后方落盘（R6B-M1：原枚举仅 3 格、漏 arl100/S4 nullB 翻转格）；
      number_manifest 行 27 判语以追加 errata 行处置（原行不改）。
P1-2  ①FILE_EXEMPT 三件豁免理由入 not_done_this_round（登记分支）；②e13 晚编辑 diff 摘要键；
      ③闭环面=申报面∪not_done 面不变量登记为 V22 claim-lint 前提。
P1-3  criteria_mtime_ledger 补 criteria_r3 行（真实 stat 双时刻）；gate3 M7 located_at 补交付时点链长注记；
      errata 前缀自本轮起 R3fix-E*（R3B-E* 与审稿人 B 席历史前缀撞名）。
"""
import json, os, datetime, subprocess
from scipy.stats import binom

BASE = '/root/autodl-tmp/projB_hetero_graph'
TODAY = datetime.datetime.now().astimezone().isoformat(timespec='seconds')
PRODUCER = 'src/r4_p0_registry_fixes.py（R4/P0-1+P1-2+P1-3）'


def load(p):
    return json.load(open(f'{BASE}/{p}'))


def save(p, d):
    json.dump(d, open(f'{BASE}/{p}', 'w'), indent=1, ensure_ascii=False)


# ---------------- P0-1①：16 格全枚举 + corrected_r4 子键 ----------------
CERT_P = 'results/EXP-R2-E15/calibration_certificate.json'
cert = load(CERT_P)
KS = {'S1_tpivot_t9_8': 0, 'S2_z3_probe': 0, 'S3_empirical_reachable': 1, 'S4_evt_hybrid_REJECTED_daywindow': 4}
enum, n_raw, flips, mismatch = {}, 0, [], []
for arl in (100, 500):
    for spec, k in KS.items():
        pa = float(binom.sf(k - 1, 9, 1.0 / arl))
        q_win = 1.0 - (1.0 - 1.0 / arl) ** 9
        pb = float(binom.sf(k - 1, 9, q_win))
        inc = cert['certificate'][f'arl{arl}'][spec]['exact_binomial']
        for tag, p_new, p_old in (('nullA', pa, inc['p_one_sided']), ('nullB', pb, inc['p_one_sided_nullB'])):
            cell = f'arl{arl}_{spec.split("_")[0]}_{tag}'
            adj = p_new * 16
            v = '拒绝' if adj < 0.05 else '不拒绝'
            if not (abs(p_new - p_old) <= 1e-12 * max(1.0, abs(p_old))):
                mismatch.append(cell)
            enum[cell] = {'p_raw': p_new, 'p_adjusted': adj, 'verdict_5pct_adjusted': v}
            if p_old < 0.05:
                n_raw += 1
                if adj >= 0.05:
                    flips.append(cell)
assert not mismatch, f'p 现算与在案不一致: {mismatch}（绝不带病落盘）'
n_adj = sum(1 for c in enum.values() if c['verdict_5pct_adjusted'] == '拒绝')
assert (n_raw, n_adj, len(flips)) == (5, 3, 2), f'枚举计数不符预期: {n_raw}/{n_adj}/{flips}'
assert sorted(flips) == ['arl100_S4_nullB', 'arl500_S3_nullA'], flips

fm = cert['family_multiplicity_stance_r3']
if 'corrected_r4' not in fm:
    fm['corrected_r4'] = {
        'status': 'APPENDED（R4/P0-1，R6B-M1：全 16 格枚举补全——原 bonferroni_x16 仅 3 格且 stance 漏 arl100/S4 nullB 翻转格）',
        'supersedes': '本键 stance 与 bonferroni_x16 的「唯一翻转格/拒绝数 3→2」表述（原文保留不改）',
        'enumeration_16': enum,
        'counts': {'n_reject_raw': n_raw, 'n_reject_bonferroni_x16': n_adj, 'flips': sorted(flips)},
        'corrected_stance': ('raw 拒绝 5 格→族级校正（Bonferroni×16）后 3 格、翻转 2 格'
                             '（arl500/S3 nullA→0.2857、arl100/S4 nullB→0.0790）；'
                             'S4 在 null_A 侧任何校正下强拒绝（1e-6/1e-9 量级）；'
                             'null_B arl100 档族级校正后不拒绝（边际 0.079）'),
        'citation_rule_writing': ('写作期引用规则：单口径 16 格 verdict 可引（随附描述性单口径声明）；'
                                  '族级立场句以本 corrected_r4 为准；勿摘句拆散（R6B-M1 Concern 防复发）'),
        'recompute': 'scipy binom.sf 现算 16/16 与在案 p_one_sided / p_one_sided_nullB 逐位一致（rel≤1e-12）断言后方落盘',
        'producer': PRODUCER, 'review_refs': 'R6B-M1 / criteria_r4 P0-1'}
    save(CERT_P, cert)
    print('1. P0-1① corrected_r4 子键落盘（16 格枚举，raw 5→adj 3，翻转 2）')
else:
    print('1. P0-1① 已存在，跳过（幂等）')

# ---------------- P0-1②：manifest 行 27 errata 行（追加，原行不改） ----------------
MAN_P = 'results/EXP-R2-REG/number_manifest.json'
man = load(MAN_P)
if not any(r.get('errata_for_row') == 27 for r in man['mappings']):
    man['mappings'].append({
        'n': 'errata R3fix-E1（R4/P0-1）：行27 claims 判语「arl500/S3=唯一翻转格」不完整——全 16 格枚举后'
             '翻转 2 格（另加 arl100/S4 nullB ×16=0.0790）；raw 拒绝 5 格→族级校正后 3 格。原行 27 原文保留',
        'errata_for_row': 27, 'errata_id': 'R3fix-E1',
        'producer': PRODUCER,
        'checks': [
            {'path': 'EXP-R2-E15/calibration_certificate.json:family_multiplicity_stance_r3.corrected_r4.counts.n_reject_raw', 'expect': 5},
            {'path': 'EXP-R2-E15/calibration_certificate.json:family_multiplicity_stance_r3.corrected_r4.counts.n_reject_bonferroni_x16', 'expect': 3},
            {'path': 'EXP-R2-E15/calibration_certificate.json:family_multiplicity_stance_r3.corrected_r4.counts.flips', 'expect': ['arl100_S4_nullB', 'arl500_S3_nullA']},
            {'path': 'EXP-R2-E15/calibration_certificate.json:family_multiplicity_stance_r3.corrected_r4.enumeration_16.arl100_S4_nullB.p_adjusted', 'expect': 0.07898370611380562}]})
    save(MAN_P, man)
    print('2. P0-1② manifest 行30 errata 行追加（行27 原文保留）')
else:
    print('2. P0-1② 已存在，跳过（幂等）')

# ---------------- P1-2①：FILE_EXEMPT 三件豁免理由入 not_done ----------------
rm = load('results/R1_metrics.json')
if not any('R4/P1-2①' in str(it.get('item', '')) for it in rm['not_done_this_round']):
    rm['not_done_this_round'].append({
        'item': 'R4/P1-2① FILE_EXEMPT 三件（keyscan_exemption_registry_r3/warn_stratification_r3/'
                'keyscan_two_phase_r3，verify_provenance.py:29-31）生产者化退出豁免 vs 豁免理由登记——取登记分支',
        'disposition': '豁免理由：三件均为闸门自身消费的登记/治理文档（keyscan_exemption_registry 与 '
                       'warn_stratification 被 verify_provenance.py:104/291 机器读取=登记语义自指；'
                       'keyscan_two_phase 为两阶段留档且其讨论串已列 src_scan_exclude）——'
                       '按设计无生产者，与 criteria* 同类；machine-consumed 事实+ FILE_EXEMPT 集合现状'
                       '在案可查，R4 不改闸门代码（改动本身即新修复面）',
        'producer': PRODUCER, 'review_refs': 'R5B 遗留/R3A2-m1/R6B-m3 / criteria_r4 P1-2'})
    save('results/R1_metrics.json', rm)
    print('3. P1-2① not_done_this_round 追加 FILE_EXEMPT 豁免理由登记')
else:
    print('3. P1-2① 已存在，跳过（幂等）')

# ---------------- P1-2②：e13 晚编辑 diff 摘要键 ----------------
E13_P = 'results/EXP-R3-REG/e13_late_edit_replay_r3.json'
e13 = load(E13_P)
if 'late_edit_change_summary_r4' not in e13:
    nsd, ssd = e13.get('non_superseded_diffs', []), e13.get('superseded_stash_diffs', [])
    e13['late_edit_change_summary_r4'] = {
        'status': 'APPENDED（R4/P1-2②：16.6 秒事后编辑「改动内容」补表征——R5B 遗留升位项）',
        'n_diffs_total': len(nsd) + len(ssd),
        'in_superseded_stash': len(ssd), 'delivery_field_diffs': len(nsd),
        'classification': ('全部差异位于 superseded_stash_diffs 留档键（语义=输入文件状态快照：R2 首跑输入 '
                           'buggy 版留 buggy 行、重放输入修正版留修正行）；交付字段'
                           '（replay/per_arm_seed/pair_directions/divergence 计数/不变量断言）零差异——'
                           '与 replay_verdict=NUMERIC_FIELDS_IDENTICAL 一致'),
        'producer': PRODUCER, 'review_refs': 'R5B 遗留二连悬空项 / criteria_r4 P1-2②'}
    save(E13_P, e13)
    print(f'4. P1-2② e13 diff 摘要键落盘（总差异 {len(nsd)+len(ssd)}，交付字段差异 {len(nsd)}）')
else:
    print('4. P1-2② 已存在，跳过（幂等）')

# ---------------- P1-2③：闭环面不变量登记（V22 前提） ----------------
if 'claim_lint_invariant_r4' not in man:
    man['claim_lint_invariant_r4'] = {
        'invariant': '闭环面 = 申报面 ∪ not_done 面——凡申报的处置/修复/遗留必须落在'
                     '（number_manifest checks 行、criteria 判据、not_done_this_round 登记）之一，无第三态',
        'scope': 'V22 写作期 claim-lint 启用时将本不变量纳入检查范围（正文数字 lint 同批；'
                 'OUTLINE 为计划文档暂不 lint）——登记层先行，前置未触发',
        'purpose': '堵「子项静默脱落」复发面（R3A2-m1/R6B-m3/R3A2 建议3）',
        'producer': PRODUCER, 'review_refs': 'R3A2-m1 / R6B-m3 / criteria_r4 P1-2③'}
    save(MAN_P, man)
    print('5. P1-2③ claim_lint_invariant_r4 登记入 manifest')
else:
    print('5. P1-2③ 已存在，跳过（幂等）')

# ---------------- P1-3b：criteria_mtime_ledger 补 criteria_r3 行 ----------------
LGR_P = 'results/EXP-R2-REG/criteria_mtime_ledger.json'
lgr = load(LGR_P)
if not any('criteria_r3' in str(r.get('file', '')) for r in lgr.get('rows_r5_append', [])):
    st = subprocess.run(['stat', '-c', '%y', f'{BASE}/results/EXP-R3-REG/criteria_r3.json'],
                        capture_output=True, text=True).stdout.strip()
    lgr['rows_r5_append'] = [{
        'file': 'results/EXP-R3-REG/criteria_r3.json',
        'mtime_filesystem': st,
        'self_reported_time': load('results/EXP-R3-REG/criteria_r3.json').get('frozen_at_filesystem'),
        'verdict': ('R3A2-m3 追加行（R4/P1-3b）：criteria_r3 双时刻申报——frozen_at_filesystem 2026-10-24T23:15:29 '
                    '（自报，先于 R3 第二轮运行）；文件 mtime 00:44:16 晚于冻结——系 r3_review2_minor_fixes.py '
                    '追加 quote_not_provenance_declaration_r3b 登记键（R3 第二轮修复动作）所致；'
                    '追加键语义=登记层补录非判据改写。append-only 声明：本行仅追加，rows 原文不动'),
        'producer': PRODUCER, 'review_refs': 'R3A2-m3 / criteria_r4 P1-3'}]
    save(LGR_P, lgr)
    print(f'6. P1-3b criteria_mtime_ledger 追加 criteria_r3 行（mtime={st}）')
else:
    print('6. P1-3b 已存在，跳过（幂等）')

# ---------------- P1-3c：M7 located_at 交付时点链长注记 ----------------
MS_P = 'results/EXP-R3-REG/gate3_mutation_score_r3.json'
ms = load(MS_P)
m7 = next(m for m in ms['mutations'] if m.get('id') == 'M7')
if 'located_at_note_r4' not in m7:
    n_rows = sum(1 for l in open(f'{BASE}/results/EXP-R2-E17/provenance_ledger.jsonl') if l.strip())
    m7['located_at_note_r4'] = (f'row 54=交付时点（2026-10-25T00:40 变异运行）链长；现链 {n_rows} 行'
                                f'（R3 收口+R4 冻结后持续增长，append-only）——located_at 为运行时点快照'
                                f'非当前行号，防跨时点误读（R6B-m5）')
    save(MS_P, ms)
    print(f'7. P1-3c M7 located_at 注记落盘（现链 {n_rows} 行）')
else:
    print('7. P1-3c 已存在，跳过（幂等）')

print('R4_P0_REGISTRY_FIXES_DONE')
