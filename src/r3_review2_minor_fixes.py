"""r3_review2_minor_fixes.py — R3 第二轮 Minor 批处理（脚本化修复+修复即入链前置落盘）。

处置清单（全部 append-only supersedes 风格：原键原词保留，只追加）：
  R3A-m4/R5B-m3  keyscan_two_phase_r3.json += phase1_definition_clarification_r3（phase1「R2 时点 src」
                 实为「R2 时点 src 快照 × 当前 results/**」；计数 70 复核无误，仅标签澄清）
  R3A-m2         calibration_certificate.json += family_multiplicity_stance_r3（16 格族级 Bonferroni×16
                 立场：arl500/S3 p=0.017857→×16=0.2857 不拒绝=唯一翻转格；S4 两档任何校正下强拒绝；
                 方案二 claims 登记层第一个登记实例）
  方案二①        number_manifest.json += 3 行 claims 类（判语-证据绑定：E15 立场/E2 三方锚/not_done+QUOTE
                 申报+mutation score 锚定，expect 现读固化禁手抄）
  R5B-m2         criteria_r3.json += quote_not_provenance_declaration_r3b（QUOTE=值巧合判定不构成出处）
  R3A-m1/R5B-m1  RESEARCH_LOG.md 四处 errata 更正注（原词唯一匹配断言后行后插入，三字段 schema）：
                 :417 59→70 / P1-7 行断根撤回 / P1-8 行 QUOTE 88→87 指针 / E17 段链行 52→以 audit 现读为准
幂等：全部追加键存在即跳过；LOG 更正注存在（grep 标记词）即跳过。纯 CPU 秒级。
"""
import json

BASE = '/root/autodl-tmp/projB_hetero_graph'
MARK = '【R3b 更正注】'


def load(p):
    return json.load(open(f'{BASE}/{p}'))


def save(p, d):
    with open(f'{BASE}/{p}', 'w') as f:
        json.dump(d, f, indent=1, ensure_ascii=False)


def main():
    # ---- 1. R3A-m4/R5B-m3 phase1 定义句澄清 ----
    kt = load('results/EXP-R3-REG/keyscan_two_phase_r3.json')
    if 'phase1_definition_clarification_r3b' not in kt:  # R4/P1-3（R6B-m3）：守卫键名补 b——查 _r3 写 _r3b 错位致重跑重复插入
        kt['phase1_definition_clarification_r3b'] = {
            'status': 'SUPERSEDES（R3 第二轮 R3A-m4/R5B-m3：定义句澄清，计数不改）',
            'clarification': 'phase1 口径实为「R2 时点 src 快照 × 当前 results/** 全部键（含 R3 追加键）」——'
                             '70 个 phase1 孤儿中部分键在 R2 时点 results 中并不存在；键名 '
                             'phase1_r2_src_snapshot 读作「R2 时点 src 侧材料」，非「R2 时点全状态快照」',
            'count_recheck': 'phase1=70 经审稿人参数化重放逐位复现（R5B-m3），本席不改数；「先升后降」演示目的'
                             '（升级前扫描口径弱→处置后 0）不受影响',
            'producer': 'src/r3_review2_minor_fixes.py（R3 第二轮）'}
        save('results/EXP-R3-REG/keyscan_two_phase_r3.json', kt)
        print('1. keyscan_two_phase phase1 定义句澄清键落盘')

    # ---- 2. R3A-m2 E15 族级 multiplicity 立场（数值现算禁手抄） ----
    from scipy.stats import binom
    p_arl500_S3 = float(binom.sf(0, 9, 1 / 500))  # P(X>=1), n=9 池, 名义 1/500——首版 sf(0)+pmf(0) 恒 1 被逐位断言拦截后修正
    cert = load('results/EXP-R2-E15/calibration_certificate.json')
    p_raw_reg = cert['certificate']['arl500']['S3_empirical_reachable']['exact_binomial']['p_one_sided']
    assert abs(p_arl500_S3 - p_raw_reg) < 1e-12, f'现算 {p_arl500_S3} 与在案 {p_raw_reg} 失配'
    x16 = lambda p: p * 16
    if 'family_multiplicity_stance_r3' not in cert:
        cert['family_multiplicity_stance_r3'] = {
            'status': 'APPENDED（R3 第二轮 R3A-m2：16 格族级多重比较立场；各格 verdict 原文保留）',
            'family_definition': '8 规格（2 arl 档 × S1-S4）× 双 null（A 保守/B 每窗派生）=16 格精确二项，'
                                 '单口径 5% 逐格判定（证书主证据=realized ARL，本层为描述性并报口径）',
            'bonferroni_x16': {
                'arl500_S3_nullA': {'p_raw': p_raw_reg, 'p_adjusted': x16(p_raw_reg),
                                    'verdict_5pct_adjusted': '不拒绝（×16=0.2857>0.05）——16 格中唯一翻转格'},
                'arl500_S4_nullA': {'p_raw': cert['certificate']['arl500']['S4_evt_hybrid_REJECTED_daywindow']
                                    ['exact_binomial']['p_one_sided'],
                                    'p_adjusted': x16(cert['certificate']['arl500']
                                                      ['S4_evt_hybrid_REJECTED_daywindow']['exact_binomial']
                                                      ['p_one_sided']),
                                    'verdict_5pct_adjusted': '拒绝（任何校正下强拒绝）'},
                'arl100_S4_nullA': {'p_raw': cert['certificate']['arl100']['S4_evt_hybrid_REJECTED_daywindow']
                                    ['exact_binomial']['p_one_sided'],
                                    'p_adjusted': x16(cert['certificate']['arl100']
                                                      ['S4_evt_hybrid_REJECTED_daywindow']['exact_binomial']
                                                      ['p_one_sided']),
                                    'verdict_5pct_adjusted': '拒绝（任何校正下强拒绝）'}},
            'stance': ('族级校正（Bonferroni×16）下拒绝数 3→2：arl500/S3 由「拒绝」翻为「不拒绝」，S4 两档不变。'
                       '结论方向（S4 档超名义误报预算、S1-S3 不超）对校正稳健；写作引用 arl500/S3 单格「拒绝」'
                       '时必须随附本立场句或声明描述性单口径（E14 自标先例同款）'),
            'producer': 'src/r3_review2_minor_fixes.py（R3 第二轮 R3A-m2）；p 现算 binom(k>=1;n=9,p=1/500) '
                        '与在案逐位断言一致后方落盘',
            'review_refs': 'R3A-m2 / 方案二 claims 登记层实例 #1'}
        save('results/EXP-R2-E15/calibration_certificate.json', cert)
        print(f'2. E15 族级 multiplicity 立场键落盘（现算 p={p_raw_reg:.15g} 与在案一致）')

    # ---- 3. 方案二①：manifest claims 类 3 行（expect 现读固化） ----
    mf = load('results/EXP-R2-REG/number_manifest.json')
    have = {r['n'] for r in mf['mappings']}
    e2 = load('results/EXP-R1-E2/e2_closure.json')
    rm = load('results/R1_metrics.json')
    new_rows = [
        {'n': 'claims/E15 族级 multiplicity 立场（R3A-m2）：arl500/S3 ×16=0.2857 不拒绝=唯一翻转格',
         'producer': 'src/r3_review2_minor_fixes.py（R3 第二轮方案二①判语-证据绑定）',
         'checks': [{'path': 'EXP-R2-E15/calibration_certificate.json:family_multiplicity_stance_r3.bonferroni_x16.arl500_S3_nullA.p_adjusted',
                     'expect': x16(p_raw_reg)}]},
        {'n': 'claims/E2 full 臂 AP 三方锚定+判语更正（R5B-M1）：E47B 锚=incase re 值=补算值逐位一致',
         'producer': 'src/r3_review2_minor_fixes.py（R3 第二轮方案二①）',
         'checks': [{'path': 'EXP-R1-E2/e2_closure.json:regression_check_full_ap_anchor_r3b.three_way_anchor.bitwise_identical',
                     'expect': True},
                    {'path': 'EXP-R1-E2/e2_closure.json:regression_check_full_ap_anchor_r3b.three_way_anchor.anchor_ap_round4',
                     'expect': e2['regression_check_full_ap_anchor_r3b']['three_way_anchor']['anchor_ap_round4']},
                    {'path': 'R1_metrics.json:not_done_this_round', 'expect': None}]},
        {'n': 'claims/QUOTE 不构成出处申报（R5B-m2）+ 闸3 拦截力变异自证锚定（R3A-M1/R5B-M2）',
         'producer': 'src/r3_review2_minor_fixes.py（R3 第二轮方案二①）',
         'checks': [{'path': 'EXP-R3-REG/criteria_r3.json:quote_not_provenance_declaration_r3b.registered',
                     'expect': 'R3b'},
                    {'path': 'EXP-R3-REG/gate3_mutation_score_r3.json:mutation_score', 'expect': '8/8'},
                    {'path': 'R1_metrics.json:_meta.not_done_appended_r3b', 'expect': None}]},
    ]
    added = [r for r in new_rows if r['n'] not in have]
    if added:
        mf['mappings'].extend(added)
        mf.setdefault('claims_layer_note_r3b', (
            'R3 第二轮方案二①：claims 类行入列——凡「全部/零/断根/等效」级判语的更正与立场必须以登记行+机器 '
            'check 锚定（DO-178C 三向追溯同思想）；无 check 又未入 not_done 的强判语为闸2 违规（写作期 '
            'claim-lint 的登记层前提）'))
        save('results/EXP-R2-REG/number_manifest.json', mf)
        print(f'3. manifest claims 行 +{len(added)}（总 {len(mf["mappings"])} 行）')
    else:
        print('3. manifest claims 行已存在，跳过（幂等）')

    # ---- 4. R5B-m2 QUOTE 申报句（criteria_r3 追加键） ----
    cr = load('results/EXP-R3-REG/criteria_r3.json')
    if 'quote_not_provenance_declaration_r3b' not in cr:
        cr['quote_not_provenance_declaration_r3b'] = {
            'registered': 'R3b',
            'declaration': '闸2 L_QUOTE 层为值巧合判定（finalize 浮点字面量按自身精度舍入后精确出现于交付 '
                           'JSON）——QUOTE 不构成出处/溯源链接，仅申报「该数字已在交付层存在」；出处判定以 '
                           'number_manifest checks 路径为准。R3A-m3 收窄后判定池与闸1 skip 同口径（排除 '
                           'criteria* 与 FILE_EXEMPT 登记文档），收窄前后分层 91/1/87/3 零变化（原 87 QUOTE '
                           '全部真命中交付 JSON，收窄为纯防御性收紧）',
            'producer': 'src/r3_review2_minor_fixes.py（R3 第二轮）',
            'review_refs': 'R5B-m2 / R3A-m3'}
        save('results/EXP-R3-REG/criteria_r3.json', cr)
        print('4. criteria_r3 QUOTE 申报键落盘')

    # ---- 5. R3A-m1/R5B-m1 LOG 四处 errata 更正注（原词唯一匹配→行后插入） ----
    logp = f'{BASE}/RESEARCH_LOG.md'
    lines = open(logp, encoding='utf-8').read().splitlines()
    notes = [
        ('R3B-E1', '59→处置后 0',
         'editorial',
         '两阶段留档 phase1 计数 70（keyscan_two_phase_r3.json phase1_r2_src_snapshot.orphan_keys 现读为准，'
         '审稿人参数化重放逐位复现）；「先升后降 70→处置后 0」',
         '59→处置后 0'),
        ('R3B-E2', 'R1A-m7 连续两轮悬空断根',
         'technical',
         '「断根」措辞撤回+参照申报更正：EXP-R1-VERIFY/incase_check.json 自 R1 起在案载有 full 臂 '
         'ap_logon_re=0.0043/0.0048/0.0056，「在案交付无 full 臂 AP 参照值」申报句被自家产物推翻（R5B-M1）；'
         '本席从 E47B 在案 npz 独立推锚三方逐位一致，状态更正 NEW_VALUE→REPRODUCED——null 本体（R1 原值）'
         '不改写；机器锚=e2_closure.json:regression_check_full_ap_anchor_r3b + R1_metrics not_done_this_round',
         '「新算值、无逐位参照」非复现校验'),
        ('R3B-E3', 'QUOTE 88=交付数舍入引述机器判定',
         'editorial',
         'QUOTE 实为 87（warn_stratification_r3.json l_quote=87，91=1+87+3 自洽）；本行写就时序早于注册表收敛，'
         '机器锚=results/EXP-R3-REG/warn_stratification_r3.json',
         'QUOTE 88'),
        ('R3B-E4', '链行 52=19 旧',
         'editorial',
         '链行数为时点值：该段写就后末跑冻结 LOG 自身致 +1（审计时 53），此后每轮 --freeze 递增——行数以 '
         'provenance_first_audit.json:gate3.ledger_rows 现读为准，本注不写死数',
         '链行 52'),
    ]
    inserted = 0
    for rid, anchor_word, eclass, corr, supersedes in notes:
        # 幂等锚=稳定 id（首版按「anchor 行内含 MARK」判重是设计错误：更正注为独立新行，
        # anchor 行永远不含 MARK → 双跑重复插入，已由幂等双跑暴露并清理）
        if any(f'{MARK}#{rid}' in l for l in lines):
            print(f'5. 更正注已在（{rid}），跳过')
            continue
        hits = [i for i, l in enumerate(lines) if anchor_word in l]
        assert len(hits) == 1, f'原词非唯一（{len(hits)} 命中）：{anchor_word}'
        i = hits[0]
        lines.insert(i + 1, f'  - {MARK}#{rid}（{eclass}）上行「{supersedes}」→ {corr}'
                            f'〔errata_class={eclass}; supersedes=原词保留于上行; producer=src/r3_review2_minor_fixes.py〕')
        inserted += 1
    with open(logp, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    print(f'5. LOG errata 更正注插入 {inserted} 处（原词 grep 仍在断言=插入后原文行未动）')
    # 原词仍在断言（V14 同款）
    txt = open(logp, encoding='utf-8').read()
    for anchor_word, *_ in notes:
        assert anchor_word in txt, f'原词丢失：{anchor_word}'
    print('R3B_MINOR_DONE 全部落盘（幂等可重入）')


if __name__ == '__main__':
    main()
