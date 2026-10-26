"""r3_e2_anchor_r3b.py — R3 第二轮 R5B-M1 修复：E2 full 臂 AP 三方锚定+判语过强更正+「断根」措辞撤回。

【R5B-M1 意见独立复核结论（本席先验证后动手）】
①意见属实：EXP-R1-VERIFY/incase_check.json 自 R1 起在案载有 arm=full ap_logon_re=0.0043/0.0048/0.0056，
  P1-7 补算键「在案交付无 full 臂 AP 参照值」申报句被自家产物推翻；E2 regression_check_verdict 与 VERIFY
  verdict 两处「AUC/AP 全部 |Δ|」判语对 full 臂 AP 过强（in 侧 null 无 Δ 可言）；R1_metrics not_done 为空。
②精确化（接受意见但口径收紧）：null 本体（regression_check.full[*].in_ap_logon=R1 原始交付值）任何重算都
  无法给增量——null 保持 null 不改写（只增不改红线）；可更正的是参照申报与判语，不是伪造 R1 原值。
③本席独立推锚：从白名单在案 npz（/root/autodl-tmp/results/E47B_fairpool_ci/full，R5B-M1 指认可推锚处）
  同式重算 AP=0.0043/0.0048/0.0056，与 P1-7 补算值、incase ap_logon_re 三方逐位一致（池计数 4/4 全同）——
  P1-7 状态由 NEW_VALUE 更正为 REPRODUCED（复现校验 PASS，参照=VERIFY 在案 re 值+E47B 独立锚）。
【动作（全 append-only supersedes 风格：原键原文不动，只追加更正键）】
e2_closure.json += regression_check_full_ap_anchor_r3b / regression_check_verdict_correction_r3b；
incase_check.json += verdict_correction_r3b；R1_metrics.json not_done_this_round: null → 正式登记清单。
幂等：锚值确定，重跑末态逐位一致。纯 CPU 秒级。判据=criteria_r3b 口径见本 docstring（先写后跑）。
"""
import json, sys

sys.path.insert(0, '/root/autodl-tmp/projB_hetero_graph/src')
from e2_closure import metrics_from_npz  # noqa: E402  同机器同式（E2 原收口函数）

BASE = '/root/autodl-tmp/projB_hetero_graph'
E47B_FULL = '/root/autodl-tmp/results/E47B_fairpool_ci/full'  # 白名单在案 npz（ask 硬边界允许）
E2P = f'{BASE}/results/EXP-R1-E2/e2_closure.json'
ICP = f'{BASE}/results/EXP-R1-VERIFY/incase_check.json'
RMP = f'{BASE}/results/R1_metrics.json'


def main():
    # ---- 三方锚定与双向逐位断言 ----
    anchor, incase_re, backfill = {}, {}, {}
    ic = json.load(open(ICP))
    e2 = json.load(open(E2P))
    for r in ic['rows']:
        if r['arm'] == 'full':
            incase_re[r['seed']] = r['ap_logon_re']
    bk = e2['regression_check_full_ap_backfill_r3']['rows']
    for s in range(3):
        m = metrics_from_npz(E47B_FULL, s)
        anchor[s] = round(m['ap_logon'], 4)
        backfill[s] = bk[f'full_s{s}']['ap_backfilled']
    print('E47B 锚      :', [anchor[s] for s in range(3)])
    print('incase re 值 :', [incase_re[s] for s in range(3)])
    print('P1-7 补算值  :', [backfill[s] for s in range(3)])
    assert all(anchor[s] == incase_re[s] == backfill[s] for s in range(3)), '三方逐位失配——停，不硬改'
    # 池计数断言（同池前提）
    m0 = metrics_from_npz(E47B_FULL, 0)
    assert (m0['n_pool'], m0['n_pos'], m0['n_pool_logon'], m0['n_pos_logon']) == (445229, 277, 131235, 277)
    print('三方逐位 PASS + E47B 池计数 (445229,277,131235,277) 与 uniform1_full_60ep 全同 → 同评估池实证')

    # ---- e2_closure.json 追加两键（原文不动）----
    e2['regression_check_full_ap_anchor_r3b'] = {
        'status': 'SUPERSEDES（R3 第二轮 R5B-M1：只增不改——backfill_r3 键原文保留；其「NEW_VALUE/无逐位参照/'
                  '在案交付无 full 臂 AP 参照值」申报句被自家产物推翻，本键更正口径）',
        'null_class_correction': 'in_ap_logon=null 本体不改写（null=R1 原始交付值，任何重算不构成其增量，'
                                 'INCASE[\'full\'][\'ap_logon\']=None 仍真）；更正的是参照申报：EXP-R1-VERIFY/'
                                 'incase_check.json 自 R1 起在案载有 full 臂 ap_logon_re=0.0043/0.0048/0.0056',
        'backfill_status_correction': 'NEW_VALUE → REPRODUCED（复现校验 PASS）：P1-7 补算值与在案 VERIFY re 值'
                                      '逐位一致（3/3），另获 E47B 独立第三方锚逐位一致（3/3）',
        'three_way_anchor': {
            'anchor_npz': 'results 白名单在案 /root/autodl-tmp/results/E47B_fairpool_ci/full/scores_seed{0,1,2}.npz',
            'anchor_ap_round4': [anchor[s] for s in range(3)],
            'incase_ap_logon_re': [incase_re[s] for s in range(3)],
            'backfill_r3_ap': [backfill[s] for s in range(3)],
            'pool_counts_E47B': [m0['n_pool'], m0['n_pos'], m0['n_pool_logon'], m0['n_pos_logon']],
            'bitwise_identical': True},
        'broken_root_claim_retraction': '「R1A-m7 连续两轮悬空在此断根」措辞撤回：实际发生=P1-7 以同值重放收口 '
                                        'null 且申报句自身有误（修复轮自伤注释级第 5 例）；悬空问题的最终收口'
                                        '以本键三方锚定为准',
        'producer': 'src/r3_e2_anchor_r3b.py（R3 第二轮 R5B-M1）；机器=sklearn average_precision_score @ '
                    'rel==LOGON（e2_closure.metrics_from_npz 同式）',
        'review_refs': 'R5B-M1 / R3A-M1（佐证面）'}
    e2['regression_check_verdict_correction_r3b'] = {
        'status': 'SUPERSEDES（R3 第二轮 R5B-M1：判语过强更正）',
        'original_verdict_kept': e2['regression_check_verdict'],
        'correction': '原判语「混合池/LOGON AUC/AP 全部 |Δ|<0.02」对 full 臂 AP 过强：full AP 的 in 侧=null，'
                      '无 |Δ| 可言（homo AP 3 seed 有 |Δ| 且全过；AUC 各边全过不受影响）。更正口径=「AUC 全部 '
                      '边+AP homo 臂 3 seed |Δ|<0.02；full 臂 AP 无在案原值可比，其重算值经三方锚定逐位一致'
                      '（见 regression_check_full_ap_anchor_r3b）」',
        'producer': 'src/r3_e2_anchor_r3b.py'}
    with open(E2P, 'w') as f:
        json.dump(e2, f, indent=1, ensure_ascii=False)

    # ---- incase_check.json 追加更正键（判语「逐边一致」过强）----
    ic['verdict_correction_r3b'] = {
        'status': 'SUPERSEDES（R3 第二轮 R5B-M1：只增不改——verdict 原文保留）',
        'original_verdict_kept': ic['verdict'],
        'correction': '原判语「3 臂×3 seed 逐边 npz 复算与在案 JSON 一致」对 full 臂 AP 边过强：该边在案 JSON '
                      '侧（ap_logon_in）=null，match 判定实际只覆盖有值边+计数；AP 边逐位一致仅成立于 re 侧'
                      '重算值之间（VERIFY ap_logon_re 与后续补算/E47B 锚三方逐位一致，见 '
                      'e2_closure.json:regression_check_full_ap_anchor_r3b）',
        'producer': 'src/r3_e2_anchor_r3b.py'}
    with open(ICP, 'w') as f:
        json.dump(ic, f, indent=1, ensure_ascii=False)

    # ---- R1_metrics.json not_done_this_round: null → 正式登记（R5B-M1「清单兜底分支未走」闭环）----
    rm = json.load(open(RMP))
    rm['not_done_this_round'] = [
        {'item': 'E2 regression_check_verdict 与 VERIFY incase_check verdict 两处「AUC/AP 全部 |Δ|/逐边一致」'
                 '判语原文未在本轮改写（只增不改红线）',
         'disposition': '以 supersedes 更正键收口：e2_closure.json:regression_check_verdict_correction_r3b + '
                        'incase_check.json:verdict_correction_r3b；写作期正文引用判语时以更正口径为准',
         'refs': 'R5B-M1 / R1A-m7'},
        {'item': 'full 臂 AP 在案原值（R1 时代 in 侧）永远缺位——R1 原始交付未算，不可补造',
         'disposition': '如实保持 null；其重算参照链（VERIFY re 值+E47B 锚+P1-7 重放三方逐位一致）见 '
                        'e2_closure.json:regression_check_full_ap_anchor_r3b',
         'refs': 'R5B-M1'}]
    rm['_meta']['not_done_appended_r3b'] = ('R3 第二轮 R5B-M1：not_done_this_round 由 null 起用（清单兜底分支'
                                            '首走）；追加键 producer=src/r3_e2_anchor_r3b.py')
    with open(RMP, 'w') as f:
        json.dump(rm, f, indent=1, ensure_ascii=False)
    print('R3B_ANCHOR_DONE：e2_closure/incase/R1_metrics 三产物追加键落盘（原文零改写）')


if __name__ == '__main__':
    main()
