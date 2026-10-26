"""r3_errata_appends.py — R3/P1-5+P1-6：三处滑笔 errata 追加 + e12 design 键 posthoc 指针。

errata 三字段 schema（errata 惯例：原文不改不删、更正独立可引）：
  errata_class ∈ {editorial, technical} / reason / supersedes。
三处（R4B-m4①②③，数值前提本席亲验）：
  ①criteria_mtime_ledger rows_r4_append[0]「晚于」→「早于」（同格括注自相矛盾；editorial）
  ②criteria_e14 conclusion_fix 上界区间起点 0.00169→0.00162（本席现算 min 上界=0.00162=uniform_16；
    0.00169 为 temporal_16 单点值误作区间起点；technical——预登记数值与产物实算不符、产物自愈）
  ③criteria_e16 hypothesis「约减半」→「约 1/3」（−0.02081→−0.00741 实为 0.356；editorial）
P1-6：e12_closure.json 顶层追加 design_posthoc_pointer_r3（原 design 键原文保留）。
幂等：键在且终态一致→no-op；全部追加键名在本脚本字面量在案（闸1 有源）。
"""
import json

B = '/root/autodl-tmp/projB_hetero_graph'
PROD = 'src/r3_errata_appends.py（R3/P1-5+P1-6）'


def ensure(path, key, final, label):
    d = json.load(open(path))
    if key in d:
        assert d[key] == final, f'{path}:{key} 在案值不符——人工核查'
        print(f'  no-op {label}')
        return
    d[key] = final
    with open(path, 'w') as f:
        json.dump(d, f, indent=1, ensure_ascii=False)
    print(f'  APPEND {label}')


def main():
    # ---- ①mtime_ledger「晚于」→「早于」（editorial）----
    p1 = f'{B}/results/EXP-R2-REG/criteria_mtime_ledger.json'
    row0 = json.load(open(p1))['rows_r4_append'][0]
    assert '晚于' in row0['verdict'] and '早于' not in row0['verdict'], '现场变化，人工核查'
    ensure(p1, 'rows_r4_append_errata_r3', {
        'errata_class': 'editorial',
        'reason': '方向词滑笔且与同格括注自相矛盾：原 verdict 写「19:42:07 晚于 E12 两臂训练启动」，'
                  '同格括注却是「logs/e12_t1_homo_60ep.log 19:46:20 之前约 4 分钟」——训练启动由产物完成'
                  '时刻（19:46:20/19:47:08）倒推 ≈19:42:5x，19:42:07 在其之前，「晚于」应为「早于」'
                  '（本席核验：mtime 证据链未变，仅方向词更正）',
        'supersedes': 'rows_r4_append[0].verdict 中的方向词「晚于」（原文保留不删改）；'
                      '实质结论不变：训练所用脚本临跑前被编辑、判据后补（「收口分析前冻结」降格维持）',
        'corrected_reading': 'src/r1_train_arm.py mtime 19:42:07 早于 E12 两臂训练启动（≈19:42:5x，倒推）；'
                             '晚于其 criteria 冻结/落盘（E12 判据 post_hoc 19:48+ 补登记）——时序病灶在判据侧不在脚本侧',
        'producer': PROD, 'review_refs': 'R4B-m4① / 本轮 P1-5'}, 'criteria_mtime_ledger errata')

    # ---- ②criteria_e14 上界起点 0.00169→0.00162（technical）----
    p2 = f'{B}/results/EXP-R2-REG/criteria_e14_tost_producer.json'
    t = json.load(open(f'{B}/results/EXP-R2-REG/tost_boundary_scan_ap.json'))
    ups = {k: v['ci90'][1] for k, v in t['points'].items()}
    assert min(ups.values()) == 0.00162 and ups['temporal_16'] == 0.00169, '数值前提变化，人工核查'
    ensure(p2, 'conclusion_fix_errata_r3', {
        'errata_class': 'technical',
        'reason': '预登记数值滑笔：原 conclusion_fix 写「上界实为 0.00169-0.00327」，但 0.00169 是 '
                  'temporal_16 单点值而非区间起点——六点实算 min 上界=0.00162（uniform_16）、max=0.00327'
                  '（uniform_4），本席从 tost_boundary_scan_ap.json 现算复核属实（R4B-m4②）',
        'supersedes': 'pre_registered_expectations.conclusion_fix 中的区间「0.00169-0.00327」（原文保留）',
        'corrected_text': '上界实为 0.00162-0.00327（min=uniform_16 0.00162；上界不承担排除语义，'
                          '起排除作用者为下界口径，min 下界=−0.00956>−0.01 不变）——预登记滑笔由产物实算'
                          '自愈，判定（Δ=0.01 排除 6/6、Δ=0.005 仅 temporal_4）不受影响',
        'producer': PROD, 'review_refs': 'R4B-m4② / 本轮 P1-5'}, 'criteria_e14 errata')

    # ---- ③criteria_e16「减半」→「约 1/3」（editorial）----
    p3 = f'{B}/results/EXP-R2-E16/criteria_e16.json'
    ensure(p3, 'hypothesis_errata_r3', {
        'errata_class': 'editorial',
        'reason': '幅度表述滑笔：原 hypothesis 写「幅度约减半」，交付实算 30ep −0.02081→60ep −0.00741，'
                  '|比值|=0.356≈1/3 而非 1/2（R4B-m4③ 数值逐位证实）',
        'supersedes': 'hypothesis 中「幅度约减半」（原文保留）',
        'corrected_text': '幅度约 1/3（−0.02081→−0.00741，比值 0.356；方向保持）——聚合 p=0.616 未检出',
        'producer': PROD, 'review_refs': 'R4B-m4③ / 本轮 P1-5'}, 'criteria_e16 errata')

    # ---- P1-6：e12 design 键 posthoc 指针 ----
    p4 = f'{B}/results/EXP-R2-E12/e12_closure.json'
    ensure(p4, 'design_posthoc_pointer_r3', {
        'status': 'APPENDED（R3/P1-6）：本键为顶层 design 键的 posthoc 指针——原 design 键'
                  '「见模块 docstring（判据双去向预登记）」自报「开工前冻结」不被 mtime 支持'
                  '（e12_closure.py 19:48:32 晚于训练产物 19:46:20/19:47:08），防单读误判事前冻结',
        'pointer': 'results/EXP-R2-E12/criteria_e12_posthoc_r4.json（判据 post_hoc 补登记）',
        'freeze_status': '收口分析前冻结（降格申报，R3B-M3）；GPU 门禁已立规：训练启动前断言 criteria 行已落盘',
        'supersedes': '仅指针补充；原 design 键与 docstring 原文保留不改',
        'producer': PROD, 'review_refs': 'R4B-m5① / 本轮 P1-6'}, 'e12 design 指针')
    print('R3_ERRATA_APPENDS_DONE')


if __name__ == '__main__':
    main()
