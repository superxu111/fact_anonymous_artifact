"""r3_manifest_append.py — R3/P0-3：number_manifest 增 4+1 行（expect 现读固化，禁手抄）+E14 postmortem。

行（R2A2-M3：本轮新数字零覆盖缺口）：①tost D4（CI 下界/Δ 排除）②E13 延迟分歧 4 对 8 格
③E15 证书 0/9-1/9-4/9 与双 null p（全精度+<1e-6 档）④E16 聚合 0.26853/0.0005 与 −0.00741/0.61619
⑤R3-P1-7 E2 full 臂 AP 补算（V16）。
粒度规则（criteria_r3.json 先登记）：expect=现读交付 JSON 字面值原样，不二次舍入。
幂等：行按 n 定位——在案且 expect==现读→no-op；expect≠现读→FAIL（交付漂移信号，禁静默改行）。
"""
import json

B = '/root/autodl-tmp/projB_hetero_graph'
MP = f'{B}/results/EXP-R2-REG/number_manifest.json'
POSTMORTEM_KEY = 'postmortem_r3_manifest_rows'


def rd(rel_and_path):
    """现读交付值（expect 固化来源，禁手抄）。"""
    fp, _, jpath = rel_and_path.partition(':')
    cur = json.load(open(f'{B}/results/{fp}'))
    for part in [x for x in jpath.split('.') if x]:
        if '[' in part:
            k, idx = part[:-1].split('[')
            cur = cur[k][int(idx)]
        else:
            cur = cur[part]
    return cur


T = 'EXP-R2-REG/tost_boundary_scan_ap.json'
F = 'EXP-R2-REG/e4_window_row_fair_replay.json'
C = 'EXP-R2-E15/calibration_certificate.json'
E16 = 'EXP-R2-E16/e12_60ep_seed_aggregate.json'
E2 = 'EXP-R1-E2/e2_closure.json'
S3, S4 = 'S3_empirical_reachable', 'S4_evt_hybrid_REJECTED_daywindow'
S1, S2 = 'S1_tpivot_t9_8', 'S2_z3_probe'

ROWS = [
    {'n': 'D4 tost 边界扫描（R3/P0-3 补行）：CI 下界 −0.00956 与 Δ 排除（起排除作用为下界）',
     'producer': 'src/r3_manifest_append.py（R3/P0-3；expect 现读固化）',
     'checks': [
         {'path': f'{T}:ci_bounds_range.ci_lower_min', 'expect': rd(f'{T}:ci_bounds_range.ci_lower_min'),
          'note': 'min 下界=−0.00956>−0.01（最紧格 uniform_1）'},
         {'path': f'{T}:ci_bounds_range.ci_upper_max', 'expect': rd(f'{T}:ci_bounds_range.ci_upper_max'),
          'note': '上界不承担排除语义（R2-E14 更正口径）'},
         {'path': f'{T}:points.uniform_1.ci90[0]', 'expect': rd(f'{T}:points.uniform_1.ci90[0]')},
         {'path': f'{T}:points.temporal_4.excludable_delta', 'expect': rd(f'{T}:points.temporal_4.excludable_delta'),
          'note': 'Δ=0.005 仅 temporal_4 可排除（预登记枚举）'}]},
    {'n': 'R2-E13 延迟规则真实分歧（R3/P0-3 补行）：4 对 8 格（R2A-M2 完整双档口径；R3B-M1 为 arl100 子集）',
     'producer': 'src/r3_manifest_append.py（R3/P0-3；expect 现读固化）',
     'checks': [
         {'path': f'{F}:delay_rule_divergence_count.pairs', 'expect': rd(f'{F}:delay_rule_divergence_count.pairs')},
         {'path': f'{F}:delay_rule_divergence_count.cells', 'expect': rd(f'{F}:delay_rule_divergence_count.cells')}]},
    {'n': 'R2-E15 校准证书（R3/P0-3 补行）：越限 0/9-1/9-4/9 与双 null p（全精度；p 档界=1e-6）',
     'producer': 'src/r3_manifest_append.py（R3/P0-3；expect 现读固化自 R3 修复后证书）',
     'checks': [
         *[{'path': f'{C}:certificate.{arl}.{s}.k_exceed_pools', 'expect': rd(f'{C}:certificate.{arl}.{s}.k_exceed_pools')}
           for arl in ('arl100', 'arl500') for s in (S1, S2, S3, S4)],
         {'path': f'{C}:certificate.arl100.{S3}.exact_binomial.p_one_sided',
          'expect': rd(f'{C}:certificate.arl100.{S3}.exact_binomial.p_one_sided'),
          'note': 'arl100/S3 不拒绝（旧静态句「边缘拒绝」矛盾的数值现场，R2A2-M2①）'},
         {'path': f'{C}:certificate.arl500.{S3}.exact_binomial.p_one_sided',
          'expect': rd(f'{C}:certificate.arl500.{S3}.exact_binomial.p_one_sided'),
          'note': 'arl500/S3 拒绝（口径依赖并报）'},
         {'path': f'{C}:certificate.arl100.{S4}.exact_binomial.p_one_sided',
          'expect': rd(f'{C}:certificate.arl100.{S4}.exact_binomial.p_one_sided'),
          'note': '全精度（旧 round6=1e-06，V2 diff 显式登记于 spec_curve v2 键）'},
         {'path': f'{C}:certificate.arl500.{S4}.exact_binomial.p_one_sided',
          'expect': rd(f'{C}:certificate.arl500.{S4}.exact_binomial.p_one_sided'),
          'note': '全精度 ≈2e-9（旧 round6 舍为 0.0，R2A2-m3）'},
         {'path': f'{C}:certificate.arl500.{S4}.exact_binomial.p_display_tier',
          'expect': rd(f'{C}:certificate.arl500.{S4}.exact_binomial.p_display_tier'),
          'note': '<1e-6 档（档界规则先登记于 criteria_r3.json）'}]},
    {'n': 'R2-E16 60ep 聚合（R3/P0-3 补行）：mixed 0.26853/0.0005 与 LOGON AUC −0.00741/0.61619',
     'producer': 'src/r3_manifest_append.py（R3/P0-3；expect 现读固化）',
     'checks': [
         {'path': f'{E16}:ep30_vs_ep60_aggregate.auc_mixed.ep60_this.obs_mean_diff',
          'expect': rd(f'{E16}:ep30_vs_ep60_aggregate.auc_mixed.ep60_this.obs_mean_diff')},
         {'path': f'{E16}:ep30_vs_ep60_aggregate.auc_mixed.ep60_this.p_b2000',
          'expect': rd(f'{E16}:ep30_vs_ep60_aggregate.auc_mixed.ep60_this.p_b2000'),
          'note': '3/3 可分保持（60ep 口径稳健；族标注=稳健性抽查未入 Holm）'},
         {'path': f'{E16}:ep30_vs_ep60_aggregate.auc_logon.ep60_this.obs_mean_diff',
          'expect': rd(f'{E16}:ep30_vs_ep60_aggregate.auc_logon.ep60_this.obs_mean_diff'),
          'note': '30ep −0.02081→60ep −0.00741 方向保持幅度≈1/3（R4B-m4③ 勘误后口径）'},
         {'path': f'{E16}:ep30_vs_ep60_aggregate.auc_logon.ep60_this.p_b2000',
          'expect': rd(f'{E16}:ep30_vs_ep60_aggregate.auc_logon.ep60_this.p_b2000'),
          'note': '未检出；红线：未检出≠等效'}]},
    {'n': 'R3-P1-7 E2 full 臂 AP 补算（V16）：新算值无逐位参照（homo 机器核验 3/3 逐位过后方算）',
     'producer': 'src/r3_manifest_append.py（R3/P0-3；expect 现读固化自补算键）',
     'checks': [
         {'path': f'{E2}:regression_check_full_ap_backfill_r3.rows.full_s{seed}.ap_backfilled',
          'expect': rd(f'{E2}:regression_check_full_ap_backfill_r3.rows.full_s{seed}.ap_backfilled')}
         for seed in range(3)]},
]


def main():
    mf = json.load(open(MP))
    by_n = {r['n']: r for r in mf['mappings']}
    added = 0
    for row in ROWS:
        if row['n'] in by_n:
            old = by_n[row['n']]
            assert old.get('checks') == row['checks'], \
                f"manifest 行在案但 expect≠现读：{row['n'][:40]}——交付漂移信号，人工核查"
            print(f"  no-op {row['n'][:48]}…（在案且 expect==现读）")
            continue
        mf['mappings'].append(row)
        added += 1
        print(f"  APPEND {row['n'][:48]}…（{len(row['checks'])} checks）")
    mf['r3_appended_rows_note'] = (
        'R3/P0-3 追加行（只增不改）：5 行 expect 全部由 src/r3_manifest_append.py 现读交付 JSON 固化（禁手抄）；'
        '粒度=JSON 字面值原样不二次舍入（规则先登记于 results/EXP-R3-REG/criteria_r3.json）；'
        '原 21 行原文保留。R2A2-M3「本轮新数字零覆盖」自此闭环。')
    with open(MP, 'w') as f:
        json.dump(mf, f, indent=1, ensure_ascii=False)
    # E14 postmortem（V10：追加方式落盘，自身满足新纪律）
    pc = f'{B}/results/EXP-R2-REG/criteria_e14_tost_producer.json'
    c = json.load(open(pc))
    if POSTMORTEM_KEY not in c:
        c[POSTMORTEM_KEY] = {
            'status': 'APPENDED（R3/P0-3）：预登记判据「number_manifest 增行」至 R2 收口未兑现（R2A2-M3）——'
                      '本句为 postmortem：判据登记了但执行层漏兑现，暴露「登记≠执行」接缝；'
                      'R3 由 src/r3_manifest_append.py 兑现（5 行 expect 现读固化）并以追加方式落盘本注，'
                      '自身满足「修复脚本化+修复即入链」纪律',
            'producer': 'src/r3_manifest_append.py（R3/P0-3）',
            'review_refs': 'R2A2-M3 / 本轮 P0-3'}
        with open(pc, 'w') as f:
            json.dump(c, f, indent=1, ensure_ascii=False)
        print('  APPEND criteria_e14 postmortem_r3_manifest_rows')
    else:
        print('  no-op criteria_e14 postmortem（在案）')
    print(f'R3_MANIFEST_APPEND_DONE added={added} total_rows={len(mf["mappings"])}')


if __name__ == '__main__':
    main()
