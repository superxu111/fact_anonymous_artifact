"""e13_delay_rule_fix.py — R2-E13（R2A-M2/R3B-M1 响应）：e4 延迟规则方向修正重生成。

【实验设计自查（开工前冻结，判据见 results/EXP-R2-E13/criteria.json）】
修正对象：src/r2_reg_layer.py::build_e4_fair_row_replay 的 delay 分支方向（旧版复用 det 的 va>vb
  比较符且 None→−1e9，把更晚告警判为更好，与声明「先告警者 better，None=最差」相反）。
小样本自测（先于全量重生成）：对 arl100/arl500 全 6 对手算预期逐格断言——
  arl100 delay 正确行 = full_vs_homoflows [tie,second_better,first_better] /
    full_vs_homo [tie,first_better,first_better] / homoflows_vs_homo [tie,first_better,tie]；
  arl500 = [tie,tie,tie] / [tie,first_better,tie] / [tie,first_better,tie]。
预登记预期：修正 vs 旧版互换恰 5 格（全在 arl100）；真实分歧（delay vs det 主规则）=4 对 8 格
  （arl100 2 对 3 格 + arl500 2 对 5 格）；不变量=per_arm_seed 逐位、det 主规则行逐位。
成功判据：断言全过→重生成 e4_window_row_fair_replay.json（原错误方向行保留于 superseded 键，只增不改）。
规模/成本：纯 CPU 秒级。
"""
import json, sys

sys.path.insert(0, '/root/autodl-tmp/projB_hetero_graph/src')
import r2_reg_layer as R

EXPECT_DELAY = {
    'arl100': {'full_vs_homoflows': ['tie', 'second_better', 'first_better'],
               'full_vs_homo': ['tie', 'first_better', 'first_better'],
               'homoflows_vs_homo': ['tie', 'first_better', 'tie']},
    'arl500': {'full_vs_homoflows': ['tie', 'tie', 'tie'],
               'full_vs_homo': ['tie', 'first_better', 'tie'],
               'homoflows_vs_homo': ['tie', 'first_better', 'tie']},
}
EXPECT_FLIP_CELLS = 5
EXPECT_DIFF = {'arl100': {'full_vs_homoflows': 2, 'full_vs_homo': 0, 'homoflows_vs_homo': 1},
               'arl500': {'full_vs_homoflows': 3, 'full_vs_homo': 2, 'homoflows_vs_homo': 0}}


def main():
    old = json.load(open(f'{R.OUT}/e4_window_row_fair_replay.json'))  # 旧版（方向错误）留档
    # ---- 小样本自测：直接在内存里跑一次重放逻辑，断言手算预期 ----
    e5 = json.load(open(f'{R.BASE}/results/EXP-R1-E5/e5_evt_calib.json'))
    e4 = json.load(open(f'{R.BASE}/results/EXP-R1-E4/e4_granularity.json'))
    incase = e4['r2_repairs']['granularity_matrix_window_row_fair']['fair_window_row']
    for arl in ('arl100', 'arl500'):
        rows = e5['g3_h1_reverdict'][arl]['rows']
        pas = {f"{r['arm']}_s{r['seed']}": r['fixed_evt']['delay'] for r in rows}
        pairs = [('full', 'homoflows'), ('full', 'homo'), ('homoflows', 'homo')]
        for a, b in pairs:
            got = []
            for s in range(3):
                va = pas[f'{a}_s{s}'] if pas[f'{a}_s{s}'] is not None else 10 ** 9
                vb = pas[f'{b}_s{s}'] if pas[f'{b}_s{s}'] is not None else 10 ** 9
                got.append('tie' if va == vb else ('first_better' if va < vb else 'second_better'))
            exp = EXPECT_DELAY[arl][f'{a}_vs_{b}']
            assert got == exp, f'小样本自测失败 {arl} {a}_vs_{b}: {got} != {exp}'
            inc = incase[arl]['pair_directions'][f'{a}_vs_{b}']
            ndiff = sum(1 for i in range(3) if got[i] != inc[i])
            assert ndiff == EXPECT_DIFF[arl][f'{a}_vs_{b}'], \
                f'分歧数不符 {arl} {a}_vs_{b}: {ndiff}'
            # 旧版 vs 正确版互换格数
            oldv = old['replay'][arl]['pair_directions_delay_rule_sensitivity'][f'{a}_vs_{b}']
            nflip = sum(1 for i in range(3) if oldv[i] != got[i] and oldv[i] != 'tie' and got[i] != 'tie')
            assert nflip == sum(1 for i in range(3) if oldv[i] != got[i]), 'tie 格受方向影响？'
            print(f'  自测过 {arl} {a}_vs_{b}: {got}（旧版 {oldv}，互换 {nflip} 格，vs 主规则分歧 {ndiff} 格）')
    total_flip = sum(1 for arl in ('arl100', 'arl500') for a, b in
                     [('full', 'homoflows'), ('full', 'homo'), ('homoflows', 'homo')]
                     for i in range(3)
                     if old['replay'][arl]['pair_directions_delay_rule_sensitivity'][f'{a}_vs_{b}'][i]
                     != EXPECT_DELAY[arl][f'{a}_vs_{b}'][i])
    assert total_flip == EXPECT_FLIP_CELLS, f'互换格数 {total_flip} != 预登记 {EXPECT_FLIP_CELLS}'
    print(f'小样本自测全过：互换 {total_flip} 格（预登记 5）、真实分歧 4 对 8 格（arl100 3 + arl500 5）')
    # ---- 全量重生成（build_e4_fair_row_replay 内部会读旧文件进 superseded 键）----
    R.build_e4_fair_row_replay()
    new = json.load(open(f'{R.OUT}/e4_window_row_fair_replay.json'))
    assert new['bitwise_match_per_arm_seed'] is True, 'per_arm_seed 不变量破坏'
    assert new['pair_directions_match_under_det_rate_rule'] is True, 'det 主规则行不变量破坏'
    for arl in ('arl100', 'arl500'):
        for pair, exp in EXPECT_DELAY[arl].items():
            assert new['replay'][arl]['pair_directions_delay_rule_sensitivity'][pair] == exp
    assert new['delay_rule_divergence_count']['pairs'] == 4 and new['delay_rule_divergence_count']['cells'] == 8, \
        new['delay_rule_divergence_count']
    assert 'superseded_r2am2_r3bm1' in new and new['superseded_r2am2_r3bm1']['old_delay_rule_note']
    print('E13_DONE 重生成通过全部不变量断言；superseded 键在位；分歧计数 pairs=4 cells=8')


if __name__ == '__main__':
    main()
