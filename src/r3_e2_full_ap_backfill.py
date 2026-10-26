"""r3_e2_full_ap_backfill.py — R3/P1-7：E2 full 臂 in_ap_logon=null 补算（R1A-m7 连续两轮悬空断根）。

【实验设计自查（判据先行，见 results/EXP-R3-REG/criteria_r3.json:P1-7_E2_full_ap_backfill）】
null_class=not_computed（在案参照缺失）；机器=与 homo 臂在案值同机器同式（e2_closure.metrics_from_npz：
sklearn average_precision_score @ rel=='LOGON' 行掩码，同一批 uniform1_*_60ep npz）。
机器核验（口径可比性门）：先重算 homo 3 seed，round(·,4) 须与在案 0.0084/0.0168/0.0056 逐位一致，
不一致即停不硬补。full 臂无在案参照值→补算值如实申报「新算值、无逐位参照」，禁冒称复现校验（V16）。
落盘=e2_closure.json 顶层追加键 regression_check_full_ap_backfill_r3（原 null 行不改写，只增不改）。
幂等：同 npz 重算值确定，重跑末态逐位一致。纯 CPU 秒级。
"""
import json, sys

sys.path.insert(0, '/root/autodl-tmp/projB_hetero_graph/src')
from e2_closure import metrics_from_npz, BASE, INCASE  # noqa: E402

OUTP = f'{BASE}/e2_closure.json'


def main():
    e2 = json.load(open(OUTP))
    rows = {}
    # ---- 机器核验（homo 臂：在案值逐位）----
    homo_ok = True
    for seed in range(3):
        m = metrics_from_npz(f'{BASE}/uniform1_homo_60ep', seed)
        ap4 = round(m['ap_logon'], 4)
        ref = INCASE['homo']['ap_logon'][seed]
        ok = ap4 == ref
        homo_ok &= ok
        rows[f'homo_s{seed}'] = {'ap_recomputed': ap4, 'incase_reference': ref,
                                 'machine_check': 'PASS' if ok else f'FAIL（{ap4}!={ref}）'}
        print(f'机器核验 homo_s{seed}: 重算 {ap4} vs 在案 {ref} -> {"PASS" if ok else "FAIL"}')
    assert homo_ok, 'homo 机器核验失败——口径不可比，停（不硬补）'
    # ---- full 臂补算（新算值，无逐位参照）----
    for seed in range(3):
        m = metrics_from_npz(f'{BASE}/uniform1_full_60ep', seed)
        ap4 = round(m['ap_logon'], 4)
        rows[f'full_s{seed}'] = {
            'ap_backfilled': ap4, 'incase_reference': None,
            'status': 'NEW_VALUE（新算值、无逐位参照——在案交付无 full 臂 AP 参照值，'
                      '本值为同机器同 npz 补算，非复现校验，V16 如实申报）',
            'n_pool_logon': m['n_pool_logon'], 'n_pos_logon': m['n_pos_logon']}
        print(f'full_s{seed}: AP(补算)={ap4}（LOGON 行 {m["n_pool_logon"]}，正例 {m["n_pos_logon"]}）')
    key = 'regression_check_full_ap_backfill_r3'
    e2[key] = {
        'status': 'APPENDED（R3/P1-7：只增不改——regression_check.full[*].in_ap_logon=null 原行不改写，'
                  '本键为补算登记；R1A-m7 连续两轮悬空在此断根，null_class=not_computed）',
        'producer': 'src/r3_e2_full_ap_backfill.py（R3/P1-7）；判据=results/EXP-R3-REG/criteria_r3.json',
        'machine_caliber': 'sklearn average_precision_score @ rel==LOGON 掩码（=src/e2_closure.py metrics_from_npz 同式；'
                           '与 homo 臂在案值同机器同 npz）——homo 3 seed 机器核验 PASS 后方算 full',
        'rows': rows,
        'review_refs': 'R1A-m7 / R2A2-m4 / 本轮 P1-7'}
    with open(OUTP, 'w') as f:
        json.dump(e2, f, indent=1, ensure_ascii=False)
    # NaN/形状断言（防白做）
    for s in range(3):
        v = rows[f'full_s{s}']['ap_backfilled']
        assert v == v and 0.0 < v < 1.0, f'full_s{s} AP 量级异常: {v}'
    print('R3_P1_7_DONE e2_closure.json 追加键落盘')


if __name__ == '__main__':
    main()
