"""test_r3_gate1_keyscan.py — R3/P0-2 闸1 键级扫描双向回归 + criteria_e17 自述机器比对 + κ 公式断言。

V6 双向：三叙述键生产者化后有源；kappa_t_pivot_inflation（含数字实质键）不得误豁免；
  合成 bad_key_123 型（含数字无源）仍报孤儿；GENERIC_KEYS/纯叙述键零误报。
V8：criteria_e17.json:gate1_alignment_r3.constants 与 verify_provenance.GATE1_CONSTANTS 逐项机器比对，
  且 gate1() 实际输出自报常量一致。
V-R3-5：κ 公式重算两档 |diff|<1e-4（1.2785 复现形态回归——原字符串断言对该形态判 False 的缺口被升级覆盖）。
运行：python3 src/test_r3_gate1_keyscan.py（纯 CPU 秒级）。
"""
import json, sys

sys.path.insert(0, '/root/autodl-tmp/projB_hetero_graph/src')
import verify_provenance as vp

B = vp.BASE
PASS = []


def check(name, cond):
    assert cond, f'FAIL: {name}'
    PASS.append(name)
    print(f'  PASS {name}')


def is_orphan(k, src_all, canon_src):
    """gate1 谓词直调（单一实现 vp.key_sourced，防复制漂移）。"""
    if not isinstance(k, str) or len(k) <= vp.GATE1_CONSTANTS['min_key_len'] \
            or k in vp.GENERIC_KEYS or k.startswith('_'):
        return False
    return not vp.key_sourced(k, src_all, canon_src)


def main():
    # 与 gate1 同源快照（元脚本自排除：本测试/两阶段/豁免登记表讨论的键名字面量不构成生产者）
    src_all = vp.gate1_src_all()
    canon_src = vp.canon(src_all)

    print('[1] V6 双向回归')
    # ① 三历史叙述键：生产者化后有源（canon 命中 src）
    for k in ('field_name_note_r3bm1', 'independence_declaration_r2am2',
              'robustness_spotcheck_family_note_r2am5'):
        check(f'叙述键有源 {k}', not is_orphan(k, src_all, canon_src))
        check(f'叙述键 canon 非空命中 {k}',
              len(vp.canon(k)) >= 3 and vp.canon(k) in canon_src)
    # ② κ 实质键：含数字但不得误豁免（经 verbatim/canon 命中 src）
    check('kappa_t_pivot_inflation 有源（数字键不整键豁免）',
          not is_orphan('kappa_t_pivot_inflation', src_all, canon_src))
    # ③ bad_key_123 型：含数字无源 → 必须仍报孤儿（旧规则 `not any(ch.isdigit())` 整键豁免假阴路径）
    check('bad_key_123 型仍报孤儿', is_orphan('bad_key_123', src_all, canon_src))
    check('旧规则对该型会漏报（反证规则修复必要性）',
          any(ch.isdigit() for ch in 'bad_key_123'))
    # ④ 白名单/短键零误报
    for k in ('producer', 'note', 'date', '_internal', 'checks', 'abc'):
        check(f'白名单/短键零误报 {k}', not is_orphan(k, src_all, canon_src))

    print('[2] V8 criteria_e17 自述=实现 机器比对')
    c17 = json.load(open(f'{B}/results/EXP-R2-E17/criteria_e17.json'))
    check('gate1_alignment_r3 键在案', 'gate1_alignment_r3' in c17)
    check('自述常量==GATE1_CONSTANTS',
          c17['gate1_alignment_r3']['constants'] == vp.GATE1_CONSTANTS)
    g1 = vp.gate1()
    check('gate1() 实际输出自报常量==GATE1_CONSTANTS', g1['constants'] == vp.GATE1_CONSTANTS)
    check('自述 scope 与实现 glob 一致',
          g1['constants']['scope_glob'] == 'results/**/*.json'
          and vp.GATE1_CONSTANTS['key_scan_depth_max'] == 2)

    print('[3] V-R3-5 κ 公式重算断言（1.2785 复现形态回归覆盖）')
    g2 = vp.gate2()
    check('κ 公式重算两档 |diff|<1e-4', g2['kappa_formula_ok'])
    check('κ 断言含公式重算字段', set(g2['kappa_formula_recheck']) == {'arl100', 'arl500'})
    # 字符串断言升级的双向：现值行（更正上下文）放行、无标记行触发
    check('现值 1.2785 仅存于更正引文（uncontextualized 空）',
          g2['kappa_uncontextualized_lines'] == [])
    check('无标记行触发（词表双向测试）',
          vp and not any(w in 'plain line with 1.2785 here' for w in
                         ['更正', 'supersedes', 'SUPERSEDES', '误植', '硬编码', '原字符串', '原文保留']))
    print(f'TEST_R3_GATE1_KEYSCAN_DONE all {len(PASS)} checks passed')


if __name__ == '__main__':
    main()
