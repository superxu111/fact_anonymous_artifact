"""r3_key_producers.py — R3/P0-2+P1-8：追加键生产者化（幂等，只增不改）。

【实验设计自查（判据=results/EXP-R3-REG/criteria_r3.json）】
对象（R2A2-M1/R4B-m3：三叙述键全仓无生产者）：
  ①e9 field_name_note_r3bm1 ②e9 independence_declaration_r2am2 ③e12 robustness_spotcheck_family_note_r2am5
  ——本脚本以字面量定义键值，断言与在案值逐位一致（防篡改），键已在且值一致→不写（末态逐位稳定=幂等）；
  键缺失或值不符→追加/修复并申报。另含 P1-8 两件：E16 首跑 bug fix_note（R2A2-m6 弱留证闭环）、
  91 WARN 三层分层注册表（待登记层 3 项勘误引用值逐条登记，棘轮只许降）、criteria_e17 闸1 自述对齐键。
幂等性判据（V7）：重跑两次，全部目标文件末态逐位一致。
"""
import json, sys

sys.path.insert(0, '/root/autodl-tmp/projB_hetero_graph/src')
B = '/root/autodl-tmp/projB_hetero_graph'
R = f'{B}/results'

E9_NOTE_1 = ('per_seed.auc_mixed_pool_n（=131,235）字段名沿自 mixed 任务复制（e9_auc_logon_agg.py:76），'
             '在 metric=auc_logon 下实为 LOGON 行掩码池的行数——数值正确、名字误导；论文映射时更名 '
             'logon_masked_pool_n 或加注。')
E9_NOTE_2 = ('cross_aggregate_calibers 的 Stouffer 单侧/Fisher 把 3 个协议点的聚合 p 合并，但三点共享同一'
             '评估池（mixed 445,229 / LOGON 行掩码 131,235 逐位相同）与同 3 个训练种子——跨点独立性不成立，'
             'Fisher/Stouffer 口径偏激进（如实申报）；ACAT 对任意相关稳健、且 seed 层聚合置换为主口径，'
             '主结论不受影响；引用三口径交叉时须带本申报。')
E12_NOTE = ('本实验九个 p 值（mixed/AUC/AP×3 seed）为稳健性抽查定位、未入任何 Holm 申报族、不进摘要；'
            'mixed 三 p Bonferroni×3 后 0.024/0.006/0.018 仍全过（抽查可接受性核）；60ep seed 层聚合补齐见 '
            'results/EXP-R2-E16/（R2 第四轮）。')


def ensure(path, key, value, producer, extra=None):
    """幂等追加（所有权语义）：键缺→追加；终态逐位一致→no-op；
    值不符但 producer 字段=本生产者（生产者自有键）→ REFRESH（如对齐键随实现常量演进）；
    值不符且 producer 非本生产者（外来内容）→ FAIL（防篡改静默覆盖）。"""
    d = json.load(open(path))
    final = dict(value) if isinstance(value, dict) else value
    if isinstance(final, dict):
        final['producer'] = producer
        if extra:
            final.update(extra)
    if key in d:
        if d[key] == final:
            print(f'  no-op {path.split("results/")[-1]}:{key}（在案且逐位一致）')
            return 'noop'
        if isinstance(d[key], dict) and d[key].get('producer') == producer:
            d[key] = final
            with open(path, 'w') as f:
                json.dump(d, f, indent=1, ensure_ascii=False)
            print(f'  REFRESH {path.split("results/")[-1]}:{key}（生产者自有键随实现常量演进）')
            return 'refresh'
        raise AssertionError(f'{path}:{key} 在案值与本生产者不一致——人工核查，禁静默覆盖')
    d[key] = final
    with open(path, 'w') as f:
        json.dump(d, f, indent=1, ensure_ascii=False)
    print(f'  APPEND {path.split("results/")[-1]}:{key}')
    return 'appended'


def main():
    print('[1] e9 两键+e12 一键生产者化（R2A2-M1/R4B-m3）')
    p9 = f'{R}/EXP-R2-E9/e9_auc_logon_agg.json'
    cur = json.load(open(p9))
    assert cur['field_name_note_r3bm1'] == E9_NOTE_1, 'e9 note1 在案值漂移'
    assert cur['independence_declaration_r2am2'] == E9_NOTE_2, 'e9 note2 在案值漂移'
    p12 = f'{R}/EXP-R2-E12/e12_closure.json'
    cur12 = json.load(open(p12))
    assert cur12['robustness_spotcheck_family_note_r2am5'] == E12_NOTE, 'e12 note 在案值漂移'
    prod9 = 'src/r3_key_producers.py（R3/P0-2 生产者化；原文=2026-10-24 R2 第四轮追加值，本脚本断言逐位一致）'
    ensure(p9, 'field_name_note_r3bm1', E9_NOTE_1, prod9)
    ensure(p9, 'independence_declaration_r2am2', E9_NOTE_2, prod9)
    ensure(p12, 'robustness_spotcheck_family_note_r2am5', E12_NOTE, prod9)

    print('[2] E16 首跑 bug fix_note（P1-8/R2A2-m6 弱留证闭环）')
    p16 = f'{R}/EXP-R2-E16/e12_60ep_seed_aggregate.json'
    fix16 = {
        'status': 'APPENDED（R3/P1-8）：首跑 log 未留档（logs/e16_seed_agg.log 仅 392 字节=最终成功运行，'
                  '首跑留痕被覆盖），以本注为准',
        'first_run_bug': '首跑 ap_logon 口径误传 metric=\'auc\'（聚合 obs 与 AUC 同值暴露），修复后重跑',
        'evidence_note': '首跑数值未交付未引用；修复重跑的逐 seed obs 与 E12 交付逐位一致（管线零漂移断言在案）；'
                         '本注为 R2A2-m6 弱留证申报的登记层闭环',
        'review_refs': 'R2A2-m6 / 本轮 P1-8'}
    ensure(p16, 'fix_note_r2a2m6', fix16,
           'src/r3_key_producers.py（R3/P1-8）')

    print('[3] criteria_e17 闸1 自述对齐键（P0-2/V8：自述=实现机器比对，常量自实现现读）')
    from verify_provenance import GATE1_CONSTANTS
    pc17 = f'{R}/EXP-R2-E17/criteria_e17.json'
    align = {
        'status': 'APPENDED（R3/P0-2）：本键为闸1 键级扫描 R3 重设计后的自述对齐声明——原 design 键'
                  '「扫全部键名」自述强于 R2 实现（R4B-m3），本键声明与 R3 实现常量逐项一致，'
                  '单测 src/test_r3_gate1_keyscan.py 机器比对',
        'supersedes_scope': '仅键级扫描语义；文件级扫描与闸2/闸3 语义不变',
        'constants': GATE1_CONSTANTS,
        'unit_test': 'src/test_r3_gate1_keyscan.py',
        'review_refs': 'R4B-m3 / R2A2-M1 / 本轮 P0-2'}
    ensure(pc17, 'gate1_alignment_r3', align,
           'src/r3_key_producers.py（R3/P0-2，常量自 verify_provenance.GATE1_CONSTANTS 现读）')

    print('[4] 91 WARN 三层分层注册表（P1-8/V18：EXEMPT/引述性/待登记；棘轮=待登记只许降不许升）')
    from verify_provenance import stratify_warns  # 单一实现（与 gate2 同口径，防双口径接缝）
    _, l_exempt, l_quote, l_register = stratify_warns()
    resolved = [{'file': 'finalize_r3.py', 'literal': '1.3122',
                 'reason': '勘误引用值（更正注内被更正的错误值原文：原字符串硬编码 arl100 κ 误植，'
                           '真值 1.3124 已入 manifest 行 E11 κ；禁进正文，仅存于更正引文）'},
                {'file': 'finalize_r3.py', 'literal': '1.2785',
                 'reason': '勘误引用值（更正注内被更正的错误值原文：原 arl500 κ 误植值，'
                           '真值 1.4616 已入 manifest；闸2 κ 断言的上下文豁免对象）'},
                {'file': 'finalize_r3.py', 'literal': '3.2905',
                 'reason': '勘误引用值（更正注内 z_{1-1/2000}=3.2905 系误植分母出处说明，非任何档位执行值）'}]
    reg = {'exp': 'R3/P1-8：finalize*.py 浮点字面量三层分层注册表',
           'producer': 'src/r3_key_producers.py[4]（R3/P1-8；闸2 verify_provenance.gate2 同规则现算比对）',
           'layer_rule': 'L_EXEMPT=结构常量（解析真值 1.000）；L_QUOTE=引述性（数值按字面量自身精度舍入后'
                         '精确出现于 results/** 交付 JSON，机器判定）；L_REGISTER=待登记（前两者皆非）',
           'ratchet': 'L_REGISTER 棘轮：只许降不许升——闸2 现算待登记项须全部见于 l_register_resolved，'
                      '新增未登记项即 FAIL；本表由 R3 起为棘轮基线',
           'counts': {'warn_total': len(l_exempt) + len(l_quote) + len(l_register),
                      'l_exempt': len(l_exempt), 'l_quote': len(l_quote), 'l_register': len(l_register)},
           'l_exempt': l_exempt, 'l_quote_count_by_file': l_quote and {} or {},
           'l_register_resolved': resolved,
           'l_register_count_by_file': {}}
    from collections import Counter
    reg['l_quote_count_by_file'] = dict(Counter(x['file'] for x in l_quote))
    regp = f'{R}/EXP-R3-REG/warn_stratification_r3.json'
    if __import__('os').path.exists(regp):
        old = json.load(open(regp))
        if len(old.get('l_register_resolved', [])) > len(resolved):
            raise AssertionError('棘轮违规：待登记已解决清单不许增长回退')
    with open(regp, 'w') as f:
        json.dump(reg, f, indent=1, ensure_ascii=False)
    print(f"  registry written: EXEMPT {reg['counts']['l_exempt']} / QUOTE {reg['counts']['l_quote']}"
          f" / REGISTER {reg['counts']['l_register']}（resolved {len(resolved)}）")
    assert reg['counts']['l_register'] == len(resolved), '待登记层非空且未逐条登记'
    print('R3_KEY_PRODUCERS_DONE')


if __name__ == '__main__':
    main()
