"""r3_keyscan_exemptions.py — R3/P0-2：闸1 键级孤儿「类级豁免登记表」（R2 设计语义的处置层落地）。

背景：闸1 扫描扩 results/** 后，动态插值键/数据实体键/叙述节键共 50 键无法以字符串匹配判定生产者
（键名含运行时 f-string 插值或来自数据本身）。criteria_e17 原设计即「无写入点进 ORPHAN 清单并给处置
（历史件豁免登记）」——本脚本把处置机器化：每条豁免=正则模式+src 模板证据（写入点行）+适用文件+样例，
证据不在 src 即拒登；gate1 逐类计数上报（可见非静默）。未匹配任何模式的键仍报孤儿（bad-key 型防线不变）。
幂等：模板证据现查 src，重跑末态逐位一致。纯 CPU 秒级。
"""
import json, re, sys, glob, os

sys.path.insert(0, '/root/autodl-tmp/projB_hetero_graph/src')
B = '/root/autodl-tmp/projB_hetero_graph'
OUTP = f'{B}/results/EXP-R3-REG/keyscan_exemption_registry_r3.json'

ENTRIES = [
    {'class': 'grid_composite',
     'pattern': r'^(uniform|temporal)_(1|4|16)_(full|homo|homoflows|noflows)_s[0-2]$',
     'evidence_file': 'src/e2_closure.py', 'evidence_line': 114,
     'evidence_fragment': "grid[f'{strat}_{ratio}_{arm}_s{seed}']",
     'reason': '协议×比率×臂×seed 四变量 f-string 动态组装键（字符串匹配不可见插值；写入点在案）',
     'applies_to': ['EXP-R1-E2/e2_closure.json'],
     'example_keys': ['uniform_1_full_s0', 'temporal_16_homo_s2']},
    {'class': 'pair_composite',
     'pattern': r'^(full|homo|homoflows|noflows)(_rerun)?_vs_(full|homo|homoflows|noflows)(_rerun)?(_edgeAP_logon|_eventAP_weighted)?$',
     'evidence_file': 'src/e4_granularity.py', 'evidence_line': 141,
     'evidence_fragment': "col = f'{arm_a}_vs_{arm_b}'",
     'evidence_extra': ['src/e6_inference.py:269 f\'{a}_vs_{b}_edgeAP_logon\'',
                        'src/e7_e4_repair.py:184 f\'{a}_vs_{b}_eventAP_weighted\''],
     'reason': '臂对比较 f-string 组装键（三处写入点在案）',
     'applies_to': ['EXP-R1-E4/e4_granularity.json', 'EXP-R1-E6/e6_inference.json',
                    'EXP-R1-E7/e7_e4_repair.json'],
     'example_keys': ['full_vs_homo_rerun', 'full_vs_homo_rerun_edgeAP_logon']},
    {'class': 'data_entity',
     'pattern': r'^U\d+@DOM\d+$',
     'evidence_file': 'src/e4_granularity.py', 'evidence_line': 49,
     'evidence_fragment': "'dropped_users': dict(rt_user_drop)",
     'reason': 'LANL 数据实体标识（用户@域）直接作键——键源自数据非代码字面量（dropped_users 计数字典）',
     'applies_to': ['EXP-R1-E4/e4_granularity.json'],
     'example_keys': ['U9763@DOM1', 'U737@DOM1']},
    {'class': 'narrative_section',
     'pattern': r'^(baselines|instrument|estimand)$',
     'evidence_file': '—', 'evidence_line': 0, 'evidence_fragment': '—',
     'reason': '纯叙述节键（无数字、无生产者属设计使然；与 R2 首轮审计「叙述性键豁免登记」同类处置，'
               '逐键列名登记而非模式泛化）',
     'applies_to': ['EXP-R1-E4/e4_granularity.json', 'EXP-R1-E5/e5_evt_calib.json',
                    'EXP-R1-E6/e6_inference.json'],
     'example_keys': ['baselines', 'instrument', 'estimand']},
]


def main():
    src_cache = {}

    def src_of(rel):
        if rel not in src_cache:
            src_cache[rel] = open(f'{B}/{rel}', encoding='utf-8', errors='ignore').read()
        return src_cache[rel]
    for e in ENTRIES:
        if e['evidence_file'] != '—':
            lines = src_of(e['evidence_file']).splitlines()
            frag = e['evidence_fragment']
            hit = any(frag in l for l in lines)
            assert hit, f"模板证据不在 src：{e['evidence_file']} 找不到 {frag!r}——拒登"
    # 现扫 PH2 残余，逐键断言可被某类覆盖且文件适用（src 快照=gate1 同源构造，元脚本自排除）
    import verify_provenance as vp
    src_all = vp.gate1_src_all()
    canon_src = vp.canon(src_all)
    reg = {'exp': 'R3/P0-2：闸1 键级豁免登记表（类级处置：模式+模板证据+适用文件；未匹配键仍报孤儿）',
           'producer': 'src/r3_keyscan_exemptions.py（R3/P0-2；gate1 运行时加载逐类计数上报）',
           'entries': ENTRIES}
    covered, uncovered = {}, []
    for p in sorted(glob.glob(f'{B}/results/**/*.json', recursive=True)):
        rel = os.path.relpath(p, f'{B}/results')
        stem = os.path.splitext(os.path.basename(p))[0]
        if any(stem.startswith(x) or stem == x for x in vp.FILE_EXEMPT) or stem.startswith('criteria'):
            continue
        try:
            d = json.load(open(p))
        except Exception:
            continue

        def walk(o, depth=0):
            if depth > vp.GATE1_CONSTANTS['key_scan_depth_max'] or not isinstance(o, dict):
                return
            for k, v in o.items():
                if isinstance(k, str) and len(k) > vp.GATE1_CONSTANTS['min_key_len'] \
                        and k not in vp.GENERIC_KEYS and not k.startswith('_') \
                        and not vp.key_sourced(k, src_all, canon_src):
                    hit_cls = None
                    for e in ENTRIES:
                        if re.match(e['pattern'], k) and (not e['applies_to'] or rel in e['applies_to']):
                            hit_cls = e['class']
                            break
                    if hit_cls:
                        covered[f'{rel}:{k}'] = hit_cls
                    else:
                        uncovered.append({'file': rel, 'key': k[:80]})
                walk(v, depth + 1)
        walk(d)
    from collections import Counter
    reg['covered_keys'] = len(covered)
    reg['covered_by_class'] = dict(Counter(covered.values()))
    reg['uncovered_after_registry'] = uncovered
    with open(OUTP, 'w') as f:
        json.dump(reg, f, indent=1, ensure_ascii=False)
    print(f"豁免登记表：覆盖 {reg['covered_keys']} 键（{reg['covered_by_class']}），登记后未覆盖残余 {len(uncovered)}")
    for u in uncovered:
        print('  UNCOVERED:', u)
    assert not uncovered, '登记后仍有未覆盖孤儿——补登记或修键'
    print('R3_KEYSCAN_EXEMPTIONS_DONE')


if __name__ == '__main__':
    main()
