"""r3_gate1_twophase.py — R3/P0-2：闸1 键级扫描「先升后降」两阶段留档（V5）。

阶段1=新剥数字规则 × R2 时点 src（剔除本轮 r3_* 生产者脚本，复现「三叙述键无生产者」的 R2 现场）
  → 预期键级孤儿含三叙述键（升）；
阶段2=新规则 × 当前全量 src（含生产者）→ 预期 0（降）。
两阶段输出均落 results/EXP-R3-REG/keyscan_two_phase_r3.json——直接报 0 而无阶段1 留档=可疑（判据明文）。
纯 CPU 秒级；只读 results/，不触碰任何产物。
"""
import json, sys, glob, os

sys.path.insert(0, '/root/autodl-tmp/projB_hetero_graph/src')
import verify_provenance as vp

B = vp.BASE


def keyscan(src_all):
    """与 verify_provenance.gate1 键级部分同式的参数化重放（src_all 可换快照）。"""
    canon_src = vp.canon(src_all)
    orphans = []
    files_scanned = 0
    for p in sorted(glob.glob(f'{B}/results/**/*.json', recursive=True)):
        stem = os.path.splitext(os.path.basename(p))[0]
        if any(stem.startswith(x) or stem == x for x in vp.FILE_EXEMPT) or stem.startswith('criteria'):
            continue
        try:
            d = json.load(open(p))
        except Exception:
            continue
        files_scanned += 1

        def walk(o, depth=0, rel=''):
            if depth > vp.GATE1_CONSTANTS['key_scan_depth_max'] or not isinstance(o, dict):
                return
            for k, v in o.items():
                if isinstance(k, str) and len(k) > vp.GATE1_CONSTANTS['min_key_len'] \
                        and k not in vp.GENERIC_KEYS and not k.startswith('_'):
                    if not vp.key_sourced(k, src_all, canon_src):
                        orphans.append({'file': rel, 'key': k[:80]})
                walk(v, depth + 1, rel)
        walk(d, 0, os.path.relpath(p, f'{B}/results'))
    return orphans, files_scanned


def src_snapshot(exclude_r3):
    """阶段1=R2 时点（再剔除 r3_* 生产者）；阶段2=当前 gate1 同源快照（元脚本自排除）。"""
    if not exclude_r3:
        return vp.gate1_src_all()
    parts = []
    for p in sorted(glob.glob(f'{B}/src/*.py')):
        b = os.path.basename(p)
        if b.startswith('r3_') or b.startswith('test_r3'):
            continue
        parts.append(open(p, encoding='utf-8', errors='ignore').read())
    return ''.join(parts)


def main():
    ph1_orphans, n1 = keyscan(src_snapshot(exclude_r3=True))
    ph2_orphans, n2 = keyscan(src_snapshot(exclude_r3=False))
    flagged = {o['key'] for o in ph1_orphans}
    three = {'field_name_note_r3bm1', 'independence_declaration_r2am2',
             'robustness_spotcheck_family_note_r2am5'}
    assert three <= flagged, f'阶段1 未复现三叙述键孤儿现场: {three & flagged}'
    # 阶段2 处置后残余：未命中豁免登记表的键
    regp = f'{B}/results/EXP-R3-REG/keyscan_exemption_registry_r3.json'
    entries = json.load(open(regp))['entries'] if os.path.exists(regp) else []
    import re as _re
    ph2_dispositioned = []
    for o in ph2_orphans:
        if not any(_re.match(e['pattern'], o['key'])
                   and (not e['applies_to'] or o['file'] in e['applies_to']) for e in entries):
            ph2_dispositioned.append(o)
    out = {'exp': 'R3/P0-2：闸1 键级扫描范围扩 results/**+剥数字豁免规则——两阶段（先升后降）留档',
           'producer': 'src/r3_gate1_twophase.py（R3/P0-2；V5：直接报 0 可疑，故阶段1 留档）',
           'phase1_r2_src_snapshot': {
               'definition': '新规则 × R2 时点 src（剔除 r3_* 生产者与 test_r3_* 测试=三键尚无生产者的现场复现；'
                             '测试/留档脚本自身的键名讨论字面量不构成生产者，同予剔除）',
               'files_scanned': n1, 'orphan_count': len(ph1_orphans),
               'three_narrative_keys_flagged': sorted(three & flagged),
               'other_orphans_sample': [o for o in ph1_orphans if o['key'] not in three][:12],
               'other_orphans_count': len([o for o in ph1_orphans if o['key'] not in three])},
           'phase2_current_src': {
               'definition': '新规则 × 当前 gate1 同源快照（元脚本自排除）= E17 复跑闸1 现场',
               'files_scanned': n2, 'orphan_count_raw_before_disposition': len(ph2_orphans),
               'orphans_raw': ph2_orphans,
               'disposition': '类级豁免登记表（results/EXP-R3-REG/keyscan_exemption_registry_r3.json）处置',
               'orphan_count_after_disposition': len(ph2_dispositioned),
               'orphans_after_disposition': ph2_dispositioned},
           'verdict': '先升后降成立' if (ph1_orphans and not ph2_dispositioned) else 'PHASE2_NOT_CLEAN',
           'review_refs': 'V5 / R4B-m3 / R2A2-M1 / 本轮 P0-2'}
    with open(f'{B}/results/EXP-R3-REG/keyscan_two_phase_r3.json', 'w') as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    print(f"阶段1（R2 src）孤儿 {len(ph1_orphans)}（含三叙述键={bool(three & flagged)}）；"
          f"阶段2（当前 src）处置前 {len(ph2_orphans)} / 登记表处置后 {len(ph2_dispositioned)}")
    print('R3_TWOPHASE_DONE', out['verdict'])


if __name__ == '__main__':
    main()
