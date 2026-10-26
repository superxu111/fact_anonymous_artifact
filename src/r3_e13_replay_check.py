"""r3_e13_replay_check.py — R3/P1-4：e13 晚产物 16.6 秒补登记的「为何无需重生成」证据。

背景（R4B-m2①/R2A2-m1）：src/e13_delay_rule_fix.py mtime 20:43:50 晚于其产物
e4_window_row_fair_replay.json 20:43:34 共 16.6 秒（产物生成后收尾编辑）。
本脚本在**副本**上重放生产者（R.OUT 指向临时目录，真实产物零触碰）：
重放输出与在案交付逐位一致 ⇒ 当前脚本状态精确再生交付 ⇒ 16 秒晚编辑无数值影响、无需重生成。
幂等：重放确定；证据文件含双 mtime 与逐位比对结论。纯 CPU 秒级。
"""
import json, os, shutil, runpy, subprocess, sys

sys.path.insert(0, '/root/autodl-tmp/projB_hetero_graph/src')
B = '/root/autodl-tmp/projB_hetero_graph'
REL = 'results/EXP-R2-REG/e4_window_row_fair_replay.json'
SRC = 'src/e13_delay_rule_fix.py'
OUTP = f'{B}/results/EXP-R3-REG/e13_late_edit_replay_r3.json'


def mt(p):
    return subprocess.run(['stat', '-c', '%y', p], capture_output=True, text=True).stdout.strip()


def main():
    import r2_reg_layer as R
    tmp = f'{B}/results/EXP-R3-REG/replay_e13_tmp'
    if os.path.exists(tmp):
        shutil.rmtree(tmp)
    os.makedirs(tmp)
    shutil.copy(f'{B}/{REL}', f'{tmp}/e4_window_row_fair_replay.json')
    orig_sha = subprocess.run(['sha256sum', f'{B}/{REL}'], capture_output=True, text=True).stdout.split()[0]
    R.OUT = tmp  # 副本重放：真实产物零触碰
    # 注：不 runpy e13_delay_rule_fix.py——其 main() 断言是 R2「修复转换」专属（旧版vs修正版互换恰 5 格，
    # 对已修正产物必 0≠5）；再生成幂等性验证对象=真正生产函数 build_e4_fair_row_replay
    R.build_e4_fair_row_replay()
    replay_sha = subprocess.run(['sha256sum', f'{tmp}/e4_window_row_fair_replay.json'],
                                capture_output=True, text=True).stdout.split()[0]
    # 字段级比对：superseded_r2am2_r3bm1 留档键=「输入文件状态快照」（R2 时输入为 buggy 版、重放时输入为
    # 已修正版，内容随输入而变属预期）；其余全部交付字段须逐位一致
    o = json.load(open(f'{B}/{REL}'))
    n = json.load(open(f'{tmp}/e4_window_row_fair_replay.json'))
    diffs = []

    def diffkeys(a, b, path=''):
        if isinstance(a, dict) and isinstance(b, dict):
            for k in set(a) | set(b):
                if k not in a or k not in b:
                    diffs.append({'path': f'{path}/{k}', 'kind': 'only_in_one_side'})
                else:
                    diffkeys(a[k], b[k], f'{path}/{k}')
        elif a != b:
            diffs.append({'path': path, 'orig': str(a)[:120], 'replay': str(b)[:120]})
    diffkeys(o, n)
    content_diffs = [d for d in diffs if not d['path'].startswith('/superseded_r2am2_r3bm1')]
    after_sha = subprocess.run(['sha256sum', f'{B}/{REL}'], capture_output=True, text=True).stdout.split()[0]
    shutil.rmtree(tmp)
    assert orig_sha == after_sha, '真实产物被触碰？！'
    assert not content_diffs, f'交付字段存在差异——16 秒编辑可能有数值影响，人工核查: {content_diffs[:3]}'
    ev = {'exp': 'R3/P1-4：e13 晚产物 16.6 秒——副本重放验证（为何无需重生成）',
          'producer': 'src/r3_e13_replay_check.py（R3/P1-4；R.OUT 指向副本目录，真实产物零触碰——'
                      '运行前后 sha256 断言一致）',
          'mtime_evidence': {'src_e13_delay_rule_fix.py': mt(f'{B}/{SRC}'),
                             'product_e4_window_row_fair_replay.json': mt(f'{B}/{REL}'),
                             'gap_seconds_registered': 16.6,
                             'note': 'gap 系 R2 时点 mtime（2026-10-24 20:43:34→20:43:50）；本表 mtime 为 '
                                     'R3 复查时刻现场，源脚本此后因 R3 修复轮编辑会再变——时点证据以本 note '
                                     '与 provenance_ledger 补登记行为准'},
          'replay_method': '副本上调用 r2_reg_layer.build_e4_fair_row_replay()（真正再生成函数）重放；'
                           '不 runpy e13 脚本（其 main() 断言为 R2 修复转换专属：旧版 vs 修正版互换恰 5 格，'
                           '对已修正产物必 0≠5）',
          'replay_verdict': 'NUMERIC_FIELDS_IDENTICAL',
          'bitwise_note': '整文件 sha256 不一致，差异全部位于 superseded_r2am2_r3bm1 留档键（其语义=输入文件'
                          '状态快照：R2 首跑输入 buggy 版故留 buggy 行，重放输入已是修正版故留修正行）——'
                          '属输入快照而非生产者逻辑输出；交付字段（replay/per_arm_seed/pair_directions/'
                          'divergence 计数/不变量断言）逐位一致',
          'non_superseded_diffs': content_diffs,
          'superseded_stash_diffs': diffs,
          'conclusion': ('16 秒晚编辑（产物生成后收尾编辑）无数值影响，无需重生成；真实产物未被本轮触碰'
                         '（前后 sha256 一致断言在案）'),
          'review_refs': 'R4B-m2① / R2A2-m1 / 本轮 P1-4'}
    with open(OUTP, 'w') as f:
        json.dump(ev, f, indent=1, ensure_ascii=False)
    print('e13 replay: 交付字段逐位一致 | superseded 留档键差异', len(diffs), '处（输入快照语义）')
    print('R3_E13_REPLAY_DONE')


if __name__ == '__main__':
    main()
