"""r3_gate3_mutation_score.py — R3 第二轮创新实验（创新方案一③）：闸3 拦截力变异自证（mutation score）。

【动机（双审 Major 交集 R3A-M1/R5B-M2）】升级前 chain_verify 仅验 prev_hash 链接、freeze 行 sha256 从不与
现文件比对——「校验器存在」≠「校验器有拦截力」。本实验以 8 类违规突变注入 /tmp 一次性副本，度量升级后
闸3 的实际拦截力（mutation testing：killed/total；迁移源=Zhang 2022 arXiv:2204.09165 比覆盖率更真实的
套件有效性度量；单机适配 RFC 9162 CT「append-only 需独立验证方」教义=审计员进程代位）。
【判据（先写后跑，随产物落盘自证）】8 类突变全部应 killed 且病灶定位正确：
  M1 非末行内容编辑保留 hash 字段 → content_hash_mismatch
  M2 冻结文件改动不补冻结行       → freeze_drift STALE_FROZEN_HASH（chain 本身仍 intact——正是升级前盲区）
  M3 删中间行                      → prev_hash_link_broken
  M4 相邻行换序                    → prev_hash_link_broken
  M5 行 hash 字段本身篡改          → content_hash_mismatch
  M6 meta 行 note 被改保留 hash    → content_hash_mismatch
  M7 伪造追加行 prev_hash 跳变     → prev_hash_link_broken
  M8 冻结行 sha256 造假+整链重算自洽（现文件不动）→ 仅 freeze_drift 可拦（链接+内容双断言全过仍 STALE）
      ——M8 是漂移断言存在意义的极间证明：无签名下「能重算链的记录造假」唯一拦在「登记值 vs 现场工件」比对
        （in-toto verify 半边 artifact matching 的教科书补全）；记录与文件合谋篡改单机无信任根不可防，
        与 gate3.verify_model_declaration 申报一致，如实登记。
【红线】生产 ledger 与被冻文件零触碰（前后 sha256 断言）；副本用后即焚。纯 CPU 秒级。
"""
import json, hashlib, shutil, os, sys, tempfile, datetime

sys.path.insert(0, '/root/autodl-tmp/projB_hetero_graph/src')
import verify_provenance as vp  # 生产校验器本体（非测试替身）——mutation testing 测的就是它

REAL_BASE, REAL_LEDGER = vp.BASE, vp.LEDGER
OUTP = f'{REAL_BASE}/results/EXP-R3-REG/gate3_mutation_score_r3.json'


def sha(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()


def make_sandbox():
    """一次性副本：ledger + 全部被冻文件按相对路径铺开（gate3 两断言所需最小树）。"""
    tmp = tempfile.mkdtemp(prefix='gate3_mut_')
    os.makedirs(f'{tmp}/results/EXP-R2-E17', exist_ok=True)
    shutil.copy(REAL_LEDGER, f'{tmp}/results/EXP-R2-E17/provenance_ledger.jsonl')
    rows = [json.loads(l) for l in open(f'{tmp}/results/EXP-R2-E17/provenance_ledger.jsonl') if l.strip()]
    for r in rows:
        if r.get('kind') != 'freeze':
            continue
        f = r['file']
        dst = f if f.startswith('/') else f'{tmp}/{f}'
        src = f if f.startswith('/') else f'{REAL_BASE}/{f}'
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy(src, dst)
    return tmp, rows


def rewrite_ledger(tmp, rows):
    with open(f'{tmp}/results/EXP-R2-E17/provenance_ledger.jsonl', 'w') as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')


def patch_env(tmp):
    vp.BASE, vp.LEDGER = tmp, f'{tmp}/results/EXP-R2-E17/provenance_ledger.jsonl'


def restore_env():
    vp.BASE, vp.LEDGER = REAL_BASE, REAL_LEDGER


def full_rechain_from(rows, i):
    """M8 用：从第 i 行（0-based）起重算全部行 hash（保持链接+内容自洽——模拟有能力重算链的造假者）。"""
    for j in range(i, len(rows)):
        rows[j]['prev_hash'] = rows[j - 1]['hash'] if j else ''
        rows[j]['hash'] = vp.row_content_hash(rows[j])
    return rows


def run_mutation(mid, name, inject, expect):
    """注入→生产校验器双断言→killed/定位判定。expect={'detect_by': 'chain'|'drift', 'why':..., 'at':...}"""
    tmp, rows = make_sandbox()
    try:
        rows2 = inject(tmp, [dict(r) for r in rows])
        rewrite_ledger(tmp, rows2)
        patch_env(tmp)
        try:
            _, ok, bad = vp.chain_verify()
            drift = vp.freeze_drift_check(vp.chain_verify()[0]) if ok else []
            killed, by, at = False, None, None
            if not ok:
                by, at = 'chain_verify', f"row {bad['row']} {bad['why']} ({bad['detail']})"
                killed = (expect['detect_by'] == 'chain' and bad['why'] == expect['why'])
                if killed and expect.get('at') is not None:
                    killed = (bad['row'] == expect['at'])
            elif drift:
                by, at = 'freeze_drift_check', f"{drift[0]['file']} {drift[0]['why']}"
                killed = (expect['detect_by'] == 'drift' and drift[0]['why'] == expect['why'])
            return {'id': mid, 'name': name, 'expected': expect, 'killed': bool(killed),
                    'detected_by': by, 'located_at': at,
                    'verdict': 'KILLED' if killed else 'SURVIVED'}
        finally:
            restore_env()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main():
    pre = sha(REAL_LEDGER)
    # 动态定位（不写死行号，防链增长后脚本失配）
    tmp0, rows0 = make_sandbox()
    shutil.rmtree(tmp0, ignore_errors=True)
    i_meta = next(i for i, r in enumerate(rows0) if r.get('kind') == 'meta')       # 首个 meta 行（0-based）
    i_freeze2 = next(i for i, r in enumerate(rows0) if r.get('kind') == 'freeze')  # 首个 freeze 行
    log_file = rows0[-1]['file']                                                   # 末行冻结文件（现值锚）
    n = len(rows0)

    def m1(tmp, rows):
        rows[i_freeze2]['note'] += '（MUTATED 内容编辑未补 hash）'
        return rows                                                  # hash 字段保留 → 内容失配

    def m2(tmp, rows):
        p = f'{tmp}/{log_file}'
        with open(p, 'a') as f:
            f.write('\n<!-- M2: frozen file silently edited, no new freeze row -->\n')
        return rows                                                  # ledger 不动 → 仅 drift 可拦

    def m3(tmp, rows):
        del rows[10]
        return rows

    def m4(tmp, rows):
        rows[2], rows[3] = rows[3], rows[2]
        return rows

    def m5(tmp, rows):
        rows[7]['hash'] = 'f' * 64
        return rows

    def m6(tmp, rows):
        rows[i_meta]['note'] += '（MUTATED meta note）'
        return rows

    def m7(tmp, rows):
        forged = {'ts_collected': datetime.datetime.now().astimezone().isoformat(timespec='seconds'),
                  'kind': 'freeze', 'file': 'results/forged_r3.json', 'sha256': '0' * 64,
                  'prev_hash': 'a' * 64, 'note': 'M7 forged append',
                  'slsa_level_self_assessed': 'L1'}
        forged['hash'] = vp.row_content_hash(forged)                 # 行自身自洽，但 prev_hash 跳变
        rows.append(forged)
        return rows

    def m8(tmp, rows):
        # 定位 log_file 的最新 freeze 行，造假其 sha256 并整链重算自洽（现文件副本不动）
        i = max(i for i, r in enumerate(rows) if r.get('kind') == 'freeze' and r['file'] == log_file)
        rows[i]['sha256'] = 'e' * 64
        return full_rechain_from(rows, i)

    results = [
        run_mutation('M1', '非末行 freeze 行内容编辑保留 hash 字段', m1,
                     {'detect_by': 'chain', 'why': 'content_hash_mismatch', 'at': i_freeze2 + 1}),
        run_mutation('M2', '冻结文件被静默改动不补冻结行', m2,
                     {'detect_by': 'drift', 'why': 'STALE_FROZEN_HASH'}),
        run_mutation('M3', '删除中间行', m3, {'detect_by': 'chain', 'why': 'prev_hash_link_broken', 'at': 11}),
        run_mutation('M4', '相邻两行换序', m4, {'detect_by': 'chain', 'why': 'prev_hash_link_broken'}),
        run_mutation('M5', '行 hash 字段本身篡改', m5,
                     {'detect_by': 'chain', 'why': 'content_hash_mismatch', 'at': 8}),
        run_mutation('M6', 'meta 行 note 被改保留 hash', m6,
                     {'detect_by': 'chain', 'why': 'content_hash_mismatch', 'at': i_meta + 1}),
        run_mutation('M7', '伪造追加行（prev_hash 跳变、行自洽）', m7,
                     {'detect_by': 'chain', 'why': 'prev_hash_link_broken', 'at': n + 1}),
        run_mutation('M8', '冻结 sha256 记录造假+整链重算自洽（现文件不动）', m8,
                     {'detect_by': 'drift', 'why': 'STALE_FROZEN_HASH'}),
    ]
    killed = sum(r['killed'] for r in results)
    post = sha(REAL_LEDGER)
    assert pre == post, '生产 ledger 被触碰——红线违规，abort'
    out = {
        'exp': 'R3b-gate3-mutation-score（创新方案一③：拦截力变异自证）',
        'date': datetime.datetime.now().astimezone().isoformat(timespec='seconds'),
        'producer': 'src/r3_gate3_mutation_score.py（测生产校验器 src/verify_provenance.py 本体，/tmp 一次性副本）',
        'mutation_score': f'{killed}/{len(results)}',
        'target': '升级后闸3=chain_verify（链接+行内容哈希双断言）+freeze_drift_check（最新冻结 sha256 vs 现文件）',
        'mutations': results,
        'boundary_declaration': ('单机无信任根：M8 证明「记录造假+链重算」唯一拦在登记值 vs 现场工件比对；'
                                 '记录与文件合谋篡改（等价合法重冻结）不可防——无签名体系的能力边界，'
                                 '与 gate3.verify_model_declaration 申报一致'),
        'production_ledger_untouched': True,
        'review_refs': 'R3A-M1 / R5B-M2 / 创新方案一③'}
    with open(OUTP, 'w') as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    for r in results:
        print(f"  {r['id']} {r['verdict']:8s} by={r['detected_by']} at={r['located_at']}")
    print(f'MUTATION_SCORE_DONE {killed}/{len(results)}  生产链零触碰（前后 sha 一致）')
    assert killed == len(results), f'SURVIVED 突变存在：{[r["id"] for r in results if not r["killed"]]}'


if __name__ == '__main__':
    main()
