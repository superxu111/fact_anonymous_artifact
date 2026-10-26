"""verify_provenance.py — R2-E17（定向创新方案一）：R2-PV 三闸证据链机器验证。

【设计自查（开工前冻结，判据见 results/EXP-R2-E17/criteria_e17.json）】
动机：三轮审稿反复失效的「自报式登记机制」升级为机器强制验证（R2A-M1/M2/M3、R3B-M2/M3 同根因）。
迁移模板：in-toto（USENIX Sec 2019）layout/link/verify 的 product→material 哈希链 + SLSA v1.0 分级
  语言 + statcheck（Nuijten 2016）自动一致性检查（教训适配：校正后统计量只比对不重算，manifest 行带
  口径说明，防误报）。
闸1 producer gate：results/**/*.json 逐文件——有 producer 键指向 src/*.py、或文件名在 src/*.py 源文
  出现 → 有源；否则 ORPHAN_FILE（拦截 R2A-M3「无源插入」复发）。键级抽查限 EXP-R2-REG 层。
闸2 SSOT lint：number_manifest 每行——prose 行查出处存在；JSON path 行现读解析 + expect 值 diff
  （只比对不重算）；扫描 src/finalize*.py 浮点字面量，未登记且未豁免 → WARN，已知复发值 → FAIL。
闸3 freeze gate：provenance_ledger.jsonl append-only 内容寻址哈希链（sha256+prev_hash+collected_at as-of）——
  R3 第二轮双审升级（R3A-M1/R5B-M2 同病灶）：验证端逐行复算行内容哈希（row_content_hash 单一实现，生成端同式
  复用，不再信任行内自报 hash 字段）+ 每文件最新冻结 sha256 vs 现文件漂移断言（freeze_drift_check，多次冻结取
  末行）+ 拦截力以 8 类违规变异注入自证（mutation score 落盘 results/EXP-R3-REG/gate3_mutation_score_r3.json）。
  GPU 门禁新规（训练启动前断言 criteria 行已落盘），E12 冻结降格登记（收口分析前冻结），SLSA 自评。
用法：python3 src/verify_provenance.py [--freeze FILE:NOTE ...] ；首轮审计报告落盘
  results/EXP-R2-E17/provenance_first_audit.json。纯 CPU 秒级。
"""
import json, hashlib, os, re, sys, glob, datetime, argparse

BASE = '/root/autodl-tmp/projB_hetero_graph'
OUT = f'{BASE}/results/EXP-R2-E17'
LEDGER = f'{BASE}/results/EXP-R2-E17/provenance_ledger.jsonl'
GENERIC_KEYS = {'exp', 'date', 'design', 'status', 'note', 'notes', 'producer', 'review_refs',
                'verdict', 'expect', 'path', 'caliber', 'n', 'machine', 'criteria', 'unit_tests',
                'summary', 'reading', 'conclusion', 'families', 'per_point', 'rows', 'points',
                'tolerance', 'mappings', 'usage', 'r4_appended_rows_note', 'certificate'}
FILE_EXEMPT = {'criteria', 'unit_tests', 'provenance_ledger', 'provenance_first_audit',
               'number_manifest', 'criteria_mtime_ledger', 'R1_metrics',
               'keyscan_exemption_registry', 'warn_stratification', 'keyscan_two_phase'}
FLOAT_RE = re.compile(r'(?<![\w.])(\d+\.\d{3,})(?![\w])')
EXEMPT_PREFIX = ('2000', '500', '0.02', '0.05', '0.0005', '100.', '300', '0.02', '24.', '48.', '18.', '6.')
# R3/P0-2：闸1 键级扫描常量（自述=实现机器比对的单测见 src/test_r3_gate1_keyscan.py；
# 声明面=results/EXP-R2-E17/criteria_e17.json:gate1_alignment_r3）
GATE1_CONSTANTS = {
    'scope_glob': 'results/**/*.json',
    'key_scan_depth_max': 2,
    'list_container_not_descended': True,
    'min_key_len': 5,
    'canon_min_len': 3,
    'canon_rule': 'canon(k)=剥全部数字；有源 ⟺ k 原文 in src_all 或 (len(canon(k))>=3 且 canon(k) in canon(src_all))——剥数字仅作豁免判定，孤儿按原键名上报',
    'composite_segment_rule': "含 '/' 复合键按段判定：每段（原文 in src_all 或 canon(段) in canon_src）全命中方为有源——f'{arm}/s{seed}/{tier}' 型动态插值键的生产者检测近似；单字符段残留误豁免风险如实申报（见 criteria_r3.json）",
    'skip_files': "stem 以 criteria 开头或命中 FILE_EXEMPT（登记文档按设计无生产者，沿用 R2 文本键豁免处置）",
    'src_scan_exclude': ['test_r3', 'r3_gate1_twophase', 'r3_keyscan_exemptions'],
    'src_scan_exclude_semantics': '元脚本（测试/两阶段留档/豁免登记表）自身讨论键名字面量——讨论串不构成生产者，剔除出键级 src 快照；真生产者脚本（r3_key_producers/r3_e2_full_ap_backfill 等）保留',
    'synthetic_guard': '含数字无源的合成键（bad-key 型）剥数字后不命中 src → 仍报孤儿（数字不再整键豁免）'}


def canon(s):
    """R3/P0-2 键名规范化：剥全部数字（PEP 503 canonicalize_name 同思想）。"""
    return re.sub(r'\d+', '', s)


def gate1_src_all():
    """闸1 键级扫描的 src 快照（单一构造点：gate1/两阶段留档/豁免登记表/单测共用，防漂移）。"""
    parts = []
    for p in sorted(glob.glob(f'{BASE}/src/*.py')):
        b = os.path.basename(p)
        if any(b.startswith(x) for x in GATE1_CONSTANTS['src_scan_exclude']):
            continue
        parts.append(open(p, encoding='utf-8', errors='ignore').read())
    return ''.join(parts)


def key_sourced(k, src_all, canon_src):
    """闸1 键级「有源」谓词——单一实现（gate1/两阶段留档/单测三处共用，防复制漂移）。"""
    if k in src_all:
        return True
    if '/' in k:
        return all(seg in src_all or (canon(seg) and canon(seg) in canon_src)
                   for seg in k.split('/'))
    ck = canon(k)
    return len(ck) >= GATE1_CONSTANTS['canon_min_len'] and ck in canon_src


def sha256(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()


# ---------------- 闸 1：producer gate ----------------
def gate1():
    src_all = gate1_src_all()
    orphans, sourced = [], []
    for p in sorted(glob.glob(f'{BASE}/results/**/*.json', recursive=True)):
        rel = os.path.relpath(p, f'{BASE}/results')
        stem = os.path.splitext(os.path.basename(p))[0]
        if any(stem.startswith(x) or stem == x for x in FILE_EXEMPT) or stem.startswith('criteria'):
            continue
        try:
            d = json.load(open(p))
        except Exception as e:
            orphans.append({'file': rel, 'reason': f'unreadable: {e}'}); continue
        prod = d.get('producer', '') if isinstance(d, dict) else ''
        prod_ok = isinstance(prod, str) and ('src/' in prod) and prod.split('src/')[-1].split('.')[0] + '.py' in src_all
        name_ok = stem in src_all or os.path.splitext(os.path.basename(p))[0] in src_all
        (sourced if (prod_ok or name_ok) else orphans).append(
            {'file': rel, 'via': 'producer_key' if prod_ok else ('filename_in_src' if name_ok else '—')})
    # 键级扫描（R3/P0-2 重设计：范围扩 results/**；剥数字豁免规则；孤儿按原键名上报）
    canon_src = canon(src_all)
    key_orphans, key_exempt_files = [], 0
    exempt_by_class = {}
    regp = f'{BASE}/results/EXP-R3-REG/keyscan_exemption_registry_r3.json'
    exemption_registry = json.load(open(regp))['entries'] if os.path.exists(regp) else []
    for p in sorted(glob.glob(f'{BASE}/results/**/*.json', recursive=True)):
        stem = os.path.splitext(os.path.basename(p))[0]
        if any(stem.startswith(x) or stem == x for x in FILE_EXEMPT) or stem.startswith('criteria'):
            key_exempt_files += 1
            continue
        rel = os.path.relpath(p, f'{BASE}/results')
        try:
            d = json.load(open(p))
        except Exception:
            continue

        def walk(o, depth=0):
            if depth > GATE1_CONSTANTS['key_scan_depth_max'] or not isinstance(o, dict):
                return
            for k, v in o.items():
                if isinstance(k, str) and len(k) > GATE1_CONSTANTS['min_key_len'] \
                        and k not in GENERIC_KEYS and not k.startswith('_'):
                    if not key_sourced(k, src_all, canon_src):
                        hit = next((e['class'] for e in exemption_registry
                                    if re.match(e['pattern'], k)
                                    and (not e['applies_to'] or rel in e['applies_to'])), None)
                        if hit:
                            exempt_by_class[hit] = exempt_by_class.get(hit, 0) + 1
                        else:
                            key_orphans.append({'file': rel, 'key': k[:80]})
                walk(v, depth + 1)
        walk(d)
    return {'orphan_files': orphans, 'sourced_count': len(sourced),
            'key_level_orphans': key_orphans, 'key_scan_exempt_files': key_exempt_files,
            'key_exempt_by_class': exempt_by_class,
            'key_exemption_registry': 'results/EXP-R3-REG/keyscan_exemption_registry_r3.json',
            'constants': GATE1_CONSTANTS}


# ---------------- 闸 2：SSOT lint ----------------
def resolve(path_expr):
    """解析 'results/...json:dot.path[+ second]' → 现值。支持 [*] 通配与多段 + 连接。"""
    vals = []
    for seg in path_expr.split(' + '):
        if '.json' not in seg:
            continue
        fp, _, jpath = seg.partition(':')
        fp = f'{BASE}/{fp.strip()}'
        if not os.path.exists(fp):
            vals.append(('MISSING_FILE', seg)); continue
        cur = json.load(open(fp))
        ok = True
        for part in [x for x in jpath.split('.') if x]:
            m = re.match(r'(\w+)\[\*\]', part)
            if m:
                cur = cur.get(m.group(1)) if isinstance(cur, dict) else cur
                if not isinstance(cur, list):
                    ok = False; break
                cur = [cur] if not isinstance(cur, list) else cur
                continue
            if isinstance(cur, dict) and part in cur:
                cur = cur[part]
            else:
                ok = False; break
        vals.append((cur if ok else 'PATH_FAIL', seg))
    return vals


def stratify_warns():
    """finalize*.py 浮点字面量 → (warns, l_exempt, l_quote, l_register)。

    单一实现（gate2 与 EXP-R3-REG/warn_stratification_r3.json 注册表共用，防双口径接缝）：
    总体=未入 manifest expect 的 finalize 浮点字面量；L_EXEMPT=解析真值 1.000；
    L_QUOTE=按字面量自身精度舍入后精确出现于 results/** 交付 JSON；L_REGISTER=其余（待登记）。"""
    mf = json.load(open(f'{BASE}/results/EXP-R2-REG/number_manifest.json'))
    expect_vals = set()
    for row in mf['mappings']:
        e = row.get('expect')
        if isinstance(e, dict):
            expect_vals |= {float(x) for x in e.values() if isinstance(x, (int, float))}
        elif isinstance(e, (int, float)):
            expect_vals.add(float(e))
    warns = []
    for p in glob.glob(f'{BASE}/src/finalize*.py'):
        for m in FLOAT_RE.finditer(open(p).read()):
            v = m.group(1)
            if v.startswith(EXEMPT_PREFIX):
                continue
            if float(v) not in expect_vals:
                warns.append({'file': os.path.basename(p), 'literal': v})
    results_pool = set()

    def collect(o):
        if isinstance(o, dict):
            for v in o.values():
                collect(v)
        elif isinstance(o, list):
            for v in o:
                collect(v)
        elif isinstance(o, float):
            results_pool.add(o)
    for p in glob.glob(f'{BASE}/results/**/*.json', recursive=True):
        # R3A-m3 收窄：L_QUOTE 判定池与闸1 skip 同口径——排除 criteria* 与 FILE_EXEMPT 登记文档，
        # 仅存于登记文档的数字不再被误分层为引述性而逃过棘轮（注册表自述规则=「精确出现于交付 JSON」）
        stem = os.path.splitext(os.path.basename(p))[0]
        if any(stem.startswith(x) or stem == x for x in FILE_EXEMPT) or stem.startswith('criteria'):
            continue
        try:
            collect(json.load(open(p)))
        except Exception:
            pass
    l_exempt, l_quote, l_register = [], [], []
    for w in warns:
        v = w['literal']
        fv = float(v)
        if v == '1.000':
            l_exempt.append(w)  # 解析真值（结构常量）
            continue
        d = len(v.split('.')[1])
        if any(round(f, d) == fv for f in results_pool):
            l_quote.append(w)  # 引述性：交付数（按字面量自身精度舍入后）的引用
        else:
            l_register.append(w)
    return warns, l_exempt, l_quote, l_register


def gate2():
    """优先读 row['checks']（E17 规范化精确路径+expect 固化）；expect=None → presence；
    无 checks 的旧式行宽解析 fallback；statcheck 适配=只比对不重算。"""
    mf = json.load(open(f'{BASE}/results/EXP-R2-REG/number_manifest.json'))
    results, hard_fail = [], []

    def read_exact(file_colon_path):
        fp, _, jpath = file_colon_path.partition(':')
        ffull = f'{BASE}/results/{fp.strip()}' if not fp.startswith('results/') else f'{BASE}/{fp}'
        if not os.path.exists(ffull):
            return 'MISSING_FILE'
        cur = json.load(open(ffull))
        if not jpath.strip():
            return cur  # presence 检查：文件可读即通过
        parts = jpath.split('.')
        i = 0
        while i < len(parts):
            cand, j, resolved = parts[i], i, False
            while True:
                if isinstance(cur, dict) and cand in cur:  # ① dict 键命中
                    cur = cur[cand]; i = j + 1; resolved = True; break
                if cand.endswith(']') and '[' in cand:  # ② 索引段解析
                    k, idx = cand[:-1].rsplit('[', 1)
                    base = cur.get(k) if (isinstance(cur, dict) and k in cur) else (cur if k == '' else None)
                    if isinstance(base, list):
                        cur = base[int(idx)]; i = j + 1; resolved = True; break
                if j + 1 < len(parts):  # ③ 贪心合并：键名本身含 . 或 /（ewma0.2@arl100、arl100/cusum_k0.5）
                    j += 1
                    cand = cand + '.' + parts[j]
                else:
                    break
            if not resolved:
                return 'PATH_FAIL'
        return cur

    for row in mf['mappings']:
        r = {'n': row['n'][:50], 'path_type': row.get('path_type', 'json')}
        if row.get('path_type') == 'prose':
            r['check'] = 'prose 出处登记（写作期映射核）'
            r['ok'] = os.path.exists(f'{BASE}/RESEARCH_LOG.md')
        elif 'checks' in row:
            bad = []
            for c in row['checks']:
                got = read_exact(c['path'])
                if got in ('MISSING_FILE', 'PATH_FAIL'):
                    bad.append({'path': c['path'], 'err': got}); continue
                if c.get('expect') is not None and got != c['expect']:
                    bad.append({'path': c['path'], 'expect': c['expect'], 'got': got})
            r['checks_verdict'] = f"{len(row['checks']) - len(bad)}/{len(row['checks'])} ok"
            r['bad'] = bad
            r['ok'] = not bad
        else:
            vals = resolve(row['path'])
            ok = all(v != 'MISSING_FILE' and v != 'PATH_FAIL' for v, _ in vals)
            r['legacy_resolve'] = 'ok' if ok else 'FAIL'
            r['ok'] = ok
        if not r['ok']:
            hard_fail.append(r)
        results.append(r)
    # finalize*.py 浮点字面量扫描（statcheck 适配：只登记比对，不重算统计量）——单一实现见 stratify_warns
    warns, l_exempt, l_quote, l_register = stratify_warns()
    # ---- R3/P1-8：91 WARN 三层分层（EXEMPT/引述性/待登记，棘轮=待登记只许降不许升）----
    regp = f'{BASE}/results/EXP-R3-REG/warn_stratification_r3.json'
    unresolved, registry_missing = [], l_register
    if os.path.exists(regp):
        registry = json.load(open(regp))
        resolved = {(x['file'], x['literal']) for x in registry.get('l_register_resolved', [])}
        unresolved = [w for w in l_register if (w['file'], w['literal']) not in resolved]
        registry_missing = False
    # ---- R3/P1-8：κ 断言升级（公式重算+更正上下文豁免，R2A2-m7 单回归形态缺口）----
    from scipy.stats import t as tdist, norm
    import math
    e11 = json.load(open(f'{BASE}/results/EXP-R2-E11/e11_g3_sensitivity.json'))
    kappa_formula = {}
    for arl, a in (('arl100', 100), ('arl500', 500)):
        tau_t = math.sqrt(1.0 + 1.0 / 9) * float(tdist.ppf(1 - 1.0 / a, 8))  # e5_evt_calib.py:160-164 同式
        kap = tau_t / float(norm.ppf(1 - 1.0 / a))
        reg = float(e11['kappa_t_pivot_inflation'][arl])
        kappa_formula[arl] = {'recomputed': round(kap, 6), 'registered': reg,
                              'absdiff': round(abs(kap - reg), 6), 'ok': abs(kap - reg) < 1e-4}
    kappa_formula_ok = all(v['ok'] for v in kappa_formula.values())
    # 字符串断言升级：'1.2785' 出现行若无更正上下文标记词 → 触发（豁免词表先登记于 criteria_r3）
    KAPPA_EXEMPT_WORDS = ['更正', 'supersedes', 'SUPERSEDES', '误植', '硬编码', '原字符串', '原文保留']
    rm_lines = open(f'{BASE}/results/R1_metrics.json').read().splitlines()
    uncontextualized = [i + 1 for i, l in enumerate(rm_lines)
                        if '1.2785' in l and not any(w in l for w in KAPPA_EXEMPT_WORDS)]
    rm_all = '\n'.join(rm_lines)
    kappa_regression = ('1.2785' in rm_all and '1.4616' not in rm_all) or bool(uncontextualized)
    return {'manifest_rows': results, 'unregistered_floats_in_finalize': warns,
            'warn_stratification': {'l_exempt': l_exempt, 'n_quote': len(l_quote),
                                    'l_register': l_register, 'unresolved_vs_registry': unresolved,
                                    'registry': 'results/EXP-R3-REG/warn_stratification_r3.json'},
            'kappa_formula_recheck': kappa_formula, 'kappa_formula_ok': kappa_formula_ok,
            'kappa_uncontextualized_lines': uncontextualized,
            'kappa_regression_12785_only': kappa_regression,
            'hard_fail': hard_fail
            + ([{'kappa': '1.2785 present, 1.4616 absent'}] if ('1.2785' in rm_all and '1.4616' not in rm_all) else [])
            + ([{'kappa_context': f'1.2785 无更正上下文标记词行: {uncontextualized}'}] if uncontextualized else [])
            + ([{'kappa_formula': '公式重算超差'}] if not kappa_formula_ok else [])
            + ([{'warn_register': f'待登记层未登记项 {len(unresolved)} 条'}] if unresolved else [])
            + ([{'warn_registry_missing': '分层注册表缺失且待登记层非空'}] if (registry_missing and l_register) else [])}


# ---------------- 闸 3：freeze gate（append-only 内容寻址哈希链） ----------------
def row_content_hash(r):
    """行内容哈希——单一实现（R3 双审升级 R3A-M1/R5B-M2）：生成端 gate3 与验证端 chain_verify 共用，
    防两端公式漂移。公式=sha256(prev_hash + json.dumps(行去 hash 字段, sort_keys=True, ensure_ascii=False))。"""
    return hashlib.sha256((r['prev_hash'] + json.dumps({k: v for k, v in r.items() if k != 'hash'},
                                                        sort_keys=True, ensure_ascii=False)).encode()).hexdigest()


def chain_verify():
    """链接 + 行内容哈希双断言（R3A-M1①：升级前仅验 prev_hash 链接，非末行内容被就地编辑而保留
    hash 字段时 chain_intact 仍 True——盲区恰是本项目连续四轮「修复轮自伤」的就地文本编辑事故类型）。
    返回 (rows, ok, bad)；bad=结构化病灶 {'row': 行号, 'why': prev_hash_link_broken|content_hash_mismatch}。"""
    rows = [json.loads(l) for l in open(LEDGER) if l.strip()]
    prev = ''
    for i, r in enumerate(rows, 1):
        if r['prev_hash'] != prev:
            return rows, False, {'row': i, 'why': 'prev_hash_link_broken',
                                 'detail': (r.get('file') or r.get('note') or '')[:60]}
        if row_content_hash(r) != r['hash']:
            return rows, False, {'row': i, 'why': 'content_hash_mismatch',
                                 'detail': (r.get('file') or r.get('note') or '')[:60]}
        prev = r['hash']
    return rows, True, None


def freeze_drift_check(rows):
    """冻结漂移断言（R3A-M1②/R5B-M2：升级前 freeze 行 sha256 从不与现文件比对，「冻结后被静默改动而
    不补冻结行」机器无感）。每文件取最新 freeze 行（多次冻结取末行）sha256 与现文件逐一比对。
    返回 STALE 清单（空=全部现值一致）；audit/ledger 自身按设计不入链（FILE_EXEMPT/自指豁免）。"""
    latest = {}
    for i, r in enumerate(rows, 1):
        if r.get('kind') == 'freeze':
            latest[r['file']] = (i, r['sha256'])
    stale = []
    for f, (i, s) in sorted(latest.items()):
        p = f if f.startswith('/') else f'{BASE}/{f}'
        if not os.path.exists(p):
            stale.append({'file': f, 'row': i, 'why': 'MISSING_FILE'}); continue
        if sha256(p) != s:
            stale.append({'file': f, 'row': i, 'why': 'STALE_FROZEN_HASH'})
    return stale


def gate3(freeze_args, meta_args):
    os.makedirs(OUT, exist_ok=True)
    rows, ok, bad = (chain_verify() if os.path.exists(LEDGER) else ([], True, None))
    if not ok:
        # 链已断时拒绝追加（append-only 前提是链完好；带病追加会污染后续全部行）
        return {'ledger_rows': len(rows), 'chain_intact': False, 'broken_at': bad,
                'freeze_drift': [], 'appended_rows': 0,
                'refused_append_reason': 'chain broken——先处置病灶再冻结（防污染后行）',
                'gpu_gate_rule': ('GPU 实验门禁（R3B-M3 立规）：训练启动前必须断言 criteria 行已落盘于 '
                                  'provenance_ledger.jsonl（先 criteria 后训练；训练所用 src/*.py 不得晚于'
                                  ' criteria 冻结时刻被编辑，违者冻结失效并在台账如实登记）'),
                'slsa_self_assessment': ('现状 SLSA v1.0 自评 L1（存在 provenance=append-only sha256 链+as-of '
                                         'collected_at）；无托管平台签名/RFC 3161 TSA 锚定，不足 L2——如实申报')}
    prev = rows[-1]['hash'] if rows else ''
    ts = datetime.datetime.now().astimezone().isoformat(timespec='seconds')
    new_rows = []
    # R3/P1-4：kind=meta 行（头部申报句/晚产物补登记等登记语义行；无 file/sha256，chain_verify 兼容；
    # 不改首行、append-only）
    for note in meta_args:
        row = {'ts_collected': ts, 'kind': 'meta', 'note': note, 'prev_hash': prev,
               'slsa_level_self_assessed': 'L1'}
        row['hash'] = row_content_hash(row)
        new_rows.append(row); prev = row['hash']
    for spec in freeze_args:
        fname, _, note = spec.partition(':')
        p = f'{BASE}/{fname}' if not fname.startswith('/') else fname
        row = {'ts_collected': ts, 'kind': 'freeze', 'file': fname, 'sha256': sha256(p),
               'prev_hash': prev, 'note': note, 'slsa_level_self_assessed': 'L1'}
        row['hash'] = row_content_hash(row)
        new_rows.append(row); prev = row['hash']
    with open(LEDGER, 'a') as f:
        for r in new_rows:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    rows2, ok2, bad2 = chain_verify()
    drift = freeze_drift_check(rows2) if ok2 else []
    n_freeze_files = len({r['file'] for r in rows2 if r.get('kind') == 'freeze'})
    return {'ledger_rows': len(rows2), 'chain_intact': ok2, 'broken_at': bad2,
            'freeze_drift': drift,
            'freeze_drift_verdict': (f'PASS：每文件最新冻结 sha256 与现文件逐一比对 '
                                     f'{n_freeze_files - len(drift)}/{n_freeze_files} 现值一致')
                                     if ok2 and not drift else f'FAIL：{len(drift)} 文件漂移/缺失',
            'appended_rows': len(new_rows),
            'verify_model_declaration': ('内容寻址（行哈希由内容复算断言）+ 漂移受检（冻结值 vs 现文件机器比对）'
                                         '+ 变异自证（拦截力度量 mutation score 见 '
                                         'results/EXP-R3-REG/gate3_mutation_score_r3.json）——对照 in-toto '
                                         'verify 半边 artifact matching（登记值必须与现场工件逐项比对）；'
                                         '单机无信任根：append-only 防中间行篡改，尾部重写防不了（无签名）——'
                                         '由变异自证代位独立审计方申报'),
            'gpu_gate_rule': ('GPU 实验门禁（R3B-M3 立规）：训练启动前必须断言 criteria 行已落盘于 '
                              'provenance_ledger.jsonl（先 criteria 后训练；训练所用 src/*.py 不得晚于'
                              ' criteria 冻结时刻被编辑，违者冻结失效并在台账如实登记）'),
            'slsa_self_assessment': ('现状 SLSA v1.0 自评 L1（存在 provenance=append-only sha256 链+as-of '
                                     'collected_at）；无托管平台签名/RFC 3161 TSA 锚定，不足 L2——如实申报')}


def main():
    ap = argparse.ArgumentParser()
    # R3 修复：action='append'（store 会在重复 flag 时覆盖——曾致 32 个 freeze 只入链 1 个）
    ap.add_argument('--freeze', action='append', nargs='*', default=[], metavar='FILE:NOTE')
    ap.add_argument('--meta', action='append', nargs='*', default=[], metavar='NOTE',
                    help='R3/P1-4：kind=meta 登记行（无 file/sha256）')
    args = ap.parse_args()
    freeze_args = [x for grp in args.freeze for x in grp]
    meta_args = [x for grp in args.meta for x in grp]
    audit = {'collected_at': datetime.datetime.now().astimezone().isoformat(timespec='seconds'),
             'gate1': gate1(), 'gate2': gate2(), 'gate3': gate3(freeze_args, meta_args)}
    os.makedirs(OUT, exist_ok=True)
    with open(f'{OUT}/provenance_first_audit.json', 'w') as f:
        json.dump(audit, f, indent=1, ensure_ascii=False)
    g1, g2, g3 = audit['gate1'], audit['gate2'], audit['gate3']
    print(f"闸1: 有源 {g1['sourced_count']} 件 / ORPHAN_FILE {len(g1['orphan_files'])} 件"
          f"（键级孤儿 {len(g1['key_level_orphans'])}，键扫描豁免登记文件 {g1['key_scan_exempt_files']}）")
    for o in g1['orphan_files']:
        print('  ORPHAN:', o)
    for o in g1['key_level_orphans']:
        print('  KEY-ORPHAN:', o)
    ws = g2['warn_stratification']
    print(f"闸2: manifest {len(g2['manifest_rows'])} 行 / FAIL {len(g2['hard_fail'])} / "
          f"finalize 未登记浮点 WARN {len(g2['unregistered_floats_in_finalize'])}"
          f"（EXEMPT {len(ws['l_exempt'])} / QUOTE {ws['n_quote']} / REGISTER {len(ws['l_register'])}"
          f" 未登记 {len(ws['unresolved_vs_registry'])}）/ κ 公式重算={'PASS' if g2['kappa_formula_ok'] else 'FAIL'}"
          f" / κ 回归={g2['kappa_regression_12785_only']}")
    for r in g2['hard_fail']:
        print('  G2FAIL:', r)
    print(f"闸3: 链行 {g3['ledger_rows']} / chain_intact={g3['chain_intact']} / 冻结漂移={g3.get('freeze_drift_verdict', '—')}")
    print('VERIFY_PROVENANCE_DONE',
          'PASS' if (not g1['orphan_files'] and not g1['key_level_orphans'] and not g2['hard_fail']
                     and g3['chain_intact'] and not g3.get('freeze_drift')) else 'HAS_FINDINGS')


if __name__ == '__main__':
    main()
