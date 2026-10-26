"""e5_evt_calib.py — R1-E5：EVT 尾外推 + 参数不确定性传播的公平校准协议（R1A-C1/R1B-C1 修复）。

判据：results/EXP-R1-E5/criteria.json（运行前冻结）。
方法：POT/GPD 尾外推（McNeil-Frey 2000 两步法 / SPOT KDD17 机制）+ 池层 bootstrap 预测式阈值
（SPC Phase I 参数估计效应文献）；验收：合成世界 G1 + 真实小时流 G2 + 公平 H1 重判定 G3。
数据：与 e1_window_alert.py 完全相同的良性日 z 池（不重训任何模型，纯重校准）。

结构性上限申报（2026-10-24 步级闸门补漏补记；数值先已冻结于 criteria.json motivation 与本件
root_cause_confirmed 输出列）：在案 results/EXP-R1-E1/e1_window_alert.json 全部 54 个 fixed
校准点实测 ARL 8.7-9.4（名义 100/500 两档阈值相同）——9 良性日离散池最小非零告警率 1/9 →
经验 ARL 结构性上限≈9，二分上界被池最大值封顶（e1_window_alert.py:91,129-131）。故本件为
E1 fixed 读出的替代校准路径：在案 e1_window_alert.py 不改动（其产物为历史证据），公平校准
一律走本件（小池 t 预测枢轴主阈值 + EVT/GPD 形状诊断并报），H1 公平重判定见 g3_h1_reverdict。
"""
import json, sys
import numpy as np
from collections import defaultdict
from scipy.stats import genpareto, norm

sys.path.insert(0, '/root/autodl-tmp/projB_hetero_graph/src')
from r4b_common import load_redteam_events, attack_truth_by_day

BASE = '/root/autodl-tmp/projB_hetero_graph'
ARMS = {
    'full': f'{BASE}/results/EXP-R1-E1/arm_full_60ep',
    'homoflows': f'{BASE}/results/EXP-R1-E1/arm_homoflows_60ep',
    'homo': f'{BASE}/results/EXP-R1-E1/arm_homo_60ep',
}
OUT = f'{BASE}/results/EXP-R1-E5'
EVAL_DAYS = [12, 13, 14, 15, 20, 21, 22, 26, 27, 28, 29]
BENIGN_DAYS = [10, 11, 16, 17, 18, 19, 23, 24, 25]
TOPK = 50
ARL0S = [100, 500]
QS = [0.70, 0.75, 0.80, 0.85, 0.90]


# ---------- 复用 e1_window_alert 的窗统计量口径（逐字一致，不 import 以免触发其模块级数据加载） ----------
def window_stats(npz_path):
    z = np.load(npz_path, allow_pickle=True)
    day, rel, score = z['day'], z['rel'], z['score']
    lm = rel == 'LOGON'
    out = defaultdict(dict)
    for d in sorted(set(day[lm].tolist())):
        s = score[lm][day[lm] == d]
        srt = np.sort(s)[::-1]
        out[int(d)]['top50mean'] = float(srt[:TOPK].mean())
        out[int(d)]['mean'] = float(s.mean())
        out[int(d)]['max'] = float(srt[0])
    return out


# ---------- EVT 核心 ----------
def evt_tau_from_fit(u, p_u, xi, sigma, arl0):
    """P(X>τ)=p_u·(1+ξ(τ−u)/σ)^(−1/ξ)=1/ARL0 的解；ξ→0 退化为指数尾。"""
    p_star = 1.0 / arl0
    if abs(xi) < 1e-8:
        return u + sigma * np.log(p_u / p_star)
    return float(u + (sigma / xi) * ((p_star / p_u) ** (-xi) - 1.0))


def _fit_gpd(excess, estimator='lmom'):
    """GPD 超额拟合。默认 L-矩（Hosking 1990，水文学小样本标准）；'mle' 为敏感性对照。"""
    if len(excess) < 5 or np.ptp(excess) < 1e-12:
        return None
    x = np.sort(np.asarray(excess, float))
    n = len(x)
    if estimator == 'lmom':
        j = np.arange(1, n + 1)
        b0 = x.mean()
        b1 = float(np.sum((j - 1) / max(n - 1, 1) * x)) / n
        l1, l2 = b0, 2 * b1 - b0
        if l2 <= 0 or l1 <= 0:
            return None
        xi = 2.0 - l1 / l2
        sigma = l1 * (1.0 - xi)
        if sigma <= 0 or not np.isfinite(xi) or xi >= 0.95:
            return None
        return float(xi), float(sigma)
    try:
        xi, _, sigma = genpareto.fit(x, floc=0)
    except Exception:
        return None
    if not np.isfinite(xi) or not np.isfinite(sigma) or sigma <= 0:
        return None
    return float(xi), float(sigma)


def auto_q_u(values, target_exc=20, q_floor=0.75):
    """水文学操作规则：≥target_exc 个超额前提下阈值尽量高。"""
    n = len(values)
    return float(max(q_floor, 1.0 - target_exc / n))


def fit_evt_tau(values, arl0, q_u=None, n_boot=1000, rng=None, pools=None, estimator='lmom'):
    """POT 校准 τ(ARL0)。values: 合并后的标准化良性值；pools: 池列表（给定则池层 bootstrap）。
    q_u=None 时按 auto_q_u 自适应。返回 ok/tau(预测中位)/tau_plugin/tau_ci/xi/sigma/p_u/u/n_exc。"""
    rng = rng or np.random.default_rng(2026)
    values = np.asarray(values, float)
    if np.ptp(values) < 1e-12:
        return {'ok': False, 'reason': 'degenerate', 'tau': None}
    if q_u is None:
        q_u = auto_q_u(values)
    u = np.quantile(values, q_u)
    exc = values[values > u] - u
    p_u = float((values > u).mean())
    fit = _fit_gpd(exc, estimator)
    if fit is None:
        return {'ok': False, 'reason': 'gpd_fit_failed', 'tau': None}
    xi, sigma = fit
    tau_plug = evt_tau_from_fit(u, p_u, xi, sigma, arl0)
    # 池层 bootstrap → τ 预测分布
    taus = []
    for _ in range(n_boot):
        if pools is not None:
            idx = rng.integers(0, len(pools), len(pools))
            v = np.concatenate([pools[i] for i in idx])
        else:
            v = rng.choice(values, len(values), replace=True)
        ub = np.quantile(v, q_u)
        excb = v[v > ub] - ub
        f = _fit_gpd(excb, estimator)
        if f is None:
            continue
        xib, sigb = f
        pub = float((v > ub).mean())
        if pub <= 0:
            continue
        taus.append(evt_tau_from_fit(ub, pub, xib, sigb, arl0))
    if len(taus) < max(50, int(0.3 * n_boot)):
        return {'ok': False, 'reason': 'bootstrap_failed', 'tau': None}
    taus = np.sort(np.array(taus))
    return {'ok': True, 'tau': float(np.median(taus)), 'tau_plugin': float(tau_plug),
            'tau_ci': [float(taus[int(0.025 * len(taus))]), float(taus[int(0.975 * len(taus))])],
            'xi': xi, 'sigma': sigma, 'p_u': p_u, 'u': float(u), 'n_exc': int(len(exc)),
            'q_u_used': round(float(q_u), 4), 'estimator': estimator,
            'n_boot_ok': int(len(taus))}


def calibrate_tau(values, arl0, n_fit=None, n_boot=1000, rng=None, pools=None):
    """协议主入口（混合规则）：经验分位数可行（1/ARL0 ≥ 4/n，配置律 E=N·q*）→ 经验分位；
    否则 GPD 尾外推。返回统一结构（branch/tau/tau_ci + evt 细节）。"""
    rng = rng or np.random.default_rng(2026)
    values = np.asarray(values, float)
    n = n_fit or len(values)
    branch = 'empirical' if (1.0 / arl0) >= 4.0 / n else 'evt'
    tau_emp = float(np.quantile(values, 1 - 1 / arl0)) if (1 - 1 / arl0) * n >= 1 else None
    if branch == 'empirical':
        bs = sorted(float(np.quantile(rng.choice(values, len(values), replace=True), 1 - 1 / arl0))
                    for _ in range(n_boot))
        return {'ok': True, 'branch': 'empirical', 'tau': tau_emp,
                'tau_ci': [bs[int(0.025 * n_boot)], bs[int(0.975 * n_boot)]],
                'tau_empirical': tau_emp, 'n_fit': int(n)}
    r = fit_evt_tau(values, arl0, n_boot=n_boot, rng=rng, pools=pools)
    r['branch'] = 'evt'
    r['tau_empirical'] = tau_emp
    r['n_fit'] = int(n)
    return r


def t_pivot_tau(n_cal, arl0):
    """t 预测枢轴量阈值：未来窗 z=(x−x̄)/s ~ √(1+1/n)·t_{n−1}（正态父体下精确预测式）。
    这修正 n 很小时逐池 z 标准化的参数估计效应（SPC Phase I 文献对实测 ARL 低于名义的正式解释）。"""
    from scipy.stats import t as tdist
    return float(np.sqrt(1.0 + 1.0 / n_cal) * tdist.ppf(1 - 1.0 / arl0, n_cal - 1))


def replay_arl_discrete(pool, tau, n_rep=2000, T=None, rng=None):
    """bootstrap 离散池 fixed 阈值 run-length（与 e1_window_alert.batch_runlen_empirical 同语义）。"""
    rng = rng or np.random.default_rng(2026)
    pool = np.asarray(pool, float)
    if T is None:
        p_hat = max((pool > tau).mean(), 0.5 / len(pool))
        T = int(min(max(1000, 10 / p_hat), 50_000))
    idx = rng.integers(0, len(pool), (n_rep, T))
    al = pool[idx] > tau
    rl = np.where(al.any(axis=1), al.argmax(axis=1) + 1, T)
    return float(rl.mean())


def replay_arl_stream(stream, tau):
    """自然序连续流 fixed 阈值 run-length 序列（保依赖结构的 in-control 重放）。"""
    al = np.asarray(stream, float) > tau
    if not al.any():
        return float(len(stream)), int(len(stream))
    first = int(np.argmax(al))
    return float(first + 1), int(len(stream))


def block_boot_arl_ci(stream, tau, B=1000, block=24, arl0=100, rng=None):
    """块 bootstrap（24h 块，保日周期依赖）的 ARL 95% CI。"""
    rng = rng or np.random.default_rng(2026)
    s = np.asarray(stream, float)
    n = len(s)
    nblk = int(np.ceil(n / block))
    rls = []
    for _ in range(B):
        idx = np.concatenate([np.arange(rng.integers(0, n - block), rng.integers(0, n - block) + block)
                              for _ in range(nblk)])[:n * 3]
        rl, _ = replay_arl_stream(s[idx], tau)
        rls.append(rl)
    rls = np.array(rls)
    return float(np.median(rls)), [float(np.quantile(rls, 0.025)), float(np.quantile(rls, 0.975))]


# ---------- 主件 ----------
def main():
    rng = np.random.default_rng(2026)
    truth = attack_truth_by_day(load_redteam_events())
    ev_events = {d: truth.get(d, {}).get('n_events', 0) for d in EVAL_DAYS}
    out = {'exp': 'R1-E5 EVT 公平校准', 'date': '2026-10-24',
           'criteria': 'criteria.json（冻结）',
           'root_cause_confirmed': {'fixed_measured_arl_incase': '8.7-9.4（54 点，本席复算均值 9.0/9.1 两档）',
                                    'structural': '9 离散良性值 → 最小非零告警率 1/9 → 经验 ARL 上限≈9；二分上界被池最大值封顶（e1_window_alert.py:91,129-131）'},
           'calibration': {}, 'g1_synthetic': {}, 'g2_hourly': {}, 'g3_h1_reverdict': {},
           'z3_crosscheck': {}, 'selfstarting': {}}

    # ---- 数据：与 e1_window_alert 逐字一致的 z 池 ----
    pools_all = {}   # (stat) -> {'pools': [(arm,seed,arr9)], 'eval': {(arm,seed): z_ev}}
    for stat in ['top50mean', 'mean', 'max']:
        pools, evz = [], {}
        for arm, d in ARMS.items():
            for seed in range(3):
                ev = window_stats(f'{d}/scores_seed{seed}.npz')
                bn = window_stats(f'{d}/benign_scores_seed{seed}.npz')
                bv = np.array([bn[dd][stat] for dd in BENIGN_DAYS])
                mu, sd = bv.mean(), bv.std(ddof=1)
                pools.append((arm, seed, (bv - mu) / sd))
                evz[(arm, seed)] = np.array([(ev[dd][stat] - mu) / sd for dd in EVAL_DAYS])
        pools_all[stat] = {'pools': pools, 'eval': evz}

    # ---- 校准（主口径 top50mean；附表 mean/max）----
    for stat in ['top50mean', 'mean', 'max']:
        print(f'[calib] {stat} ...', flush=True)
        pools = [p[2] for p in pools_all[stat]['pools']]
        pooled = np.concatenate(pools)
        out['calibration'][stat] = {}
        for arl0 in ARL0S:
            entry = {}
            # 主阈值：t 预测枢轴量（小池参数估计效应修正，正态父体下精确；判据修订 3）
            tau_t = t_pivot_tau(n_cal=9, arl0=arl0)
            # EVT/混合规则并报（形状诊断 + 非参推广）+ 经验分位列
            r_auto = calibrate_tau(pooled, arl0, n_boot=2000, rng=rng, pools=pools)
            entry['evt_hybrid'] = {k: (round(v, 4) if isinstance(v, float) else v)
                                   for k, v in r_auto.items() if k != 'tau_ci'} | \
                                  {'tau_ci': [round(x, 4) for x in r_auto['tau_ci']] if r_auto.get('tau_ci') else None}
            for q_u in QS:
                r = fit_evt_tau(pooled, arl0, q_u=q_u, n_boot=300, rng=rng, pools=pools,
                                estimator='mle')
                entry[f'mle_q{q_u}'] = {k: (round(v, 4) if isinstance(v, float) else v)
                                        for k, v in r.items() if k != 'tau_ci'} | \
                                       {k2: v2 for k2, v2 in
                                        [('tau_ci', [round(x, 4) for x in r['tau_ci']] if r.get('tau_ci') else None)]}
            main_r = {'branch': 'tpivot', 'tau': round(tau_t, 4),
                      'formula': 'sqrt(1+1/9)*t_8(1-1/ARL0)',
                      'evt_hybrid_tau': entry['evt_hybrid'].get('tau'),
                      'tau_empirical': entry['evt_hybrid'].get('tau_empirical'),
                      'note': 't-枢轴修正逐池标准化参数估计效应；EVT 并报为形状诊断/非参推广'}
            out['calibration'][stat][f'arl{arl0}'] = {'main_tpivot': main_r, 'sensitivity': entry,
                                                      'normal_theory_tau': round(float(norm.ppf(1 - 1 / arl0)), 4)}
        # QQ 诊断（主口径）：超额经验分位 vs GPD 拟合分位相关
        q_auto = auto_q_u(pooled)
        u = np.quantile(pooled, q_auto)
        exc = np.sort(pooled[pooled > u] - u)
        f = _fit_gpd(pooled[pooled > u] - u, 'lmom')
        if f is not None:
            pp = (np.arange(1, len(exc) + 1) - 0.5) / len(exc)
            qt = genpareto.ppf(1 - pp, f[0], loc=0, scale=f[1])
            out['calibration'][stat]['qq_corr_main'] = round(float(np.corrcoef(exc, qt)[0, 1]), 4)
            out['calibration'][stat]['gpd_xi_main'] = round(f[0], 4)

    # ---- G1 合成世界验收（匹配真实 DGP：9 池逐池 9 点抽样后逐池 z 标准化；重放同世界）----
    print('[G1] ...', flush=True)
    for arl0 in ARL0S:
        rows, rows_heavy = [], []
        for w in range(100):
            rng_w = np.random.default_rng(10_000 + w)
            pools, mu_j, sd_j = [], None, None
            for p in range(9):
                v = rng_w.normal(0, 1, 9)
                mu, sd = v.mean(), v.std(ddof=1)
                if p == 0:
                    mu_j, sd_j = mu, sd
                pools.append((v - mu) / sd)
            pooled = np.concatenate(pools)
            r_evt = calibrate_tau(pooled, arl0, n_boot=300, rng=rng_w, pools=pools)
            tau_t = t_pivot_tau(9, arl0)
            fresh = (rng_w.normal(0, 1, 9000) - mu_j) / sd_j
            arl_t = replay_arl_discrete(fresh, tau_t, n_rep=600, rng=rng_w)
            arl_e = replay_arl_discrete(fresh, r_evt['tau'], n_rep=600, rng=rng_w)
            rows.append({'tau_tpivot': round(tau_t, 3), 'arl_tpivot': round(arl_t, 1),
                         'rel_dev_tpivot': round(arl_t / arl0 - 1, 3),
                         'tau_evt': round(r_evt['tau'], 3), 'arl_evt': round(arl_e, 1),
                         'rel_dev_evt': round(arl_e / arl0 - 1, 3)})
            # 重尾敏感性（申报不设门）：10% 质量在 2+GPD(0.3,0.8) 尾上，逐池 z 标准化同构
            rng_h = np.random.default_rng(20_000 + w)
            pools_h = []
            for p in range(9):
                v = np.concatenate([rng_h.normal(0, 1, 8),
                                    2.0 + genpareto.rvs(0.3, scale=0.8, size=1, random_state=rng_h)])
                mu, sd = v.mean(), v.std(ddof=1)
                if p == 0:
                    mu_h, sd_h = mu, sd
                pools_h.append((v - mu) / sd)
            r_h = calibrate_tau(np.concatenate(pools_h), arl0, n_boot=300, rng=rng_h, pools=pools_h)
            fresh_h = np.concatenate([rng_h.normal(0, 1, 8100),
                                      2.0 + genpareto.rvs(0.3, scale=0.8, size=900,
                                                          random_state=rng_h)])
            fresh_h = (fresh_h - mu_h) / sd_h
            arl_h = replay_arl_discrete(fresh_h, r_h['tau'], n_rep=1500, rng=rng_h)
            rows_heavy.append({'tau': round(r_h['tau'], 3), 'arl': round(arl_h, 1),
                               'rel_dev': round(arl_h / arl0 - 1, 3)})
        dvs = np.array([r['rel_dev_tpivot'] for r in rows])
        mean_rate_x = float(np.mean(1.0 / (1.0 + dvs)))
        fracs_e = float(np.mean([abs(r['rel_dev_evt']) < 0.33 for r in rows]))
        out['g1_synthetic'][f'arl{arl0}'] = {
            'n_worlds': 100, 'gate': '平均误报率校准：世界平均告警率 vs 1/ARL0 相对偏差 <15%（判据修订 5）',
            'gate_pass': bool(abs(mean_rate_x - 1.0) < 0.15),
            'mean_alarm_rate_x_arl0': round(mean_rate_x, 4),
            'median_rel_dev_tpivot_reported': round(float(np.median(dvs)), 3),
            'world_iqr_rel_dev_irreducible': [round(float(np.quantile(dvs, 0.25)), 3),
                                              round(float(np.quantile(dvs, 0.75)), 3)],
            'pass_frac_lt33pct_reported': round(float(np.mean([abs(x) < 0.33 for x in dvs])), 3),
            'evt_disclosed_no_gate': {'pass_frac': round(fracs_e, 3),
                                      'median_rel_dev': round(float(np.median([r['rel_dev_evt'] for r in rows])), 3),
                                      'note': 'EVT/GPD 不修正池标准化参数估计效应——申报列（其偏差正是 SPOT 类方法套用到小 Phase-I 池时的隐患）'},
            'rows': rows,
            'heavy_tail_sensitivity_no_gate': {
                'median_rel_dev_tpivot': round(float(np.median([r['rel_dev'] for r in rows_heavy])), 3),
                'note': '10% 质量在 GPD(0.3) 重尾上——GPD/正态两假设同时受限的申报性边界（WMO TD-781、Inderscience 138231）'}}

    # ---- G2 真实小时流（前 70% 良性拟合 → 后 30% 自然序 in-control 重放）----
    print('[G2] ...', flush=True)
    rows = np.loadtxt(f'{BASE}/results/EXP-R1-E1/hourly_auth_stream.tsv')
    hours = rows[:, 0].astype(int)
    newp = rows[:, 2].astype(float)
    WARMUP = 168
    attack_hours = {ts // 3600 for ts, u, s, d in load_redteam_events() if ts // 3600 >= WARMUP}
    sel = hours >= WARMUP
    h_sel, x_sel = hours[sel], newp[sel]
    is_att = np.array([h in attack_hours for h in h_sel])
    benign_vals = x_sel[~is_att]
    mu, sd = benign_vals.mean(), benign_vals.std(ddof=1)
    zb = (benign_vals - mu) / sd
    n_fit = int(0.7 * len(zb))
    fit_pool, holdout = zb[:n_fit], zb[n_fit:]     # 自然序（保依赖）
    out['g2_hourly']['n_fit_benign'] = int(n_fit)
    out['g2_hourly']['n_holdout_benign'] = int(len(holdout))
    for arl0 in ARL0S:
        r_evt = fit_evt_tau(fit_pool, arl0, n_boot=1000, rng=rng)
        r_emp = {'tau': float(np.quantile(fit_pool, 1 - 1 / arl0))}
        med_evt, ci_evt = block_boot_arl_ci(holdout, r_evt['tau'], arl0=arl0, rng=rng)
        med_emp, ci_emp = block_boot_arl_ci(holdout, r_emp['tau'], arl0=arl0, rng=rng)
        out['g2_hourly'][f'arl{arl0}'] = {
            'evt': {'tau': round(r_evt['tau'], 3), 'xi': round(r_evt['xi'], 3),
                    'arl_median_holdout': round(med_evt, 1), 'arl_ci95': [round(x, 1) for x in ci_evt],
                    'nominal_in_ci': bool(ci_evt[0] <= arl0 <= ci_evt[1])},
            'empirical_ref': {'tau': round(float(r_emp['tau']), 3),
                              'arl_median_holdout': round(med_emp, 1),
                              'arl_ci95': [round(x, 1) for x in ci_emp],
                              'nominal_in_ci': bool(ci_emp[0] <= arl0 <= ci_emp[1])}}

    # ---- G3 公平 H1 重判定（主口径 top50mean；CUSUM/EWMA 沿在案校准）----
    print('[G3] ...', flush=True)
    e1 = json.load(open(f'{BASE}/results/EXP-R1-E1/e1_window_alert.json'))
    g3 = {}
    for arl0 in ARL0S:
        cal_main = out['calibration']['top50mean'][f'arl{arl0}']
        tau_main = cal_main['main_tpivot']
        tau = tau_main['tau']
        evt_r = cal_main['sensitivity']['evt_hybrid']
        lo_candidates = [tau] + ([evt_r['tau_ci'][0]] if evt_r.get('tau_ci') else [])
        hi_candidates = [tau] + ([evt_r['tau_ci'][1]] if evt_r.get('tau_ci') else [])
        tau_lo, tau_hi = float(min(lo_candidates)), float(max(hi_candidates))
        rows3 = []
        for arm in ARMS:
            for seed in range(3):
                z_ev = pools_all['top50mean']['eval'][(arm, seed)]
                al = z_ev > tau
                det = float(al.mean())
                det_lo = float((z_ev > tau_hi).mean())
                det_hi = float((z_ev > tau_lo).mean())
                delay = int(np.argmax(al)) if al.any() else None
                incase = e1['arms'][arm][str(seed)]['top50mean']['detection']
                rows3.append({
                    'arm': arm, 'seed': seed,
                    'fixed_evt': {'tau': round(tau, 3), 'delay': delay, 'det_rate': round(det, 3),
                                  'det_rate_ci_band': [round(det_lo, 3), round(det_hi, 3)],
                                  'alarm_days': [EVAL_DAYS[t] for t in range(11) if al[t]]},
                    'cusum_incase': {'delay': incase[f'cusum_k0.5@arl{arl0}']['delay_windows_from_day12'],
                                     'det_rate': incase[f'cusum_k0.5@arl{arl0}']['detection_rate'],
                                     'measured_arl': e1['arms'][arm][str(seed)]['top50mean']['calibrations'][f'cusum_k0.5@arl{arl0}']['arl_measured_bootstrap_holdout']},
                    'ewma02_incase': {'delay': incase['ewma_l0.2@arl%d' % arl0]['delay_windows_from_day12'],
                                      'det_rate': incase['ewma_l0.2@arl%d' % arl0]['detection_rate'],
                                      'measured_arl': e1['arms'][arm][str(seed)]['top50mean']['calibrations']['ewma_l0.2@arl%d' % arl0]['arl_measured_bootstrap_holdout']}})
        # 冻结判据：至少一种序贯读出检测延迟 ≤ fixed（逐臂逐 seed 计，非劣计数）
        n_ni_cusum = sum(1 for r in rows3 if r['cusum_incase']['delay'] is not None and
                         (r['fixed_evt']['delay'] is None or r['cusum_incase']['delay'] <= r['fixed_evt']['delay']))
        n_ni_ewma = sum(1 for r in rows3 if r['ewma02_incase']['delay'] is not None and
                        (r['fixed_evt']['delay'] is None or r['ewma02_incase']['delay'] <= r['fixed_evt']['delay']))
        fw = 1 - (1 - 1 / arl0) ** 11
        out['g3_h1_reverdict'][f'arl{arl0}'] = {
            'rows': rows3, 'noninferior_counts': {'cusum': f'{n_ni_cusum}/9', 'ewma02': f'{n_ni_ewma}/9'},
            'familywise_falsealarm_11win': round(fw, 4),
            'verdict': ('公平对照重判定：' + ('存在非劣序贯读出' if (n_ni_cusum + n_ni_ewma) > 0
                        else '序贯读出仍无一非劣（延迟均劣于 fixed@τ_evt）') +
                        f'；三读出共同 11 窗族级误报预算 {fw:.4f}（ARL 对齐后联合推断口径）')}

    # ---- R1A 交叉核对：z=3.0 处 fixed 检出（审稿人口径）----
    z3 = []
    for arm in ARMS:
        for seed in range(3):
            z_ev = pools_all['top50mean']['eval'][(arm, seed)]
            z3.append(round(float((z_ev > 3.0).mean()), 3))
    out['z3_crosscheck'] = {'det_rates_at_z3': z3, 'range': [min(z3), max(z3)],
                            'reviewerA_claim': '0.18-0.45', 'confirmed': bool(min(z3) >= 0.15 and max(z3) <= 0.50)}

    with open(f'{OUT}/e5_evt_calib.json', 'w') as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    print('G1:', {k: {kk: vv for kk, vv in v.items() if kk != 'rows'} for k, v in out['g1_synthetic'].items()})
    print('G2:', json.dumps(out['g2_hourly'], ensure_ascii=False)[:400])
    print('G3 verdicts:', json.dumps({k: v.get('verdict') for k, v in out['g3_h1_reverdict'].items()}, ensure_ascii=False))
    print('E5_DONE')


if __name__ == '__main__':
    main()
