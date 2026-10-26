"""e1_window_alert.py — R1-E1 窗级 CUSUM/EWMA 告警读出主件（日窗模型分数流）。

【实验设计自查（开工前冻结）】
H1：对边级分数流做窗级序贯检验（阈值在纯良性窗校准），实测 ARL 对齐（ARL0≈100/500 窗两档）下，
  窗级检测延迟非劣于固定阈值读出。
自变量：读出{fixed, CUSUM(k=0.5), EWMA(λ=0.2), EWMA(λ=0.3)} × 臂{full,homo,homoflows}(60ep 自产) ×
  3 seeds × ARL0{100,500}。
因变量：首告警窗/检测延迟、检测率（11 攻击窗覆盖率）、per-alert 事件强度、实测 ARL（名义并报，W2）。
窗统计量（预登记）：s_d = 当日 LOGON 观测边异常分 top-50 mean（附表 mean/max）。
良性校准窗：9 个无 redteam 事件日（day10,11,16-19,23-25）的窗统计量；标准化后 bootstrap 有放回
  构造校准流（保留经验分布重尾）；holdout 流实测 ARL（W2 两列）。
对照：①同 ARL0 固定阈值（同误报预算公平比较）；②随机置换基线（1000 次，检测率经验 p 值）。
成功判据（冻结）：至少一种序贯读出在 ARL 对齐下检测延迟 ≤ 固定阈值 → 主件；否则如实报降级附表
  （"读出方式不改变臂间结论"）。
红线：检测延迟非劣≠检测率更优；未检出显著差异≠等效。
"""
import json, sys
import numpy as np
from collections import defaultdict
sys.path.insert(0, '/root/autodl-tmp/projB_hetero_graph/src')
from seq_readout import cusum_stream, ewma_stream, fixed_stream, goel_wu_arl0
from r4b_common import load_redteam_events, attack_truth_by_day

ARMS = {
    'full': '/root/autodl-tmp/projB_hetero_graph/results/EXP-R1-E1/arm_full_60ep',
    'homoflows': '/root/autodl-tmp/projB_hetero_graph/results/EXP-R1-E1/arm_homoflows_60ep',
    'homo': '/root/autodl-tmp/projB_hetero_graph/results/EXP-R1-E1/arm_homo_60ep',
}
OUT = '/root/autodl-tmp/projB_hetero_graph/results/EXP-R1-E1'
EVAL_DAYS = [12, 13, 14, 15, 20, 21, 22, 26, 27, 28, 29]
BENIGN_DAYS = [10, 11, 16, 17, 18, 19, 23, 24, 25]
TOPK = 50
ARL0S = [100, 500]
READOUTS = ['fixed', 'cusum_k0.5', 'ewma_l0.2', 'ewma_l0.3']

truth = attack_truth_by_day(load_redteam_events())
ev_events = {d: truth.get(d, {}).get('n_events', 0) for d in EVAL_DAYS}


def window_stats(npz_path):
    """逐日窗统计量：top-50 mean / mean / max（LOGON 行）。返回 {day: {stat: value}}。"""
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


def batch_runlen_empirical(idx_mat, pool, kind, p1, p2):
    """bootstrap 经验流批量 run-length。idx_mat: (n_rep,T) 索引→pool 取 z。"""
    Z = pool[idx_mat]
    n, T = Z.shape
    if kind == 'cusum':
        S = np.zeros(n); hit = np.zeros(n, dtype=bool); rl = np.full(n, T, dtype=np.int64)
        for t in range(T):
            S = np.maximum(0.0, S + Z[:, t] - p1)
            al = S > p2
            rl[np.where(al & ~hit)[0]] = t + 1
            hit |= al
            S[al] = 0.0
    elif kind == 'ewma':
        acc = np.zeros(n); thr = p2 * np.sqrt(p1 / (2 - p1))
        hit = np.zeros(n, dtype=bool); rl = np.full(n, T, dtype=np.int64)
        for t in range(T):
            acc = p1 * Z[:, t] + (1 - p1) * acc
            al = acc > thr
            rl[np.where(al & ~hit)[0]] = t + 1
            hit |= al
    else:
        al = Z > p2
        rl = np.where(al.any(axis=1), al.argmax(axis=1) + 1, T).astype(np.int64)
    return rl


def calibrate_empirical(kind, arl_target, pool, rng, lam=None, k=0.5, n_rep=300, T=None,
                        max_iter=26, tol=0.06):
    """在良性 bootstrap 池上二分校准到 ARL0。返回 (p1, p2, measured_arl)。"""
    T = T or int(arl_target * 25)
    if kind == 'cusum':
        p1 = float(k)
    elif kind == 'ewma':
        p1 = float(lam)
    else:
        # fixed：直接在经验池上二分 τ
        lo, hi = np.percentile(pool, 50), np.percentile(pool, 99.999)
        for _ in range(24):
            mid = (lo + hi) / 2
            rl = batch_runlen_empirical(rng.integers(0, len(pool), (n_rep, T)), pool, 'fixed', None, mid)
            m = rl.mean()
            if abs(m - arl_target) / arl_target < tol:
                return None, float(mid), float(m)
            if m > arl_target:
                hi = mid
            else:
                lo = mid
        return None, float((lo + hi) / 2), float(m)
    lo, hi = 0.5, 60.0
    m = None
    for _ in range(max_iter):
        mid = (lo + hi) / 2
        rl = batch_runlen_empirical(rng.integers(0, len(pool), (n_rep, T)), pool, kind, p1, mid)
        m = rl.mean()
        if abs(m - arl_target) / arl_target < tol:
            return p1, float(mid), float(m)
        if m > arl_target:
            hi = mid
        else:
            lo = mid
    return p1, float((lo + hi) / 2), float(m)


def main():
    rng = np.random.default_rng(2026)
    results = {'design': '见模块 docstring（预登记冻结）', 'TOPK': TOPK, 'ARL0S': ARL0S,
               'arms': {}}
    for arm, d in ARMS.items():
        results['arms'][arm] = {}
        for seed in range(3):
            ev = window_stats(f'{d}/scores_seed{seed}.npz')
            bn = window_stats(f'{d}/benign_scores_seed{seed}.npz')
            res_seed = {}
            for stat in ['top50mean', 'mean', 'max']:
                benign_vals = np.array([bn[dd][stat] for dd in BENIGN_DAYS])
                mu, sd = benign_vals.mean(), benign_vals.std(ddof=1)
                zpool = (benign_vals - mu) / sd          # 标准化良性窗池（bootstrap 源）
                z_ev = np.array([(ev[dd][stat] - mu) / sd for dd in EVAL_DAYS])
                seed_stat = {'benign_mu': float(mu), 'benign_sd': float(sd),
                             'z_eval_windows': [round(float(x), 3) for x in z_ev],
                             'calibrations': {}, 'detection': {}}
                for arl0 in ARL0S:
                    for ro in READOUTS:
                        kind = ('cusum' if 'cusum' in ro else
                                'ewma' if 'ewma' in ro else 'fixed')
                        lam = float(ro.split('l')[1]) if 'ewma' in ro else None
                        p1, p2, m_cal = calibrate_empirical(kind, arl0, zpool, rng, lam=lam)
                        # holdout 实测 ARL（W2：名义 vs 实测两列）
                        rl = batch_runlen_empirical(
                            rng.integers(0, len(zpool), (2000, int(arl0 * 20))),
                            zpool, kind, p1, p2)
                        arl_meas = float(rl.mean())
                        # 真实 11 窗检测
                        if kind == 'cusum':
                            _, al = cusum_stream(z_ev, p1, p2)
                        elif kind == 'ewma':
                            _, al = ewma_stream(z_ev, lam, p2)
                        else:
                            al = fixed_stream(z_ev, p2)
                        first_alarm = int(np.argmax(al)) if al.any() else None
                        det_rate = float(al.mean())
                        alarm_ev = [ev_events[EVAL_DAYS[t]] for t in range(len(EVAL_DAYS)) if al[t]]
                        noalarm_ev = [ev_events[EVAL_DAYS[t]] for t in range(len(EVAL_DAYS)) if not al[t]]
                        # 点二列相关（告警窗 vs 事件强度）
                        pb = None
                        if 0 < al.sum() < len(al):
                            pb = float(np.corrcoef(al.astype(float), 
                                                   np.array([ev_events[d] for d in EVAL_DAYS]))[0, 1])
                        seed_stat['calibrations'][f'{ro}@arl{arl0}'] = {
                            'p1': round(p1, 4) if p1 is not None else None,
                            'p2': round(p2, 4),
                            'arl_nominal': arl0, 'arl_measured_bootstrap_holdout': round(arl_meas, 1),
                            'arl_calibration': round(m_cal, 1)}
                        seed_stat['detection'][f'{ro}@arl{arl0}'] = {
                            'first_alarm_win_idx': first_alarm,
                            'first_alarm_day': EVAL_DAYS[first_alarm] if first_alarm is not None else None,
                            'delay_windows_from_day12': first_alarm if first_alarm is not None else None,
                            'detection_rate': round(det_rate, 3),
                            'n_alarm_windows': int(al.sum()),
                            'alarm_windows_days': [EVAL_DAYS[t] for t in range(len(EVAL_DAYS)) if al[t]],
                            'mean_events_in_alarm_windows': (round(float(np.mean(alarm_ev)), 1)
                                                             if alarm_ev else None),
                            'mean_events_in_nonalarm_windows': (round(float(np.mean(noalarm_ev)), 1)
                                                                if noalarm_ev else None),
                            'pointbiserial_alarm_vs_events': round(pb, 3) if pb is not None else None}
                # 随机置换基线（同读出配置下首告警分布的随机参照——检测率经验 p 值）
                perm_rates = {f'{ro}@arl{arl0}': [] for arl0 in ARL0S for ro in READOUTS}
                for pi in range(1000):
                    perm = rng.permutation(z_ev)
                    for arl0 in ARL0S:
                        for ro in READOUTS:
                            c = seed_stat['calibrations'][f'{ro}@arl{arl0}']
                            if 'cusum' in ro:
                                _, alp = cusum_stream(perm, c['p1'], c['p2'])
                            elif 'ewma' in ro:
                                _, alp = ewma_stream(perm, float(ro.split('l')[1]), c['p2'])
                            else:
                                alp = fixed_stream(perm, c['p2'])
                            perm_rates[f'{ro}@arl{arl0}'].append(float(alp.mean()))
                for key, arr in perm_rates.items():
                    obs = seed_stat['detection'][key]['detection_rate']
                    seed_stat['detection'][key]['random_perm_ref_mean_rate'] = round(float(np.mean(arr)), 3)
                    seed_stat['detection'][key]['empirical_p_vs_random'] = round(
                        float((np.array(arr) >= obs).mean()), 4)
                res_seed[stat] = seed_stat
            results['arms'][arm][seed] = res_seed
            print(f'{arm} seed{seed} done', flush=True)
    # H1 判据汇总：CUSUM/EWMA vs fixed 的检测延迟（主口径 top50mean）
    h1 = {}
    for arm in ARMS:
        for seed in range(3):
            for arl0 in ARL0S:
                dd = {ro: results['arms'][arm][seed]['top50mean']['detection'][f'{ro}@arl{arl0}']
                      for ro in READOUTS}
                key = f'{arm}/s{seed}/arl{arl0}'
                h1[key] = {ro: {'delay': dd[ro]['delay_windows_from_day12'],
                                'det_rate': dd[ro]['detection_rate']} for ro in READOUTS}
    results['h1_delay_comparison_top50mean'] = h1
    with open(f'{OUT}/e1_window_alert.json', 'w') as f:
        json.dump(results, f, indent=1, ensure_ascii=False)
    print('E1_DONE')


if __name__ == '__main__':
    main()
