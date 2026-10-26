"""e11_g3_sensitivity.py — R2-E11（P0-1 / R2B-M2+R1A2-m7）：G3 单向敏感性 + det_rate_ci_band 重定义。

【实验设计自查（开工前冻结）】
验证假设（单向敏感性，R2B-M2 resolution test）：H1「存在非劣序贯读出」的非劣计数（arl500 9/9、
  arl100 7/9）对「fixed 用 t-枢轴保守阈值、序贯读出用在案 bootstrap 校准」的校准哲学不对称是否稳健——
  把 CUSUM/EWMA 阈值按同一枢轴保守因子 κ 等效收紧一档后，非劣计数是否保持。
自变量：读出{cusum_k0.5, ewma_l0.2} × κ 档{1.0（还原对照）, κ_t(保守档)} × ARL0{100,500} × 臂×seed。
  κ_t = τ_t(n=9,ARL0) / z_{1−1/ARL0}（t-枢轴相对正态理论阈值的膨胀因子 = 参数估计效应保守余量；
  arl100: 3.0531/2.3263=1.3124、arl500: 4.2068/2.8782=1.4616）——与 fixed 侧保守化同源、单向（只收紧序贯侧，
  对「存在非劣序贯」不利方向的稳健性检验）。
  〔R2 第四轮更正（R2A-M1/R3B-M2）：原 docstring 误写分母 3.2905（实为 z_{1−1/2000}）得 arl500 κ=1.2785，
  代码 :84 snorm.ppf(1−1/a) 一直正确（arl500 实为 z_{0.998}=2.8782→κ=1.4616，arl100 1.3124）；执行值以
  results/EXP-R2-E11/e11_g3_sensitivity.json:kappa_t_pivot_inflation={arl100:1.3124, arl500:1.4616} 为准
  （number_manifest 已入底账，E17 闸 2 lint 覆盖）。实际收紧强于登记，结论方向被强化而非削弱。〕
因变量：11 评估窗检出率/首告警延迟/非劣计数（判据与 e5_evt_calib.py:395-399 完全同式）。
对照：κ=1 档须还原在案 g3 计数（9/9、7/9），否则本实验无效（校准机不一致）。
规模/成本：纯 CPU，2 读出×2 档×2 ARL0×9 臂seed 校准（二分×批量 run-length），预计 <10 分钟。
成功判据：κ=1 还原在案；κ>1 结果如实申报——保持→H1 条件句强度确认；不保持→H1 条件句再弱化
  （预先登记两种去向，防有利方向修剪）。
断言：阈值↑→实测 ARL 单调不降；κ=1 还原计数逐位核对。

det_rate_ci_band 处置（R2B-M2③/R1A2-m7，先登记后处置）：
  旧字段 INVALIDATED——det_rate_ci_band=[det(τ_hi), det(τ_lo)] 的 τ_lo 混入已否决 EVT 分支
  tau_ci 下界（e5_evt_calib.py:371-373，EVT 日窗已被 ξ̂=−3.92 自家诊断否决）→ 带宽上界虚高至 1.0 无信息量。
  重定义（V-R2-7）：假设扰动带两种规格——①sd±10%（校准池标准差扰动）；②df∈{7,9} 自由度敏感性
  （t₇/t₉ 分位）——计算 fixed 检出率带（全网格 9 格口径），产出守卫带式申报表。历史 e5_evt_calib.json
  原文不改写（supersedes 留痕原则），处置登记于本产物与 R1_metrics 修复层。
"""
import json, os, sys
import numpy as np
from scipy.stats import t as tdist, norm as snorm

sys.path.insert(0, '/root/autodl-tmp/projB_hetero_graph/src')
from e1_window_alert import (ARMS, EVAL_DAYS, BENIGN_DAYS, window_stats,
                             calibrate_empirical)
from e5_evt_calib import t_pivot_tau

BASE = '/root/autodl-tmp/projB_hetero_graph'
OUT = f'{BASE}/results/EXP-R2-E11'
ARL0S = [100, 500]
SEQ = {'cusum_k0.5': ('cusum', None, 0.5), 'ewma_l0.2': ('ewma', 0.2, None)}


def build_pools():
    """与 e5_evt_calib.py:217-229 逐字一致的 z 池/评估 z（top50mean 主口径）。"""
    pools, evz = {}, {}
    for arm, d in ARMS.items():
        for seed in range(3):
            ev = window_stats(f'{d}/scores_seed{seed}.npz')
            bn = window_stats(f'{d}/benign_scores_seed{seed}.npz')
            bv = np.array([bn[dd]['top50mean'] for dd in BENIGN_DAYS])
            mu, sd = bv.mean(), bv.std(ddof=1)
            pools[(arm, seed)] = (bv - mu) / sd
            evz[(arm, seed)] = np.array([(ev[dd]['top50mean'] - mu) / sd for dd in EVAL_DAYS])
    return pools, evz


def replay_eval(z_ev, kind, lam, k, p1, p2):
    """11 评估窗首告警（与 e1_window_alert.py:148-151 同语义）。返回 (delay, det_rate)。"""
    from seq_readout import cusum_stream, ewma_stream
    if kind == 'cusum':
        _, al = cusum_stream(z_ev, p1, p2)
    else:
        _, al = ewma_stream(z_ev, lam, p2)
    delay = int(np.argmax(al)) if al.any() else None
    return delay, float(al.mean())


def _one_task(args):
    """单（arl0, 读出, 臂, seed, 档）校准+重放。并行粒度。"""
    arl0, ro, arm, seed, mode, kappa_v, pool, z_ev = args
    kind, lam, k = SEQ[ro]
    target = arl0 if mode == 'kappa1' else kappa_v * arl0
    rng = np.random.default_rng(90_000 + arl0 + (0 if mode == 'kappa1' else 777))
    p1, p2, m = calibrate_empirical(kind, target, pool, rng, lam=lam, k=k,
                                    n_rep=300, T=int(target * 25), max_iter=30, tol=0.06)
    d, det = replay_eval(z_ev, kind, lam, k, p1, p2)
    return (arl0, ro, arm, seed, mode,
            {'thr': round(float(p2), 3), 'arl_target': round(target, 1),
             'arl_measured': round(float(m), 1), 'delay': d, 'det': round(float(det), 3)})


def main():
    from concurrent.futures import ProcessPoolExecutor
    os.makedirs(OUT, exist_ok=True)
    pools, evz = build_pools()
    g3_incase = json.load(open(f'{BASE}/results/EXP-R1-E5/e5_evt_calib.json'))['g3_h1_reverdict']
    kappa = {a: float(t_pivot_tau(9, a) / snorm.ppf(1 - 1.0 / a)) for a in ARL0S}
    out = {'exp': 'R2-E11 G3 单向敏感性 + det_rate_ci_band 重定义（P0-1/R2B-M2+R1A2-m7）',
           'date': '2026-10-24', 'design': '见模块 docstring（开工前冻结，判据双去向预先登记）',
           'kappa_t_pivot_inflation': {f'arl{a}': round(kappa[a], 4) for a in ARL0S},
           'kappa_def': 'κ = τ_t(9,ARL0)/z_{1−1/ARL0}（t-枢轴相对正态理论阈值的保守膨胀因子；'
                        '序贯侧收紧目标 = κ·ARL0，与 fixed@τ_t 同源单向保守化）',
           'sensitivity': {}, 'det_rate_ci_band_disposition': {}}

    for arl0 in ARL0S:
        for ro in SEQ:
            tasks = []
            for (arm, seed), pool in sorted(pools.items()):
                for mode in ('kappa1', 'kappa_t'):
                    tasks.append((arl0, ro, arm, seed, mode, kappa[arl0], pool, evz[(arm, seed)]))
            res = {}
            with ProcessPoolExecutor(max_workers=18) as ex:
                for (a, r_, arm, seed, mode, info) in ex.map(_one_task, tasks):
                    res[(arm, seed, mode)] = info
                    print(f"  arl{a} {r_} {arm} s{seed} {mode}: thr={info['thr']} arl={info['arl_measured']} "
                          f"delay={info['delay']}", flush=True)
            rows = []
            for (arm, seed) in sorted(pools):
                inc = next(r for r in g3_incase[f'arl{arl0}']['rows']
                           if r['arm'] == arm and r['seed'] == seed)
                fd = inc['fixed_evt']['delay']
                o, t_ = res[(arm, seed, 'kappa1')], res[(arm, seed, 'kappa_t')]
                ni = lambda dd: (dd is not None and (fd is None or dd <= fd))
                assert t_['arl_measured'] >= o['arl_measured'] - max(6.0, 0.10 * o['arl_measured']), \
                    f'ARL 单调性断言失败 {arm}{seed} {ro} arl{arl0}'
                rows.append({'arm': arm, 'seed': seed,
                             'fixed_tpivot_delay': fd, 'fixed_tpivot_det': inc['fixed_evt']['det_rate'],
                             'kappa1': {**o, 'noninferior': bool(ni(o['delay']))},
                             'kappa_t': {**t_, 'noninferior': bool(ni(t_['delay']))}})
            inc_cnt = g3_incase[f'arl{arl0}']['noninferior_counts']
            key = {'cusum_k0.5': 'cusum', 'ewma_l0.2': 'ewma02'}[ro]
            cnt1 = sum(1 for r in rows if r['kappa1']['noninferior'])
            cntt = sum(1 for r in rows if r['kappa_t']['noninferior'])
            out['sensitivity'][f'arl{arl0}/{ro}'] = {
                'rows': rows,
                'noninferior_kappa1_restore': f'{cnt1}/9',
                'incase_reference': inc_cnt[key],
                'restore_match': f'{cnt1}/9' == inc_cnt[key],
                'noninferior_kappa_tightened': f'{cntt}/9',
                'count_maintained_under_tightening': bool(cntt == cnt1)}
            print(f"  == arl{arl0} {ro}: κ=1 还原 {cnt1}/9（在案 {inc_cnt[key]}，match={f'{cnt1}/9' == inc_cnt[key]}）"
                  f"→ κ_t 收紧 {cntt}/9", flush=True)

    # ---- det_rate_ci_band 处置（先登记后处置；历史产物不改写）----
    e5 = json.load(open(f'{BASE}/results/EXP-R1-E5/e5_evt_calib.json'))
    cal = e5['calibration']['top50mean']
    for arl0 in ARL0S:
        tau_t = t_pivot_tau(9, arl0)
        evt_tau_ci = cal[f'arl{arl0}']['sensitivity']['evt_hybrid'].get('tau_ci')
        g3rows = g3_incase[f'arl{arl0}']['rows']
        old_band = [r['fixed_evt']['det_rate_ci_band'] for r in g3rows]
        tau_lo_sd, tau_hi_sd = tau_t / 1.10, tau_t * 1.10
        tau_lo_df = float(np.sqrt(1 + 1 / 9) * tdist.ppf(1 - 1.0 / arl0, 9))
        tau_hi_df = float(np.sqrt(1 + 1 / 9) * tdist.ppf(1 - 1.0 / arl0, 7))
        spec_out = {}
        for spec, (tlo, thi) in {'sd_pm10pct': (min(tau_lo_sd, tau_hi_sd), max(tau_lo_sd, tau_hi_sd)),
                                 'df_7_to_9': (min(tau_lo_df, tau_hi_df), max(tau_lo_df, tau_hi_df))}.items():
            dets_lo, dets_hi = [], []
            for (arm, seed), z_ev in sorted(evz.items()):
                dets_hi.append(float((z_ev > tlo).mean()))
                dets_lo.append(float((z_ev > thi).mean()))
            spec_out[spec] = {'tau_band': [round(float(min(tlo, thi)), 4), round(float(max(tlo, thi)), 4)],
                              'det_rate_band_grid9': [round(min(dets_lo), 3), round(max(dets_hi), 3)],
                              'note': '检出率带=9 格（3 臂×3 seed）在 τ 带两端的检出率范围（全网格口径，'
                                      '不再取单格并报；R1A2-m1 教训）'}
        out['det_rate_ci_band_disposition'][f'arl{arl0}'] = {
            'old_field_status': 'INVALIDATED（supersedes 登记，历史 e5_evt_calib.json 原文不改写）',
            'invalidation_reason': f'det_rate_ci_band 由主阈值 τ_t 与已否决 EVT 分支 tau_ci 混合生成'
                                   f'（e5_evt_calib.py:371-373；EVT tau_ci={evt_tau_ci}，日窗 EVT 分支已被'
                                   f'ξ̂=−3.92 自家诊断否决）→ 带宽上界虚高至 1.0 无信息量（R1A2-m7/R2B-M2③）',
            'old_band_sample': old_band[:3],
            'redefined_guard_band': spec_out,
            'point_estimate': {'tau_t': round(float(tau_t), 4)}}

    with open(f'{OUT}/e11_g3_sensitivity.json', 'w') as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    summ = {k: {kk: v[kk] for kk in ('noninferior_kappa1_restore', 'incase_reference', 'restore_match',
                                     'noninferior_kappa_tightened', 'count_maintained_under_tightening')}
            for k, v in out['sensitivity'].items()}
    print('E11_DONE', json.dumps(summ, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
