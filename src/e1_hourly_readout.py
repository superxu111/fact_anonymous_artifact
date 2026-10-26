"""e1_hourly_readout.py — E1 附表：小时窗粒度对照（数据驱动流，无模型分数）。

【预登记口径】
流统计量 = 每小时新见 (user,dst) 对计数（pair novelty，经典 SOC 特征；auth.txt.gz 全量解析
EXP-R1-E1/hourly_auth_stream.tsv，1392 小时）。warm-up=前 168 小时（部署冷启动新对爆发，排除）。
攻击小时 = warm-up 后有 redteam 事件的小时；良性小时 bootstrap 构造校准流。
三读出（fixed/CUSUM k=0.5/EWMA λ=0.2）× ARL0{100,500}，同 e1_window_alert 主件校准程序。
语义：读出模块在更细运维粒度上的可用性对照（模型分数只有日粒度，此处为纯数据流）。
"""
import json, sys
import numpy as np
sys.path.insert(0, '/root/autodl-tmp/projB_hetero_graph/src')
from seq_readout import cusum_stream, ewma_stream, fixed_stream
from e1_window_alert import batch_runlen_empirical, calibrate_empirical
from r4b_common import load_redteam_events

OUT = '/root/autodl-tmp/projB_hetero_graph/results/EXP-R1-E1'
WARMUP = 168
ARL0S = [100, 500]
READOUTS = [('fixed', None), ('cusum', 0.5), ('ewma', 0.2)]

rows = np.loadtxt(f'{OUT}/hourly_auth_stream.tsv')
hours = rows[:, 0].astype(int)
newp = rows[:, 2].astype(float)
assert len(hours) == 1392 and hours[0] == 0
attack_hours = set()
for ts, u, s, d in load_redteam_events():
    if ts // 3600 >= WARMUP:
        attack_hours.add(ts // 3600)
sel = hours >= WARMUP
h_sel = hours[sel]
x_sel = newp[sel]
is_attack = np.array([h in attack_hours for h in h_sel])
benign_vals = x_sel[~is_attack]
mu, sd = benign_vals.mean(), benign_vals.std(ddof=1)
zpool = ((benign_vals - mu) / sd)
z_ev = (x_sel - mu) / sd
print(f'warm-up {WARMUP}h 后窗数={len(h_sel)}, 攻击小时={int(is_attack.sum())}, '
      f'良性小时={int((~is_attack).sum())}, new_pairs 良性 μ={mu:.0f} σ={sd:.0f}')

rng = np.random.default_rng(2026)
out = {'design': '见模块 docstring（预登记）', 'warmup_h': WARMUP,
       'n_windows': int(len(h_sel)), 'n_attack_windows': int(is_attack.sum()),
       'stat': 'hourly new (user,dst) pair count', 'readouts': {}}
for arl0 in ARL0S:
    for kind, lam in READOUTS:
        p1, p2, m_cal = calibrate_empirical(kind, arl0, zpool, rng, lam=lam)
        rl = batch_runlen_empirical(rng.integers(0, len(zpool), (2000, arl0 * 20)),
                                    zpool, kind, p1, p2)
        if kind == 'cusum':
            _, al = cusum_stream(z_ev, p1, p2)
        elif kind == 'ewma':
            _, al = ewma_stream(z_ev, lam, p2)
        else:
            al = fixed_stream(z_ev, p2)
        # 告警后重置（CUSUM 内建）；检测率=攻击小时覆盖率（任一时刻处于告警=覆盖）
        # EWMA/fixed 无状态——覆盖率=告警小时∩攻击小时
        cov = float((al & is_attack).sum() / is_attack.sum())
        # 首告警延迟（相对首个攻击小时）
        first_attack = int(np.argmax(is_attack))
        first_alarm = int(np.argmax(al)) if al.any() else None
        delay = (first_alarm - first_attack) if first_alarm is not None else None
        prec = float((al & is_attack).sum() / max(al.sum(), 1))
        out['readouts'][f'{kind}{lam if lam else ""}@arl{arl0}'] = {
            'p1': p1, 'p2': round(p2, 4), 'arl_measured_holdout': round(float(rl.mean()), 1),
            'n_alarm_windows': int(al.sum()), 'attack_coverage': round(cov, 3),
            'per_alert_precision': round(prec, 3),
            'first_alarm_hour_idx': first_alarm, 'first_attack_hour_idx': first_attack,
            'delay_h': delay}
        print(f"{kind}{lam}@{arl0}: cov={cov:.2f} prec={prec:.2f} delay={delay} "
              f"arl_meas={rl.mean():.0f}")
with open(f'{OUT}/e1_hourly_readout.json', 'w') as f:
    json.dump(out, f, indent=1)
print('E1_HOURLY_DONE')
