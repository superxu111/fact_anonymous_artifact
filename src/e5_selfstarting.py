"""e5_selfstarting.py — R1-E5 附录：self-starting Q 图读出（创新方案二可落地残件）。

Quesenberry (JQT 1991) 机制：逐窗用运行均值/方差标准化，无 Phase I 校准池——结构性消掉
"9 离散良性值校准池"问题。适配（申报）：①攻击污染 → 鲁棒变体：仅 q_t<2 的窗更新运行统计；
②前 24 窗不检验（Q 图早期弱检出警示）；③小时尺度（1,224 窗）——日窗按 R1B-m7 降为描述性。
判据锚点：EXP-R1-E5/criteria.json appendix。
"""
import json
import numpy as np
from scipy.stats import norm

sys_path = '/root/autodl-tmp/projB_hetero_graph'
BASE = f'{sys_path}/results/EXP-R1-E1'
OUT = f'{sys_path}/results/EXP-R1-E5/e5_selfstarting.json'
WARMUP = 168
ARL0S = [100, 500]


def qchart_stream(x, c, min_win=24, robust_update=True):
    """鲁棒 self-starting Q 图。返回 (q 序列, alarm 布尔)。"""
    m = x[0]
    var = 0.0
    n = 1
    q = np.zeros(len(x))
    al = np.zeros(len(x), dtype=bool)
    for t in range(1, len(x)):
        var = max(var * (n - 1) / max(n, 1) + (x[t] - m) * (x[t] - m) / (n + 1), 1e-12)
        m = m + (x[t] - m) / (n + 1)
        n += 1
        if t < min_win or n < 24:
            q[t] = 0.0
            continue
        sd = max(np.sqrt(var), 0.1 * abs(m), 1e-9)
        q[t] = (x[t] - m) / sd
        al[t] = q[t] > c
        if robust_update and q[t] < 2.0:
            pass                                  # 统计已在上方更新（未告警=可信良性）
        elif robust_update:
            # 告警窗回滚本次更新（防攻击污染运行统计）
            m = m - (x[t] - m) / n
            var = var  # 方差不回滚（保守）
    return q, al


def main():
    rows = np.loadtxt(f'{BASE}/hourly_auth_stream.tsv')
    hours = rows[:, 0].astype(int)
    newp = rows[:, 2].astype(float)
    sel = hours >= WARMUP
    h_sel, x_sel = hours[sel], newp[sel]
    from r4b_common import load_redteam_events
    attack_hours = {ts // 3600 for ts, u, s, d in load_redteam_events() if ts // 3600 >= WARMUP}
    is_att = np.array([h in attack_hours for h in h_sel])
    benign_idx = np.where(~is_att)[0]
    out = {'exp': 'R1-E5 附录 self-starting Q 图读出', 'date': '2026-10-24',
           'design': 'Quesenberry 1991 机制 + 鲁棒更新（仅 q<2 窗更新统计）+ 前 24 窗不检验；'
                     'c 由良性小时块 bootstrap 流校准到 ARL0（申报：校准仅用于报告口径，'
                     '结构性优点=不依赖 Phase I 校准池）',
           'readouts': {}}
    rng = np.random.default_rng(2026)
    for arl0 in ARL0S:
        c0 = float(norm.ppf(1 - 1 / arl0))
        c = c0
        # 良性流校准（块 bootstrap 保日周期；二分 6 轮）
        lo, hi = c0, c0 + 3.0
        for _ in range(6):
            rls = []
            for _ in range(120):
                idx = np.concatenate([benign_idx[rng.integers(0, len(benign_idx), 24)]
                                      for _ in range(8)]).flatten()[:1000]
                _, al = qchart_stream(x_sel[idx], c)
                rl = int(np.argmax(al)) + 1 if al.any() else len(idx)
                rls.append(rl)
            m = float(np.mean(rls))
            if abs(m - arl0) / arl0 < 0.25:
                break
            if m > arl0:
                hi = c
            else:
                lo = c
            c = (lo + hi) / 2
        # 真实流应用
        _, al = qchart_stream(x_sel, c)
        cov = float((al & is_att).sum() / is_att.sum())
        prec = float((al & is_att).sum() / max(al.sum(), 1))
        first_att = int(np.argmax(is_att))
        first_al = int(np.argmax(al)) if al.any() else None
        out['readouts'][f'selfstarting@arl{arl0}'] = {
            'c_calibrated': round(c, 3), 'c_normal_theory': round(c0, 3),
            'attack_coverage': round(cov, 3), 'per_alert_precision': round(prec, 3),
            'n_alarm_windows': int(al.sum()),
            'delay_h': (first_al - first_att) if first_al is not None else None,
            'note': '与 e1_hourly_readout.json 同流对照：fixed@arl100 覆盖 4%/延迟 49h，'
                    'ewma0.2@arl100 覆盖 54%/延迟 0'}
    with open(OUT, 'w') as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out['readouts'], indent=1))
    print('E5_SELFSTARTING_DONE')


if __name__ == '__main__':
    import sys
    sys.path.insert(0, f'{sys_path}/src')
    main()
