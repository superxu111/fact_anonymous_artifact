"""seq_readout.py — R1-E1 窗级序贯告警读出：手写 CUSUM / EWMA / 固定阈值（statsmodels 缺，手写实现）。

公式口径（预登记冻结，对照 NIST e-Handbook §6.3.1.3 CUSUM / §6.3.1.4 EWMA）：
  标准化：z_t = (s_t − μ_b) / σ_b，(μ_b, σ_b) 由良性校准窗估计。
  CUSUM（单侧上侧，分数高=异常）：
      S_t = max(0, S_{t-1} + z_t − ν)，S_0 = 0；告警 S_t > h；告警后 S 重置 0（restart 语义）。
      ν = k·σ（标准化后 k=0.5 惯例），h 由目标 ARL0 在良性流上二分校准。
  EWMA：Z_t = λ·z_t + (1−λ)·Z_{t-1}，Z_0 = 0；告警 Z_t > L·σ_ewma，σ_ewma = sqrt(λ/(2−λ))；
      L 由 ARL0 二分校准；λ ∈ {0.2, 0.3}（NIST 推荐域）。
  固定阈值：z_t > τ；τ = Φ^{-1}(1 − 1/ARL0)（零漂正态理论），实测校验。
  ARL0 = 良性流平均 run-length（首次告警窗索引+1，截断按流长计）。
  解析对照：Goel-Wu/Wald 零漂近似 ARL0 ≈ (e^{2kh} − 1 − 2kh)/(2k²)，与蒙特卡洛的偏差实测申报。
"""
import numpy as np
from scipy.stats import norm


# ---------- 单条流递推（真实数据用，语义参考实现） ----------
def cusum_stream(z, nu, h, reset_on_alarm=True):
    z = np.asarray(z, dtype=np.float64)
    T = len(z)
    S = np.zeros(T)
    alarm = np.zeros(T, dtype=bool)
    s = 0.0
    for t in range(T):
        s = max(0.0, s + z[t] - nu)
        if s > h:
            alarm[t] = True
            if reset_on_alarm:
                s = 0.0
        S[t] = s
    return S, alarm


def ewma_stream(z, lam, L):
    z = np.asarray(z, dtype=np.float64)
    T = len(z)
    Z = np.zeros(T)
    alarm = np.zeros(T, dtype=bool)
    thr = L * np.sqrt(lam / (2.0 - lam))
    zacc = 0.0
    for t in range(T):
        zacc = lam * z[t] + (1 - lam) * zacc
        Z[t] = zacc
        alarm[t] = zacc > thr
    return Z, alarm


def fixed_stream(z, tau):
    return np.asarray(z, dtype=np.float64) > tau


# ---------- 批量蒙特卡洛 run-length（校准用，n_rep 条流并行递推） ----------
def batch_runlen(Z, kind, param1, param2):
    """Z: (n_rep, T) 良性标准化流。kind∈{cusum,ewma,fixed}。返回 run-length 向量 (n_rep,)。
    run-length = 首次告警窗索引+1；无告警 = T（截断计，ARL 估计偏保守下界——校准目标含此口径）。"""
    n, T = Z.shape
    if kind == 'cusum':
        S = np.zeros(n)
        hit = np.zeros(n, dtype=bool)
        rl = np.full(n, T, dtype=np.int64)
        for t in range(T):
            S = np.maximum(0.0, S + Z[:, t] - param1)
            al = S > param2
            rl[np.where(al & ~hit)[0]] = t + 1
            hit |= al
            S[al] = 0.0
    elif kind == 'ewma':
        acc = np.zeros(n)
        thr = param2 * np.sqrt(param1 / (2.0 - param1))
        hit = np.zeros(n, dtype=bool)
        rl = np.full(n, T, dtype=np.int64)
        for t in range(T):
            acc = param1 * Z[:, t] + (1 - param1) * acc
            al = acc > thr
            rl[np.where(al & ~hit)[0]] = t + 1
            hit |= al
    elif kind == 'fixed':
        al = Z > param2
        hit = al.any(axis=1)
        idx = np.where(al.any(axis=1), al.argmax(axis=1) + 1, T)
        rl = idx
    else:
        raise ValueError(kind)
    return rl


def calibrate(kind, arl_target, rng, lam=None, k=0.5, n_rep=300, seg_mult=25, tol=0.05, max_iter=26):
    """二分校准阈值参数到 ARL0≈arl_target（良性 iid N(0,1) 流）。
    返回 dict(kind, p1, p2, measured_arl, n_rep)。p1=ν( cusum)/λ(ewma)/None(fixed)。"""
    T = int(arl_target * seg_mult)
    if kind == 'cusum':
        p1 = float(k)

        def measure(p2):
            return batch_runlen(rng.standard_normal((n_rep, T)), 'cusum', p1, p2).mean()
        lo, hi = 0.5, 40.0
    elif kind == 'ewma':
        p1 = float(lam)

        def measure(p2):
            return batch_runlen(rng.standard_normal((n_rep, T)), 'ewma', p1, p2).mean()
        lo, hi = 0.5, 40.0
    elif kind == 'fixed':
        p1 = None
        p2 = float(norm.ppf(1 - 1.0 / arl_target))
        m = batch_runlen(rng.standard_normal((n_rep, T)), 'fixed', None, p2).mean()
        return {'kind': kind, 'p1': None, 'p2': p2, 'measured_arl': float(m), 'n_rep': n_rep,
                'T': T}
    else:
        raise ValueError(kind)
    for _ in range(max_iter):
        mid = (lo + hi) / 2
        m = measure(mid)
        if abs(m - arl_target) / arl_target < tol:
            return {'kind': kind, 'p1': p1, 'p2': float(mid), 'measured_arl': float(m),
                    'n_rep': n_rep, 'T': T}
        if m > arl_target:
            hi = mid
        else:
            lo = mid
    return {'kind': kind, 'p1': p1, 'p2': float((lo + hi) / 2), 'measured_arl': float(m),
            'n_rep': n_rep, 'T': T}


def measure_arl_stream(kind, p1, p2, rng, arl_target, n_rep=2000, seg_mult=20):
    """高精度实测 ARL（校准后复核）。截断口径同 batch_runlen。"""
    T = int(arl_target * seg_mult)
    rl = batch_runlen(rng.standard_normal((n_rep, T)), kind, p1, p2)
    return float(rl.mean()), float(np.percentile(rl, 10)), float(np.percentile(rl, 90))


def goel_wu_arl0(k, h):
    """Goel-Wu/Wald 零漂解析近似（量级对照用，近似偏差由测试实测申报）。"""
    kh = k * h
    return (np.exp(2 * kh) - 1 - 2 * kh) / (2 * k ** 2)
