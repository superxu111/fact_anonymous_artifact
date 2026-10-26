"""smoke_r1_m2_calibration.py — M2 冒烟：MAGIC 距离分数上的阈值协议对比。
数据：FDUDSDE/MAGIC 官方发布 distance_save_{ds}.pkl + metadata.json + test pkl（作者预处理产物）。
坐标重建 = eval.py:105-121 协议（skip_benign=前 n_test-1 图节点和；test_idx=i>=skip_benign or malicious）。
对比协议：
  P1 MAGIC 代码协议：PR 曲线扫最优 F1 + 数据集硬编码召回下限（model/eval.py:199-205）
  P2 MAGIC 论文协议：朴素最优 F1（无 rec 下限）——纯测试标签线性搜索
  P3 固定分位数（无标签）：θ=score 的 {99%, 99.9%, 99.99%} 分位
  P4 POT-EVT（无标签）：95% 分位以上 GPD 拟合，取尾部概率 τ∈{1e-2,1e-3,1e-4} 对应阈值
另报 precision@K, K∈{1000, 10000}（预算受限口径）。
"""
import json, pickle, sys
import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_curve, precision_score, recall_score, f1_score

DS_REC_FLOOR = {'trace': 0.99979, 'theia': 0.99996, 'cadets': 0.9976}
BASE = '/root/autodl-tmp/projB_hetero_graph/external/MAGIC/data'
RES = '/root/autodl-tmp/projB_hetero_graph/external/MAGIC/eval_result'

def load(ds):
    m = json.load(open(f'{BASE}/{ds}/metadata.json'))
    mal_idx, mal_names = m['malicious']
    sizes = []
    for i in range(m['n_test']):
        g = pickle.load(open(f'{BASE}/{ds}/test{i}.pkl', 'rb'))
        sizes.append(len(g['nodes']) if isinstance(g, dict) else g.number_of_nodes())
        del g
    skip_benign = sum(sizes[:-1])
    y_full = np.zeros(sum(sizes)); y_full[np.asarray(mal_idx)] = 1.0
    idx = np.arange(len(y_full))
    test_idx = idx[(idx >= skip_benign) | (y_full == 1)]
    y = y_full[test_idx].astype(int)
    mean_dist, dist = pickle.load(open(f'{RES}/distance_save_{ds}.pkl', 'rb'))
    score = np.asarray(dist) / mean_dist
    assert len(score) == len(y), f'{ds}: {len(score)} vs {len(y)}'
    return y, score

def pot_evt_threshold(score, p0=0.95, tau=1e-3):
    """GPD POT：u=95% 分位；超额拟合 GPD（SMOME 简化：Grimshaw 一步估计用矩法）。
    x_q = u + (sigma/hat)( (n/Nu * tau)^(-xi) - 1 )，tau=尾部概率。"""
    u = np.quantile(score, p0)
    exc = score[score > u] - u
    Nu = len(exc); n = len(score)
    m = exc.mean(); v = exc.var()
    xi_hat = 0.5 * (1 - m**2 / v)
    sigma_hat = m * (1 - xi_hat)
    xq = u + sigma_hat / xi_hat * ((n / Nu * tau) ** (-xi_hat) - 1)
    return xq

def eval_at(y, score, th):
    pred = (score >= th).astype(int)
    if pred.sum() == 0: return dict(p=0.0, r=0.0, f1=0.0, n_flag=0)
    return dict(p=precision_score(y, pred, zero_division=0), r=recall_score(y, pred),
                f1=f1_score(y, pred, zero_division=0), n_flag=int(pred.sum()))

def run(ds):
    y, score = load(ds)
    n_pos = int(y.sum()); base_rate = n_pos / len(y)
    auc = roc_auc_score(y, score)
    print(f'\n===== {ds.upper()} =====')
    print(f'n_test={len(y)} n_pos={n_pos} base_rate={base_rate:.6f} AUC={auc:.4f}')
    # P1: MAGIC 代码协议（带硬编码 rec 下限）
    prec, rec, th = precision_recall_curve(y, score)
    f1s = 2 * prec * rec / (rec + prec + 1e-9)
    floor = DS_REC_FLOOR[ds]
    best_idx = -1
    for i in range(len(f1s)):
        if rec[i] < floor:
            best_idx = i - 1; break
    if best_idx < 0:
        for i in range(len(f1s) - 2, -1, -1):
            if prec[i] >= prec.max() - 1e-9 and rec[i] >= floor:
                best_idx = i; break
    best_idx = max(best_idx, 0)
    print(f'P1 MAGIC代码协议(rec>={floor}): F1={f1s[best_idx]:.4f} P={prec[best_idx]:.4f} R={rec[best_idx]:.4f} θ={th[best_idx] if best_idx < len(th) else float("inf"):.4f}')
    # P2: 朴素最优 F1（无约束测试标签搜索）
    bi = int(np.argmax(f1s))
    print(f'P2 朴素最优F1:      F1={f1s[bi]:.4f} P={prec[bi]:.4f} R={rec[bi]:.4f} θ={th[bi] if bi < len(th) else float("inf"):.4f}')
    # P3: 固定分位数
    for q in (0.99, 0.999, 0.9999):
        th_q = np.quantile(score, q)
        r = eval_at(y, score, th_q)
        print(f'P3 分位数 q={q}:     F1={r["f1"]:.4f} P={r["p"]:.4f} R={r["r"]:.4f} 告警数={r["n_flag"]} θ={th_q:.4f}')
    # P4: POT-EVT
    for tau in (1e-2, 1e-3, 1e-4):
        th_e = pot_evt_threshold(score, tau=tau)
        r = eval_at(y, score, th_e)
        print(f'P4 POT-EVT τ={tau}:  F1={r["f1"]:.4f} P={r["p"]:.4f} R={r["r"]:.4f} 告警数={r["n_flag"]} θ={th_e:.4f}')
    # 预算受限
    order = np.argsort(-score)
    for K in (1000, 10000):
        topk = order[:K]
        print(f'P@K K={K}: P={y[topk].mean():.4f} (随机基线 {base_rate:.4f}, 倍数 {y[topk].mean()/base_rate:.1f}x) 命中={int(y[topk].sum())}/{n_pos}')
    # 天花板检查：分数能完全分离吗
    print(f'分数分布: benign p50={np.median(score[y==0]):.3f} p99={np.quantile(score[y==0],0.99):.3f} | pos p50={np.median(score[y==1]):.3f}')
    return dict(ds=ds, n=len(y), n_pos=n_pos, auc=float(auc))

if __name__ == '__main__':
    out = {}
    for ds in ['trace', 'cadets', 'theia']:
        try:
            out[ds] = run(ds)
        except FileNotFoundError as e:
            print(f'{ds}: 数据未解压/缺失 {e}')
    with open('/root/autodl-tmp/projB_hetero_graph/logs/smoke_r1_m2_result.json', 'w') as f:
        json.dump(out, f, indent=1)
    print('\nDONE')
