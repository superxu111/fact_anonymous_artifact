"""r1_train_arm.py — R1-E2 负采样/候选池协议敏感性臂 + R1-E3 宽度补偿臂（统一训练脚本）。

【实验设计自查（开工前冻结）】
E2（H2）：full vs homo 的「LOGON 同池更差 + 混合池不可分」结论不随
  训练负采样策略{uniform, temporal} × 比例{1:1,1:4,1:16} × 评估候选池构造{同视图 LOGON, 混合}翻转，
  或给出翻转边界。
  自变量：训练负采样策略×比例（6 点）；评估池构造 2 口径（每次 run 同时产出）。
  因变量：LOGON 同池 AP/AUC、混合池 AUC、评估侧采样池 AP/AUC、配对置换 p（B=500，日块，收口脚本算）。
  对照（回归校验）：uniform 1:1 60ep 须带内复现在案数值——
    full 混合池 AUC {0.8958,0.8994,0.9210}；homo_rerun {0.9314,0.8873,0.8867}；
    homo_rerun LOGON AP {0.0084,0.0168,0.0056}；池计数 (445229,277,131235,277)。
  成功判据（冻结）：复现校验带内 + 完整协议×臂×seed 表 + 翻转边界表或稳定区间结论。
E3（H3）：homo 臂参数缺口 6,913（0.819%）不驱动主结论。
  自变量：homo 宽度 d'=33（+2.1% 过冲）vs 未补偿 homo(d=32)。
  因变量：LOGON 同池 AP/AUC、混合池 AUC（60ep×3 seeds，与 homo_rerun 同口径）。
  成功判据（冻结）：差异在 seed 波动带内 → 容量伪影排除条款；超出 → 如实申报改写为容量 vs 归纳偏置讨论。

【口径冻结】
- uniform 训练负采样 = 两端独立均匀 randint（user∈[0,n_users), host∈[n_users,N)），无碰撞过滤，
  与在案 r4_e42b_hetero.py:206-207 逐行一致（复现校验的有效性前提）。
- temporal 训练负采样（换 destination 型，对接 arXiv:2610.04869 解释框架）=
  负边=(真 user, 当日活跃 host 随机)，排除当日观测对，最多 3 轮补采，实际使用数落盘申报。
  〔更正登记 2026-10-24 R2-P1-8，回应 R1A2-m5：本行「排除当日观测对」与实现不符——实现为
  **全局**已观测对过滤（filter_obs/OBS_ARR，见 sample_neg_temporal 函数级 docstring 更正），
  uniform 无过滤 vs temporal 全局过滤的协议差异照旧申报。〕
- 评估主口径 = 无采样观测边池（在案口径）；评估侧附加 sweep = 非观测 (user,host) 对采样池
  {uniform, temporal(host 限当日活跃)} × {1:1,1:4,1:16}，全部打分算 AP/AUC（2506.12588：采样下
  AUC 是唯一一致指标 → AP/AUC 并报）。
- 训练：day0-9 快照，60ep（校验链/E3）或 30ep（网格）；L = BCE_LOGON + BCE_FLOW + 0.5·CE_type。
"""
import argparse, json, os, sys, time
import numpy as np
import torch
from torch import nn
from torch.optim import Adam
from torch_geometric.nn import GCNConv
from sklearn.metrics import average_precision_score, roc_auc_score
sys.path.insert(0, '/root/autodl-tmp/projB_hetero_graph/src')
from r4b_common import (load_logon_days, load_flow_days, load_redteam_events,
                        attack_truth_by_day, TRAIN_DAYS, EVAL_DAYS)

p = argparse.ArgumentParser()
p.add_argument('--arm', default='full', choices=['full', 'homo', 'homoflows'])
p.add_argument('--width', type=int, default=32, help='E3: 33')
p.add_argument('--neg-strategy', default='uniform', choices=['uniform', 'temporal'])
p.add_argument('--neg-ratio', type=int, default=1)
p.add_argument('--seeds', default='0,1,2')
p.add_argument('--epochs', type=int, default=60)
p.add_argument('--type-w', type=float, default=0.5)
p.add_argument('--out', required=True)
p.add_argument('--sweep-ratios', default='1,4,16')
p.add_argument('--smoke', action='store_true', help='1ep+1seed+sweep 截断，全流程验证')
p.add_argument('--benign-days', default='', help='逗号分隔良性日（E1 校准流打分），如 10,11,16,17,18,19,23,24,25')
args = p.parse_args()
if args.smoke:
    args.epochs, args.seeds = 1, '0'
args.homo = (args.arm == 'homo')
if args.arm == 'homoflows':
    # 同构机制（无类型嵌入/共享GRU/等权聚合/对称解码）+ FLOW 可见（E47B homoflows 臂同构型）
    args.homo = True
    args.no_flows = False
args.no_type_emb = args.homo
args.shared_gru = args.homo
args.no_cross_attn = args.homo
args.sym_decode = args.homo
if args.arm != 'homoflows':
    args.no_flows = args.homo
OUT = args.out
os.makedirs(OUT, exist_ok=True)
DEV = torch.device('cuda:0')
D, EMB = args.width, 8
TAG = f'{args.arm}_w{args.width}_{args.neg_strategy}{args.neg_ratio}'

# ---------------- 数据 ----------------
L = load_logon_days(ntlm=True)
F = load_flow_days()
RTev = load_redteam_events()
truth = attack_truth_by_day(RTev)
users, hosts = L['users'], L['hosts']
fhosts_only = [h for h in F['hosts'] if h not in L['hidx']]
ALL_HOSTS = hosts + fhosts_only
hidx = dict(L['hidx'])
for i, h in enumerate(fhosts_only):
    hidx[h] = L['n_users'] + len(hosts) + i
N = L['n_users'] + len(ALL_HOSTS)
NTYPE = torch.zeros(N, dtype=torch.long)
NTYPE[L['n_users']:] = 1
tr_days = [d for d in TRAIN_DAYS if d in L['logon_days']]
# 全局观测 LOGON 对集合（int64 编码 u*N+v；评估采样负池过滤用）
OBS_PAIRS = set()
for d_, (ei_, _c) in L['logon_days'].items():
    a_, b_ = ei_[0].numpy(), ei_[1].numpy()
    OBS_PAIRS.update((a_ * N + b_).tolist())
OBS_ARR = np.array(sorted(OBS_PAIRS), dtype=np.int64)


def filter_obs(pu, pv):
    """np searchsorted 过滤全局观测对。pu/pv: np int64 数组。"""
    codes = pu.astype(np.int64) * N + pv.astype(np.int64)
    posl = np.clip(np.searchsorted(OBS_ARR, codes), 0, len(OBS_ARR) - 1)
    return OBS_ARR[posl] != codes
print(f'N={N} users={L["n_users"]} hosts={len(ALL_HOSTS)} tr_days={tr_days} '
      f'obs_pairs={len(OBS_PAIRS)} arm={args.arm} w={args.width} '
      f'neg={args.neg_strategy}x{args.neg_ratio} ep={args.epochs}', flush=True)


class HeteroEuler(nn.Module):
    def __init__(self, N, d=D, emb=EMB):
        super().__init__()
        self.node_emb = nn.Embedding(N, d)
        self.type_emb = nn.Embedding(2, emb)
        din = d + (emb if not args.no_type_emb else 0)
        self.gcn_logon = GCNConv(din, d, add_self_loops=True)
        self.gcn_flow = GCNConv(din, d, add_self_loops=True)
        self.gate = nn.Linear(2 * d, 1)
        hin = d
        if args.shared_gru:
            self.gru = nn.GRU(2 * d if not args.no_flows else d, d, batch_first=True)
        else:
            self.gru_l = nn.GRU(d, d, batch_first=True)
            self.gru_f = nn.GRU(d, d, batch_first=True)
            self.mgate = nn.Linear(2 * d, 1)
        if args.sym_decode:
            self.b = nn.Parameter(torch.zeros(1))
        else:
            self.Wl = nn.Parameter(torch.randn(d, d) * 0.1)
            self.Wf = nn.Parameter(torch.randn(d, d) * 0.1)
            self.bl = nn.Parameter(torch.zeros(1))
            self.bf = nn.Parameter(torch.zeros(1))
        self.type_head = nn.Sequential(nn.Linear(3 if not args.sym_decode else 1, 8),
                                       nn.ReLU(), nn.Linear(8, 3))

    def encode_seq(self, x, seq_logon, seq_flow):
        zl, zf = [], []
        for ei in seq_logon:
            zl.append(self.gcn_logon(x, ei))
        for ei in seq_flow:
            zf.append(self.gcn_flow(x, ei))
        return torch.stack(zl), torch.stack(zf)

    def memory(self, zl, zf):
        if args.shared_gru:
            inp = zl if args.no_flows else torch.cat([zl, zf], dim=-1)
            out, _ = self.gru(inp)
            return out
        hl, _ = self.gru_l(zl)
        hf, _ = self.gru_f(zf)
        if args.no_flows:
            return hl
        if args.no_cross_attn:
            return 0.5 * hl + 0.5 * hf
        g = torch.sigmoid(self.mgate(torch.cat([hl, hf], dim=-1)))
        return g * hl + (1 - g) * hf

    def decode(self, u, v, h):
        if args.sym_decode:
            s = (h[u] * h[v]).sum(-1) + self.b
            return s, s
        sl = torch.einsum('bi,ij,bj->b', h[u], self.Wl, h[v]) + self.bl
        sf = torch.einsum('bi,ij,bj->b', h[u], self.Wf, h[v]) + self.bf
        return sl, sf

    def type_logits(self, u, v, h):
        if args.sym_decode:
            s = (h[u] * h[v]).sum(-1) + self.b
            feat = s[:, None]
        else:
            sl = torch.einsum('bi,ij,bj->b', h[u], self.Wl, h[v]) + self.bl
            sf = torch.einsum('bi,ij,bj->b', h[u], self.Wf, h[v]) + self.bf
            feat = torch.stack([sl, sf, torch.zeros_like(sl)], dim=-1)
        return self.type_head(feat)


def make_x():
    if args.no_type_emb:
        return model.node_emb.weight
    return torch.cat([model.node_emb.weight, model.type_emb(NTYPE.to(DEV))], dim=-1)


def sample_neg_temporal(u_pos, v_pos, k_target, active_hosts_t):
    """temporal 负采样：真 source + 当日活跃 host，滤除**全局**已观测对（filter_obs/OBS_ARR，
    与 uniform 臂同一观测对登记表）；≤3 轮补采。
    【更正登记 2026-10-24，回应 R1B-M5】原 docstring 称"排除当日观测对"与实现不符——实现为
    全局观测对过滤，无"当日"限定；原第 179 行 dead 变量 obs（构造后未使用）随本次更正移除
    （行为无变化）。uniform 路径无过滤、temporal 路径全局过滤——协议间过滤差异在论文申报。"""
    u_out, v_out = [], []
    for _ in range(3):
        need = k_target - len(u_out)
        if need <= 0:
            break
        idx = torch.randint(0, len(active_hosts_t), (int(need * 1.2) + 16,), device=DEV)
        cand_v = active_hosts_t[idx]
        cand_u = u_pos[torch.randint(0, len(u_pos), (len(idx),), device=DEV)]
        pairs = cand_u.cpu().numpy() * N + cand_v.cpu().numpy()
        keep = filter_obs(cand_u.cpu().numpy(), cand_v.cpu().numpy())
        if keep.any():
            kv = cand_v.cpu().numpy()[keep]
            ku = cand_u.cpu().numpy()[keep]
            # 再过全局观测对
            keep2 = filter_obs(ku, kv)
            u_out.extend(ku[keep2].tolist())
            v_out.extend(kv[keep2].tolist())
    u_t = torch.tensor(u_out, dtype=torch.long, device=DEV)
    v_t = torch.tensor(v_out, dtype=torch.long, device=DEV)
    return u_t[:k_target], v_t[:k_target]


def eval_sweep_day(d, u, v, h, sl, ratios, cap=None):
    """评估侧采样池 sweep：观测 LOGON 边(良性+攻击) ∪ 采样非观测负。返回各比例点指标。"""
    inv_u = {i: u_ for u_, i in L['uidx'].items()}
    inv_h = {i: h_ for h_, i in hidx.items()}
    td = truth_day.get(d, {'logon_pairs': {}})
    y_obs = np.array([1 if (inv_u[int(a)], inv_h[int(b)]) in td['logon_pairs'] else 0
                      for a, b in zip(u.cpu(), v.cpu())])
    s_obs = (-sl).cpu().numpy()
    n_obs = len(y_obs)
    active_t = torch.unique(L['logon_days'][d][0][1]).to(DEV)  # 当日活跃 host
    out = {}
    for strat in ['uniform', 'temporal']:
        for r in ratios:
            m = r * n_obs
            if cap:
                m = min(m, cap)
            got = 0
            su, sv = [], []
            for _ in range(4):
                need = m - got
                if need <= 0:
                    break
                cu = torch.randint(0, L['n_users'], (int(need * 1.15) + 16,), device=DEV)
                if strat == 'uniform':
                    cv = torch.randint(L['n_users'], N, (len(cu),), device=DEV)
                else:
                    aidx = torch.randint(0, len(active_t), (len(cu),), device=DEV)
                    cv = active_t[aidx]
                pu = cu.cpu().numpy()
                pv = cv.cpu().numpy()
                kp = filter_obs(pu, pv)
                su.extend(pu[kp].tolist())
                sv.extend(pv[kp].tolist())
                got = len(su)
            su_t = torch.tensor(su, dtype=torch.long, device=DEV)
            sv_t = torch.tensor(sv, dtype=torch.long, device=DEV)
            with torch.no_grad():
                sn, _ = model.decode(su_t, sv_t, h)
            s_all = np.concatenate([s_obs, (-sn).cpu().numpy()])
            y_all = np.concatenate([y_obs, np.zeros(len(su))])
            ap = float(average_precision_score(y_all, s_all))
            auc = float(roc_auc_score(y_all, s_all))
            out[f'{strat}_{r}'] = {'ap': round(ap, 6), 'auc': round(auc, 6),
                                   'n_neg': len(su), 'ratio_target': r,
                                   'base_rate': round(float(y_all.mean()), 8)}
    return out


truth_day = truth
model = None


def run_seed(seed):
    torch.manual_seed(seed); np.random.seed(seed)
    global model
    model = HeteroEuler(N).to(DEV)
    n_params = sum(pp.numel() for pp in model.parameters())
    opt = Adam(model.parameters(), lr=0.01)
    hist = []
    seq_logon = [L['logon_days'][d][0].to(DEV) for d in tr_days]
    seq_flow = ([F['flow_days'][d][0].to(DEV) for d in tr_days if d in F['flow_days']]
                if not args.no_flows else [torch.zeros((2, 0), dtype=torch.long, device=DEV)] * len(tr_days))
    neg_stats = {}
    for ep in range(args.epochs):
        model.train(); opt.zero_grad()
        x = make_x()
        zl, zf = model.encode_seq(x, seq_logon, seq_flow)
        loss, l_l, l_f, l_t = 0.0, 0.0, 0.0, 0.0
        mem = model.memory(zl, zf)
        for i, d in enumerate(tr_days):
            h = mem[i]
            ei, ec = L['logon_days'][d]
            u, v = ei[0].to(DEV), ei[1].to(DEV)
            k = ei.size(1)
            if args.neg_strategy == 'uniform':
                # 与在案 r4_e42b_hetero.py:206-207 逐行一致（复现校验前提）；1:1 时 kn=k 逐位同
                kn = k * args.neg_ratio
                nu = torch.randint(0, L['n_users'], (kn,), device=DEV)
                nv = torch.randint(L['n_users'], N, (kn,), device=DEV)
            else:
                active_t = torch.unique(v)
                nu, nv = sample_neg_temporal(u, v, k * args.neg_ratio, active_t)
                neg_stats[f'day{d}_ep{ep}'] = int(len(nu))
            slp, sfp = model.decode(u, v, h)
            sln, sfn = model.decode(nu, nv, h)
            EPS = 1e-6
            l_l = l_l + (-(torch.log(torch.sigmoid(slp) + EPS)).mean()
                         - (torch.log(1 - torch.sigmoid(sln) + EPS)).mean()) / 2
            if not args.no_flows and d in F['flow_days']:
                fei, _, _ = F['flow_days'][d]
                fu, fv = fei[0].to(DEV) + L['n_users'], fei[1].to(DEV) + L['n_users']
                kf = fei.size(1)
                nf1 = torch.randint(L['n_users'], N, (kf,), device=DEV)
                nf2 = torch.randint(L['n_users'], N, (kf,), device=DEV)
                bad = nf1 == nf2
                nf2[bad] = (nf2[bad] + 1 - L['n_users']) % (N - L['n_users']) + L['n_users']
                sfp2, _ = model.decode(fu, fv, h)
                sfn2, _ = model.decode(nf1, nf2, h)
                l_f = l_f + (-(torch.log(torch.sigmoid(sfp2) + EPS)).mean()
                             - (torch.log(1 - torch.sigmoid(sfn2) + EPS)).mean()) / 2
                tu = torch.cat([u, fu, nf1])
                tv = torch.cat([v, fv, nf2])
                tt = torch.cat([torch.zeros(k, dtype=torch.long), torch.ones(kf, dtype=torch.long),
                                2 * torch.ones(kf, dtype=torch.long)]).to(DEV)
                lt = model.type_logits(tu, tv, h)
                l_t = l_t + torch.nn.functional.cross_entropy(lt, tt)
            else:
                tu = torch.cat([u, nu]); tv = torch.cat([v, nv])
                tt = torch.cat([torch.zeros(k, dtype=torch.long),
                                2 * torch.ones(len(nu), dtype=torch.long)]).to(DEV)
                lt = model.type_logits(tu, tv, h)
                l_t = l_t + torch.nn.functional.cross_entropy(lt, tt)
        loss = l_l / len(tr_days) + l_f / len(tr_days) + args.type_w * l_t / len(tr_days)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
        opt.step()
        hist.append({'ep': ep, 'loss': float(loss), 'l_logon': float(l_l) / len(tr_days),
                     'l_flow': float(l_f) / len(tr_days), 'l_type': float(l_t) / len(tr_days)})
        # NaN 断言（防带病）
        assert np.isfinite(float(loss)), f'NaN loss at ep{ep} seed{seed}'
        if ep % 10 == 0:
            print(f'seed{seed} ep{ep} loss={float(loss):.4f} '
                  f'(L={float(l_l)/len(tr_days):.4f} F={float(l_f)/len(tr_days):.4f} '
                  f'T={float(l_t)/len(tr_days):.4f})', flush=True)
    # ---------------- 评估 ----------------
    model.eval()
    ev_days = [d for d in EVAL_DAYS if d in L['logon_days']]
    rows = []
    pos_r, neg_r = {}, {}
    ratios = [int(r) for r in args.sweep_ratios.split(',')]
    sweep_out = {}
    inv_u = {i: u_ for u_, i in L['uidx'].items()}
    inv_h = {i: h_ for h_, i in hidx.items()}
    with torch.no_grad():
        for d in ev_days:
            seq = seq_logon + [L['logon_days'][d][0].to(DEV)]
            seqf = seq_flow + ([F['flow_days'][d][0].to(DEV)] if (not args.no_flows and d in F['flow_days'])
                               else [torch.zeros((2, 0), dtype=torch.long, device=DEV)])
            zl2, zf2 = model.encode_seq(make_x(), seq, seqf)
            mem2 = model.memory(zl2, zf2)
            h = mem2[-1]
            ei, _ = L['logon_days'][d]
            u, v = ei[0].to(DEV), ei[1].to(DEV)
            sl, _sf = model.decode(u, v, h)
            if d in F['flow_days']:
                # 评测期 FLOW 打分与训练解耦（在案 C-R2-2 口径：homo 训练无 FLOW，
                # 评测仍对 FLOW 边打对称分 → 同池 445,229 行）
                fei, _, _ = F['flow_days'][d]
                fu, fv = fei[0].to(DEV) + L['n_users'], fei[1].to(DEV) + L['n_users']
                _sl2, sf = model.decode(fu, fv, h)
            else:
                fu = fv = torch.zeros(0, dtype=torch.long, device=DEV)
                sf = torch.zeros(0, device=DEV)
            td = truth.get(d, {'logon_pairs': {}, 'host_pairs': {}})
            yl = np.array([1 if (inv_u[int(a)], inv_h[int(b)]) in td['logon_pairs'] else 0
                           for a, b in zip(u.cpu(), v.cpu())])
            yf = np.array([1 if (inv_h[int(a)], inv_h[int(b)]) in td['host_pairs'] else 0
                           for a, b in zip(fu.cpu(), fv.cpu())]) if len(fu) else np.zeros(0)
            anom_l = (-sl).cpu().numpy()
            anom_f = (-sf).cpu().numpy()
            for yy, ss_ in zip(yl, anom_l):
                (pos_r if yy else neg_r).setdefault(d, []).append(float(ss_))
                rows.append((d, 'LOGON', float(ss_), int(yy)))
            for yy, ss_ in zip(yf, anom_f):
                (pos_r if yy else neg_r).setdefault(d, []).append(float(ss_))
                rows.append((d, 'FLOW', float(ss_), int(yy)))
            # 评估侧采样池 sweep（smoke 截断）
            sweep_out[f'day{d}'] = eval_sweep_day(d, u, v, h, sl, ratios,
                                                  cap=20000 if args.smoke else None)
    allrows = np.array(rows, dtype=object)
    ys = allrows[:, 3].astype(float)
    ss = allrows[:, 2].astype(float)
    ap = float(average_precision_score(ys, ss)) if len(set(ys)) > 1 else None
    auc = float(roc_auc_score(ys, ss)) if len(set(ys)) > 1 else None
    lmask = allrows[:, 1] == 'LOGON'
    ysl, ssl = ys[lmask], ss[lmask]
    ap_logon = float(average_precision_score(ysl, ssl)) if len(set(ysl)) > 1 else None
    auc_logon = float(roc_auc_score(ysl, ssl)) if len(set(ysl)) > 1 else None
    neg_r_sorted = {d: np.sort(np.array(v)) for d, v in neg_r.items()}
    days_u = list(pos_r.keys())
    num = 0.0
    wp = sum(len(pos_r[d]) for d in days_u)
    wn = sum(len(neg_r_sorted[d]) for d in days_u)
    for d1 in days_u:
        p1 = pos_r[d1]
        for d2 in days_u:
            nr = neg_r_sorted[d2]
            lo = np.searchsorted(nr, p1, side='left')
            hi = np.searchsorted(nr, p1, side='right')
            num += float((lo + 0.5 * (hi - lo)).sum())
    pool_auc = num / (wp * wn) if wp and wn else None
    # 逐边分数落盘（在案格式，E1/E4 复用）
    np.savez_compressed(f'{OUT}/scores_seed{seed}.npz',
                        day=allrows[:, 0].astype(np.int64), rel=allrows[:, 1],
                        score=ss, y=ys)
    # E1 良性日打分（同 DETECT 程序 seq=tr+[d]；良性日无 redteam 事件，y 恒 0）
    if args.benign_days:
        bd = [int(x) for x in args.benign_days.split(',') if int(x) in L['logon_days']]
        brows = []
        with torch.no_grad():
            for d in bd:
                seq = seq_logon + [L['logon_days'][d][0].to(DEV)]
                seqf = seq_flow + ([F['flow_days'][d][0].to(DEV)]
                                   if (not args.no_flows and d in F['flow_days'])
                                   else [torch.zeros((2, 0), dtype=torch.long, device=DEV)])
                zl3, zf3 = model.encode_seq(make_x(), seq, seqf)
                h3 = model.memory(zl3, zf3)[-1]
                ei3, _ = L['logon_days'][d]
                u3, v3 = ei3[0].to(DEV), ei3[1].to(DEV)
                sl3, _ = model.decode(u3, v3, h3)
                for ss_ in (-sl3).cpu().numpy():
                    brows.append((d, 'LOGON', float(ss_), 0))
                if not args.no_flows and d in F['flow_days']:
                    fei3, _, _ = F['flow_days'][d]
                    fu3 = fei3[0].to(DEV) + L['n_users']
                    fv3 = fei3[1].to(DEV) + L['n_users']
                    _x, sf3 = model.decode(fu3, fv3, h3)
                    for ss_ in (-sf3).cpu().numpy():
                        brows.append((d, 'FLOW', float(ss_), 0))
        barr = np.array(brows, dtype=object)
        np.savez_compressed(f'{OUT}/benign_scores_seed{seed}.npz',
                            day=barr[:, 0].astype(np.int64), rel=barr[:, 1],
                            score=barr[:, 2].astype(np.float64))
        print(f'seed{seed} benign days scored: {bd}', flush=True)
    r = {'seed': seed, 'attack_ap': ap, 'attack_auc': auc,
         'attack_ap_logon': ap_logon, 'attack_auc_logon': auc_logon,
         'base_rate_logon': float(ysl.mean()) if len(ysl) else None,
         'n_pool_logon': int(lmask.sum()), 'n_pos_logon': int(ysl.sum()),
         'n_pool': int(len(ys)), 'n_pos': int(ys.sum()),
         'auc_mixed_pool_unweighted': pool_auc, 'n_params': int(n_params),
         'eval_sweep': sweep_out, 'train_hist_tail': hist[-3:],
         'neg_strategy': args.neg_strategy, 'neg_ratio': args.neg_ratio,
         'n_neg_temporal_used': (sum(neg_stats.values()) if neg_stats else None)}
    print(f"seed{seed} ap_logon={ap_logon:.4f} auc_logon={auc_logon:.4f} "
          f"auc_mixed={auc:.4f} params={n_params} ({time.time():.0f})", flush=True)
    return r


results = {'tag': TAG, 'arm': args.arm, 'width': args.width,
           'neg_strategy': args.neg_strategy, 'neg_ratio': args.neg_ratio,
           'epochs': args.epochs, 'protocol': {
               'train_days': tr_days, 'eval_days': [d for d in EVAL_DAYS if d in L['logon_days']],
               'logon_ntlm': True, 'anomaly': '-logit',
               'neg_uniform': 'both-end iid randint, no collision filter (incase-identical path)',
               'neg_temporal': 'true-source + active-host-of-day, exclude observed pairs'},
           'seeds': {}}
for seed in [int(s) for s in args.seeds.split(',')]:
    t0 = time.time()
    r = run_seed(seed)
    r['wall_s'] = round(time.time() - t0, 1)
    results['seeds'][str(seed)] = r
    with open(f'{OUT}/results_partial.json', 'w') as f:
        json.dump(results, f, indent=1)
with open(f'{OUT}/results.json', 'w') as f:
    json.dump(results, f, indent=1)
with open(f'{OUT}/DONE.sentinel', 'w') as f:
    f.write('done')
print(f'R1TRAIN_{TAG}_DONE', flush=True)
