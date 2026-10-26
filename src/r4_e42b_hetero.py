"""r4_e42b_hetero.py — R4方向B E42B 主实验 + E43B 消融 + E41B 同构臂 + E44B 类型预测。

实验设计（自查）：
  模型 HeteroEuler：节点嵌入 Embedding(N,d)+类型嵌入；编码器=per-relation GCN（LOGON 图/FLOW 图）
        + per-node 跨类型门控聚合；记忆=per-relation GRU（LOGON 序列/FLOW 序列）+类型门控融合；
        读出=per-relation 有向双线性（LOGON 头/FLOW 头）+类型头 softmax{LOGON,FLOW,none}（KAIROS 式）。
  消融开关：--no-type-emb / --shared-gru / --no-cross-attn / --sym-decode / --no-flows；
        --homo = 同构 Euler(auth-only) 臂（= 全部 off：无类型嵌入+单 GRU 拼接输入+对称内积+无 flows）。
  训练：day0-9 快照；L = BCE_LOGON + BCE_FLOW + 0.5·CE_type；负=随机对（同数）。
  评估（E5 协议对齐）：红队日 DETECT（seq=tr+[d]）；评分池=当日 LOGON∪FLOW 观测边；
        攻击标签=事件级红队对（LOGON 池 (user,dst)；FLOW 池 (src,dst) host 对）；
        指标=AP（异常分=−logit）+日块化池化 AUC（E25 函数）+攻击边中位低分百分位；
        FLOW 结构 AP=FLOW 观测边 vs 随机负（当日）；类型 macro-F1 vs majority；混淆矩阵。
  判据（冻结 5.4）：GO=3 seeds 中 ≥2 满足 攻击边 AP ≥ 同构臂+0.02 且 FLOW AP 显著>random 且
        macro-F1 ≥ majority+0.10。AP 绝对量级必报。
"""
import argparse, json, os, sys, time
import numpy as np
import torch
from torch import nn
from torch.optim import Adam
from torch_geometric.nn import GCNConv
from torch_geometric.utils import to_undirected
from sklearn.metrics import average_precision_score, roc_auc_score, f1_score, confusion_matrix
# projB 副本（2026-10-24）：输出一律写 projB 内；输入读取走用户批准白名单（见 R0_数据可行性.md §6）
sys.path.insert(0, '/root/autodl-tmp/projB_hetero_graph/src')
from r4b_common import (load_logon_days, load_flow_days, load_redteam_events,
                        attack_truth_by_day, TRAIN_DAYS, EVAL_DAYS)

p = argparse.ArgumentParser()
p.add_argument('--tag', default='full')
p.add_argument('--homo', action='store_true', help='同构 Euler(auth-only) 臂')
p.add_argument('--no-type-emb', action='store_true')
p.add_argument('--shared-gru', action='store_true')
p.add_argument('--no-cross-attn', action='store_true')
p.add_argument('--sym-decode', action='store_true')
p.add_argument('--no-flows', action='store_true')
p.add_argument('--seeds', default='0,1,2')
p.add_argument('--type-w', type=float, default=0.5)
p.add_argument('--epochs', type=int, default=60)  # keep
p.add_argument('--out', default=None)
args = p.parse_args()
args.no_type_emb = args.no_type_emb or args.homo
args.shared_gru = args.shared_gru or args.homo
args.no_cross_attn = args.no_cross_attn or args.homo
args.sym_decode = args.sym_decode or args.homo
args.no_flows = args.no_flows or args.homo
OUT = args.out or f'/root/autodl-tmp/projB_hetero_graph/results/E42B_hetero/{args.tag}'
os.makedirs(OUT, exist_ok=True)
DEV = torch.device('cuda:0')
D, EMB = 32, 8
if args.homo:
    OUT = '/root/autodl-tmp/projB_hetero_graph/results/E41B_homo_arm'
    os.makedirs(OUT, exist_ok=True)


def auc_pooled_weighted(pos_ranks, neg_ranks, c):
    """日块加权池化 AUC（E25 先例 r3_e25_attack_bed.py:77）。pos/neg = {day: 分数升序秩数组}。"""
    days = list(pos_ranks.keys())
    wp = sum(c.get(d, 1) * len(pos_ranks[d]) for d in days)
    wn = sum(c.get(d, 1) * len(neg_ranks[d]) for d in days)
    if wp == 0 or wn == 0:
        return None
    num = 0.0
    for d1 in days:
        p1 = pos_ranks[d1]
        if len(p1) == 0:
            continue
        for d2 in days:
            nr = neg_ranks[d2]
            if len(nr) == 0:
                continue
            lo = np.searchsorted(nr, p1, side='left')
            hi = np.searchsorted(nr, p1, side='right')
            num += c.get(d1, 1) * c.get(d2, 1) * float((lo + 0.5 * (hi - lo)).sum())
    return num / (wp * wn)


# ---------------- 数据 ----------------
L = load_logon_days(ntlm=True)
F = load_flow_days()
RTev = load_redteam_events()
truth = attack_truth_by_day(RTev)
users, hosts = L['users'], L['hosts']
hmap = L['hidx']
# 全局 host 索引并集（auth NTLM hosts + flows hosts）
fhosts_only = [h for h in F['hosts'] if h not in L['hidx']]
ALL_HOSTS = hosts + fhosts_only
hidx = dict(L['hidx'])
for i, h in enumerate(fhosts_only):
    hidx[h] = L['n_users'] + len(hosts) + i  # 全局偏移：users 区 + auth hosts 区之后
N = L['n_users'] + len(ALL_HOSTS)
NTYPE = torch.zeros(N, dtype=torch.long)
NTYPE[L['n_users']:N] = 1  # 0=USER 1=HOST
print(f'N={N} (users={L["n_users"]} hosts={len(ALL_HOSTS)}, flows-only hosts={len(fhosts_only)}) '
      f'dropped={L["dropped"]} days_logon={len(L["logon_days"])} days_flow={len(F["flow_days"])}', flush=True)

tr_days = [d for d in TRAIN_DAYS if d in L['logon_days']]
print('train days:', tr_days, flush=True)


class HeteroEuler(nn.Module):
    def __init__(self, N, d=D, emb=EMB):
        super().__init__()
        self.node_emb = nn.Embedding(N, d)
        self.type_emb = nn.Embedding(2, emb)
        din = d + (emb if not args.no_type_emb else 0)
        self.gcn_logon = GCNConv(din, d, add_self_loops=True)
        self.gcn_flow = GCNConv(din, d, add_self_loops=True)
        self.gate = nn.Linear(2 * d, 1)          # per-node 跨类型门控
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
        """seq_*: list of edge_index (LongTensor)。返回 hidden 序列 (T,N,d)（GRU 前）。"""
        zl, zf = [], []
        for ei in seq_logon:
            z = self.gcn_logon(x, ei)
            zl.append(z)
        for ei in seq_flow:
            z = self.gcn_flow(x, ei)
            zf.append(z)
        zl = torch.stack(zl); zf = torch.stack(zf)
        return zl, zf

    def memory(self, zl, zf):
        """(T,N,d)×2 → (T,N,d) 记忆输出。"""
        T, Nn, _ = zl.shape
        if args.shared_gru:
            inp = zl if args.no_flows else torch.cat([zl, zf], dim=-1)
            out, _ = self.gru(inp)
            return out
        hl, _ = self.gru_l(zl)
        hf, _ = self.gru_f(zf)
        if args.no_flows:
            return hl
        if args.no_cross_attn:
            return 0.5 * hl + 0.5 * hf   # 跨类型聚合 off：等权平均（无学习门控）
        g = torch.sigmoid(self.mgate(torch.cat([hl, hf], dim=-1)))
        return g * hl + (1 - g) * hf

    def decode(self, u, v, h):
        """h: (N,d) 某时刻记忆。返回 (logon_logit, flow_logit)。"""
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
    return torch.cat([model.node_emb.weight,
                      model.type_emb(NTYPE.to(DEV))], dim=-1)


def run_seed(seed):
    torch.manual_seed(seed); np.random.seed(seed)
    rng = np.random.RandomState(seed)
    global model
    model = HeteroEuler(N).to(DEV)
    opt = Adam(model.parameters(), lr=0.01)
    hist = []
    # 预编码训练日边（NDL）；FLOW 训练日=F['flow_days'] 中的 tr_days
    seq_logon = [L['logon_days'][d][0].to(DEV) for d in tr_days]
    seq_flow = ([F['flow_days'][d][0].to(DEV) for d in tr_days if d in F['flow_days']]
                if not args.no_flows else [torch.zeros((2, 0), dtype=torch.long, device=DEV)] * len(tr_days))
    for ep in range(args.epochs):
        model.train(); opt.zero_grad()
        x = make_x()
        zl, zf = model.encode_seq(x, seq_logon, seq_flow)
        loss, l_l, l_f, l_t = 0.0, 0.0, 0.0, 0.0
        mem = model.memory(zl, zf)
        for i, d in enumerate(tr_days):
            h = mem[i]
            # LOGON BCE
            ei, ec = L['logon_days'][d]
            u, v = ei[0].to(DEV), ei[1].to(DEV)
            k = ei.size(1)
            nu = torch.randint(0, L['n_users'], (k,), device=DEV)
            nv = torch.randint(L['n_users'], N, (k,), device=DEV)
            slp, sfp = model.decode(u, v, h)
            sln, sfn = model.decode(nu, nv, h)
            EPS = 1e-6
            l_l = l_l + (-(torch.log(torch.sigmoid(slp) + EPS)).mean()
                         - (torch.log(1 - torch.sigmoid(sln) + EPS)).mean()) / 2
            # FLOW BCE
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
                # 类型 CE：pos logon=0, pos flow=1, none=2
                tu = torch.cat([u, fu, nf1])
                tv = torch.cat([v, fv, nf2])
                tt = torch.cat([torch.zeros(k, dtype=torch.long), torch.ones(kf, dtype=torch.long),
                                2 * torch.ones(kf, dtype=torch.long)]).to(DEV)
                lt = model.type_logits(tu, tv, h)
                l_t = l_t + torch.nn.functional.cross_entropy(lt, tt)
            else:
                tu = torch.cat([u, nu]); tv = torch.cat([v, nv])
                tt = torch.cat([torch.zeros(k, dtype=torch.long),
                                2 * torch.ones(k, dtype=torch.long)]).to(DEV)
                lt = model.type_logits(tu, tv, h)
                l_t = l_t + torch.nn.functional.cross_entropy(lt, tt)
        loss = l_l / len(tr_days) + l_f / len(tr_days) + args.type_w * l_t / len(tr_days)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
        opt.step()
        hist.append({'ep': ep, 'loss': float(loss), 'l_logon': float(l_l) / len(tr_days),
                     'l_flow': float(l_f) / len(tr_days), 'l_type': float(l_t) / len(tr_days)})
        if ep % 10 == 0:
            print(f'seed{seed} ep{ep} loss={float(loss):.4f} '
                  f'(L={float(l_l)/len(tr_days):.4f} F={float(l_f)/len(tr_days):.4f} T={float(l_t)/len(tr_days):.4f})',
                  flush=True)
    # ---------------- 评估 ----------------
    model.eval()
    with torch.no_grad():
        x = make_x()
        seq_logon_e = seq_logon
        seq_flow_e = seq_flow
        zl, zf = model.encode_seq(x, seq_logon_e, seq_flow_e)
        mem_tr = model.memory(zl, zf)
    ev_days = [d for d in EVAL_DAYS if d in L['logon_days']]
    # 收集逐日分数/标签
    rows = []
    type_true, type_pred = [], []
    flow_ap_pool_s, flow_ap_pool_y = [], []
    pos_r, neg_r = {}, {}
    med_pct = []
    rng_ev = np.random.RandomState(seed + 100)
    with torch.no_grad():
        for d in ev_days:
            seq = seq_logon + ([L['logon_days'][d][0].to(DEV)])
            seqf = seq_flow + ([F['flow_days'][d][0].to(DEV)] if (not args.no_flows and d in F['flow_days'])
                               else [torch.zeros((2, 0), dtype=torch.long, device=DEV)])
            zl2, zf2 = model.encode_seq(make_x(), seq, seqf)
            mem2 = model.memory(zl2, zf2)
            h = mem2[-1]
            # LOGON 池
            ei, _ = L['logon_days'][d]
            u, v = ei[0].to(DEV), ei[1].to(DEV)
            sl, _sf = model.decode(u, v, h)
            # FLOW 池
            if not args.no_flows and d in F['flow_days']:
                fei, _, _ = F['flow_days'][d]
                fu, fv = fei[0].to(DEV) + L['n_users'], fei[1].to(DEV) + L['n_users']
                _sl2, sf = model.decode(fu, fv, h)
            else:
                fu = fv = torch.zeros(0, dtype=torch.long, device=DEV)
                sf = torch.zeros(0, device=DEV)
            truth_d = truth.get(d, {'logon_pairs': {}, 'host_pairs': {}, 'users': set(), 'hosts': set()})
            inv_u = {i: u_ for u_, i in L['uidx'].items()}
            inv_h = {i: h_ for h_, i in hidx.items()}
            # LOGON 标签（事件级 (user,dst) 对）
            yl = np.array([1 if (inv_u[int(a)], inv_h[int(b)]) in truth_d['logon_pairs'] else 0
                           for a, b in zip(u.cpu(), v.cpu())])
            # FLOW 标签（host 对）
            yf = np.array([1 if (inv_h[int(a)], inv_h[int(b)]) in truth_d['host_pairs'] else 0
                           for a, b in zip(fu.cpu(), fv.cpu())]) if len(fu) else np.zeros(0)
            anom_l = (-sl).cpu().numpy()
            anom_f = (-sf).cpu().numpy()
            for yy, ss_ in zip(yl, anom_l):
                (pos_r if yy else neg_r).setdefault(d, []).append(float(ss_))
                rows.append((d, 'LOGON', float(ss_), int(yy)))
            for yy, ss_ in zip(yf, anom_f):
                (pos_r if yy else neg_r).setdefault(d, []).append(float(ss_))
                rows.append((d, 'FLOW', float(ss_), int(yy)))
            # FLOW 结构 AP（观测 vs 随机负，当日）
            alls = np.concatenate([anom_l, anom_f]) if len(anom_f) else anom_l  # 异常分
            ally = np.concatenate([yl, yf]) if len(yf) else yl
            order = np.argsort(-alls)   # 异常分降序 = 最异常在前（E5 语义）
            rank_of = np.empty(len(alls), np.int64)
            rank_of[order] = np.arange(len(alls))
            ap_idx = np.where(ally == 1)[0]
            if len(ap_idx):
                med_pct.append(float(np.median(rank_of[ap_idx]) / max(len(alls) - 1, 1)))
            # FLOW 结构 AP（观测 vs 随机负，当日）
            if len(fu) and not args.no_flows:
                kf = len(fu)
                nf1 = torch.randint(L['n_users'], N, (kf,), device=DEV)
                nf2 = torch.randint(L['n_users'], N, (kf,), device=DEV)
                bad = nf1 == nf2
                nf2[bad] = (nf2[bad] + 1 - L['n_users']) % (N - L['n_users']) + L['n_users']
                _, sfn = model.decode(nf1, nf2, h)
                flow_ap_pool_s.extend(sf.cpu().tolist()); flow_ap_pool_y.extend([1] * kf)
                flow_ap_pool_s.extend(sfn.cpu().tolist()); flow_ap_pool_y.extend([0] * kf)
            # 类型头（当日观测边 vs 随机 none 对，排除观测对）
            obs = set(zip(u.cpu().tolist(), v.cpu().tolist())) | set(zip(fu.cpu().tolist(), fv.cpu().tolist()))
            kn = 2 * (len(u) + len(fu))
            nu = torch.randint(0, L['n_users'], (kn,), device=DEV)
            nv = torch.randint(L['n_users'], N, (kn,), device=DEV)
            keep = [i for i in range(kn) if (int(nu[i]), int(nv[i])) not in obs]
            keep = keep[:max(len(u) + len(fu), 1)]
            tu3 = torch.cat([u, fu, nu[keep]])
            tv3 = torch.cat([v, fv, nv[keep]])
            tt3 = torch.cat([torch.zeros(len(u), dtype=torch.long),
                             torch.ones(len(fu), dtype=torch.long),
                             2 * torch.ones(len(keep), dtype=torch.long)]).to(DEV)
            lt = model.type_logits(tu3, tv3, h)
            pred = lt.argmax(-1).cpu().numpy()
            type_true.extend(tt3.cpu().tolist()); type_pred.extend(pred.tolist())
    # ---- 汇总 ----
    allrows = np.array(rows, dtype=object)
    ys = allrows[:, 3].astype(float)
    ss = allrows[:, 2].astype(float)   # ss = 异常分（-logit，攻击边应高分）
    ap = float(average_precision_score(ys, ss)) if len(set(ys)) > 1 else None
    auc = float(roc_auc_score(ys, ss)) if len(set(ys)) > 1 else None
    # LOGON 同池 AP（两臂池构成一致，判据主口径）；混合池 AP 为辅助
    lmask = allrows[:, 1] == 'LOGON'
    ysl, ssl = ys[lmask], ss[lmask]
    ap_logon = float(average_precision_score(ysl, ssl)) if len(set(ysl)) > 1 else None
    auc_logon = float(roc_auc_score(ysl, ssl)) if len(set(ysl)) > 1 else None
    base_rate_logon = float(ysl.mean()) if len(ysl) else None
    neg_r_sorted = {d: np.sort(np.array(v)) for d, v in neg_r.items()}
    pool_auc = auc_pooled_weighted(pos_r, neg_r_sorted, {})
    # FLOW AP vs random
    if flow_ap_pool_y:
        fap = float(average_precision_score(flow_ap_pool_y, flow_ap_pool_s))
        fauc = float(roc_auc_score(flow_ap_pool_y, flow_ap_pool_s))
    else:
        fap = fauc = None
    f1m = float(f1_score(type_true, type_pred, average='macro'))
    cm = confusion_matrix(type_true, type_pred, labels=[0, 1, 2]).tolist()
    majority = float(max(np.bincount(type_true, minlength=3)) / len(type_true))
    return {'seed': seed, 'attack_ap': ap, 'attack_auc': auc, 'pool_auc_dayweighted': pool_auc,
            'attack_ap_logon': ap_logon, 'attack_auc_logon': auc_logon, 'base_rate_logon': base_rate_logon,
            'n_pool_logon': int(lmask.sum()), 'n_pos_logon': int(ysl.sum()),
            'median_pct': (float(np.median(med_pct)) if med_pct else None),
            'n_pool': int(len(ys)), 'n_pos': int(ys.sum()),
            'flow_ap': fap, 'flow_auc': fauc, 'flow_ap_random_ref': 0.5 * 0 + float(np.mean(flow_ap_pool_y)) if flow_ap_pool_y else None,
            'type_macro_f1': f1m, 'type_majority_base': majority, 'type_cm': cm,
            'train_hist_tail': hist[-3:]}


results = {'tag': args.tag, 'homo': args.homo, 'args': vars(args), 'protocol': {
    'logon_ntlm': True, 'eval': 'E5 DETECT seq=tr+[d]', 'truth': 'event-level (user,dst) / (src,dst)',
    'anomaly': '-logit'}, 'seeds': {}}
for seed in [int(s) for s in args.seeds.split(',')]:
    t0 = time.time()
    r = run_seed(seed)
    results['seeds'][str(seed)] = r
    print(f"seed{seed} AP={r['attack_ap'] if r['attack_ap'] is None else round(r['attack_ap'],4)} "
          f"AUC={r['attack_auc'] if r['attack_auc'] is None else round(r['attack_auc'],4)} "
          f"poolAUC={r['pool_auc_dayweighted'] if r['pool_auc_dayweighted'] is None else round(r['pool_auc_dayweighted'],4)} "
          f"med_pct={r['median_pct'] if r['median_pct'] is None else round(r['median_pct'],4)} "
          f"FLOW_AP={r['flow_ap'] if r['flow_ap'] is None else round(r['flow_ap'],4)} "
          f"F1={r['type_macro_f1']:.3f} maj={r['type_majority_base']:.3f} ({time.time()-t0:.0f}s)", flush=True)
    with open(f'{OUT}/results_partial.json', 'w') as f:
        json.dump(results, f, indent=1)
with open(f'{OUT}/results.json', 'w') as f:
    json.dump(results, f, indent=1)
with open(f'{OUT}/DONE.sentinel', 'w') as f:
    f.write('done')
print(f'E42B_{args.tag}_DONE')
