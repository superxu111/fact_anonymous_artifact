"""r4b_common.py — R4 方向B 共享模块：USER/HOST+LOGON/FLOW 异构时序图加载与攻击真值。

口径纪律（与 E40B 一致）：
  - auth 9 列实测：time,user@dom,acct@dom,src_comp,dst_comp,auth_type,direction,op,success；
    LOGON 边 = user(列2) -> dst_computer(列5)（E4 取证 p1=$4|$5 与 redteam src|dst 匹配成立即证）。
  - LOGON 主口径 = NTLM 行（E5 官方过滤协议）；全量口径敏感性。
  - USER 节点 = U\\d+@ 账户 + ANONYMOUS LOGON；机器账户($ 结尾)丢弃并申报丢失率。
  - HOST 节点 = C\\d+（auth+flows 并集）。
  - FLOW 边 = (src,dst) day 级聚合（proto 众数/字节分位在 E40B 统计，训练图用 cnt+bytes_log）。
  - 攻击真值 = redteam 事件 (ts,user,src,dst)：事件级 (user,dst) 对 @day（749 行全量）；
    host-host 口径 (src,dst) 对照 E5。
"""
import gzip, json
import numpy as np
import torch
from collections import defaultdict

E40B = '/root/autodl-tmp/results/E40B_hetero_graph'
RT = '/root/autodl-tmp/data/raw/lanl/redteam.txt.gz'

_DAY_CACHE = {}


def load_logon_days(ntlm=True, user_min_events=0):
    """返回 logon_days{day: (edge_index LongTensor [2,E] user->host, cnt FloatTensor[E])},
    以及 users list / hosts list / 丢弃统计。"""
    key = f'logon_{ntlm}'
    if key in _DAY_CACHE:
        return _DAY_CACHE[key]
    users_c, hosts_c = defaultdict(int), set()
    byday = defaultdict(list)
    dropped = defaultdict(int)
    with gzip.open(f'{E40B}/e40b_auth_userhost_day.tsv.gz', 'rt') as f:
        for ln in f:
            d, u, h, c, nt = ln.rstrip('\n').split('\t')
            c, nt = int(c), int(nt)
            if ntlm:
                if nt == 0:
                    continue
                c = nt
            if u.endswith('$'):
                dropped['machine_acct'] += c
                continue
            if u.startswith('ANONYMOUS'):
                u = 'ANONYMOUS'
            elif not u.startswith('U'):
                dropped['non_U_user'] += c
                continue
            users_c[u] += c
            hosts_c.add(h)
            byday[int(d)].append((u, h, c))
    users = sorted(users_c)
    hosts = sorted(hosts_c)
    uidx = {u: i for i, u in enumerate(users)}
    hidx = {h: len(users) + i for i, h in enumerate(hosts)}
    logon_days = {}
    for d, es in byday.items():
        eu = torch.tensor([uidx[u] for u, _, _ in es], dtype=torch.long)
        eh = torch.tensor([hidx[h] for _, h, _ in es], dtype=torch.long)
        ec = torch.tensor([float(c) for _, _, c in es])
        logon_days[d] = (torch.stack([eu, eh]), ec)
    out = {'logon_days': logon_days, 'users': users, 'hosts': hosts,
           'uidx': uidx, 'hidx': hidx, 'n_users': len(users), 'n_hosts': len(hosts),
           'dropped': dict(dropped), 'user_event_counts': users_c}
    _DAY_CACHE[key] = out
    return out


def load_flow_days():
    """返回 flow_days{day: (edge_index [2,E] host->host, cnt, bytes_log)} 与 flows hosts 集。"""
    key = 'flow_days'
    if key in _DAY_CACHE:
        return _DAY_CACHE[key]
    hosts = set()
    byday = defaultdict(list)
    import gzip as _g
    with _g.open(f'{E40B}/e40b_flows_hosthost_day.tsv.gz', 'rt') as f:
        for ln in f:
            d, s, t, pr, c, b, pk = ln.rstrip('\n').split('\t')
            c, b = int(c), float(b)
            hosts.add(s); hosts.add(t)
            byday[int(d)].append((s, t, c, b))
    hosts = sorted(hosts)
    hidx = {h: i for i, h in enumerate(hosts)}
    flow_days = {}
    for d, es in byday.items():
        ei = torch.tensor([[hidx[s] for s, _, _, _ in es], [hidx[t] for _, t, _, _ in es]], dtype=torch.long)
        ec = torch.tensor([float(c) for _, _, c, _ in es])
        eb = torch.tensor([np.log1p(b) for _, _, _, b in es])
        flow_days[d] = (ei, ec, eb)
    out = {'flow_days': flow_days, 'hosts': hosts, 'hidx': hidx, 'n_hosts': len(hosts)}
    _DAY_CACHE[key] = out
    return out


def load_redteam_events():
    """返回 749 事件列表 (ts, user_norm, src_host, dst_host)；user 归一化同 load_logon_days。"""
    evs = []
    with gzip.open(RT, 'rt') as f:
        for ln in f:
            ts, u, s, d = ln.strip().split(',')
            if u.endswith('$'):
                continue
            if u.startswith('ANONYMOUS'):
                u = 'ANONYMOUS'
            evs.append((int(ts), u, s, d))
    return evs


def attack_truth_by_day(evs):
    """{day: {'logon_pairs': {(user,dst): cnt}, 'host_pairs': {(src,dst): cnt}, 'n_events': int,
              'users': set, 'hosts': set}}"""
    out = defaultdict(lambda: {'logon_pairs': defaultdict(int), 'host_pairs': defaultdict(int),
                               'n_events': 0, 'users': set(), 'hosts': set()})
    for ts, u, s, d in evs:
        day = ts // 86400
        o = out[day]
        o['logon_pairs'][(u, d)] += 1
        o['host_pairs'][(s, d)] += 1
        o['n_events'] += 1
        o['users'].add(u); o['hosts'].add(s); o['hosts'].add(d)
    return dict(out)


TRAIN_DAYS = list(range(0, 10))
EVAL_DAYS = [12, 13, 14, 15, 20, 21, 22, 26, 27, 28, 29]
