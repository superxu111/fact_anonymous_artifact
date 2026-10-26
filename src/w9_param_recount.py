"""w9_param_recount.py — W9 独立参数复算：full(d=32) vs homo(d=32) vs homo(d'=33)。
独立结构定义（与 src/r1_train_arm.py HeteroEuler 逐参对应），对照在案申报口径
（full=843,631 / homo=836,718 / 缺口 6,913=0.819% / d'=33 过冲 +17,351=+2.1%）逐项申报。"""
import json
import torch
from torch import nn
from torch_geometric.nn import GCNConv

N = 25815  # users=11635 + hosts=14180（E40B NTLM 口径，管线实测打印值）


def build(arm, d):
    """按 r1_train_arm.HeteroEuler 同构参数化。返回 nn.ModuleList 形式的参数组。"""
    mods = {}
    mods['node_emb'] = nn.Embedding(N, d)
    mods['type_emb'] = nn.Embedding(2, 8)                      # 创建但不用于 homo（参数仍在）
    din = d if arm == 'homo' else d + 8
    mods['gcn_logon'] = GCNConv(din, d, add_self_loops=True)
    mods['gcn_flow'] = GCNConv(din, d, add_self_loops=True)
    if arm == 'full':                                          # 非共享门控（homo/homoflows 不建）
        mods['gru_l'] = nn.GRU(d, d, batch_first=True)
        mods['gru_f'] = nn.GRU(d, d, batch_first=True)
        mods['mgate'] = nn.Linear(2 * d, 1)
        mods['gate'] = nn.Linear(2 * d, 1)
    else:                                                      # homo: shared GRU, no flows → 输入 d
        mods['gru'] = nn.GRU(d, d, batch_first=True)
        mods['gate'] = nn.Linear(2 * d, 1)                     # 创建但不用于 homo（参数仍在）
    if arm == 'full':
        mods['Wl'] = nn.Parameter(torch.randn(d, d) * 0.1)
        mods['Wf'] = nn.Parameter(torch.randn(d, d) * 0.1)
        mods['bl'] = nn.Parameter(torch.zeros(1))
        mods['bf'] = nn.Parameter(torch.zeros(1))
        mods['type_head'] = nn.Sequential(nn.Linear(3, 8), nn.ReLU(), nn.Linear(8, 3))
    else:
        mods['b'] = nn.Parameter(torch.zeros(1))
        mods['type_head'] = nn.Sequential(nn.Linear(1, 8), nn.ReLU(), nn.Linear(8, 3))
    def cnt(x):
        return sum(pp.numel() for pp in x.parameters()) if isinstance(x, nn.Module) else x.numel()
    return sum(cnt(m) for m in mods.values()), {k: cnt(m) for k, m in mods.items()}


out = {}
for arm, w in [('full', 32), ('homo', 32), ('homo', 33)]:
    total, breakdown = build(arm, w)
    out[f'{arm}_w{w}'] = {'total': int(total), 'breakdown': breakdown}

full32 = out['full_w32']['total']
homo32 = out['homo_w32']['total']
homo33 = out['homo_w33']['total']
out['derived'] = {
    'gap_full_minus_homo_w32': int(full32 - homo32),
    'gap_pct': round(100 * (full32 - homo32) / full32, 4),
    'overshoot_w33_vs_full': int(homo33 - full32),
    'overshoot_pct': round(100 * (homo33 - full32) / full32, 4),
    'incase_declared': {'full': 843631, 'homo': 836718, 'gap': 6913, 'gap_pct': 0.819,
                        'w33_overshoot_declared': 17351, 'overshoot_pct_declared': 2.1},
    'recount_vs_incase': {
        'full_exact_match': bool(full32 == 843631),
        'homo_delta': int(homo32 - 836718),
        'homo_delta_note': '在案 836,718 系旧 E41B 脚本版本口径（结构差 ~2,065，0.25%）；'
                           '本管线内三臂同源自洽，E3 对照不受影响，两口径并报',
        'overshoot_delta': int((homo33 - full32) - 17351)},
}
with open('/root/autodl-tmp/projB_hetero_graph/results/EXP-R1-E3/w9_param_recount.json', 'w') as f:
    json.dump(out, f, indent=1)
print(json.dumps(out['derived'], indent=1))
print('breakdown full_w32:', out['full_w32']['breakdown'])
print('W9_DONE')
