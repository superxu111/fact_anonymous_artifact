"""verify_incase.py — R1 核验优先：从在案逐边分数 npz 复算指标，对照在案 JSON。

核验对象（白名单内）：
  - results/E47B_fairpool_ci/full/         （full 60ep×3s）
  - results/E47B_fairpool_ci_rerun/homo/   （homo_rerun 60ep×3s，C-R2-2 修正版）
  - results/E47B_fairpool_ci_rerun/homoflows/
对照：verdict_c_r2_2.json 的 auc_mixed/auc_logon/ap_logon；ci_and_pool_audit.json 的池计数。
判据：确定性复算，容浮点尾差（|Δ|<5e-4 视为一致）。
"""
import json
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

BASE = '/root/autodl-tmp'
ARMS = {
    'full': f'{BASE}/results/E47B_fairpool_ci/full',
    'homo_rerun': f'{BASE}/results/E47B_fairpool_ci_rerun/homo',
    'homoflows_rerun': f'{BASE}/results/E47B_fairpool_ci_rerun/homoflows',
}
VERDICT = f'{BASE}/results/E47B_fairpool_ci_rerun/verdict_c_r2_2.json'

verdict = json.load(open(VERDICT))
rows = []
ok_all = True
for arm, d in ARMS.items():
    for s in range(3):
        z = np.load(f'{d}/scores_seed{s}.npz', allow_pickle=True)
        day, rel, score, y = z['day'], z['rel'], z['score'], z['y']
        # 池计数核验
        n_pool, n_pos = len(y), int(y.sum())
        lmask = rel == 'LOGON'
        n_pool_logon, n_pos_logon = int(lmask.sum()), int(y[lmask].sum())
        # 指标复算（score=异常分，攻击边应高分，与在案口径一致）
        auc_mixed = float(roc_auc_score(y, score))
        ap_logon = float(average_precision_score(y[lmask], score[lmask]))
        auc_logon = float(roc_auc_score(y[lmask], score[lmask]))
        v = verdict['arms'][arm][str(s)]
        d_auc = abs(auc_mixed - v['auc_mixed'])
        d_logon = abs(auc_logon - v['auc_logon'])
        ap_v = v.get('ap_logon')
        d_ap = abs(ap_logon - ap_v) if (ap_v is not None and ap_logon is not None) else None
        cnt_ok = (n_pool, n_pos, n_pool_logon, n_pos_logon) == (445229, 277, 131235, 277)
        ok = d_auc < 5e-4 and d_logon < 5e-4 and (d_ap is None or d_ap < 5e-4) and cnt_ok
        ok_all &= ok
        rows.append({'arm': arm, 'seed': s, 'n_pool': n_pool, 'n_pos': n_pos,
                     'n_pool_logon': n_pool_logon, 'n_pos_logon': n_pos_logon,
                     'auc_mixed_re': round(auc_mixed, 4), 'auc_mixed_in': v['auc_mixed'],
                     'auc_logon_re': round(auc_logon, 4), 'auc_logon_in': v['auc_logon'],
                     'ap_logon_re': round(ap_logon, 4), 'ap_logon_in': ap_v,
                     'counts_match_incase': cnt_ok, 'match': bool(ok)})
        print(f"{arm} seed{s}: auc_mixed {auc_mixed:.4f} vs {v['auc_mixed']} | "
              f"auc_logon {auc_logon:.4f} vs {v['auc_logon']} | ap_logon {ap_logon:.4f} vs {ap_v} | "
              f"counts({n_pool},{n_pos},{n_pool_logon},{n_pos_logon}) {'OK' if ok else 'MISMATCH'}")

out = {'exp': 'R1-verify-incase', 'date': '2026-10-24',
       'verdict': 'PASS：3 臂×3 seed 逐边 npz 复算与在案 JSON 一致（容差 5e-4），池计数逐位一致' if ok_all
                  else 'FAIL：见 rows',
       'tolerance': 5e-4, 'rows': rows}
with open('/root/autodl-tmp/projB_hetero_graph/results/EXP-R1-VERIFY/incase_check.json', 'w') as f:
    json.dump(out, f, indent=1, ensure_ascii=False)
print('ALL_MATCH' if ok_all else 'HAS_MISMATCH')
