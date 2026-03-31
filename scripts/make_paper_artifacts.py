import csv
from pathlib import Path
import json

root = Path('paper')
metrics_path = root / 'results' / 'metrics.csv'
rows = list(csv.DictReader(metrics_path.open()))

by_exp = {}
for r in rows:
    by_exp.setdefault(r['experiment'], []).append(r)
summary = {exp: {f"{i['variant']}::{i['metric']}": float(i['value']) for i in items} for exp, items in by_exp.items()}
(root / 'results' / 'summary.json').write_text(json.dumps(summary, indent=2))

lines = ['| Experiment | Key result |', '|---|---|']
core = summary.get('core_adaptation', {})
if core:
    lines.append(f"| Core adaptation | recall_mse baseline={core.get('baseline::recall_mse', float('nan')):.4f}, alc={core.get('alc::recall_mse', float('nan')):.4f} |")
    lines.append(f"|  | recall_acc baseline={core.get('baseline::recall_acc', float('nan')):.3f}, alc={core.get('alc::recall_acc', float('nan')):.3f} |")
stab = summary.get('stability', {})
if stab:
    lines.append(f"| Stability | nan_rate={stab.get('alc::nan_rate', float('nan')):.6f}, final_norm={stab.get('alc::slot_norm_final', float('nan')):.3f} |")
pers = summary.get('persistence', {})
if pers:
    lines.append(f"| Persistence | slot_diff={pers.get('alc::slot_max_abs_diff', float('nan')):.2e}, behavior_diff={pers.get('alc::behavior_max_abs_diff', float('nan')):.2e} |")
train = summary.get('trainability', {})
if train:
    lines.append(f"| Trainability | before={train.get('interface_trainable::loss_before', float('nan')):.4f}, after={train.get('interface_trainable::loss_after', float('nan')):.4f} |")
eff = summary.get('efficiency', {})
if eff:
    lines.append(f"| Efficiency | baseline={eff.get('baseline::forward_ms', float('nan')):.3f}ms, alc={eff.get('alc::forward_ms', float('nan')):.3f}ms, ratio={eff.get('alc::slowdown_ratio', float('nan')):.2f}x |")
(root / 'tables' / 'main_results.md').write_text('\n'.join(lines) + '\n')

abl_rows = sorted([r for r in rows if r['experiment'] == 'ablation'], key=lambda r: float(r['value']))
(root / 'tables' / 'ablation_results.csv').write_text('variant,recall_mse\n' + '\n'.join(f"{r['variant']},{float(r['value']):.6f}" for r in abl_rows) + '\n')

try:
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(4.5,3.0))
    ax.bar(['Baseline', 'ALC'], [core.get('baseline::recall_mse', 0.0), core.get('alc::recall_mse', 0.0)], color=['#999999', '#2E86AB'])
    ax.set_ylabel('Recall MSE (lower is better)')
    ax.set_title('Core adaptation result')
    fig.tight_layout()
    fig.savefig(root / 'figures' / 'core_adaptation_mse.png', dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(4.5,3.0))
    ax.bar(['Baseline', 'ALC'], [eff.get('baseline::forward_ms', 0.0), eff.get('alc::forward_ms', 0.0)], color=['#999999', '#E07A5F'])
    ax.set_ylabel('Forward time (ms)')
    ax.set_title('Inference overhead')
    fig.tight_layout()
    fig.savefig(root / 'figures' / 'efficiency_forward_ms.png', dpi=180)
    plt.close(fig)
    print('Generated tables and figures.')
except Exception as e:
    (root / 'figures' / 'FIGURE_GENERATION_WARNING.txt').write_text(f'matplotlib unavailable or failed: {e}\n')
    print('Generated tables. Figure generation skipped:', e)
