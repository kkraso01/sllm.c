import csv
import json
from pathlib import Path

root = Path('paper')
metrics_path = root / 'results' / 'metrics.csv'
rows = list(csv.DictReader(metrics_path.open()))

by_exp = {}
for r in rows:
    by_exp.setdefault(r['experiment'], []).append(r)
summary = {exp: {f"{i['variant']}::{i['metric']}": float(i['value']) for i in items} for exp, items in by_exp.items()}
(root / 'results' / 'summary.json').write_text(json.dumps(summary, indent=2))

multiseed_path = root / 'results' / 'multiseed_summary.csv'
multiseed = {}
if multiseed_path.exists():
    for r in csv.DictReader(multiseed_path.open()):
        multiseed[(r['experiment'], r['variant'], r['metric'])] = {
            'n': int(r['n_seeds']),
            'mean': float(r['mean']),
            'std': float(r['std']),
            'ci95': float(r['ci95']),
        }


def fmt_single(exp, key):
    return summary.get(exp, {}).get(key, float('nan'))


def fmt_ms(exp, variant, metric, precision=4):
    k = (exp, variant, metric)
    if k not in multiseed:
        return 'n/a'
    s = multiseed[k]
    return f"{s['mean']:.{precision}f} ± {s['std']:.{precision}f} (95% CI ± {s['ci95']:.{precision}f}, n={s['n']})"


lines = ['| Experiment | Key result |', '|---|---|']
sessionkv = summary.get('sessionkv', {})
if sessionkv:
    lines.append(f"| SessionKV-DR delayed recall (delay=6,facts=4) | baseline={sessionkv.get('baseline::delayed_acc_delay_6_facts_4', float('nan')):.3f}, alc_no_write={sessionkv.get('alc_no_write::delayed_acc_delay_6_facts_4', float('nan')):.3f}, alc={sessionkv.get('alc::delayed_acc_delay_6_facts_4', float('nan')):.3f} |")
    lines.append(f"| SessionKV-DR overwrite | baseline={sessionkv.get('baseline::overwrite_acc', float('nan')):.3f}, alc_no_write={sessionkv.get('alc_no_write::overwrite_acc', float('nan')):.3f}, alc={sessionkv.get('alc::overwrite_acc', float('nan')):.3f} |")
core = summary.get('core_adaptation', {})
if core:
    lines.append(f"| Core adaptation | recall_mse baseline={core.get('baseline::recall_mse', float('nan')):.4f}, alc_no_write={core.get('alc_no_write::recall_mse', float('nan')):.4f}, alc={core.get('alc::recall_mse', float('nan')):.4f} |")
    lines.append(f"|  | recall_acc baseline={core.get('baseline::recall_acc', float('nan')):.3f}, alc_no_write={core.get('alc_no_write::recall_acc', float('nan')):.3f}, alc={core.get('alc::recall_acc', float('nan')):.3f} |")
lang = summary.get('language_benchmark', {})
if lang:
    lines.append(f"| Language-shaped benchmark | recall_mse baseline={lang.get('baseline::recall_mse', float('nan')):.4f}, alc_no_write={lang.get('alc_no_write::recall_mse', float('nan')):.4f}, alc={lang.get('alc::recall_mse', float('nan')):.4f} |")
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

row_end = r"\\"
main_tex = [
    f'Metric & Result {row_end}',
    '\\midrule',
    f"SessionKV delayed acc d=6,f=4 (baseline) & {fmt_ms('sessionkv', 'baseline', 'delayed_acc_delay_6_facts_4', 3)} {row_end}",
    f"SessionKV delayed acc d=6,f=4 (ALC no-write) & {fmt_ms('sessionkv', 'alc_no_write', 'delayed_acc_delay_6_facts_4', 3)} {row_end}",
    f"SessionKV delayed acc d=6,f=4 (ALC full) & {fmt_ms('sessionkv', 'alc', 'delayed_acc_delay_6_facts_4', 3)} {row_end}",
    f"SessionKV overwrite acc (ALC full) & {fmt_ms('sessionkv', 'alc', 'overwrite_acc', 3)} {row_end}",
    f"SessionKV persistence recall acc (ALC full) & {fmt_ms('sessionkv', 'alc', 'persistence_recall_acc', 3)} {row_end}",
    f"Language-shaped recall MSE (baseline) & {fmt_ms('language_benchmark', 'baseline', 'recall_mse')} {row_end}",
    f"Language-shaped recall MSE (ALC no-write) & {fmt_ms('language_benchmark', 'alc_no_write', 'recall_mse')} {row_end}",
    f"Language-shaped recall MSE (ALC full) & {fmt_ms('language_benchmark', 'alc', 'recall_mse')} {row_end}",
    f"Stability NaN rate & {fmt_ms('stability', 'alc', 'nan_rate', 6)} {row_end}",
    f"Trainability loss before & {fmt_ms('trainability', 'interface_trainable', 'loss_before')} {row_end}",
    f"Trainability loss after & {fmt_ms('trainability', 'interface_trainable', 'loss_after')} {row_end}",
    f"Persistence behavior max diff (single-seed) & {fmt_single('persistence', 'alc::behavior_max_abs_diff'):.2e} {row_end}",
    f"Forward-time slowdown (single-seed tiny CPU) & {fmt_single('efficiency', 'alc::slowdown_ratio'):.2f}x {row_end}",
]
(root / 'tables' / 'main_results.tex').write_text('\n'.join(main_tex) + '\n')

ablation_tex_rows = [f'Variant & recall MSE {row_end}', '\\midrule']
for r in abl_rows[:10]:
    ablation_tex_rows.append(f"{r['variant'].replace('_', '\\_')} & {float(r['value']):.4f} {row_end}")
(root / 'tables' / 'ablation_top10.tex').write_text('\n'.join(ablation_tex_rows) + '\n')

tinyqa_tex = [f'Delay & Baseline & ALC no-write & ALC full {row_end}', '\\midrule']
for d in [0, 4, 8]:
    tinyqa_tex.append(
        f"{d} & {fmt_ms('tinyqa', 'baseline', f'qa_acc_delay_{d}', 3)} & "
        f"{fmt_ms('tinyqa', 'alc_no_write', f'qa_acc_delay_{d}', 3)} & "
        f"{fmt_ms('tinyqa', 'alc', f'qa_acc_delay_{d}', 3)} {row_end}"
    )
(root / 'tables' / 'tinyqa_results.tex').write_text('\n'.join(tinyqa_tex) + '\n')

sessionkv_main_tex = [f'Delay/Facts & Baseline & ALC no-write & ALC full {row_end}', '\\midrule']
for delay in [0, 2, 6, 12]:
    metric = f'delayed_acc_delay_{delay}_facts_4'
    sessionkv_main_tex.append(
        f"d={delay}, f=4 & {fmt_ms('sessionkv', 'baseline', metric, 3)} & "
        f"{fmt_ms('sessionkv', 'alc_no_write', metric, 3)} & "
        f"{fmt_ms('sessionkv', 'alc', metric, 3)} {row_end}"
    )
(root / 'tables' / 'sessionkv_main.tex').write_text('\n'.join(sessionkv_main_tex) + '\n')

sessionkv_overwrite_tex = [f'Metric & Baseline & ALC no-write & ALC full {row_end}', '\\midrule']
for metric, label in [('overwrite_acc', 'Overwrite accuracy'), ('stale_confusion_rate', 'Stale confusion rate'), ('persistence_recall_acc', 'Persistence recall acc')]:
    sessionkv_overwrite_tex.append(
        f"{label} & {fmt_ms('sessionkv', 'baseline', metric, 3)} & "
        f"{fmt_ms('sessionkv', 'alc_no_write', metric, 3)} & "
        f"{fmt_ms('sessionkv', 'alc', metric, 3)} {row_end}"
    )
(root / 'tables' / 'sessionkv_overwrite.tex').write_text('\n'.join(sessionkv_overwrite_tex) + '\n')

sessionkv_capacity_tex = [f'Setting & Baseline & ALC no-write & ALC full {row_end}', '\\midrule']
for facts in [1, 4, 8]:
    metric = f'delayed_acc_delay_6_facts_{facts}'
    sessionkv_capacity_tex.append(
        f"facts={facts}, delay=6 & {fmt_ms('sessionkv', 'baseline', metric, 3)} & "
        f"{fmt_ms('sessionkv', 'alc_no_write', metric, 3)} & "
        f"{fmt_ms('sessionkv', 'alc', metric, 3)} {row_end}"
    )
for distractors in [0, 4, 12, 24]:
    metric = f'acc_distractors_{distractors}'
    sessionkv_capacity_tex.append(
        f"distractors={distractors}, facts=4 & {fmt_ms('sessionkv', 'baseline', metric, 3)} & "
        f"{fmt_ms('sessionkv', 'alc_no_write', metric, 3)} & "
        f"{fmt_ms('sessionkv', 'alc', metric, 3)} {row_end}"
    )
(root / 'tables' / 'sessionkv_capacity.tex').write_text('\n'.join(sessionkv_capacity_tex) + '\n')

try:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5.4, 3.2))
    labels = ['Baseline', 'ALC no-write', 'ALC']
    vals = [fmt_single('core_adaptation', 'baseline::recall_mse'), fmt_single('core_adaptation', 'alc_no_write::recall_mse'), fmt_single('core_adaptation', 'alc::recall_mse')]
    ax.bar(labels, vals, color=['#999999', '#74A57F', '#2E86AB'])
    ax.set_ylabel('Recall MSE (lower is better)')
    ax.set_title('Core adaptation (seed 42 reference run)')
    fig.tight_layout()
    fig.savefig(root / 'figures' / 'core_adaptation_mse.png', dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.4, 3.2))
    labels = ['Baseline', 'ALC no-write', 'ALC']
    vals = [fmt_single('language_benchmark', 'baseline::recall_mse'), fmt_single('language_benchmark', 'alc_no_write::recall_mse'), fmt_single('language_benchmark', 'alc::recall_mse')]
    ax.bar(labels, vals, color=['#999999', '#74A57F', '#2E86AB'])
    ax.set_ylabel('Recall MSE (lower is better)')
    ax.set_title('Language-shaped delayed recall benchmark')
    fig.tight_layout()
    fig.savefig(root / 'figures' / 'language_benchmark_mse.png', dpi=220)
    plt.close(fig)

    print('Generated tables and figures.')
except Exception as e:
    (root / 'figures' / 'FIGURE_GENERATION_WARNING.txt').write_text(f'matplotlib unavailable or failed: {e}\n')
    print('Generated tables. Figure generation skipped:', e)
