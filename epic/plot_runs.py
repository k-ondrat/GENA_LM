"""Кривые обучения из логов TensorBoard нескольких запусков на одном графике + CSV.

    python epic/plot_runs.py OUT_PREFIX ИМЯ=ПАПКА_ЗАПУСКА [ИМЯ=ПАПКА ...]
    python epic/plot_runs.py $GENALM_HOME/runs/plots/bh \
        A=$GENALM_HOME/runs/expression/epic_basehead_A/20261031-231908 \
        reg_shift=$(ls -td $GENALM_HOME/runs/expression/epic_basehead_A_reg_shift/*/ | head -1)

Пишет OUT_PREFIX.png и OUT_PREFIX.csv (run, tag, step, value). Train сглажен (EMA 0.9), сырые точки бледно.
Train-loss запусков с разными окнами напрямую не сравним; valid у всех один и тот же — его и сравнивать.
"""
import csv, glob, os, sys
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

out, runs = sys.argv[1], dict(a.split("=", 1) for a in sys.argv[2:])
PANELS = [("other_loss", "побуквенный MSE (other_loss)"), ("tok_loss", "токенный MSE (tok_loss)"), ("loss", "loss = other + tok")]
data = {}
for name, d in runs.items():
    files = sorted(glob.glob(os.path.join(d, "**", "events.out.tfevents.*"), recursive=True))
    assert files, f"нет логов TensorBoard в {d}"
    for f in files:
        ea = EventAccumulator(f, size_guidance={"scalars": 0}); ea.Reload()
        for tag in ea.Tags()["scalars"]:
            data.setdefault((name, tag), {}).update({e.step: e.value for e in ea.Scalars(tag)})

os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
with open(out + ".csv", "w", newline="") as fh:
    w = csv.writer(fh); w.writerow(["run", "tag", "step", "value"])
    for (name, tag), sv in sorted(data.items()):
        w.writerows([name, tag, s, v] for s, v in sorted(sv.items()))

fig, axes = plt.subplots(2, len(PANELS), figsize=(5 * len(PANELS), 7), sharex=True)
for j, (key, title) in enumerate(PANELS):
    for i, split in enumerate(("train", "valid")):
        ax = axes[i, j]
        for k, name in enumerate(runs):
            sv = data.get((name, f"{key}/iterations/{split}"))
            if not sv: continue
            x, y = zip(*sorted(sv.items())); c = f"C{k}"
            if split == "train":
                ema, sm = None, []
                for v in y: ema = v if ema is None else .9 * ema + .1 * v; sm.append(ema)
                ax.plot(x, y, color=c, alpha=.2, lw=.8); ax.plot(x, sm, color=c, lw=1.6, label=name)
            else:
                ax.plot(x, y, "o-", color=c, lw=1.6, ms=4, label=name)
                ax.annotate(f"{y[-1]:.4f}", (x[-1], y[-1]), textcoords="offset points", xytext=(4, 4), fontsize=8, color=c)
        ax.set_title(f"{title}, {split}", fontsize=10); ax.grid(alpha=.3)
        if i == 1: ax.set_xlabel("шаг")
        if ax.lines: ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(out + ".png", dpi=120)
print(f"{out}.png, {out}.csv")
for (name, tag), sv in sorted(data.items()):
    if tag.endswith("/iterations/valid"):
        s = max(sv); print(f"  {name:12s} {tag.split('/')[0]:11s} valid: шаг {s}, {sv[s]:.4f}; лучший {min(sv.values()):.4f}")
