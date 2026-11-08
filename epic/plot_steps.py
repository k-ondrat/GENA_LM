"""AUPRC и Spearman по шагам из predictions/<ВИД>_valid_<ТЕГ>_<ШАГ>.eval.txt (строка FULL).

    python epic/plot_steps.py cgigas_uk_roslin_v1 [--base B_0] [--out PNG]

Каждый ТЕГ (A, reg, shift, ...) — своя кривая. --base: файл-старт, добавляется шагом 0 во все кривые
(по умолчанию B_0, если есть). Пишет PNG и CSV рядом (по умолчанию runs/plots/steps_<ВИД>.png).
"""
import argparse, glob, os, re
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

H = os.environ["GENALM_HOME"]
LABELS = {"A": "A (регулярные + TSS)", "reg": "запуск 2 (регулярные + сдвиг)", "shift": "запуск 1 (регулярные + TSS + сдвиг)"}
p = argparse.ArgumentParser()
p.add_argument("species"); p.add_argument("--base", default="B_0"); p.add_argument("--out")
a = p.parse_args()
out = a.out or f"{H}/runs/plots/steps_{a.species}.png"
os.makedirs(os.path.dirname(out), exist_ok=True)

def full(path):
    for line in open(path):
        if line.startswith("FULL "):
            return tuple(map(float, line.split()[1:3]))

runs, base = {}, None
for f in glob.glob(f"{H}/predictions/{a.species}_valid_*.eval.txt"):
    tag = os.path.basename(f)[len(f"{a.species}_valid_"):-len(".eval.txt")]
    v = full(f)
    if v is None: continue
    if tag == a.base: base = v; continue
    m = re.fullmatch(r"(.+)_(\d+)", tag)
    if m: runs.setdefault(m[1], {})[int(m[2])] = v

with open(os.path.splitext(out)[0] + ".csv", "w") as fh:
    fh.write("run,step,auprc,spearman\n")
    if base: fh.write(f"{a.base},0,{base[0]},{base[1]}\n")
    for r, d in sorted(runs.items()):
        for s, (ap, sp) in sorted(d.items()): fh.write(f"{r},{s},{ap},{sp}\n")

fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
for k, (ax, name) in enumerate(zip(axes, ("AUPRC", "Spearman"))):
    for i, (r, d) in enumerate(sorted(runs.items())):
        pts = sorted(({0: base} if base else {}) | d)
        ys = [(({0: base} if base else {}) | d)[s][k] for s in pts]
        ax.plot(pts, ys, "o-", color=f"C{i}", label=LABELS.get(r, r), lw=1.6, ms=5)
        ax.annotate(f"{ys[-1]:.4f}", (pts[-1], ys[-1]), textcoords="offset points", xytext=(5, 0), fontsize=8, color=f"C{i}")
    if base: ax.axhline(base[k], color="gray", ls=":", lw=1, label=f"старт ({a.base})")
    ax.set_title(f"{a.species}, valid: {name}", fontsize=11); ax.set_xlabel("шаг"); ax.grid(alpha=.3)
axes[0].legend(fontsize=8)
fig.tight_layout(); fig.savefig(out, dpi=120)
print(out)
for r, d in sorted(runs.items()):
    print(f"  {r:6s} " + "  ".join(f"{s}:{ap:.4f}/{sp:.4f}" for s, (ap, sp) in sorted(d.items())))
