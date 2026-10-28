"""bigWig-треки для просмотра в IGV (обучающие треки не трогает).

    python epic/make_igv.py ASSEMBLY [ASSEMBLY ...]

Пишет в epic_train/A/igv/:
  1. csrna_full.{forward,reverse}.bw   - r1 + r2 из *-unfiltered.bed.gz, БЕЗ обрезки по whitelist
                                          (RLE-строки [start,end) учтены, реплики суммируются)
  2. *.reverse.neg.bw                    - копия reverse со знаком минус (цепь "-" рисуется вниз):
                                          для csrna_full, для обучающего csrna_train и для предсказаний
     csrna_train.{forward,reverse}.bw   - ссылки на обучающие треки (только template.train)
  3. pred.<NAME>.{forward,reverse}.bw   - предсказания модели из predictions/<A>_<NAME>/*.npz
                                          (все найденные каталоги: combo, csrnaA, valid_A/B/ATAC ...)
"""
import glob, os, sys
import numpy as np, pandas as pd, pyBigWig, pysam

H = os.environ["GENALM_HOME"]

def write_bw(path, sizes, runs):
    """runs: {contig: (starts, ends, values)}; записи в порядке заголовка."""
    bw = pyBigWig.open(path, "w")
    bw.addHeader(sizes)
    n = 0
    for c, _ in sizes:
        if c in runs and len(runs[c][0]):
            s, e, v = runs[c]
            bw.addEntries([c] * len(s), s.tolist(), ends=e.tolist(), values=v.astype(float).tolist())
            n += len(s)
    bw.close()
    print(f"  {os.path.basename(path)}: {n:,} интервалов", flush=True)

def neg(runs):
    return {c: (s, e, -v) for c, (s, e, v) in runs.items()}

def bed_runs(df):
    """Сумма перекрывающихся RLE-интервалов (реплики) -> кусочно-постоянный сигнал > 0."""
    pos = np.r_[df.s.to_numpy(), df.e.to_numpy()]
    d = np.r_[df.n.to_numpy(), -df.n.to_numpy()].astype(float)
    o = np.argsort(pos, kind="stable"); pos, d = pos[o], d[o]
    u, idx = np.unique(pos, return_index=True)
    val = np.cumsum(np.add.reduceat(d, idx))
    s, e, v = u[:-1], u[1:], val[:-1]
    k = v > 0
    return s[k], e[k], v[k]

def array_runs(track, lo):
    """Массив по основаниям (NaN = нет предсказания) -> RLE без NaN."""
    SENT = np.float32(-3.0e38)
    t = np.where(np.isfinite(track), track, SENT).astype(np.float32)
    if len(t) == 0:
        return np.array([], int), np.array([], int), np.array([])
    i0 = np.r_[0, np.flatnonzero(t[1:] != t[:-1]) + 1]
    i1 = np.r_[i0[1:], len(t)]
    k = t[i0] != SENT
    return i0[k] + lo, i1[k] + lo, t[i0][k]

def bw_runs(path):
    bw = pyBigWig.open(path)
    out = {}
    for c in bw.chroms():
        iv = bw.intervals(c)
        if iv:
            a = np.array(iv)
            out[c] = (a[:, 0].astype(int), a[:, 1].astype(int), a[:, 2])
    bw.close()
    return out

for A in sys.argv[1:]:
    print(f"== {A}", flush=True)
    d = f"{H}/epic_data/{A}"
    od = f"{H}/epic_train/{A}/igv"
    os.makedirs(od, exist_ok=True)
    with pysam.FastaFile(f"{d}/genome/genome.fa") as F:
        sizes = list(zip(F.references, F.lengths))

    # 1. полный трек из unfiltered bed
    files = sorted(glob.glob(f"{d}/for_participants/csRNA*r[12]*unfiltered.bed.gz"))
    print("  unfiltered:", [os.path.basename(f) for f in files], flush=True)
    if files:
        b = pd.concat(pd.read_csv(f, sep="\t", header=None, usecols=[0, 1, 2, 4, 5],
                                  names=["c", "s", "e", "n", "st"]) for f in files)
        for st, name in (("+", "forward"), ("-", "reverse")):
            runs = {c: bed_runs(g) for c, g in b[b.st == st].groupby("c")}
            write_bw(f"{od}/csrna_full.{name}.bw", sizes, runs)
            if st == "-":
                write_bw(f"{od}/csrna_full.reverse.neg.bw", sizes, neg(runs))
        del b

    # 2. обучающий трек: ссылки + отрицательный reverse
    tb = f"{H}/epic_train/{A}/bw/{A}.pool"
    if os.path.exists(f"{tb}.reverse.bw"):
        for name in ("forward", "reverse"):
            link = f"{od}/csrna_train.{name}.bw"
            if not os.path.lexists(link):
                os.symlink(f"{tb}.{name}.bw", link)
        write_bw(f"{od}/csrna_train.reverse.neg.bw", sizes, neg(bw_runs(f"{tb}.reverse.bw")))

    # 3. предсказания модели
    for pd_dir in sorted(glob.glob(f"{H}/predictions/{A}*/")):
        npz = glob.glob(f"{pd_dir}*.npz")
        if not npz:
            continue
        name = os.path.basename(pd_dir.rstrip("/"))[len(A):].strip("_") or "pred"
        print(f"  предсказания {name}: {len(npz)} файлов", flush=True)
        for st, sname in (("+", "forward"), ("-", "reverse")):
            runs = {}
            for f in glob.glob(f"{pd_dir}*{st}.npz"):
                c = os.path.basename(f)[:-len(f"{st}.npz")]
                z = np.load(f)
                runs[c] = array_runs(z["track"], int(z["lo"]))
            write_bw(f"{od}/pred.{name}.{sname}.bw", sizes, runs)
            if st == "-":
                write_bw(f"{od}/pred.{name}.reverse.neg.bw", sizes, neg(runs))
    print(f"== {A}: готово -> {od}", flush=True)
