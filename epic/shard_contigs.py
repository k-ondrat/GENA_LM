import os, sys, pandas as pd
A, k, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
b = pd.read_csv(f"{os.environ['GENALM_HOME']}/epic_data/{A}/for_participants/whitelist.test.bed.gz",
                sep="\t", header=None, usecols=[0, 1, 2])
g = b.groupby(0); L = (g[2].max() - g[1].min()).sort_values(ascending=False)
bins, tot = [[] for _ in range(k)], [0] * k
for c, l in L.items():
    i = tot.index(min(tot)); bins[i].append(c); tot[i] += l
os.makedirs(out, exist_ok=True)
for i, cs in enumerate(bins): open(f"{out}/shard{i}.txt", "w").write("\n".join(cs) + "\n")
print(A, [f"{t/1e6:.0f} Mb" for t in tot])
