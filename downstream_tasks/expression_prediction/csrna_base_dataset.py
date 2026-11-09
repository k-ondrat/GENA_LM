"""csRNA (EPIC) с побуквенными метками для нуклеотидной головы.

CsRNABaseDatasetTSSWindow = CsRNAWindowDataset + четыре поля на каждое основание окна
(в порядке чтения, как input_ids; для '-' это reverse complement):
  base_seq    (Lb,) long  — A,C,G,T,N = 0..4
  base_tok    (Lb,) long  — позиция токена в input_ids (CLS = 0, поэтому токены 1..L)
  base_labels (Lb,) float — число 5'-концов из bigWig (сырые счётчики, без log)
  base_mask   (Lb,) bool  — основание размечено (внутри интервалов bigWig)

Кэши токенов и токенных сигналов — те же, что у CsRNAWindowDataset с тем же max_shift
(при max_shift=0 — прежние кэши CsRNADatasetTSSWindow). Окно берётся со сдвигом win_lo.
Метки по основаниям читаются из bigWig на лету, последовательность — из genome.fa.
"""
import os

import numpy as np
import pyBigWig
import pysam
import torch

from downstream_tasks.expression_prediction.csrna_dataset import CsRNAWindowDataset

LUT = np.full(256, 4, np.int64)
for _i, _c in enumerate(b"ACGT"):
    LUT[_c] = LUT[_c + 32] = _i
COMP = np.array([3, 2, 1, 0, 4], np.int64)


class CsRNABaseDatasetTSSWindow(CsRNAWindowDataset):
    both_strands = False   # True: ещё base_labels_anti / base_mask_anti — противоположная цепь в том же порядке чтения

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert len(self.all_keys) == 1, "побуквенные метки поддержаны только для одного трека (n_keys = 1)"
        self._base_pid = None

    def _base_handles(self):
        # свои дескрипторы в каждом процессе-воркере (pysam/pyBigWig не переживают fork)
        if self._base_pid != os.getpid():
            fwd_rev = self.paths[self.all_keys[0]][0]
            self._base_bw = {s: pyBigWig.open(p) for s, p in fwd_rev.items()}
            self._base_fa = pysam.FastaFile(self.genome)
            self._base_pid = os.getpid()
        return self._base_bw, self._base_fa

    def __getitem__(self, idx):
        f = super().__getitem__(idx)
        g = self.h5_cache[self.genes.iloc[self.valid_indices[idx // self.n_cell_chunks]]["gene_id"]]
        L, lo = f["input_ids"].shape[1] - 2, f["win_lo"]   # окно = токены кэша [lo, lo + L)
        starts, ends = np.asarray(g["starts"][lo:lo + L]), np.asarray(g["ends"][lo:lo + L])
        chrom, strand = g.attrs["chrom"], g.attrs["strand"]
        r0, r1 = (int(starts[0]), int(ends[L - 1])) if strand == "+" else (int(starts[L - 1]), int(ends[0]))
        lens = ends - starts
        assert lens.sum() == r1 - r0, f"токены не покрывают окно сплошь: {chrom}:{r0}-{r1}"

        bw, fa = self._base_handles()
        y = np.asarray(bw[strand].values(chrom, r0, r1, numpy=True), np.float32)
        s = LUT[np.frombuffer(fa.fetch(chrom, r0, r1).encode(), np.uint8)]
        if strand == "-":
            y, s = y[::-1], COMP[s[::-1]]
        m = ~np.isnan(y)

        f["base_seq"] = torch.from_numpy(np.ascontiguousarray(s))
        f["base_tok"] = torch.from_numpy(np.repeat(np.arange(1, L + 1), lens))
        f["base_labels"] = torch.from_numpy(np.where(m, y, 0).astype(np.float32))
        f["base_mask"] = torch.from_numpy(np.ascontiguousarray(m))
        if self.both_strands:
            ya = np.asarray(bw["-" if strand == "+" else "+"].values(chrom, r0, r1, numpy=True), np.float32)
            ya = ya[::-1] if strand == "-" else ya
            ma = ~np.isnan(ya)
            f["base_labels_anti"] = torch.from_numpy(np.where(ma, ya, 0).astype(np.float32))
            f["base_mask_anti"] = torch.from_numpy(np.ascontiguousarray(ma))
        return f


class CsRNAProfileDataset(CsRNABaseDatasetTSSWindow):
    both_strands = True
