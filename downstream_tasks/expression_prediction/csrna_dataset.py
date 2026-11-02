"""csRNA-seq (EPIC): стрэндовый трек 5'-концов, размеченный только внутри template.train.

В конфиге: _target_: downstream_tasks.expression_prediction.csrna_dataset.CsRNAWindowDataset
(старое имя CsRNADatasetTSSWindow оставлено как синоним, при max_shift=0 поведение и кэши прежние).

Соглашение о bigWig (bed2bw.py):
  * интервалы покрывают ТОЛЬКО размеченные позиции, включая явные нулевые интервалы без ридов;
  * позиция вне интервалов = нет разметки (а не ноль);
  * значения - число 5'-концов ридов (r1 + r2), положительные на обеих цепях.
"""
import numpy as np
import pandas as pd
import torch

from downstream_tasks.expression_prediction.expression_dataset_final import (
    ExpressionDatasetFixedDesc,
    TSSWindowMixin,
)


class CsRNADataset(ExpressionDatasetFixedDesc):
    """Метки. Токен размечен, только если ВСЕ его основания лежат в интервалах bigWig.
    Цель размеченного токена - сумма 5'-концов (как в базовом классе),
    неразмеченного - NaN, который убирается из loss через labels_mask
    (та же схема, что в MethylationDataset)."""

    def __init__(self, *args, **kwargs):
        kwargs.pop("repeat_factor", None)
        tr = kwargs.get("transform_targets_bw")
        if tr is not None:
            probe = np.asarray(tr(np.array([np.nan, 1.0], dtype=np.float32)))
            assert np.isnan(probe[0]), "transform_targets_bw must preserve NaN"
        assert not kwargs.get("tpm"), "CsRNADataset works with bw targets only"
        super().__init__(*args, **kwargs)

    def get_signals_hash_path(self):
        # не переиспользовать кэш, посчитанный базовым классом (там вне интервалов нули)
        return super().get_signals_hash_path() + ".csrnamask"

    def process_region_signals(self, bw_handler, chrom, starts, ends, l, strand):
        if strand == "+":
            r0, r1 = int(starts[0]), int(ends[l - 1])
        else:
            r0, r1 = int(starts[l - 1]), int(ends[0])
        out = np.full(l, np.nan, dtype=np.float32)
        if r0 >= r1:
            return out

        size = r1 - r0
        val = np.zeros(size, dtype=np.float64)
        cov = np.zeros(size, dtype=np.float64)
        for s, e, v in bw_handler.intervals(chrom, r0, r1) or []:
            a, b = max(0, s - r0), min(size, e - r0)
            if a < b:
                val[a:b] = v
                cov[a:b] = 1.0

        pv = np.concatenate(([0.0], np.cumsum(val)))
        pc = np.concatenate(([0.0], np.cumsum(cov)))
        ts = np.clip(np.asarray(starts[:l], dtype=np.int64) - r0, 0, size)
        te = np.clip(np.asarray(ends[:l], dtype=np.int64) - r0, 0, size)

        labeled = (te > ts) & ((pc[te] - pc[ts]) == (te - ts))
        return np.where(labeled, pv[te] - pv[ts], np.nan).astype(np.float32)

    def __getitem__(self, idx):
        f = super().__getitem__(idx)
        undefined = torch.isnan(f["labels"])
        # NaN * 0 = NaN, поэтому NaN нужно занулить, а не только замаскировать
        f["labels"] = torch.nan_to_num(f["labels"], nan=0.0)
        f["labels_mask"] = f["labels_mask"] & ~undefined
        return f


class _Win(dict):
    """Кусок h5-группы кэша токенов: токены [lo, lo + n), атрибуты как у группы."""

    def __init__(self, g, lo, n):
        super().__init__({k: g[k][lo:lo + n] for k in ("input_ids", "starts", "ends")})
        self.attrs = dict(g.attrs)


class CsRNAWindowDataset(CsRNADataset):
    """Окно вокруг точки (колонка TSS интервалов) со случайным сдвигом. Вся геометрия окна - здесь.

    Кэш (один раз): от точки отдельно токенизируются num_before + max_shift токенов вверх
    и (gen_max_seq_len - num_before) + max_shift вниз, поэтому точка всегда на границе токенов.
    Сигнал считается для всех токенов кэша.

    __getitem__: из кэша вырезается W = gen_max_seq_len - 2 токенов подряд (+ CLS, SEP).
      * random_shift=False: num_before токенов до точки (как TSSWindowMixin);
      * random_shift=True: начало окна сдвигается на k целых токенов, k ~ U[-max_shift, max_shift],
        заново при каждом обращении. Токены и сигнал режутся одним срезом, новой токенизации нет.
    У краёв контига сдвиг упирается в кэш (окно прижимается к имеющимся токенам).

    max_shift=0: токены, сигнал и имена кэшей совпадают с CsRNADatasetTSSWindow (TSSWindowMixin),
    поэтому старые кэши и valid-кривые переиспользуются. Колонка TES не используется.
    """

    def __init__(self, *args, max_shift=0, random_shift=False, **kwargs):
        self.max_shift, self.random_shift = int(max_shift), bool(random_shift)
        assert kwargs.get("bw"), "CsRNAWindowDataset работает только с bw-кэшем сигналов"
        assert not kwargs.get("norm_bw"), "norm_bw не поддержан"
        super().__init__(*args, **kwargs)   # здесь же строятся кэши: max_shift нужен заранее

    @property
    def num_after(self):
        return self.gen_max_seq_len - self.num_before

    def _shift_tag(self):
        return f".shift{self.max_shift}" if self.max_shift else ""

    def get_hash_path(self):
        return super().get_hash_path() + f".after{self.num_after}" + self._shift_tag()

    def get_signals_hash_path(self):
        return super().get_signals_hash_path() + self._shift_tag()

    def tokenize_genome(self, i):
        self._ensure_sequences_open("genome tokenization")
        row = self.genes.iloc[i]
        chrom, x, minus = row["chromosome"], int(row["TSS"]), row["strand"] == "-"
        size = self.sequences.get_reference_length(chrom)
        n_up, n_dn = self.num_before + self.max_shift, self.num_after + self.max_shift

        def fetch(a, b, rc):
            s = self.sequences.fetch(chrom, max(a, 0), min(b, size)).upper()
            return self.reverse_complement(s) if rc else s

        span = lambda n: n * self.token_len_for_fetch   # грубая оценка: букв с запасом на n токенов
        # «до точки» по ходу чтения: для '+' слева от x, для '-' справа (reverse complement)
        up = TSSWindowMixin._tokens_from(
            self, fetch(x, x + span(n_up), True) if minus else fetch(x - span(n_up), x, False),
            take_last=True, limit=n_up) if n_up else []
        dn = TSSWindowMixin._tokens_from(
            self, fetch(x - span(n_dn), x, True) if minus else fetch(x, x + span(n_dn), False),
            take_last=False, limit=n_dn) if n_dn else []

        toks = up + dn                                       # порядок чтения
        left = x - sum(t[2] for t in (dn if minus else up))  # левая геномная координата кэша
        if minus:
            toks = toks[::-1]
        df = pd.DataFrame(toks, columns=["token_id", "token", "length"])
        df["start"] = df["length"].cumsum().shift(fill_value=0) + left
        df["end"] = df["start"] + df["length"]
        df["chrom"] = chrom
        return left, (df[::-1].reset_index(drop=True) if minus else df)

    def precompute_signals(self):
        # базовый метод считает сигнал только для первых gen_max_seq_len токенов, а в кэше их больше
        g = self.gen_max_seq_len
        self.gen_max_seq_len = g + 2 * self.max_shift
        try:
            super().precompute_signals()
        finally:
            self.gen_max_seq_len = g

    def window_start(self, gene_id, x, strand):
        """Индекс первого токена окна в кэше (в порядке чтения)."""
        g = self.h5_cache[gene_id]
        n, W = g["input_ids"].shape[0], self.gen_max_seq_len - 2
        if n <= W:
            return 0
        # сколько токенов кэша лежит до точки (у начала контига может быть меньше num_before + max_shift)
        nb = int((g["ends"][:] <= x).sum()) if strand == "+" else int((g["starts"][:] >= x).sum())
        k = int(torch.randint(-self.max_shift, self.max_shift + 1, ())) if self.random_shift else 0
        return min(max(nb - self.num_before + k, 0), n - W)

    def __getitem__(self, idx):
        row = self.genes.iloc[self.valid_indices[idx // self.n_cell_chunks]]
        gid, W = row["gene_id"], self.gen_max_seq_len - 2
        lo = self.window_start(gid, int(row["TSS"]), row["strand"])
        # базовый __getitem__ берёт токены и сигнал с начала кэша ([:L]); подставляем ему вырезанное окно
        tok, sig = self.h5_cache, self.signals_cache
        self.h5_cache = {gid: _Win(tok[gid], lo, W)}
        self.signals_cache = {gid: {"signals": sig[gid]["signals"][lo:lo + W]}}
        try:
            f = super().__getitem__(idx)
        finally:
            self.h5_cache, self.signals_cache = tok, sig
        f["win_lo"] = lo   # для наследников (побуквенные метки); collate это поле отбрасывает
        return f


# старое имя: прежние конфиги работают, при max_shift=0 токены, сигнал и кэши те же
CsRNADatasetTSSWindow = CsRNAWindowDataset
