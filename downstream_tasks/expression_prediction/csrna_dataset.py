"""csRNA-seq (EPIC): стрэндовый трек 5'-концов, размеченный только внутри whitelist.

Положить в downstream_tasks/expression_prediction/csrna_dataset.py.
В конфиге: _target_: downstream_tasks.expression_prediction.csrna_dataset.CsRNADatasetTSSWindow
В run_expression_finetuning_final.py (_is_expression_dataset_cfg) добавить "CsRNADatasetTSSWindow".

Соглашение о bigWig (bed2bw.py):
  * интервалы покрывают ТОЛЬКО whitelist.train, включая явные нулевые интервалы без ридов;
  * позиция вне интервалов = нет разметки (а не ноль);
  * значения - число 5'-концов ридов (r1 + r2), положительные на обеих цепях.
"""
import numpy as np
import torch

from downstream_tasks.expression_prediction.expression_dataset_final import (
    ExpressionDatasetFixedDesc,
    TSSWindowMixin,
)


class CsRNADataset(ExpressionDatasetFixedDesc):
    """Токен размечен, только если ВСЕ его основания лежат в интервалах bigWig.
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


class CsRNADatasetTSSWindow(TSSWindowMixin, CsRNADataset):
    """Окно num_before токенов выше позиции и num_after ниже (по умолчанию
    gen_max_seq_len - num_before); колонка TES не используется."""
    pass
