"""Профильная голова ProCapNet поверх GENA (EPIC, шаг B).

Ствол головы = BaseHead (токены GENA → основания + свёртка по one-hot + дилатированные блоки) —
имена слоёв те же, поэтому из чекпоинта CsRNABaseHead он грузится; новые слои prof / cnt учатся с нуля.
Выход по тайлам T оснований (в порядке чтения окна):
  prof (B, 2, Lb) — логиты профиля: канал 0 = цепь окна, 1 = противоположная (обе в порядке чтения);
  cnt  (B, n)     — log1p суммы стартов тайла по обеим цепям.
λ = expm1(cnt) · softmax(prof по 2·T позициям тайла); base_logits = log1p(λ канала 0) — как у CsRNABaseHead,
поэтому predict_base.py / assemble.py работают без изменений.

Loss (как epic/procap.py): мультиномиальный NLL профиля по размеченным позициям тайла (вес тайла min(N, cap)/N,
среднее по тайлам с ридами) + alpha · MSE(cnt, log1p N) по тайлам, размеченным >= 50%, + token_weight · tok_loss.
В лог: prof_loss, cnt_loss, prof_gain = (U − M) на рид — выигрыш профиля над равномерным (растёт, пока учится).
"""
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from downstream_tasks.expression_prediction.csrna_base_head import BaseHead, BaseHeadOutput, CsRNABaseHead
except ImportError:
    from csrna_base_head import BaseHead, BaseHeadOutput, CsRNABaseHead


@dataclass
class ProfileHeadOutput(BaseHeadOutput):
    base_logits_anti: Optional[torch.FloatTensor] = None
    prof_loss: Optional[torch.FloatTensor] = None
    cnt_loss: Optional[torch.FloatTensor] = None
    prof_gain: Optional[torch.FloatTensor] = None


class ProfileHead(BaseHead):
    def __init__(self, hidden, d=64, dil=(1, 2, 4, 8, 16, 32), tile=1000):
        super().__init__(hidden, d, dil)
        del self.out
        self.T = tile
        self.prof = nn.Conv1d(d, 2, 75, padding=37)
        self.cnt = nn.Linear(d, 1)

    def forward(self, dec, tok_pred, base_seq, base_tok):
        t = self.tok_proj(torch.cat([dec, tok_pred], -1))
        t = torch.gather(t, 1, base_tok[..., None].expand(-1, -1, t.size(-1)))
        x = t.transpose(1, 2) + self.seq_inp(F.one_hot(base_seq, 5).to(t.dtype).transpose(1, 2))
        for b in self.blocks:
            x = b(x)                                                     # (B, d, Lb)
        B, d, Lb = x.shape
        n = -(-Lb // self.T)
        valid = F.pad(base_tok > 0, (0, n * self.T - Lb))                # реальные основания (не паддинг)
        xt = F.pad(x, (0, n * self.T - Lb)).view(B, d, n, self.T)
        v = valid.view(B, 1, n, self.T).to(x.dtype)
        cnt = self.cnt((xt * v).sum(-1).transpose(1, 2) / v.sum(-1).transpose(1, 2).clamp(min=1)).squeeze(-1)
        prof = F.pad(self.prof(x), (0, n * self.T - Lb))                 # (B, 2, n·T)
        return prof, cnt, valid


def tiles(a, T):  # (B, 2, n·T) -> (B, n, 2, T)
    B, _, L = a.shape
    return a.view(B, 2, L // T, T).transpose(1, 2)


def masked_logp(p, m):  # log softmax по (2, T) внутри тайла среди m; вне m — 0
    lp = p.float().masked_fill(~m, -1e30).flatten(2).log_softmax(-1).view_as(p)
    return lp.masked_fill(~m, 0)


def profile_loss(prof, cnt, y, m, T, cap=100.0):
    """prof (B,2,nT), cnt (B,n), y (B,2,nT) счётчики, m (B,2,nT) bool размечено -> prof, cnt, gain."""
    P, Y, M = tiles(prof, T), tiles(y.float(), T), tiles(m, T)
    Y = Y * M
    k, N = M.sum((-1, -2)).float(), Y.sum((-1, -2))
    good = k >= T
    nll = -(Y * masked_logp(P, M)).sum((-1, -2))                         # (B, n)
    pos = good & (N > 0)
    w = N.clamp(max=cap) / N.clamp(min=1)
    prof_l = (nll * w)[pos].sum() / pos.sum().clamp(min=1)
    cnt_l = F.mse_loss(cnt.float()[good], torch.log1p(N[good])) if good.any() else cnt.sum() * 0
    gain = ((N * torch.log(k.clamp(min=1)))[pos].sum() - nll[pos].sum()) / N[pos].sum() if pos.any() \
        else torch.tensor(float("nan"), device=prof.device)
    return prof_l, cnt_l, gain


def to_lambda(prof, cnt, valid, T):  # ожидаемый счёт на основание (B, 2, nT)
    P, V = tiles(prof, T), tiles(valid[:, None].expand(-1, 2, -1), T)
    p = masked_logp(P, V).exp() * V
    lam = torch.expm1(cnt.float().clamp(max=20)).clamp(min=0)[..., None, None] * p
    B, n = cnt.shape
    return lam.transpose(1, 2).reshape(B, 2, n * T)


class CsRNAProfileHead(CsRNABaseHead):

    def __init__(self, *args, base_tile=1000, alpha=20.0, count_cap=100.0, **kwargs):
        super().__init__(*args, **kwargs)
        h = self.base_head
        self.base_head = ProfileHead(self.decoder.config.hidden_size, h.out.in_channels,
                                     tuple(b.conv.dilation[0] for b in h.blocks), base_tile).to(h.out.weight.device)
        self.alpha, self.count_cap = alpha, count_cap
        n = sum(p.numel() for p in self.base_head.parameters())
        print(f"[profile_head] {n:,} параметров, тайл {base_tile}, alpha {alpha}, count_cap {count_cap}")

    def forward(self, input_ids=None, attention_mask=None, labels_mask=None, labels=None, return_dict=None,
                desc_input_ids=None, desc_attention_mask=None, dataset_flag=None, base_seq=None, base_tok=None,
                base_labels=None, base_mask=None, base_labels_anti=None, base_mask_anti=None):
        assert dataset_flag.shape[1] == 1, "профильная голова: n_keys = 1"
        with torch.set_grad_enabled(torch.is_grad_enabled() and not self.freeze_backbone):
            out = super(CsRNABaseHead, self).forward(
                input_ids=input_ids, attention_mask=attention_mask, labels_mask=labels_mask, labels=labels,
                return_dict=True, desc_input_ids=desc_input_ids, desc_attention_mask=desc_attention_mask,
                dataset_flag=dataset_flag)
        res = dict(loss=out.loss, logits=out.logits, labels_reshaped=out.labels_reshaped,
                   labels_mask_reshaped=out.labels_mask_reshaped, cls_loss=out.cls_loss,
                   other_loss=out.other_loss, tok_loss=out.other_loss)
        if base_seq is None:
            return ProfileHeadOutput(**res)

        w = self.base_head.prof.weight.dtype
        dec, tok_pred = out.hidden_states[0].to(w), out.logits.to(w)
        prof, cnt, valid = self.base_head(dec, tok_pred, base_seq.long(), base_tok.long())
        Lb, T = base_seq.shape[1], self.base_head.T
        lam = to_lambda(prof, cnt, valid, T)[..., :Lb]
        res.update(base_logits=torch.log1p(lam[:, 0]), base_logits_anti=torch.log1p(lam[:, 1]))
        if base_labels is not None:
            pad = (0, prof.shape[-1] - Lb)
            y = F.pad(torch.stack([base_labels, base_labels_anti], 1).float(), pad)
            m = F.pad(torch.stack([base_mask, base_mask_anti], 1).bool(), pad)
            pl, cl, gain = profile_loss(prof, cnt, y, m, T, self.count_cap)
            loss = pl + self.alpha * cl
            if self.token_weight and out.other_loss is not None:
                loss = loss + self.token_weight * out.other_loss
            res.update(loss=loss, other_loss=(pl + self.alpha * cl).detach(), prof_loss=pl.detach(),
                       cnt_loss=cl.detach(), prof_gain=gain.detach())
        return ProfileHeadOutput(**res)
