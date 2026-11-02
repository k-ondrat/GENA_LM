"""Нуклеотидная голова поверх ExpressionCounts (EPIC, шаг 3).

Выход декодера (вектор на токен) + токенное предсказание → Linear → d каналов →
размножение на основания токена (base_tok) → + свёртка по one-hot последовательности →
остаточные дилатированные свёртки (рецептивное поле ±67 п.н.) → Softplus → число на основание.

Loss — тот же, что у токенной модели, но по основаниям: MSE(предсказание, log1p(счётчик))
по всем размеченным основаниям; пишется в other_loss (по нему выбирается model_best).
Плюс token_weight · прежний токенный loss (tok_loss) — при совместном обучении, чтобы
не потерять токенный уровень.

freeze_backbone: true  — вариант Б: учится только base_head; GENA, Qwen и декодер в no_grad и eval().
freeze_backbone: false — вариант А: учится всё.
Чекпоинт грузится с strict=False: если в нём нет base_head, она инициализируется заново.
"""
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from downstream_tasks.expression_prediction.expression_model_final import ExpressionCounts, ExpressionModelOutput
except ImportError:  # загрузка по пути к файлу
    from expression_model_final import ExpressionCounts, ExpressionModelOutput


@dataclass
class BaseHeadOutput(ExpressionModelOutput):
    base_logits: Optional[torch.FloatTensor] = None   # (B, Lb) ≈ log1p(счётчик)
    tok_loss: Optional[torch.FloatTensor] = None


class _Block(nn.Module):
    def __init__(self, d, r):
        super().__init__()
        self.norm = nn.LayerNorm(d)
        self.conv = nn.Conv1d(d, d, 3, padding=r, dilation=r)
        self.mix = nn.Conv1d(d, d, 1)

    def forward(self, x):  # (B, d, L)
        h = self.norm(x.transpose(1, 2)).transpose(1, 2)
        return x + self.mix(F.gelu(self.conv(F.gelu(h))))


class BaseHead(nn.Module):
    def __init__(self, hidden, d=64, dil=(1, 2, 4, 8, 16, 32)):
        super().__init__()
        self.tok_proj = nn.Linear(hidden + 1, d)
        self.seq_inp = nn.Conv1d(5, d, 9, padding=4)
        self.blocks = nn.ModuleList(_Block(d, r) for r in dil)
        self.out = nn.Conv1d(d, 1, 1)

    def forward(self, dec, tok_pred, base_seq, base_tok):
        # dec (B, L, H), tok_pred (B, L, 1), base_seq / base_tok (B, Lb)
        t = self.tok_proj(torch.cat([dec, tok_pred], -1))                       # (B, L, d)
        t = torch.gather(t, 1, base_tok[..., None].expand(-1, -1, t.size(-1)))  # (B, Lb, d)
        x = t.transpose(1, 2) + self.seq_inp(F.one_hot(base_seq, 5).to(t.dtype).transpose(1, 2))
        for b in self.blocks:
            x = b(x)
        return F.softplus(self.out(x)[:, 0])                                    # (B, Lb)


class CsRNABaseHead(ExpressionCounts):

    def __init__(self, *args, base_d=64, base_dil=(1, 2, 4, 8, 16, 32), freeze_backbone=True,
                 token_weight=0.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.freeze_backbone, self.token_weight = freeze_backbone, token_weight
        if freeze_backbone:
            for p in self.parameters():
                p.requires_grad = False
        p0 = next(self.decoder.parameters())
        self.base_head = BaseHead(self.decoder.config.hidden_size, base_d, tuple(base_dil)).to(p0.device)
        n = sum(p.numel() for p in self.base_head.parameters())
        print(f"[base_head] {n:,} параметров, freeze_backbone={freeze_backbone}, token_weight={token_weight}")

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_backbone:
            for name, mod in self.named_children():
                if name != "base_head":
                    mod.eval()
        return self

    def forward(self, input_ids=None, attention_mask=None, labels_mask=None, labels=None, return_dict=None,
                desc_input_ids=None, desc_attention_mask=None, dataset_flag=None,
                base_seq=None, base_tok=None, base_labels=None, base_mask=None):
        assert dataset_flag.shape[1] == 1, "нуклеотидная голова: n_keys = 1"
        with torch.set_grad_enabled(torch.is_grad_enabled() and not self.freeze_backbone):
            out = super().forward(input_ids=input_ids, attention_mask=attention_mask, labels_mask=labels_mask,
                                  labels=labels, return_dict=True, desc_input_ids=desc_input_ids,
                                  desc_attention_mask=desc_attention_mask, dataset_flag=dataset_flag)
        dec, tok_pred = out.hidden_states[0], out.logits

        res = dict(loss=out.loss, logits=out.logits, labels_reshaped=out.labels_reshaped,
                   labels_mask_reshaped=out.labels_mask_reshaped, cls_loss=out.cls_loss,
                   other_loss=out.other_loss, tok_loss=out.other_loss)
        if base_seq is None:
            return BaseHeadOutput(**res)

        w = self.base_head.out.weight.dtype
        pred = self.base_head(dec.to(w), tok_pred.to(w), base_seq.long(), base_tok.long())
        res["base_logits"] = pred
        if base_labels is not None:
            m = base_mask.bool()
            base_loss = F.mse_loss(pred.float()[m], torch.log1p(base_labels.float()[m])) if m.any() else pred.sum() * 0
            loss = base_loss
            if self.token_weight and out.other_loss is not None:
                loss = loss + self.token_weight * out.other_loss
            res.update(loss=loss, other_loss=base_loss.detach())
        return BaseHeadOutput(**res)
