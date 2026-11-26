"""Mạng SASRec, chuyển từ pmixer/SASRec.pytorch, python/model.py ở commit 26a8fa8 (Apache-2.0, xem NOTICE.md).

Giữ nguyên kiến trúc: embedding bài (nhân √d) + embedding vị trí, các khối self-attention nhân quả có
LayerNorm sau (post-norm) và feed-forward theo từng vị trí (Conv1d kernel 1), LayerNorm cuối.
Thay đổi: nhận tensor thay vì mảng numpy, không có user_ids, bỏ predict (điểm tính ở model.py), bỏ tùy chọn norm_first.
"""
from __future__ import annotations

import torch


class PointWiseFeedForward(torch.nn.Module):
    def __init__(self, hidden_units: int, dropout_rate: float):
        super().__init__()
        self.conv1 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout1 = torch.nn.Dropout(p=dropout_rate)
        self.relu = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout2 = torch.nn.Dropout(p=dropout_rate)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = self.dropout2(self.conv2(self.relu(self.dropout1(self.conv1(inputs.transpose(-1, -2))))))
        return outputs.transpose(-1, -2)  # Conv1d cần (N, C, L)


class SASRecNetwork(torch.nn.Module):
    """item_num bài mã 1…item_num, mã 0 là đệm. forward(seqs (B, T)) → đặc trưng từng vị trí (B, T, hidden)."""

    def __init__(self, item_num: int, maxlen: int, hidden_units: int, num_blocks: int, num_heads: int, dropout_rate: float):
        super().__init__()
        self.item_emb = torch.nn.Embedding(item_num + 1, hidden_units, padding_idx=0)
        self.pos_emb = torch.nn.Embedding(maxlen + 1, hidden_units, padding_idx=0)
        self.emb_dropout = torch.nn.Dropout(p=dropout_rate)

        self.attention_layernorms = torch.nn.ModuleList()
        self.attention_layers = torch.nn.ModuleList()
        self.forward_layernorms = torch.nn.ModuleList()
        self.forward_layers = torch.nn.ModuleList()
        self.last_layernorm = torch.nn.LayerNorm(hidden_units, eps=1e-8)
        for _ in range(num_blocks):
            self.attention_layernorms.append(torch.nn.LayerNorm(hidden_units, eps=1e-8))
            self.attention_layers.append(torch.nn.MultiheadAttention(hidden_units, num_heads, dropout_rate))
            self.forward_layernorms.append(torch.nn.LayerNorm(hidden_units, eps=1e-8))
            self.forward_layers.append(PointWiseFeedForward(hidden_units, dropout_rate))

        # Khởi tạo như main.py của bản gốc: xavier_normal cho tham số nhiều chiều, hàng đệm bằng 0
        for param in self.parameters():
            if param.dim() > 1:
                torch.nn.init.xavier_normal_(param.data)
        self.pos_emb.weight.data[0, :] = 0
        self.item_emb.weight.data[0, :] = 0

    def forward(self, log_seqs: torch.Tensor) -> torch.Tensor:
        seqs = self.item_emb(log_seqs) * self.item_emb.embedding_dim**0.5
        positions = torch.arange(1, log_seqs.shape[1] + 1, device=log_seqs.device).expand_as(log_seqs)
        seqs = seqs + self.pos_emb(positions * (log_seqs != 0))
        seqs = self.emb_dropout(seqs)

        tl = seqs.shape[1]  # độ dài theo thời gian, cho mặt nạ nhân quả
        attention_mask = ~torch.tril(torch.ones((tl, tl), dtype=torch.bool, device=log_seqs.device))
        for i in range(len(self.attention_layers)):
            seqs = torch.transpose(seqs, 0, 1)
            mha_outputs, _ = self.attention_layers[i](seqs, seqs, seqs, attn_mask=attention_mask)
            seqs = self.attention_layernorms[i](seqs + mha_outputs)
            seqs = torch.transpose(seqs, 0, 1)
            seqs = self.forward_layernorms[i](seqs + self.forward_layers[i](seqs))
        return self.last_layernorm(seqs)
