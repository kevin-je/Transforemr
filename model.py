import math
import copy
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

import utils

class Embedding(nn.Module):
    def __init__(self, vocab_size, d_model):
        super(Embedding, self).__init__()

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.max_len = max_len
        self.dropout = nn.Dropout(p=dropout)

        """
        PE(pos, 2i)   = sin(pos / 10000 ^ (2i / d))
        PE(pos, 2i + 1) = cos(pos / 10000 ^ (2i / d))
        """

        pe = torch.zeros(max_len, d_model)
        position = torch.unsqueeze(torch.arange(0, max_len, dtype=torch.float), 1)
        div_term = torch.exp(-4 * torch.arange(0, d_model, 2).float() * math.log(10) / d_model)

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)

        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor):
        return self.dropout(self.pe[:, :x.size(1)] + x)

def attention(Q, K, V, mask=None, dropout=None):
        d_k = Q.size(-1)
        scores = torch.matmul(Q, K.transpose(-1, -2)) / math.sqrt(d_k)

        if mask is not None:
            scores = scores.masked_fill(mask==0, -1e9)

        A = F.softmax(scores, dim=-1)

        if dropout is not None:
            A = dropout(A)

        return torch.matmul(A, V), A


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.dropout = nn.Dropout(p=dropout)

        assert d_model % num_heads == 0, f"{d_model=} is not divisible by {num_heads=}!"

        self.d_k: int = d_model // num_heads
        self.W_qs = nn.Linear(d_model, d_model)
        self.W_ks = nn.Linear(d_model, d_model)
        self.W_vs = nn.Linear(d_model, d_model)
        self.W_os = nn.Linear(d_model, d_model)

    def forward(self, X1, X2, X3, mask=None):
        Qs, Ks, Vs = self.W_qs(X1), self.W_ks(X2), self.W_vs(X3)

        batch_size, L_q, _ = Qs.shape
        _, L_k, _ = Ks.shape

        Qs = Qs.view(batch_size, L_q, self.num_heads, self.d_k).transpose(1, 2)
        Ks = Ks.view(batch_size, L_k, self.num_heads, self.d_k).transpose(1, 2)
        Vs = Vs.view(batch_size, L_k, self.num_heads, self.d_k).transpose(1, 2)

        if mask is not None and mask.dim() == 3:
            mask = mask.unsqueeze(1)

        As, _ = attention(Qs, Ks, Vs, mask=mask, dropout=self.dropout)

        A = self.W_os(As.transpose(1, 2).contiguous().view(batch_size, L_q, self.d_model))
        return A

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(FeedForward, self).__init__()

        self.d_model = d_model

        self.linear_1 = nn.Linear(d_model, d_ff)
        self.linear_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, X):
        X = F.relu(self.linear_1(X))
        X = self.dropout(X)

        return self.linear_2(X)

class Connection(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super(Connection, self).__init__()

        self.dropout = nn.Dropout(p=dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, X, module):
        return X + self.dropout(module(self.layer_norm(X)))

class EncoderCell(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(EncoderCell, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads

        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout=dropout)
        self.ff = FeedForward(d_model, d_ff, dropout=dropout)

        self.connection1 = Connection(d_model, dropout=dropout)
        self.connection2 = Connection(d_model, dropout=dropout)

    def forward(self, X, mask):
        module1 = lambda x: self.self_attn(x, x, x, mask=mask)
        module2 = lambda x: self.ff(x)

        X = self.connection1(X, module1)
        X = self.connection2(X, module2)

        return X

def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class Encoder(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1,  num_layers=6):
        super(Encoder, self).__init__()

        encoder_cell = EncoderCell(d_model, num_heads, d_ff, dropout=dropout)
        self.encoder_lst = clones(encoder_cell, num_layers)

    def forward(self, X, mask):
        for encoder in self.encoder_lst:
            X = encoder(X, mask)

        return X

class DecoderCell(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(DecoderCell, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads

        self.self_attn1 = MultiHeadAttention(d_model, num_heads, dropout=dropout)
        self.self_attn2 = MultiHeadAttention(d_model, num_heads, dropout=dropout)
        self.ff = FeedForward(d_model, d_ff, dropout=dropout)

        self.connection1 = Connection(d_model, dropout=dropout)
        self.connection2 = Connection(d_model, dropout=dropout)
        self.connection3 = Connection(d_model, dropout=dropout)

    def forward(self, X, memory, src_mask, trg_mask):

        module1 = lambda x: self.self_attn1(x, x, x, mask=trg_mask)
        module2 = lambda x: self.self_attn2(x, memory, memory, mask=src_mask)
        module3 = lambda x: self.ff(x)

        X = self.connection1(X, module1)
        X = self.connection2(X, module2)
        X = self.connection3(X, module3)

        return X

class Decoder(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1, num_layers=6):
        super(Decoder, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads

        decoder_cell = DecoderCell(d_model, num_heads, d_ff, dropout=dropout)
        self.decoder_lst = clones(decoder_cell, num_layers)

    def forward(self, X, memory, src_mask, trg_mask):
        for decoder in self.decoder_lst:
            X = decoder(X, memory, src_mask, trg_mask)

        return X

class OutputLayer(nn.Module):
    def __init__(self, d_model, vocab_size):
        super(OutputLayer, self).__init__()
        self.linear = nn.Linear(d_model, vocab_size)

    def forward(self, X):
        return self.linear(X)



class Transformer(nn.Module):
    def __init__(self, vocab_size_en, vocab_size_zh, d_model, num_heads, d_ff, num_layers, dropout=0.1):
        super(Transformer, self).__init__()

        self.d_model = d_model
        self.num_heads = num_heads

        self.pos_encoder1 = PositionalEncoding(d_model, dropout=dropout)
        self.pos_encoder2 = PositionalEncoding(d_model, dropout=dropout)

        self.embedding1 = Embedding(vocab_size_en, d_model)
        self.embedding2 = Embedding(vocab_size_zh, d_model)

        self.encoder = Encoder(d_model, num_heads, d_ff, num_layers=num_layers, dropout=dropout)
        self.decoder = Decoder(d_model, num_heads, d_ff, num_layers=num_layers, dropout=dropout)

        self.output = OutputLayer(d_model, vocab_size_zh)

    def forward(self, X, Label, src_mask, trg_mask, sp_zh, max_len=50, is_inference=False):

        if is_inference:

            _batch_size = X.size(0)
            finished = torch.zeros(_batch_size, dtype=torch.bool, device=X.device)
            ys = torch.full((_batch_size, 1), 2, dtype=torch.long, device=X.device)

            X = self.pos_encoder1(self.embedding1(X))
            X = self.encoder(X, src_mask)

            for _ in range(max_len - 1):
                tgt_mask_infer = utils.get_tgt_mask(ys)

                logits = self.pos_encoder2(self.embedding2(ys))
                logits = self.decoder(logits, X, src_mask, trg_mask=tgt_mask_infer)
                logits = self.output(logits)

                # 取最后一位token
                next_token = logits[:, -1].argmax(dim=-1)
                next_token = torch.where(finished, torch.full_like(next_token, 3), next_token)

                ys = torch.cat([ys, next_token.unsqueeze(1)], dim=1)

                finished = finished | (next_token == 3)
                if finished.all():
                    break

            sentences = []
            for tokens in ys:
                sentence = utils.tokens_to_str(tokens, sp_zh)
                sentences.append(sentence)
            return sentences

        else:
            X = self.pos_encoder1(self.embedding1(X))
            X = self.encoder(X, src_mask)

            Label = self.pos_encoder2(self.embedding2(Label))

            X = self.decoder(Label, X, src_mask, trg_mask)

            return self.output(X)