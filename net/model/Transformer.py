import math

import numpy as np
import torch
from torch import nn

padding_value = 192


def get_attn_pad_mask(query, key):
    batch_size, len_query = query.shape
    batch_size, len_key = key.shape
    pad_attn_mask = key.data.eq(padding_value).unsqueeze(1)
    return pad_attn_mask.expand(batch_size, len_query, len_key)


def get_attn_subsequence_mask(sequence):
    device = sequence.device
    attn_shape = [sequence.shape[0], sequence.shape[1], sequence.shape[1]]
    subsequence_mask = np.triu(np.ones(attn_shape), k=1).astype(np.uint8)
    subsequence_mask = torch.autograd.Variable(torch.from_numpy(subsequence_mask) == 1).to(device)
    return subsequence_mask


class CalAttention(nn.Module):
    def __init__(self, attn_dim):
        super(CalAttention, self).__init__()
        self.attn_dim = attn_dim

    def forward(self, query, key, value, self_attn_mask):
        score = torch.matmul(query, key.transpose(-1, -2)) / np.sqrt(
            self.attn_dim)
        score = score.type(torch.float32)
        score.masked_fill_(self_attn_mask, -1e9)
        attention = nn.Softmax(dim=-1)(score)

        association_degree = torch.matmul(attention, value)
        return association_degree


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, attn_dim, head_num):
        super(MultiHeadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.attn_dim = attn_dim
        self.head_num = head_num
        self.Query = nn.Linear(embed_dim, attn_dim * head_num, bias=False)
        self.Key = nn.Linear(embed_dim, attn_dim * head_num, bias=False)
        self.Value = nn.Linear(embed_dim, attn_dim * head_num, bias=False)
        self.fc = nn.Linear(head_num * attn_dim, embed_dim, bias=False)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, query, key, value, self_attn_mask):
        identity = query
        batch_size = query.shape[0]
        query = self.Query(query).view(batch_size, -1, self.head_num, self.attn_dim).transpose(1, 2)
        key = self.Key(key).view(batch_size, -1, self.head_num, self.attn_dim).transpose(1, 2)
        value = self.Value(value).view(batch_size, -1, self.head_num, self.attn_dim).transpose(1, 2)
        self_attn_mask = self_attn_mask.unsqueeze(1).repeat(1, self.head_num, 1, 1)

        degree = CalAttention(self.attn_dim)(query, key, value, self_attn_mask)
        degree = degree.transpose(1, 2).reshape(batch_size, -1, self.head_num * self.attn_dim)
        out = self.fc(degree)

        out = self.norm(out + identity)
        return out


class FeedForward(nn.Module):
    def __init__(self, embed_dim, ffn_dim):
        super(FeedForward, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim, bias=False),
            nn.ReLU(),
            nn.Linear(ffn_dim, embed_dim, bias=False)
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        identity = x
        out = self.fc(x)
        out = self.norm(out + identity)
        return out


class EncoderLayer(nn.Module):
    def __init__(self, embed_dim, attn_dim, ffn_dim, head_num):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(embed_dim, attn_dim, head_num)
        self.ffn = FeedForward(embed_dim, ffn_dim)

    def forward(self, inputs, self_attn_mask):
        attention = self.self_attn(inputs, inputs, inputs, self_attn_mask)
        out = self.ffn(attention)
        return out


class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, ffn_dim, attn_dim, head_num, layer_num, width, height):
        super(Encoder, self).__init__()
        self.seq_embedding = nn.Embedding(vocab_size, embed_dim)
        self.image_pos_encoding = PositionEncoding2D(embed_dim, width, height)
        self.layers = nn.ModuleList([EncoderLayer(embed_dim, attn_dim, ffn_dim, head_num) for _ in range(layer_num)])

    def forward(self, series, feature_map):
        batch_size, image_dim, height, width = feature_map.shape

        feature_pos = self.image_pos_encoding(feature_map)
        feature_series = feature_pos.permute(0, 2, 3, 1).contiguous().reshape(batch_size, width * height, image_dim)
        self_attn_mask = get_attn_pad_mask(series, series)

        out = feature_series
        for layer in self.layers:
            out = layer(out, self_attn_mask)
        return out


class DecoderLayer(nn.Module):
    def __init__(self, embed_dim, attn_dim, ffn_dim, head_num):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(embed_dim, attn_dim, head_num)
        self.encoder_attn = MultiHeadAttention(embed_dim, attn_dim, head_num)
        self.ffn = FeedForward(embed_dim, ffn_dim)

    def forward(self, inputs, encoder_outputs, self_attn_mask, encoder_self_attn_mask):
        outputs = self.self_attn(inputs, inputs, inputs, self_attn_mask)
        outputs = self.encoder_attn(outputs, encoder_outputs, encoder_outputs, encoder_self_attn_mask)
        outputs = self.ffn(outputs)
        return outputs


class Decoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, ffn_dim, attn_dim, head_num, layer_num, max_seq_length):
        super(Decoder, self).__init__()
        self.seq_embedding = nn.Embedding(vocab_size, embed_dim)
        self.seq_pos_encoding = PositionEncoding1D(embed_dim, max_seq_length)
        self.layers = nn.ModuleList([DecoderLayer(embed_dim, attn_dim, ffn_dim, head_num) for _ in range(layer_num)])

    def forward(self, inputs, encoder_inputs, encoder_outputs):
        outputs = self.seq_embedding(inputs)
        outputs = self.seq_pos_encoding(outputs.transpose(0, 1)).transpose(0, 1)
        self_attn_pad_mask = get_attn_pad_mask(inputs, inputs)
        self_attn_subsequence_mask = get_attn_subsequence_mask(inputs)
        self_attn_mask = torch.gt((self_attn_pad_mask + self_attn_subsequence_mask), 0)

        encoder_attn_mask = get_attn_pad_mask(inputs, encoder_inputs)

        for layer in self.layers:
            outputs = layer(outputs, encoder_outputs, self_attn_mask, encoder_attn_mask)
        return outputs


class PositionEncoding1D(nn.Module):
    def __init__(self, dim, max_length):
        super(PositionEncoding1D, self).__init__()
        if not dim % 2 == 0:
            raise ValueError("Dimension Should Be Divided By 2")
        self.max_length = max_length

        w = torch.exp(torch.arange(0., dim, 2) * (-math.log(10000.0) / dim))
        t = torch.arange(0., max_length).unsqueeze(1)
        pos = torch.zeros(1, max_length, dim)
        pos[0, :, 0::2] = torch.sin(w * t)
        pos[0, :, 1::2] = torch.cos(w * t)
        self.register_buffer('pos', pos)

    def forward(self, x):
        _, t, _ = x.shape
        x = x + self.pos[:, :t]
        return x


class PositionEncoding2D(nn.Module):
    def __init__(self, dim, width, height):
        if not dim % 4 == 0:
            raise ValueError("Dimension Should Be Divided By 4")
        super(PositionEncoding2D, self).__init__()
        self.width = width
        self.height = height

        dim = dim // 2
        w = torch.exp(torch.arange(0., dim, 2) * -(math.log(10000.0) / dim))
        t_w = torch.arange(0., width).unsqueeze(1)
        t_h = torch.arange(0., height).unsqueeze(1)
        pos = torch.zeros(1, dim * 2, height, width)

        pos[0, 0:dim:2, :, :] = torch.sin(t_w * w).transpose(0, 1).unsqueeze(1).repeat(1, 1, height, 1)
        pos[0, 1:dim:2, :, :] = torch.cos(t_w * w).transpose(0, 1).unsqueeze(1).repeat(1, 1, height, 1)
        pos[0, dim + 0::2, :, :] = torch.sin(t_h * w).transpose(0, 1).unsqueeze(2).repeat(1, 1, 1, width)
        pos[0, dim + 1::2, :, :] = torch.cos(t_h * w).transpose(0, 1).unsqueeze(2).repeat(1, 1, 1, width)
        self.register_buffer('pos', pos)

    def forward(self, x):
        _, _, h, w = x.shape
        x = x + self.pos[:, :, :h, :w]
        return x
