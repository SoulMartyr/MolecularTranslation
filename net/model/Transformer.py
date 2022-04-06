import math

import torch
from fairseq.models import FairseqEncoder, FairseqIncrementalDecoder
from fairseq.modules import TransformerEncoderLayer, TransformerDecoderLayer
from torch.nn import ModuleList, LayerNorm, Module


class Namespace(object):
    def __init__(self, param_dict):
        self.__dict__.update(param_dict)


class TransformerEncoder(FairseqEncoder):

    def __init__(self, embed_dim, ffn_dim, head_num, layer_num):
        super(TransformerEncoder, self).__init__({})

        self.layer = ModuleList([TransformerEncoderLayer(
            Namespace({"encoder_embed_dim": embed_dim,
                       "encoder_ffn_dim": ffn_dim,
                       "encoder_attention_heads": head_num,
                       "attention_dropout": 0.1,
                       "dropout": 0.1,
                       "encoder_normalize_before": True})) for _ in range(layer_num)])
        self.layer_norm = LayerNorm(embed_dim)

    def forward(self, x, *args, **kwargs):
        for layer in self.layer:
            x = layer(x, encoder_padding_mask=None)
        x = self.layer_norm(x)
        return x

    def reorder_encoder_out(self, encoder_out, new_order):
        pass


class TransformerDecoder(FairseqIncrementalDecoder):
    def __init__(self, embed_dim, ffn_dim, head_num, layer_num):
        super(TransformerDecoder, self).__init__({})

        self.layer = ModuleList([TransformerDecoderLayer(
            Namespace({"encoder_embed_dim": embed_dim,
                       "decoder_embed_dim": embed_dim,
                       "decoder_ffn_embed_dim": ffn_dim,
                       "decoder_attention_heads": head_num,
                       "attention_dropout": 0.1,
                       "dropout": 0.1,
                       "decoder_normalize_before": True})) for _ in range(layer_num)])
        self.layer_norm = LayerNorm(embed_dim)

    def forward(self, x, memory, x_mask):
        for layer in self.layer:
            x = layer(x, memory, self_attn_mask=x_mask)[0]
        x = self.layer_norm(x)
        return x

    def forward_predict(self, x, memory, incremental_state):
        x = x[-1:]
        for layer in self.layer:
            x = layer(x, memory, incremental_state=incremental_state)
        x = self.layer_norm(x)
        return x


class PositionEncoding1D(Module):
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


class PositionEncoding2D(Module):
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
