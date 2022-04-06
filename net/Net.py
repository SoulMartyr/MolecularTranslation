from typing import Optional, Dict

import numpy as np
from torch import nn, Tensor

from net.model.ResNet import ResNet
from net.model.Transformer import *
from utils.Util import Swish

image_dim = 1024
sequence_dim = 1024
decode_dim = 1024

pixel_num = 7 * 7
image_size = 224

layer_num = 2
head_num = 8
ffn_dim = 1024
vocab_size = 193
max_seq_length = 300

sign_dict = {
    '<sos>': 190,
    '<eos>': 191,
    '<pad>': 192,
}


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 使用ResNet提取特征
        self.feature_extraction = ResNet()
        # pretrain_state_dict = torch.load("checkpoint/", map_location=lambda storage, loc: storage)
        # print(self.feature_extraction.load_state_dict(pretrain_state_dict, strict=True))
        # 1×1卷积压缩通道
        self.feature_embedding = nn.Sequential(
            nn.Conv2d(2048, image_dim, kernel_size=1, bias=None),
            nn.BatchNorm2d(image_dim),
            Swish()
        )
        # 图像二维位置编码
        self.image_pos_encoding = PositionEncoding2D(image_dim, int(np.sqrt(pixel_num)),
                                                     int(np.sqrt(pixel_num)))
        # 文本一维位置编码
        self.seq_pos_encoding = PositionEncoding1D(sequence_dim, max_seq_length)
        # TransformerEncoder
        self.encoder = TransformerEncoder(image_dim, ffn_dim, head_num, layer_num)
        # 词嵌入
        self.seq_embedding = nn.Embedding(vocab_size, sequence_dim)
        # TransformerDecoder
        self.decoder = TransformerDecoder(decode_dim, ffn_dim, head_num, layer_num)
        # 输出映射
        self.mapping = nn.Linear(decode_dim, vocab_size)
        self.dropout = nn.Dropout(p=0.5)

        # 初始化
        self.seq_embedding.weight.data.uniform_(-0.1, 0.1)
        self.mapping.bias.data.fill_(0)
        self.mapping.weight.data.uniform_(-0.1, 0.1)

    # 训练
    def forward(self, image, sequence):
        device = image.device
        batch_size = len(image)
        # 特征提取
        feature_map = self.feature_extraction(image)
        feature_embed = self.feature_embedding(feature_map)
        feature_pos = self.image_pos_encoding(feature_embed)
        # 展开成时间序列
        feature_series = feature_pos.permute(2, 3, 0, 1).contiguous().reshape(pixel_num, batch_size, image_dim)
        # Encode编码
        sequence_encode = self.encoder(feature_series)
        # Decoder解码
        sequence_embed = self.seq_embedding(sequence)
        sequence_pos = self.seq_pos_encoding(sequence_embed).permute(1, 0, 2).contiguous()
        text_mask = np.triu(np.ones((max_seq_length, max_seq_length)), k=1).astype(np.uint8)
        text_mask = torch.autograd.Variable(torch.from_numpy(text_mask) == 1).to(device)
        sequence_decode = self.decoder(sequence_pos, sequence_encode, text_mask)
        sequence_decode = sequence_decode.permute(1, 0, 2).contiguous()
        # 映射回原词库大小
        result = self.mapping(sequence_decode)
        return result

    # 测试
    def forward_predict(self, image):
        device = image.device
        batch_size = len(image)

        feature_map = self.feature_extraction(image)
        feature_embed = self.feature_embedding(feature_map)
        feature_pos = self.image_pos_encoding(feature_embed)
        feature_series = feature_pos.permute(2, 3, 0, 1).contiguous().reshape(pixel_num, batch_size, image_dim)
        sequence_encode = self.encoder(feature_series)

        predict = torch.full((batch_size, max_seq_length), sign_dict['<pad>'], dtype=torch.long).to(device)
        sequence_pos = self.text_pos.pos

        predict[:, 0] = sign_dict['<sos>']
        incremental_state = torch.jit.annotate(
            Dict[str, Dict[str, Optional[Tensor]]],
            torch.jit.annotate(Dict[str, Dict[str, Optional[Tensor]]], {}),
        )

        for idx in range(max_seq_length - 1):
            last_token = predict[:, idx]
            sequence_embed = self.seq_embedding(last_token)
            sequence_embed = sequence_embed + sequence_pos[:, idx]
            sequence_embed = sequence_embed.reshape(1, batch_size, sequence_dim)
            new_token = self.decoder.forward_predict(sequence_embed, sequence_encode, incremental_state)
            new_token = new_token.reshape(batch_size, decode_dim)
            new_token = self.mapping(new_token)

            token = torch.argmax(new_token, -1)
            predict[:, idx + 1] = token

            if ((token == sign_dict['<eos>']) | (token == sign_dict['<pad>'])).all():
                break
        return predict[:, 1:]


class AmpNet(Net):
    @torch.cuda.amp.autocast()
    def forward(self, *args):
        return super(AmpNet, self).forward(*args)


