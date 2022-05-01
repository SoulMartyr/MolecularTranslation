from net.model.ResNet import ResNet
from net.model.Transformer import *
from utils.Util import Swish

image_dim = 1024
sequence_dim = 1024
decode_dim = 1024

height = 7
width = 7
image_size = 224

layer_num = 2
head_num = 8
attn_dim = 64
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
        self.feature_to_series = nn.Sequential(
            nn.Conv2d(2048, 1, kernel_size=1, bias=None),
            nn.BatchNorm2d(1),
            Swish()
        )
        self.encoder = Encoder(vocab_size, image_dim, ffn_dim, attn_dim, head_num, layer_num, width, height)
        # 词嵌入
        # TransformerDecoder
        self.decoder = Decoder(vocab_size, decode_dim, ffn_dim, attn_dim, head_num, layer_num, max_seq_length)
        # 输出映射
        self.mapping = nn.Linear(decode_dim, vocab_size)
        self.dropout = nn.Dropout(p=0.5)

        # 初始化
        self.mapping.bias.data.fill_(0)
        self.mapping.weight.data.uniform_(-0.1, 0.1)

    # 训练
    def forward(self, image, sequence):
        batch_size = len(image)
        # 特征提取
        feature_map = self.feature_extraction(image)
        feature_embed = self.feature_embedding(feature_map)
        feature_series = self.feature_to_series(feature_map)
        feature_series = feature_series.squeeze(1).reshape(batch_size, width * height)
        # Encode编码
        sequence_encode = self.encoder(feature_series, feature_embed)
        # Decoder解码
        sequence_decode = self.decoder(sequence, feature_series, sequence_encode)
        # 映射回原词库大小
        result = self.mapping(sequence_decode)
        return result

    # 测试
    def forward_predict(self, image):
        device = image.device
        batch_size = len(image)

        feature_map = self.feature_extraction(image)
        feature_embed = self.feature_embedding(feature_map)
        feature_series = self.feature_to_series(feature_map)
        feature_series = feature_series.squeeze(1).reshape(batch_size, width * height)
        sequence_encode = self.encoder(feature_series, feature_embed)

        predict = torch.full((batch_size, max_seq_length), sign_dict['<pad>'], dtype=torch.long).to(device)

        predict[:, 0] = sign_dict['<sos>']

        for idx in range(max_seq_length - 1):
            last_sequence = predict[:, :idx + 1]
            new_sequence = self.decoder(last_sequence, feature_series, sequence_encode)
            new_sequence = self.mapping(new_sequence)
            new_sequence = new_sequence.squeeze(0)
            token = torch.argmax(new_sequence, -1)[-1]

            predict[:, idx + 1] = token
            if ((token == sign_dict['<eos>']) | (token == sign_dict['<pad>'])).all():
                break
        return predict[:, 1:]


class AmpNet(Net):
    @torch.cuda.amp.autocast()
    def forward(self, *args):
        return super(AmpNet, self).forward(*args)



