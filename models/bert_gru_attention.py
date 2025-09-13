# coding: UTF-8
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_pretrained import BertModel, BertTokenizer
import matplotlib.pyplot as plt
import seaborn as sns


# 注意：Config类保持不变，这里为了完整性再次列出
class Config(object):
    """配置参数"""

    def __init__(self, dataset):
        self.model_name = 'bert-gru-attention'  # 修改模型名称以便保存
        self.train_path = dataset + '/data/train.txt'
        self.dev_path = dataset + '/data/dev.txt'
        self.test_path = dataset + '/data/test.txt'
        self.class_list = [x.strip() for x in open(
            dataset + '/data/class.txt', encoding='utf-8').readlines()]
        self.save_path = dataset + '/saved_dict/' + self.model_name + '.ckpt'
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.require_improvement = 500
        self.num_classes = len(self.class_list)
        self.num_epochs = 8
        self.batch_size = 16
        self.pad_size = 64
        self.learning_rate = 1e-5
        self.bert_path = './bert_pretrain'
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_path)
        self.hidden_size = 768
        self.dropout = 0.5
        self.rnn_hidden = 768
        self.num_layers = 2


class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        self.bert = BertModel.from_pretrained(config.bert_path)
        for param in self.bert.parameters():
            param.requires_grad = True

        self.gru = nn.GRU(
            input_size=config.hidden_size,
            hidden_size=config.rnn_hidden,
            num_layers=config.num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=config.dropout if config.num_layers > 1 else 0
        )

        ### === Attention层 === ###
        # 定义一个全连接层，用于将GRU的输出转换为一个中间表示
        self.attention_W = nn.Linear(config.rnn_hidden * 2, config.rnn_hidden * 2)
        # 定义一个可学习的“查询”向量，用于计算注意力分数
        self.attention_v = nn.Parameter(torch.FloatTensor(config.rnn_hidden * 2, 1))
        # 初始化查询向量
        nn.init.xavier_uniform_(self.attention_v)
        ### =================== ###

        self.dropout_layer = nn.Dropout(config.dropout)

        # 最终的分类层，输入维度仍然是 rnn_hidden * 2
        self.fc = nn.Linear(config.rnn_hidden * 2, config.num_classes)

    def forward(self, x, seq_len, mask, return_attention=False):
        context = x  # shape: [batch_size, pad_size]
        # mask = mask  # shape: [batch_size, pad_size]

        # 1. BERT层
        encoder_out, _ = self.bert(context, attention_mask=mask, output_all_encoded_layers=False)
        # encoder_out shape: [batch_size, pad_size, hidden_size]

        # 2. GRU层
        gru_out, _ = self.gru(encoder_out)
        # gru_out shape: [batch_size, pad_size, rnn_hidden * 2]

        ### === Attention计算 === ###
        # 3. 计算Attention分数
        #    a. 将GRU输出通过一个全连接层并用tanh激活
        u = torch.tanh(self.attention_W(gru_out))
        #    u shape: [batch_size, pad_size, rnn_hidden * 2]
        #    b. 与查询向量v做矩阵乘法，得到每个时间步的分数
        scores = u.matmul(self.attention_v).squeeze(-1)
        #    scores shape: [batch_size, pad_size]

        # 4. Mask掉padding部分的分数
        #    将mask中为0（即padding）的位置在scores中设置为一个极小的负数
        #    这样在softmax后，这些位置的权重会趋近于0
        scores = scores.masked_fill(mask == 0, -1e9)

        # 5. 计算Attention权重
        attention_weights = F.softmax(scores, dim=1)
        #    attention_weights shape: [batch_size, pad_size]

        # 6. 计算加权的上下文向量
        #    使用权重对GRU的输出进行加权求和
        context_vector = torch.sum(attention_weights.unsqueeze(-1) * gru_out, dim=1)
        #    context_vector shape: [batch_size, rnn_hidden * 2]
        ### ======================= ###

        # 7. 分类
        out = self.dropout_layer(context_vector)
        out = self.fc(out)

        if return_attention:
            return out, attention_weights
        else:
            return out