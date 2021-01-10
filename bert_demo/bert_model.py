# % % time
import torch
import torch.nn as nn
from transformers import BertModel


# 创建 BertClassfier 类
class BertClassifier(nn.Module):
    def __init__(self, freeze_bert=False):
        """
        实例化BertClassifier需要三个参数
        @param1    bert: a BertModel object （BertModel对象）
        @param2    classifier: a torch.nn.Module classifier （一个自定义的分类器，该分类器继承nn.Module）
        @param3    freeze_bert (bool): Set `False` to fine-tune the BERT model （设置是否冻结BERT里的权重参数）
        """
        super(BertClassifier, self).__init__()
        # 指定 hidden size of BERT(默认768维), hidden size of our classifier（自己设置为50）, and number of labels（2分类问题）
        D_in, H, D_out = 768, 50, 2

        # 实例化BERT模型（Instantiate BERT model）
        self.bert = BertModel.from_pretrained('bert-base-uncased')

        # 实例化一层前向传播分类器（Instantiate an one-layer feed-forward classifier）
        self.classifier = nn.Sequential(  # Sequential 就是一个有序容器，添加一层层神经网络
            nn.Linear(D_in, H),
            nn.ReLU(),
            # nn.Dropout(0.5),
            nn.Linear(H, D_out)
        )

        # 冻结 BERT model（是否让BERT的权重参数进行更新）
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

    def forward(self, input_ids, attention_mask):
        """
        将输入传进BERT中，让classifier来计算logits，logits类似于未进行归一化的softmax的概率
        @输入(param1):input_ids (torch.Tensor): 传入一个id张量tensor，其形状为(batch_size, max_length)
        @输入(param2)：attention_mask (torch.Tensor): 传入一个mask张量，形状为(batch_size, max_length)
        @返回(return): logits (torch.Tensor): 一个输出张量，类似于softmax(batch_size, num_labels)
        """
        # 将input_ids,和attention_mask传入 BERT
        outputs = self.bert(input_ids=input_ids,
                            attention_mask=attention_mask)

        # 提取用于分类任务的特殊token[CLS]的最后一个隐层参数 （Extract the last hidden state of the token `[CLS]` for classification task）
        last_hidden_state_cls = outputs[0][:, 0, :]

        # 将上面获得CLS的最后一个隐层参数传入classifier计算logits值
        logits = self.classifier(last_hidden_state_cls)  # 可以将logits理解成未进入softmax时的概率

        return logits

