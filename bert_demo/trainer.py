import os
import re
from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# %matplotlib inline

# 加载数据并设置标签
from bert_demo.bert_model import BertClassifier

data_complaint = pd.read_csv('/Users/weijinqian/Documents/nlp/DF-Net/bert_demo/data/complaint1700.csv')
data_complaint['label'] = 0
data_non_complaint = pd.read_csv('/Users/weijinqian/Documents/nlp/DF-Net/bert_demo/data/noncomplaint1700.csv')
data_non_complaint['label'] = 1

# 将抱怨和不抱怨的两个数据合成一块
data = pd.concat([data_complaint, data_non_complaint], axis=0).reset_index(drop=True)

# 删除 'airline' 列
data.drop(['airline'], inplace=True, axis=1)

# 展示随机的5个样本
data.sample(5)

from sklearn.model_selection import train_test_split

X = data['tweet'].values
y = data['label'].values

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=2020)

# Load test data
test_data = pd.read_csv('/Users/weijinqian/Documents/nlp/DF-Net/bert_demo/data/test_data.csv')

# Keep important columns
test_data = test_data[['id', 'tweet']]

# Display 5 samples from the test data
test_data.sample(5)

import torch

if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f'There are {torch.cuda.device_count()} GPU(s) available.')
    print('Device name:', torch.cuda.get_device_name(0))

else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")


def text_preprocessing(text):
    """
    - 删除命名实体(例如 '@united'联合航空)
    - 纠正错误 (例如： '&amp;' 改成 '&')
    @该函数input：传进文本字符串
    @该函数return：返回处理过的文本字符串
    """
    # Remove '@name'
    text = re.sub(r'(@.*?)[\s]', ' ', text)

    # Replace '&amp;' with '&'
    text = re.sub(r'&amp;', '&', text)

    # Remove trailing whitespace 删除空格
    text = re.sub(r'\s+', ' ', text).strip()

    return text


from transformers import BertTokenizer, BertModel

# 加载 BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
MAX_LEN = 64


# 创建一个函数来tokenize一串文本
def preprocessing_for_bert(data):
    """
    @传入参数(param)  一串存储在np.array格式下的文本数据: Array of texts to be processed.
    @该函数返回（return1）：   input_ids (torch.Tensor格式): Tensor of token ids to be fed to a model.
    @该函数返回（return2）：   attention_masks (torch.Tensor格式): 用于指示句子中的哪些token用于模型训练
    """
    # 创建空列表来存储output数据
    input_ids = []
    attention_masks = []

    # 对存储在data(np.array)中的每个句子....
    for sent in data:
        encoded_sent = tokenizer.encode_plus(  # 进行编码
            text=text_preprocessing(sent),  # 调用上面创建的略微预处理文本的函数
            add_special_tokens=True,  # Add `[CLS]` and `[SEP]`
            max_length=MAX_LEN,  # 指定max_length（后面会指定）
            pad_to_max_length=True,  # 补长 padding
            # return_tensors='pt',           # Return PyTorch tensor 是否返回PyTorch张量
            return_attention_mask=True,  # Return attention mask
            truncation=True  # 截短
        )

        # 从上面编码得到的对象中用get获取input_ids和attention_mask存储到各自的列表中
        input_ids.append(encoded_sent.get('input_ids'))
        attention_masks.append(encoded_sent.get('attention_mask'))

    # 再将input_ids列表和attention_masks列表转换成torch的张量格式
    input_ids = torch.tensor(input_ids)
    attention_masks = torch.tensor(attention_masks)
    # 返回所有句子的input_ids, attention_masks（tensor格式）
    return input_ids, attention_masks


# 将训练数据集和测试数据集合并
all_tweets = np.concatenate([data.tweet.values, test_data.tweet.values])

# 对合并的数据进行编码
encoded_tweets = [tokenizer.encode(sent, add_special_tokens=True) for sent in all_tweets]

# 将编码后的句子长度,存储到一个列表中,找最大值
max_len = max([len(sent) for sent in encoded_tweets])
print('Max length: ', max_len)

# 打印示例：第一个句子的token_id
token_ids = list(preprocessing_for_bert([X[0]])[0].squeeze().numpy())
print('Original: ', X[0])
print('Token IDs: ', token_ids)

# 运行函数 `preprocessing_for_bert`来处理训练集和验证集
print('Tokenizing data...')
train_inputs, train_masks = preprocessing_for_bert(X_train)
val_inputs, val_masks = preprocessing_for_bert(X_val)

from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

# 将训练集和验证集的label转化成 torch.Tensor格式
train_labels = torch.tensor(y_train)
val_labels = torch.tensor(y_val)

# 针对微调fine-tuning BERT, 作者推荐 batch size 16或32
batch_size = 32

# 为训练集创建DataLoader
train_data = TensorDataset(train_inputs, train_masks, train_labels)  # 将训练集的input_id，mask和label都封装进TensorDataset
train_sampler = RandomSampler(train_data)  # 将封装好的数据洗牌
train_dataloader = DataLoader(train_data, sampler=train_sampler,
                              batch_size=batch_size)  # 将洗牌好的数据传进DataLoader，并指定batch_size

# 为验证集创建DataLoader
val_data = TensorDataset(val_inputs, val_masks, val_labels)
val_sampler = SequentialSampler(val_data)
val_dataloader = DataLoader(val_data, sampler=val_sampler, batch_size=batch_size)

from transformers import AdamW, get_linear_schedule_with_warmup


def initialize_model(epochs=4):
    """
    初始化Bert Classifier, optimizer ，learning rate scheduler.
    """
    # 实例化 Bert Classifier
    bert_classifier = BertClassifier(freeze_bert=False)

    # 告诉这个实例化的分类器，使用gpu还是cpu
    bert_classifier.to(device)

    # 创建优化器optimizer
    optimizer = AdamW(bert_classifier.parameters(),
                      lr=5e-5,  # Default learning rate
                      eps=1e-8  # Default epsilon value
                      )

    # 总共训练步数是多少？Total number of training steps
    total_steps = len(train_dataloader) * epochs

    # 设置learning rate scheduler
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=0,  # 默认值是0，意思是预热期要几步达到预设的学习率
                                                num_training_steps=total_steps)
    return bert_classifier, optimizer, scheduler


import random
import time

# 指定 loss function
loss_fn = torch.nn.CrossEntropyLoss()  # 交叉熵损失函数


def set_seed(seed_value=42):
    """设置随机种子，为了之后复现。Set seed for reproducibility.
    """
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)


def train(model, train_dataloader, val_dataloader=None, optimizer=None, epochs=4, evaluation=False):
    """正式 BertClassifier model.
    """
    # 开始training loop
    print("Start training...\n")
    for epoch_i in range(epochs):
        # =================================================================================================
        #                                            Training
        # =================================================================================================
        # 打印结果表格的表头，epoch显示当前训练是第几个epoch，训练到第几个batch了，Elapsed是耗时多少秒
        print(f"{'Epoch':^7} | {'Batch':^7} | {'Train Loss':^12} | {'Val Loss':^10} | {'Val Acc':^9} | {'Elapsed':^9}")
        print("-" * 70)

        # 开始计时，测算每轮epoch耗时多长时间
        t0_epoch, t0_batch = time.time(), time.time()

        # 每轮epoch开始前将各个计数器归零
        total_loss, batch_loss, batch_counts = 0, 0, 0

        # 这个train函数需要往里传入一个model参数，而这个model参数接收的就是之前initialize_model函数会返回的一个bert分类器模型
        model.train()  # 这个model = 一个实例化的bert_classifier

        # For each batch of training data... 从dataloader读取数据
        for step, batch in enumerate(train_dataloader):
            batch_counts += 1
            # 加载 batch到GPU/CPU
            b_input_ids, b_attn_mask, b_labels = tuple(t.to(device) for t in batch)

            # 将累计梯度清零
            model.zero_grad()

            # 往模型中传入从上面得到的input_id和mask，模型会进行前向传播得到logits值
            logits = model(b_input_ids, b_attn_mask)

            # 通过损失函数计算logits跟label之间的差距得到损失值，Compute loss and accumulate the loss values
            loss = loss_fn(logits, b_labels)
            batch_loss += loss.item()
            total_loss += loss.item()

            # 执行后向传播计算梯度
            loss.backward()

            # 修剪梯度进行归一化防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # 更新model参数，更新学习率
            optimizer.step()
            scheduler.step()

            # 每20个batch打印损失值和时间消耗
            if (step % 20 == 0 and step != 0) or (step == len(train_dataloader) - 1):
                # Calculate time elapsed for 20 batches
                time_elapsed = time.time() - t0_batch

                # Print training results
                print(
                    f"{epoch_i + 1:^7} | {step:^7} | {batch_loss / batch_counts:^12.6f} | {'-':^10} | {'-':^9} | {time_elapsed:^9.2f}")

                # 将计数器清零
                batch_loss, batch_counts = 0, 0
                t0_batch = time.time()

        # 计算整个训练数据集的平均损失（Calculate the average loss over the entire training data）
        avg_train_loss = total_loss / len(train_dataloader)

        print("-" * 70)
        # =========================================================================================
        #               Evaluation
        # =========================================================================================
        if evaluation == True:
            # 在每个epoch结束后会用验证集来测试模型的表现
            val_loss, val_accuracy = evaluate(model, val_dataloader)  # 这个evaluate函数下面有定义

            # 打印这一轮epoch下，在训练集上训练完所有数据后所耗得总体时间
            time_elapsed = time.time() - t0_epoch

            print(
                f"{epoch_i + 1:^7} | {'-':^7} | {avg_train_loss:^12.6f} | {val_loss:^10.6f} | {val_accuracy:^9.2f} | {time_elapsed:^9.2f}")
            print("-" * 70)
        print("\n")

    print("Training complete!")


def evaluate(model, val_dataloader):
    """在每个epoch结束后会用验证集来测试模型的表现
    """
    # Put the model into the evaluation mode. The dropout layers are disabled during
    # the test time.
    model.eval()

    # 创建空集，为了之后记录每一个batch的accuracy和loss
    val_accuracy = []
    val_loss = []

    # F在验证集中，每个batch....
    for batch in val_dataloader:
        # 加载 batch 数据到 GPU/CPU
        b_input_ids, b_attn_mask, b_labels = tuple(t.to(device) for t in batch)

        # 计算 logits
        with torch.no_grad():
            logits = model(b_input_ids, b_attn_mask)

        # 计算损失值
        loss = loss_fn(logits, b_labels)
        val_loss.append(loss.item())

        # 获取预测值
        preds = torch.argmax(logits, dim=1).flatten()

        # 计算准确率
        accuracy = (preds == b_labels).cpu().numpy().mean() * 100
        val_accuracy.append(accuracy)

    # 计算验证集的accuracy和loss
    val_loss = np.mean(val_loss)
    val_accuracy = np.mean(val_accuracy)

    return val_loss, val_accuracy


set_seed(42)  # Set seed for reproducibility
bert_classifier, optimizer, scheduler = initialize_model(epochs=2)
# 训练
train(bert_classifier, train_dataloader, val_dataloader, optimizer, epochs=2, evaluation=True)

import torch.nn.functional as F


def bert_predict(model, test_dataloader):
    """Perform a forward pass on the trained BERT model to predict probabilities
    on the test set.
    """
    # Put the model into the evaluation mode. The dropout layers are disabled during
    # the test time.
    model.eval()

    all_logits = []

    # For each batch in our test set...
    for batch in test_dataloader:
        # Load batch to GPU
        b_input_ids, b_attn_mask = tuple(t.to(device) for t in batch)[:2]

        # Compute logits
        with torch.no_grad():
            logits = model(b_input_ids, b_attn_mask)
        all_logits.append(logits)

    # Concatenate logits from each batch
    all_logits = torch.cat(all_logits, dim=0)

    # Apply softmax to calculate probabilities
    probs = F.softmax(all_logits, dim=1).cpu().numpy()

    return probs


from sklearn.metrics import accuracy_score, roc_curve, auc


def evaluate_roc(probs, y_true):
    """
    - Print AUC and accuracy on the test set
    - Plot ROC
    @params    probs (np.array): an array of predicted probabilities with shape (len(y_true), 2)
    @params    y_true (np.array): an array of the true values with shape (len(y_true),)
    """
    preds = probs[:, 1]
    fpr, tpr, threshold = roc_curve(y_true, preds)
    roc_auc = auc(fpr, tpr)
    print(f'AUC: {roc_auc:.4f}')

    # Get accuracy over the test set
    y_pred = np.where(preds >= 0.5, 1, 0)
    accuracy = accuracy_score(y_true, y_pred)
    print(f'Accuracy: {accuracy * 100:.2f}%')

    # Plot ROC AUC
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()

