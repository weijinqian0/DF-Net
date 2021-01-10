#Concatenate the train set and the validation set
full_train_data = torch.utils.data.ConcatDataset([train_data, val_data])# 调用torch库中的合并数据集函数
full_train_sampler = RandomSampler(full_train_data) #将数据洗牌，打乱顺序
full_train_dataloader = DataLoader(full_train_data, sampler=full_train_sampler, batch_size=32) #将打乱的数据放进DataLoader中

# Train the Bert Classifier on the entire training data
set_seed(42)
bert_classifier, optimizer, scheduler = initialize_model(epochs=2)
train(bert_classifier, full_train_dataloader, epochs=2)

#再浏览下测试集长什么样
test_data.sample(5)

# Run `preprocessing_for_bert` on the test set
print('Tokenizing data...')
test_inputs, test_masks = preprocessing_for_bert(test_data.tweet)

# Create the DataLoader for our test set
test_dataset = TensorDataset(test_inputs, test_masks)
test_sampler = SequentialSampler(test_dataset)
test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=32)

# Compute predicted probabilities on the test set
probs = bert_predict(bert_classifier, test_dataloader) #这里的bert_classifier是上面训练好的分类器

# Get predictions from the probabilities
threshold = 0.9
preds = np.where(probs[:, 1] > threshold, 1, 0) #满足判断预测值大于阈值就使出1，不满足输出0

# Number of tweets predicted non-negative
print("Number of tweets predicted non-negative: ", preds.sum())



