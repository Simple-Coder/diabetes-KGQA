"""
Created by xiedong
@Date: 2023/6/12 20:33
"""


def compute_gradient_w1():
    return 1


def compute_gradient_w2():
    return 2


w1 = 1
w2 = 2
# 假设模型中有两个参数：w1和w2
# 初始化梯度为0
w1_grad = 0
w2_grad = 0

# 第一次反向传播计算梯度
w1_grad += compute_gradient_w1()  # 假设第一次计算得到的梯度为1
w2_grad += compute_gradient_w2()  # 假设第一次计算得到的梯度为2

# 第二次反向传播计算梯度
w1_grad += compute_gradient_w1()  # 假设第二次计算得到的梯度为3
w2_grad += compute_gradient_w2()  # 假设第二次计算得到的梯度为4

# 参数更新
learning_rate = 0.1
w1 -= learning_rate * w1_grad  # 根据梯度更新参数w1
w2 -= learning_rate * w2_grad  # 根据梯度更新参数w2

# 下一轮迭代前，需要将梯度清零
w1_grad = 0
w2_grad = 0

# 继续进行下一轮迭代...
from seqeval.metrics import accuracy_score as ner_accuracy_score
from seqeval.metrics import precision_score as ner_precision_score
from seqeval.metrics import recall_score as ner_recall_score
from seqeval.metrics import f1_score as ner_f1_score
from seqeval.metrics import classification_report as ner_classification_report
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, hamming_loss
from sklearn.metrics import multilabel_confusion_matrix, classification_report

y_pred = [['B-name', 'I-name', 'I-name', 'I-name'], ['B-ingredient', 'I-ingredient', 'O', 'O', 'O', 'O'],
          ['O', 'B-startLoc_city', 'I-Dest', 'O', 'O', 'O', 'O', 'B-endLoc_city', 'I-endLoc_city', 'O', 'O', 'O'],
          ['B-keyword', 'I-keyword', 'O'],
          ['O', 'B-name', 'I-name', 'I-name', 'O', 'O', 'O', 'B-content', 'I-content', 'I-content', 'I-content',
           'I-content', 'I-content'], ['B-name', 'I-name', 'I-name', 'I-name', 'I-name'],
          ['O', 'O', 'O', 'O', 'O', 'O'],
          ['O', 'O', 'O', 'B-dishName', 'I-dishName', 'I-dishName', 'I-dishName', 'I-dishName', 'O', 'O', 'O', 'O'],
          ['O', 'O', 'O', 'O', 'B-name', 'I-name', 'I-name', 'B-category', 'I-name', 'O', 'O', 'O', 'B-content',
           'I-content', 'I-content', 'I-content', 'I-content', 'I-content', 'I-content'],
          ['O', 'O', 'O', 'B-name', 'I-name']]
y_true = [['B-name', 'I-name', 'I-name', 'I-name'], ['B-ingredient', 'I-ingredient', 'O', 'O', 'O', 'O'],
          ['O', 'B-Src', 'I-Src', 'O', 'O', 'O', 'O', 'B-Dest', 'I-Dest', 'O', 'O', 'O'],
          ['B-keyword', 'I-keyword', 'O'],
          ['O', 'B-name', 'I-name', 'I-name', 'O', 'O', 'O', 'B-content', 'I-content', 'I-content', 'I-content',
           'I-content', 'I-content'], ['B-name', 'I-name', 'I-name', 'I-name', 'I-name'],
          ['O', 'O', 'O', 'O', 'O', 'O'],
          ['O', 'O', 'O', 'B-dishName', 'I-dishName', 'I-dishName', 'I-dishName', 'I-dishName', 'O', 'O', 'O', 'O'],
          ['O', 'O', 'O', 'O', 'B-name', 'I-name', 'I-name', 'B-teleOperator', 'I-teleOperator', 'O', 'O', 'O',
           'B-content', 'I-content', 'I-content', 'I-content', 'I-content', 'I-content', 'I-content'],
          ['O', 'O', 'O', 'B-name', 'I-name']]


def test_ner_report():
    acc = ner_accuracy_score(y_true, y_pred)
    precision = ner_precision_score(y_true, y_pred)
    recall = ner_recall_score(y_true, y_pred)
    f1 = ner_f1_score(y_true, y_pred)
    report = ner_classification_report(y_true, y_pred)

    print(acc, precision, recall, f1)
    print(report)


def muti_accuracy(y_true, y_pred):
    count = 0
    for i in range(y_true.shape[0]):
        p = sum(np.logical_and(y_true[i], y_pred[i]))
        q = sum(np.logical_or(y_true[i], y_pred[i]))
        count += p / q
    return count / y_true.shape[0]


def muti_precision(y_true, y_pred):
    count = 0
    for i in range(y_true.shape[0]):
        if sum(y_pred[i]) == 0:
            continue
        count += sum(np.logical_and(y_true[i], y_pred[i])) / sum(y_pred[i])
    return count / y_true.shape[0]


def muti_recall(y_true, y_pred):
    count = 0
    for i in range(y_true.shape[0]):
        if sum(y_true[i]) == 0:
            continue
        count += sum(np.logical_and(y_true[i], y_pred[i])) / sum(y_true[i])
    return count / y_true.shape[0]


def muti_f1Measure(y_true, y_pred):
    count = 0
    for i in range(y_true.shape[0]):
        if (sum(y_true[i]) == 0) and (sum(y_pred[i]) == 0):
            continue
        p = sum(np.logical_and(y_true[i], y_pred[i]))
        q = sum(y_true[i]) + sum(y_pred[i])
        count += (2 * p) / q
    return count / y_true.shape[0]


if __name__ == '__main__':
    # test_ner_report()
    y_true = np.array([[0, 1, 0, 1],
                       [0, 1, 1, 0],
                       [1, 0, 1, 1]])

    y_pred = np.array([[0, 1, 1, 0],
                       [0, 1, 1, 0],
                       [0, 1, 0, 1]])

    # 计算准确度
    acc = accuracy_score(y_true, y_pred)

    # 计算每个标签的指标（精确度、召回率、F1值）
    precision = precision_score(y_true, y_pred, average='micro')
    recall = recall_score(y_true, y_pred, average='micro')
    f1 = f1_score(y_true, y_pred, average='micro')

    # 计算Hamming Loss
    hamming_loss = hamming_loss(y_true, y_pred)

    # 计算多标签分类报告
    classification_report = classification_report(y_true, y_pred)

    # 计算多标签混淆矩阵
    confusion_matrix = multilabel_confusion_matrix(y_true, y_pred)

    # 打印结果
    print("Accuracy:", acc)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)
    print("Hamming Loss:", hamming_loss)
    print("Classification Report:")
    print(classification_report)
    print("Confusion Matrix:")
    print(confusion_matrix)
    
    
    
    # https://zhuanlan.zhihu.com/p/385475273
    # 准确率
    accuracy = muti_accuracy(y_true, y_pred)
    print(accuracy)
    # 精确率
    print(muti_precision(y_true, y_pred))  # 0.6666
    # 召回率
    print(muti_recall(y_true, y_pred))  # 0.6111
    # F1
    print(muti_f1Measure(y_true, y_pred))  # 0.6333
    print("------")
    print(precision_score(y_true=y_true, y_pred=y_pred, average='samples'))  # 0.6666
    print(recall_score(y_true=y_true, y_pred=y_pred, average='samples'))  # 0.6111
    print(f1_score(y_true, y_pred, average='samples'))  # 0.6333
    # print(hamming_loss(y_true, y_pred))  # 0.4166
    # print(accuracy_score(y_true, y_pred))  # 0.33333333
    # print(accuracy_score(np.array([[0, 1], [1, 1]]), np.ones((2, 2))))  # 0.5
