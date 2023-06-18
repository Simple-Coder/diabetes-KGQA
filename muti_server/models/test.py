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

acc = ner_accuracy_score(y_true, y_pred)
precision = ner_precision_score(y_true, y_pred)
recall = ner_recall_score(y_true, y_pred)
f1 = ner_f1_score(y_true, y_pred)
report = ner_classification_report(y_true, y_pred)

print(acc, precision, recall, f1)
print(report)
