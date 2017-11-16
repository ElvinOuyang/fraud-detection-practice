import numpy as np
import pandas as pd
import itertools
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_score, roc_curve, auc

file_name = '../creditcard.csv'
rng = np.random.RandomState(1122)

raw_data = pd.read_csv(file_name)
x_train, x_test, y_train, y_test = train_test_split(
    raw_data.iloc[:, :-1].as_matrix(),
    raw_data.iloc[:, -1].values.astype(str),
    random_state=rng, test_size=0.1)

print('Data loaded...')

logfraud = LogisticRegression(
    penalty='l2',
    class_weight={'1': 1000, '0': 1},
    random_state=rng,
    solver='liblinear')

logfraud.fit(x_train, y_train)
print('Model trained...')

y_pred = logfraud.predict(x_test)
y_pred[y_pred == '0'] = 0
y_pred[y_pred == '1'] = 1
print('Prediction made...')

# Evaluation 1: Scores
cm = confusion_matrix(y_test, y_pred)
prec = precision_score(y_test, y_pred)  # TODO: debug this line
print("Model prediction's confusion matrix:")
print(cm)
print("Model achieved precision score of %.2f percent" % (prec * 100))

# Evaluation 2: Plot ROC
fpr, tpr, _ = roc_curve(y_test, y_pred)
roc_auc = auc(fpr, tpr)
plt.figure(1)
lw = 2
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()
