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

fraud_case = raw_data[raw_data['Class'] == 1]
print('There are %i fraudulent cases in the dataset...' % len(fraud_case))
genuine_case = raw_data[raw_data['Class'] == 0]
print('There are %i genuine cases in the dataset...' % len(genuine_case))

print('The data target class is highly unbalanced...')
print('Generating a balanced sample with roughly 1:1 for fraud:genuine...')
adjusted_raw = fraud_case.append(genuine_case.sample(n=2460, random_state=rng))

x_train, x_test, y_train, y_test = train_test_split(
    adjusted_raw.iloc[:, :-1].as_matrix(),
    adjusted_raw.iloc[:, -1].values.astype(str),
    random_state=rng, test_size=0.25)
print('Training and testing set splited...')
print('Data loaded...')

logfraud = LogisticRegression(
    penalty='l2',
    random_state=rng,
    class_weight={"1": 5, "0": 1},
    solver='liblinear')
logfraud.fit(x_train, y_train)
print('Model trained...')

y_pred = logfraud.predict(x_test).astype(int)
print('Predictions made...')

y_test = y_test.astype(int)

# Evaluation 1: Scores
cm = confusion_matrix(y_test, y_pred)
prec = precision_score(y_test, y_pred)
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
