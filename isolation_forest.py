import numpy as np
import pandas as pd
import itertools
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_score, roc_curve, auc

file_name = '../creditcard.csv'
rng = np.random.RandomState(1122)

raw_data = pd.read_csv(file_name)
x_train, x_test, y_train, y_test = train_test_split(
    raw_data.iloc[:, :-1].as_matrix(),
    raw_data.iloc[:, -1].values,
    random_state=rng, test_size=0.1)
print('Data loaded...')

isofraud = IsolationForest(random_state=rng)
print('Model trained...')

isofraud.fit(x_train, y=y_train)
print('Prediction made...')

y_pred = isofraud.predict(x_test)
y_pred[y_pred > 0] = 0
y_pred[y_pred < 0] = 1

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
