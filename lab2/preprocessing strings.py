from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import random

# data
with open('../data/SMSSpamCollection', 'r') as f:
    data = f.read().split('\n')


random.shuffle(data[1:])

# Split the data into descriptors and target
X = [line.split('\t')[1] for line in data[:-1]]
y = [line.split('\t')[0] for line in data[:-1]]

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(X)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

# Logistic regression model
log_regression = LogisticRegression()
log_regression.fit(X_train,y_train)


#le = LabelEncoder()
#X_test = le.fit_transform(X_test)
y_pred = log_regression.predict(X_test)

# Explore logistic model
print('Coefficients:', log_regression.coef_)
print('Intercept:', log_regression.intercept_)

# Top-of-the-line metrics
print('Classification Report with threshold 0.5:\n', metrics.confusion_matrix(y_test, y_pred))
print('Classification Report with threshold 0.5:\n', metrics.classification_report(y_test, y_pred))

#Probabilistic metrics
y_pred_proba = log_regression.predict_proba(X_test)[::,1]

le = LabelEncoder()
y_test = le.fit_transform(y_test)
fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba)
auc = metrics.roc_auc_score(y_test, y_pred_proba)

#create ROC curve
plt.plot(fpr,tpr,label="AUC="+str(auc))
plt.legend(loc=4)
plt.show()
