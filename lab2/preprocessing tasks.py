from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import matplotlib.pyplot as plt

# data
data = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.3)

# Logistic regression model
log_regression = LogisticRegression()
log_regression.fit(X_train,y_train)
y_pred = log_regression.predict(X_test)

# Explore logistic model
print('Coefficients:', log_regression.coef_)
print('Intercept:', log_regression.intercept_)

# Top-of-the-line metrics
print('Classification Report with threshold 0.5:\n', metrics.confusion_matrix(y_test, y_pred))
print('Classification Report with threshold 0.5:\n', metrics.classification_report(y_test, y_pred))

#Probabilistic metrics
y_pred_proba = log_regression.predict_proba(X_test)[::,1]
fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba)
auc = metrics.roc_auc_score(y_test, y_pred_proba)

#create ROC curve
plt.plot(fpr,tpr,label="AUC="+str(auc))
plt.legend(loc=4)
plt.show()


