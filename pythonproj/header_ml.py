import pandas as pd
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt
plt.rc("font", size=14)
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
import seaborn as sns
sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)
data = pd.read_csv('header_training_set.csv',header =0)
data = data.dropna()
print(data.shape)
print(data.columns)
data.head()
data['caps'].unique()
data['is_header'].unique()
data['is_header'].value_counts()
sns.countplot(x = "is_header",data = data,palette = 'hls')
# plt.show()
data.groupby("is_header").mean()
data.groupby("caps").mean()
# pd.crosstab(data.job,data.y).plot(kind='bar')
# plt.title('Purchase Frequency for Job Title')
# plt.xlabel('Job')
# plt.ylabel('Frequency of Purchase')
# plt.savefig('purchase_fre_job')
from sklearn import datasets
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
# rfe = RFE(logreg, 18)
# rfe = rfe.fit(data_final[X], data_final[y] )
data_final=data
data_final.columns.values
y = ['is_header']
data_final_vars=data_final.columns.values.tolist()

X=[i for i in data_final_vars if i not in y]
X = data_final[X]
y = data_final[y]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(X_test, y_test)))
from sklearn import model_selection
from sklearn.model_selection import cross_val_score
kfold = model_selection.KFold(n_splits=10, random_state=7)
modelCV = LogisticRegression()
scoring = 'accuracy'
results = model_selection.cross_val_score(modelCV, X_train, y_train, cv=kfold, scoring=scoring)
print("10-fold cross validation average accuracy: %.3f" % (results.mean()))
from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test, y_pred)
print(confusion_matrix)
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))