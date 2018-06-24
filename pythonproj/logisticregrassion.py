# import pandas as pd
# import numpy as np
# import pandas as pd
# import numpy as np
# from sklearn import preprocessing
# import matplotlib.pyplot as plt
# plt.rc("font", size=14)
# from sklearn.linear_model import LogisticRegression
# from sklearn.cross_validation import train_test_split
# import seaborn as sns
# sns.set(style="white")
# sns.set(style="whitegrid", color_codes=True)
# data = pd.read_csv('header_training_set.csv',header =0)
# import pdb;pdb.set_trace()
# data['education']=np.where(data['education'] =='basic.9y', 'Basic', data['education'])
# data['education']=np.where(data['education'] =='basic.6y', 'Basic', data['education'])
# data['education']=np.where(data['education'] =='basic.4y', 'Basic', data['education'])
# # %matplotlib inline
# pd.crosstab(data.job,data.y).plot(kind='bar')
# plt.title('Purchase Frequency for Job Title')
# plt.xlabel('Job')
# plt.ylabel('Frequency of Purchase')
# plt.savefig('purchase_fre_job')
# table=pd.crosstab(data.marital,data.y)
# table.div(table.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True)
# plt.title('Stacked Bar Chart of Marital Status vs Purchase')
# plt.xlabel('Marital Status')
# plt.ylabel('Proportion of Customers')
# plt.savefig('mariral_vs_pur_stack')
# table=pd.crosstab(data.education,data.y)
# table.div(table.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True)
# plt.title('Stacked Bar Chart of Education vs Purchase')
# plt.xlabel('Education')
# plt.ylabel('Proportion of Customers')
# plt.savefig('edu_vs_pur_stack')
# pd.crosstab(data.day_of_week,data.y).plot(kind='bar')
# plt.title('Purchase Frequency for Day of Week')
# plt.xlabel('Day of Week')
# plt.ylabel('Frequency of Purchase')
# plt.savefig('pur_dayofweek_bar')
# pd.crosstab(data.month,data.y).plot(kind='bar')
# plt.title('Purchase Frequency for Month')
# plt.xlabel('Month')
# plt.ylabel('Frequency of Purchase')
# plt.savefig('pur_fre_month_bar')
# data.age.hist()
# plt.title('Histogram of Age')
# plt.xlabel('Age')
# plt.ylabel('Frequency')
# plt.savefig('hist_age')
# pd.crosstab(data.poutcome,data.y).plot(kind='bar')
# plt.title('Purchase Frequency for Poutcome')
# plt.xlabel('Poutcome')
# plt.ylabel('Frequency of Purchase')
# plt.savefig('pur_fre_pout_bar')
# cat_vars=['job','marital','education','default','housing','loan','contact','month','day_of_week','poutcome']
# for var in cat_vars:
#     cat_list='var'+'_'+var
#     cat_list = pd.get_dummies(data[var], prefix=var)
#     data1=data.join(cat_list)
#     data=data1
# cat_vars=['job','marital','education','default','housing','loan','contact','month','day_of_week','poutcome']
# data_vars=data.columns.values.tolist()
# to_keep=[i for i in data_vars if i not in cat_vars]
# data_final=data[to_keep]
# data_final.columns.values
#
# data_final_vars=data_final.columns.values.tolist()
# y=['y']
# X=[i for i in data_final_vars if i not in y]
#
# from sklearn import datasets
# from sklearn.feature_selection import RFE
# from sklearn.linear_model import LogisticRegression
# logreg = LogisticRegression()
# rfe = RFE(logreg, 18)
# rfe = rfe.fit(data_final[X], data_final[y] )
# print(rfe.support_)
# print(rfe.ranking_)
# cols=["previous", "euribor3m", "job_blue-collar", "job_retired", "job_services", "job_student", "default_no",
#       "month_aug", "month_dec", "month_jul", "month_nov", "month_oct", "month_sep", "day_of_week_fri", "day_of_week_wed",
#       "poutcome_failure", "poutcome_nonexistent", "poutcome_success"]
# X=data_final[cols]
# y=data_final['y']
# import statsmodels.api as sm
# logit_model=sm.Logit(y,X)
# result=logit_model.fit()
# print(result.summary())
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
# from sklearn.linear_model import LogisticRegression
# from sklearn import metrics
# logreg = LogisticRegression()
# logreg.fit(X_train, y_train)
# y_pred = logreg.predict(X_test)
# print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(X_test, y_test)))
#
# from sklearn import model_selection
# from sklearn.model_selection import cross_val_score
# kfold = model_selection.KFold(n_splits=10, random_state=7)
# modelCV = LogisticRegression()
# scoring = 'accuracy'
# results = model_selection.cross_val_score(modelCV, X_train, y_train, cv=kfold, scoring=scoring)
# print("10-fold cross validation average accuracy: %.3f" % (results.mean()))
# from sklearn.metrics import confusion_matrix
# confusion_matrix = confusion_matrix(y_test, y_pred)
# print(confusion_matrix)
# from sklearn.metrics import classification_report
# print(classification_report(y_test, y_pred))
# from sklearn.metrics import roc_auc_score
# from sklearn.metrics import roc_curve
# logit_roc_auc = roc_auc_score(y_test, logreg.predict(X_test))
# fpr, tpr, thresholds = roc_curve(y_test, logreg.predict_proba(X_test)[:,1])
# plt.figure()
# plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
# plt.plot([0, 1], [0, 1],'r--')
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('Receiver operating characteristic')
# plt.legend(loc="lower right")
# plt.savefig('Log_ROC')
# plt.show()

# First XGBoost model for Pima Indians dataset
from numpy import loadtxt
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
# load data
dataset = loadtxt('header_training_set1.csv', delimiter=",")
# split data into X and y
X = dataset[:,0:14]
Y = dataset[:,-1]
# split data into train and test sets
seed = 7
import pdb;pdb.set_trace()
test_size = 0.33
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)
# fit model no training data
model = XGBClassifier()
model.fit(X_train, y_train)
# make predictions for test data
y_pred = model.predict(X_test)
predictions = [round(value) for value in y_pred]
# evaluate predictions
accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))