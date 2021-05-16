#!/usr/bin/env python
# coding: utf-8

# In[3]:


# Breast cancer project


# In[ ]:


#import libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


#load the data

from sklearn.datasets import load_breast_cancer
cancer_dataset = load_breast_cancer()


# In[3]:


cancer_dataset


# In[4]:


type(cancer_dataset)


# In[5]:


#keys in dataset
cancer_dataset.keys()


# In[6]:


#features of each cells in numeric format
cancer_dataset['feature_names']


# In[7]:


#description of data 
print(cancer_dataset['DESCR'])


# In[8]:


#name of features
print(cancer_dataset['feature_names'])


# In[9]:


#location of data file
print(cancer_dataset['filename'])


# In[10]:


#create dataframe
import pandas as pd
import numpy as np
cancer_df = pd.DataFrame(np.c_[cancer_dataset['data'], cancer_dataset['target']],
                      columns=np.append(cancer_dataset['feature_names'],['target']))


# In[11]:


#dataframe to csv file
cancer_df.to_csv('breast_cancer_dataframe.csv')


# In[12]:


#head of cancer dataframe
cancer_df.head(6)


# In[13]:


#tail of cancer dataframe
cancer_df.tail(6)


# In[14]:


#information of cancer dataframe
cancer_df.info()


# In[15]:


#numerical distribution of data
cancer_df.describe()


# In[16]:


cancer_df.isnull()


# In[17]:


cancer_df.sum()


# In[ ]:


#data visualization


# In[ ]:


# pairplot of cancer detection
#import seaborn as sns


# In[ ]:


sns.pairplot(cancer_df, hue = 'target')


# In[ ]:


#pairplot of sample feature
sns.pairplot(cancer_df, hue = 'target' , vars =['mean radius','mean texture','mean perimeter'])


# In[ ]:


# count the target class
sns.countplot(cancer_df['target'])


# In[ ]:


# counterplot of feature mean radius
plt.figure(figsize=(20,8))
sns.countplot(cancer_df['mean radius'])


# In[ ]:


#heatmap


# In[1]:


#heatmap of Dataframe
plt.figure(figsize=(16,9))
sns.heatmap(cancer_df)


# In[ ]:


#heatmap of a correlation matrix
cancer_df.corr()


# In[ ]:


#heatmap of correlation matrix of breast cancer Dataframe
plt.figure(figsize=(20,20))
sns.heatmap(cancer_df.corr(), annot=True ,cmap='coolwarm', linewidth=2)


# In[ ]:


#correlation barplot


# In[ ]:


#create second dataframe by dropping target
cancer_df2 = cancer_df.drop(['target'], axis=1)
print("The shape of cancer_df2 is:", cancer_df2.shape)


# In[ ]:


#cancer_df2.corrwith(cancer_df.target)


# In[ ]:


#visualize correlation barplot
import matplotlib.pyplot as plt
plt.figure(figsize=(16,5))
ax= sns.barplot(cancer_df2.corrwith(cancer_df.target).index, cancer_df2.corrwith(cancer_df.target))
ax.tick_params(labelrotation=90)


# In[ ]:


cancer_df2.corrwith(cancer_df.target).index


# In[ ]:


#split dataframe in train and test


# In[ ]:


#input variable 
x = cancer_df.drop(['target'],axis=1)
x.head(6)


# In[ ]:


#output variable
y=cancer_df['target']
y.head(6)


# In[ ]:


#split dataset into train and test
from sklearn.model_selection import train_test_split
x_train, x_test,y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=5)


# In[ ]:


x_train


# In[ ]:


x_test


# In[ ]:


y_train


# In[ ]:


y_test


# In[ ]:


#feature scaling


# In[ ]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train_sc = sc.fit_transform(x_train)
x_test_sc = sc.transform(x_test)


# In[ ]:


from sklearn.metrics import confusion_matrix, classification_report, accuracy_score


# In[ ]:


# Support vector classifier
from sklearn.svm import SVC
svc_classifier = SVC()
svc_classifier.fit(X_train, y_train)
y_pred_scv = svc_classifier.predict(X_test)
accuracy_score(y_test, y_pred_scv)


# In[ ]:


# Train with Standard scaled Data
svc_classifier2 = SVC()
svc_classifier2.fit(X_train_sc, y_train)
y_pred_svc_sc = svc_classifier2.predict(X_test_sc)
accuracy_score(y_test, y_pred_svc_sc)


# In[ ]:


# Logistic Regression
from sklearn.linear_model import LogisticRegression
lr_classifier = LogisticRegression(random_state = 51, penalty = 'l1')
lr_classifier.fit(X_train, y_train)
y_pred_lr = lr_classifier.predict(X_test)
accuracy_score(y_test, y_pred_lr)


# In[ ]:


# Train with Standard scaled Data
lr_classifier2 = LogisticRegression(random_state = 51, penalty = 'l1')
lr_classifier2.fit(X_train_sc, y_train)
y_pred_lr_sc = lr_classifier.predict(X_test_sc)
accuracy_score(y_test, y_pred_lr_sc)


# In[ ]:


# K – Nearest Neighbor Classifier
from sklearn.neighbors import KNeighborsClassifier
knn_classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
knn_classifier.fit(X_train, y_train)
y_pred_knn = knn_classifier.predict(X_test)
accuracy_score(y_test, y_pred_knn)


# In[ ]:


# Train with Standard scaled Data
knn_classifier2 = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
knn_classifier2.fit(X_train_sc, y_train)
y_pred_knn_sc = knn_classifier.predict(X_test_sc)
accuracy_score(y_test, y_pred_knn_sc)


# In[2]:


# Naive Bayes Classifier
from sklearn.naive_bayes import GaussianNB
nb_classifier = GaussianNB()
nb_classifier.fit(X_train, y_train)
y_pred_nb = nb_classifier.predict(X_test)
accuracy_score(y_test, y_pred_nb)


# In[ ]:


# Train with Standard scaled Data
nb_classifier2 = GaussianNB()
nb_classifier2.fit(X_train_sc, y_train)
y_pred_nb_sc = nb_classifier2.predict(X_test_sc)
accuracy_score(y_test, y_pred_nb_sc)


# In[ ]:


# Decision Tree Classifier
from sklearn.tree import DecisionTreeClassifier
dt_classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 51)
dt_classifier.fit(X_train, y_train)
y_pred_dt = dt_classifier.predict(X_test)
accuracy_score(y_test, y_pred_dt)


# In[ ]:


# Train with Standard scaled Data
dt_classifier2 = DecisionTreeClassifier(criterion = 'entropy', random_state = 51)
dt_classifier2.fit(X_train_sc, y_train)
y_pred_dt_sc = dt_classifier.predict(X_test_sc)
accuracy_score(y_test, y_pred_dt_sc)


# In[ ]:


# Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier
rf_classifier = RandomForestClassifier(n_estimators = 20, criterion = 'entropy', random_state = 51)
rf_classifier.fit(X_train, y_train)
y_pred_rf = rf_classifier.predict(X_test)
accuracy_score(y_test, y_pred_rf)


# In[ ]:


# Train with Standard scaled Data
rf_classifier2 = RandomForestClassifier(n_estimators = 20, criterion = 'entropy', random_state = 51)
rf_classifier2.fit(X_train_sc, y_train)
y_pred_rf_sc = rf_classifier.predict(X_test_sc)
accuracy_score(y_test, y_pred_rf_sc)


# In[ ]:


# Adaboost Classifier
from sklearn.ensemble import AdaBoostClassifier
adb_classifier = AdaBoostClassifier(DecisionTreeClassifier(criterion = 'entropy', random_state = 200),
                                    n_estimators=2000,
                                    learning_rate=0.1,
                                    algorithm='SAMME.R',
                                    random_state=1,)
adb_classifier.fit(X_train, y_train)
y_pred_adb = adb_classifier.predict(X_test)
accuracy_score(y_test, y_pred_adb)


# In[ ]:


# Train with Standard scaled Data
adb_classifier2 = AdaBoostClassifier(DecisionTreeClassifier(criterion = 'entropy', random_state = 200),
                                    n_estimators=2000,
                                    learning_rate=0.1,
                                    algorithm='SAMME.R',
                                    random_state=1,)
adb_classifier2.fit(X_train_sc, y_train)
y_pred_adb_sc = adb_classifier2.predict(X_test_sc)
accuracy_score(y_test, y_pred_adb_sc)


# In[ ]:


# XGBoost Classifier
from xgboost import XGBClassifier
xgb_classifier = XGBClassifier()
xgb_classifier.fit(X_train, y_train)
y_pred_xgb = xgb_classifier.predict(X_test)
accuracy_score(y_test, y_pred_xgb)


# In[ ]:


# Train with Standard scaled Data
xgb_classifier2 = XGBClassifier()
xgb_classifier2.fit(X_train_sc, y_train)
y_pred_xgb_sc = xgb_classifier2.predict(X_test_sc)
accuracy_score(y_test, y_pred_xgb_sc)


# In[ ]:


# XGBoost classifier most required parameters
params={
 "learning_rate"    : [0.05, 0.10, 0.15, 0.20, 0.25, 0.30 ] ,
 "max_depth"        : [ 3, 4, 5, 6, 8, 10, 12, 15],
 "min_child_weight" : [ 1, 3, 5, 7 ],
 "gamma"            : [ 0.0, 0.1, 0.2 , 0.3, 0.4 ],
 "colsample_bytree" : [ 0.3, 0.4, 0.5 , 0.7 ] 
}


# In[ ]:


# Randomized Search
from sklearn.model_selection import RandomizedSearchCV
random_search = RandomizedSearchCV(xgb_classifier, param_distributions=params, scoring= 'roc_auc', n_jobs= -1, verbose= 3)
random_search.fit(X_train, y_train)


# In[ ]:


random_search.best_params_


# In[ ]:


random_search.best_estimator_


# In[ ]:


# training XGBoost classifier with best parameters
xgb_classifier_pt = XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
       colsample_bynode=1, colsample_bytree=0.4, gamma=0.2,
       learning_rate=0.1, max_delta_step=0, max_depth=15,
       min_child_weight=1, missing=None, n_estimators=100, n_jobs=1,
       nthread=None, objective='binary:logistic', random_state=0,
       reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
       silent=None, subsample=1, verbosity=1)
 
xgb_classifier_pt.fit(X_train, y_train)
y_pred_xgb_pt = xgb_classifier_pt.predict(X_test)


# In[ ]:


accuracy_score(y_test, y_pred_xgb_pt)


# In[ ]:


cm = confusion_matrix(y_test, y_pred_xgb_pt)
plt.title('Heatmap of Confusion Matrix', fontsize = 15)
sns.heatmap(cm, annot = True)
plt.show()


# In[ ]:


print(classification_report(y_test, y_pred_xgb_pt))


# In[ ]:


# Cross validation
from sklearn.model_selection import cross_val_score
cross_validation = cross_val_score(estimator = xgb_model_pt2, X = X_train_sc, y = y_train, cv = 10)
print("Cross validation of XGBoost model = ",cross_validation)
print("Cross validation of XGBoost model (in mean) = ",cross_validation.mean())
from sklearn.model_selection import cross_val_score
cross_validation = cross_val_score(estimator = xgb_classifier_pt, X = X_train_sc,y = y_train, cv = 10)
print("Cross validation accuracy of XGBoost model = ", cross_validation)
print("\nCross validation mean accuracy of XGBoost model = ", cross_validation.mean())


# In[ ]:


## Pickle
import pickle
 
# save model
pickle.dump(xgb_classifier_pt, open('breast_cancer_detector.pickle', 'wb'))
 
# load model
breast_cancer_detector_model = pickle.load(open('breast_cancer_detector.pickle', 'rb'))
 
# predict the output
y_pred = breast_cancer_detector_model.predict(X_test)
 
# confusion matrix
print('Confusion matrix of XGBoost model: \n',confusion_matrix(y_test, y_pred),'\n')
 
# show the accuracy
print('Accuracy of XGBoost model = ',accuracy_score(y_test, y_pred))


# In[ ]:





# In[ ]:





# In[ ]:




