import numpy as np
import pandas as pd

# read the csv file with header in the 0th row
ad = pd.read_csv('.../add.csv',low_memory=False, header=0)
# drop the first column
ad = ad.iloc[:,1:1560]

# rename the headers
cont = ['height', 'width', 'aratio', 'local']
url = ["url"+str(i+1) for i in range(457)]
origurl = ["origurl"+str(i+1) for i in range(495)]
ancurl = ["ancurl"+str(i+1) for i in range(472)]
alt = ["alt"+str(i+1) for i in range(111)]
caption = ["caption"+str(i+1) for i in range(19)]
new_name = cont + url + origurl + ancurl + alt + caption + ['label']
ad.columns = new_name

# describe the continuous variables information
ad['height'].describe()
ad['width'].describe()
ad.aratio.describe()

# drop rows where height and width contain missing values (non-numeric values)
ad = ad.loc[pd.to_numeric(ad.loc[:,'local'], errors='coerce').notnull()]
ad = ad.loc[pd.to_numeric(ad.loc[:,'height'], errors='coerce').notnull()]
ad = ad.loc[pd.to_numeric(ad.loc[:,'width'], errors='coerce').notnull()]
ad.shape    # 2359 x 1559

# convert the labels into numeric values
ad.label = ad.label.map({'ad.':1, 'nonad.':0})

# separate the features and the labels
Y = ad.label
X = ad.loc[:, 'height':'caption19']

# variable screening using chi-square test for categorical variables
from scipy import stats
col_names = []
for i in range(1555):
    cont_table = pd.crosstab(Y, X.iloc[:,i+3], margins=True)
    f_obs = np.array([cont_table.iloc[0][0:2].values, cont_table.iloc[1][0:2].values])
    pvalue = stats.chi2_contingency(f_obs, correction=False)[1]
    if pvalue <= 0.05:
        col_names = np.append(col_names, X.columns[i+3])
len(col_names)  # 581 binary variables are significant at 0.05

# new subset of data containing the continuous variables and the significant binary variables
var_names = ['height', 'width', 'aratio']
var_names.extend(col_names)
X_new = X.loc[:,var_names]

# split the data into test set and training set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X_new, Y)

# change the continuous variables to data type float
X_train_cont = X_train.iloc[:,0:3].astype(float)
X_test_cont = X_test.iloc[:,0:3].astype(float)

# standardize/normalize the continuous variables
from sklearn.preprocessing import StandardScaler
scaled = StandardScaler()
X_train_cont_scaled = pd.DataFrame(scaled.fit_transform(X_train_cont),
                columns=X_train_cont.columns, index = X_train_cont.index)
X_test_cont_scaled = pd.DataFrame(scaled.fit_transform(X_test_cont),
                columns=X_test_cont.columns, index = X_test_cont.index)
X_train_scaled = pd.concat([X_train_cont_scaled,X_train.iloc[:,3:]], axis=1)
X_test_scaled = pd.concat([X_test_cont_scaled,X_test.iloc[:,3:]], axis=1)

# Random Forests
from sklearn.ensemble import RandomForestClassifier
import time
start_timeRF = time.time()
RF = RandomForestClassifier(n_estimators=100) # create 100 decision trees
RF.fit(X_train_scaled,Y_train)  # fit the model with Random Forests
end_timeRF = time.time()
print("RF training time: ", end_timeRF-start_timeRF)

# feature importance
feature_importance = pd.DataFrame(RF.feature_importances_, index =X_train_scaled.columns,
                                  columns=['importance']).sort_values('importance', ascending=False)
print("Top 10 important features:")
print(feature_importance.head(10))

'''
Top 10 important features:
           importance
width        0.126815
aratio       0.064006
ancurl288    0.058157
height       0.043336
url348       0.037996
ancurl444    0.033018
alt56        0.028808
ancurl188    0.019383
alt28        0.018657
alt8         0.018525
'''

# Calculate the accuracy of the test set
from sklearn import metrics
predictions = RF.predict(X_test_scaled)
print("RF Accuracy:", metrics.accuracy_score(Y_test, predictions))  # 0.9763
print("RF Confusion Matrix")
print(metrics.confusion_matrix(Y_test, predictions))
print("RF Results")
print(metrics.classification_report(Y_test, predictions))
'''
RF Confusion Matrix
[[496   4]
 [ 10  80]]
 
 RF Results
              precision    recall  f1-score   support

           0       0.98      0.99      0.99       500
           1       0.95      0.89      0.92        90

   micro avg       0.98      0.98      0.98       590
   macro avg       0.97      0.94      0.95       590
weighted avg       0.98      0.98      0.98       590
'''

# Naive Bayes
from sklearn.naive_bayes import GaussianNB, BernoulliNB
X_train_dis = X_train.iloc[:,3:]
start_timeNB = time.time()
GNB = GaussianNB()
GNB.fit(X_train_cont_scaled, Y_train)
BNB = BernoulliNB()
BNB.fit(X_train_dis,Y_train)
end_timeNB = time.time()
print("NB training time: ", end_timeNB-start_timeNB)

Gthetas = GNB.theta_
Gsigma = GNB.sigma_
Bfeaturelogprob =BNB.feature_log_prob_
classprior = GNB.class_prior_

# return the posterior probabilities for binary(BernoulliNB) and continuous cases(GaussianNB)
GNBpost = GNB.predict_proba(X_test_cont_scaled)
BNBpost = BNB.predict_proba(X_test_scaled.iloc[:,3:])

# classify
num = (GNBpost*BNBpost)/classprior
predictions1 = np.zeros(len(num))
for i in range(len(num)):
    if num[:,0][i] > num[:,1][i]:
        predictions1[i] = 0
    else:
        predictions1[i] = 1
print("NB Accuracy:", metrics.accuracy_score(Y_test, predictions1))
print("NB Confusion Matrix")
print(metrics.confusion_matrix(Y_test, predictions1))
print("NB Results")
print(metrics.classification_report(Y_test, predictions1))

'''
NB Confusion Matrix
[[498   2]
 [ 16  74]]
NB Results
              precision    recall  f1-score   support

           0       0.97      1.00      0.98       500
           1       0.97      0.82      0.89        90

   micro avg       0.97      0.97      0.97       590
   macro avg       0.97      0.91      0.94       590
weighted avg       0.97      0.97      0.97       590
'''