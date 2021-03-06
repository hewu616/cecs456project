import pandas as pd
import numpy as np
import time

# load csv file
file = ".../add.csv"
ad = pd.read_csv(file,low_memory=False, header=0)

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

# drop rows where local contains missing values
ad = ad.loc[pd.to_numeric(ad.loc[:,'local'], errors='coerce').notnull()]

# replace all '?' with NaN
ad = ad.applymap(lambda val: np.nan if str(val).strip() == '?' else val)
dims = ['height', 'width']
ad[dims] = ad[dims].fillna(ad[dims].median())

# calculate missing aratio values
import statistics
ad[['height', 'width']] = ad[['height', 'width']].astype(int)
ar = ad['width'] / ad['height']
rar = round(ar,4)
ad['aratio'] = ad['aratio'].fillna(rar)
ad[['aratio']] = ad[['aratio']].astype(float)

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
print("Significant binary variables at 0.05:",len(col_names),)  # 600 binary variables are significant at 0.05

# new subset of data containing the continuous variables
# and the significant binary variables
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

# Random Forest
start_timeRF=time.time()
from sklearn.ensemble import RandomForestClassifier
RF = RandomForestClassifier(n_estimators=100) # create 100 decision trees
RF.fit(X_train_scaled,Y_train)  # fit the model with Random Forests
end_timeRF=time.time()
tot_timeRF=end_timeRF-start_timeRF

# feature importance
feature_importance = pd.DataFrame(RF.feature_importances_, index =X_train_scaled.columns,
                columns=['importance']).sort_values('importance', ascending=False)
print("")
print("Top 10 important features:")
print(feature_importance.head(10))
# output
"""
Top 10 important features:
           importance
width        0.098135
ancurl288    0.059456
aratio       0.054134
url348       0.046260
ancurl444    0.039142
height       0.033793
ancurl274    0.028151
alt56        0.023112
alt28        0.021707
alt8         0.018218
"""

# Calculate the accuracy of the test set
from sklearn import metrics
predictionsRF = RF.predict(X_test_scaled)
print("")
print("Random Forest Runtime:", round(tot_timeRF,3))
print("")
print("Random Forest Accuracy:", round(metrics.accuracy_score(Y_test, predictionsRF),3))
print("")
print("Random Forest Confusion Matrix:")
print(metrics.confusion_matrix(Y_test, predictionsRF))
print("")
print("Random Forest Results:")
print(metrics.classification_report(Y_test, predictionsRF))
"""
Random Forest Runtime: 1.224

Random Forest Accuracy: 0.98

Random Forest Confusion Matrix:
[[699   3]
 [ 13 101]]

Random Forest Results:
              precision    recall  f1-score   support

           0       0.98      1.00      0.99       702
           1       0.97      0.89      0.93       114

   micro avg       0.98      0.98      0.98       816
   macro avg       0.98      0.94      0.96       816
weighted avg       0.98      0.98      0.98       816
"""

# Naive Bayes
start_timeNB=time.time()

# Naive Bayes (Gaussian)
from sklearn.naive_bayes import GaussianNB
GNB = GaussianNB()
GNB.fit(X_train_cont_scaled,Y_train)

# Naive Bayes (Bernoulli)
from sklearn.naive_bayes import BernoulliNB
BNB = BernoulliNB()
BNB.fit(X_train_scaled.iloc[:,3:],Y_train)

end_timeNB=time.time()
tot_timeNB=end_timeNB-start_timeNB

Gthetas = GNB.theta_
Gsigma = GNB.sigma_
GNBpost = GNB.predict_proba(X_test_cont_scaled)
classprior = GNB.class_prior_

Bflp =BNB.feature_log_prob_
BNBpost = BNB.predict_proba(X_test_scaled.iloc[:,3:])

# Classify
NB = (GNBpost*BNBpost)/classprior
predictionsNB = np.zeros(len(NB))
for i in range(len(NB)):
    if NB[:,0][i] > NB[:,1][i]:
        predictionsNB[i] = 0
    else:
        predictionsNB[i] = 1
print("")
print("Naive Bayes Runtime:", round(tot_timeNB,3))
print("")
print("Naive Bayes Accuracy:", round(metrics.accuracy_score(Y_test, predictionsNB),3))
print("")
print("Naive Bayes Confusion Matrix:")
print(metrics.confusion_matrix(Y_test, predictionsNB))
print("")
print("Naive Bayes Results:")
print(metrics.classification_report(Y_test, predictionsNB))
"""
Naive Bayes Runtime: 0.087

Naive Bayes Accuracy: 0.967

Naive Bayes Confusion Matrix:
[[700   2]
 [ 25  89]]

Naive Bayes Results:
              precision    recall  f1-score   support

           0       0.97      1.00      0.98       702
           1       0.98      0.78      0.87       114

   micro avg       0.97      0.97      0.97       816
   macro avg       0.97      0.89      0.92       816
weighted avg       0.97      0.97      0.97       816
"""
