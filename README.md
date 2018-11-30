# cecs456project
Internet Advertisements Data Set comes from UCI Machine Learning Repository 
(Dua, D. and Karra Taniskidou, E. (2017). UCI Machine Learning Repository [http://archive.ics.uci.edu/ml]. Irvine, CA: University of California, School of Information and Computer Science.)

Dataset contains 3279 instances with a label of either 'ad' or 'nonad'

Features include 3 continuous variables, and 1555 binary variables.

28% of the data are missing.

Task is to classify whether an image is an advertisement or not.

Random Forests and Naive Bayes are used here.

Two approaches are used to account for the missing data.

1. Observations of missing values are removed from the dataset - See 'Missing Values Removed.py'

2. Observations of missing values are imputed using the median - See 'MissingValuesImputed.py'
