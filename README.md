# CECS456 Project

CECS456 Project is a machine learning course project that involves classification of whether an image is an advertisement or not given geometry of the images, texts in the URL, image's URL, alt text, the anchor text, and text near the anchor text. (Dua, D. and Karra Taniskidou, E. (2017). UCI Machine Learning Repository [http://archive.ics.uci.edu/ml]. Irvine, CA: University of California, School of Information and Computer Science.)

## Getting Started

- Download the Internet Advertisement Data set 'add.csv' from [Kaggle by UCI Machine Learning](https://www.kaggle.com/uciml/internet-advertisements-data-set)
- Ensure Python is installed on your computer.
- run [MissingValuesImputed.py](./MissingValuesImputed.py)
- run [MissingValuesRemoved.py](./MissingValuesRemoved.py)

## Directory

[MissingValuesImputed.py](./MissingValuesImputed.py) replaces all observations containing missing values by the median of the existing values of a particular feature. Two models are built using Random Forest and Naive Bayes to classify whether an image is an advertisement or not. The accuracy, run-time, confusion matrix, and classification report of each model are returned.

[MissingValuesRemoved.py](./MissingValuesRemoved.py) removes all observations containing missing values. Two models are built using Random Forest and Naive Bayes to classify whether an image is an advertisement or not. The accuracy, run-time, confusion matrix, and classification report of each model are returned.
