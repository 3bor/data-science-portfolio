# Run with Python2.7

#
# Predicting the arXiv class 'hep-ph' or 'hep-th' of a paper
# depending on its abstract
#

import random
import glob
import re
import numpy as np
from datetime import datetime
from xml.etree import ElementTree
from sklearn.model_selection import train_test_split
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.svm import SVC


from utilities import *  
# utilities contains custom helper functions:
# get_files(folder) - get all files with .txt extension from a folder
# get_records(file) - get the records from an input file in xml format
# get_label(record) - get the label of each record 
# get_abstract(record) - get abstract of each record
# parse_text(text) - Parse text by stemming words
# print_word_freq() - Print words and their frequencies
def myprint(*args):
    print(str(datetime.now())+' | '+''.join(map(str,args)))
    

### Load paper metadata
# 2012-2012:  6337 records
# 2012-2013: 16772 records
# 2012-2014: 34174 records
# 2012-2015: 69293 records
# 2012-2016: ///// records
# 2012-2017: ///// records
# 2012-2018: ///// records
records = []
for year in range(2012,2015):
	myprint('Loading hep-ph abstracts from '+str(year)) # first category
	folder = 'hidden/datasets/hepph-'+str(year)+'/'
	files = get_files(folder) # labeled data first category
	for file in files:
		records = records + get_records(file)
	myprint('Loading hep-th abstracts from '+str(year)) # second category
	folder = 'hidden/datasets/hepth-'+str(year)+'/'
	files = get_files(folder) 
	for file in files:
		records = records + get_records(file)

myprint('> '+str(len(records))+' records in total')


### Extract labels (hep-th / hep-ph) and features (abstracts)
labels = [get_label(rec) for rec in records]
features = [get_abstract(rec) for rec in records]
myprint('Extracted labels and features')

### Parse the abstracts by extracting all stemmed words
stemmer = SnowballStemmer("english")
features = [parse_text(text, stemmer) for text in features]
myprint('Stemmed abstracts')

### Split into train and test set
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size = 4000, random_state = 42)
myprint('Split into train and test set')

### TfIdf vectorization of features
vectorizer = TfidfVectorizer(stop_words = 'english')
trans_features_train = vectorizer.fit_transform(features_train)
trans_features_test = vectorizer.transform(features_test)
myprint('Vectorized features')
myprint('> '+str(trans_features_train.shape[0])+' abstracts in train set')
myprint('> '+str(trans_features_train.shape[1])+' words per abstract in train set')
myprint('> '+str(trans_features_test.shape[0])+' abstracts in test set')
myprint('> '+str(trans_features_test.shape[1])+' words per abstract in test set')


### Reduce feature dimensionality
# Set a sensible upper bound of the number of words that is
# required to capture the difference between the two categories
feature_dim = 1000
selector = SelectKBest(chi2, k = feature_dim)
selector.fit(trans_features_train, labels_train)
trans_features_train = selector.transform(trans_features_train).toarray()
trans_features_test = selector.transform(trans_features_test).toarray()
myprint('Reduced dimensionality')
myprint('> '+str(trans_features_train.shape[1])+' words per abstract')


### Classification with support vector machine 
# Pro: effective in high-dimensional feature spaces (i.e. large dictionary)
#
# Idea:
# Given training vectors x_i in R^p for i=1,...,n in two classes 
# and a vector y in {-1,1}^n, the goal of SVM is to find w in R^p 
# and b in R, such that the prediction sign(w^T.phi(x)+b) 
# is correct for most samples
#
# Problem:
# min_{w,b,z} 1/2 w^T w + C sum_i z_i
# subject to y_i(w^T.phi(x_i)+b) >= 1-z_i with z_i >= 0 for all i
# 
# Default settings SVC():
# C = 1.0
# kernel = 'rbf'  -> exp(-gamma||x-x'||^2)
# gamma = 'scale' -> 1 / (n_features * X.var())


###
### EXPERIMENTS
###

### Run 1: Vary training set size m
# print('|     m | train |  test |')
# for m in [100,500,1000,5000,10000,15000]:
# 	subset_features_train = trans_features_train[:m]
# 	subset_labels_train = labels_train[:m]
# 	clf = SVC()
# 	clf.fit(subset_features_train, subset_labels_train)
# 	train_score = clf.score(subset_features_train, subset_labels_train)
# 	test_score = clf.score(trans_features_test, labels_test)
# 	print('| %5d | %0.3f | %0.3f |' % (m, train_score, test_score))
#
# Training curve:
# |     m | train |  test |
# |   100 | 0.530 | 0.442 |
# |   500 | 0.524 | 0.558 |
# |  1000 | 0.537 | 0.558 |
# |  5000 | 0.547 | 0.558 |
# | 10000 | 0.549 | 0.558 |
# | 15000 | 0.548 | 0.558 |
#
# Best value: 0.558
# Interpret: 	train ~ test ~ constant => Indicates high bias
# Advice: 		Reduce regularization 	=> Increase C  (beyond default 1.0)


### Run 2: Vary regularization parameter C
# print('|     m |     C | train |  test |')
# for m in [100,500,1000,5000,10000,15000]:
# 	for C in [1,10,100,1000,10000]:
# 		subset_features_train = trans_features_train[:m]
# 		subset_labels_train = labels_train[:m]
# 		clf = SVC(C = C)
# 		clf.fit(subset_features_train, subset_labels_train)
# 		train_score = clf.score(subset_features_train, subset_labels_train)
# 		test_score = clf.score(trans_features_test, labels_test)
# 		print('| %5d | %5d | %0.3f | %0.3f |' % (m, C, train_score, test_score))
# 
# Training curve:
# |     m |     C | train |  test |
# |   100 |     1 | 0.530 | 0.442 | \
# |   100 |    10 | 0.530 | 0.442 |  \
# |   100 |   100 | 0.530 | 0.442 |   } Best C = 1000
# |   100 |  1000 | 0.970 | 0.793*|  /
# |   100 | 10000 | 1.000 | 0.774 | /
# |   500 |     1 | 0.524 | 0.558 | \
# |   500 |    10 | 0.524 | 0.558 |  \
# |   500 |   100 | 0.882 | 0.857*|   } Best C = 100
# |   500 |  1000 | 0.944 | 0.852 |  /
# |   500 | 10000 | 0.992 | 0.808 | /
# |  1000 |     1 | 0.537 | 0.558 | \
# |  1000 |    10 | 0.537 | 0.558 |  \
# |  1000 |   100 | 0.872 | 0.860*|   } Best C = 100
# |  1000 |  1000 | 0.928 | 0.841 |  /
# |  1000 | 10000 | 0.987 | 0.791 | /
# |  5000 |     1 | 0.547 | 0.558 | \
# |  5000 |    10 | 0.815 | 0.804 |  \
# |  5000 |   100 | 0.863 | 0.863*|   } Best C = 100-1000
# |  5000 |  1000 | 0.882 | 0.862*|  /
# |  5000 | 10000 | 0.902 | 0.834 | /
# | 10000 |     1 | 0.549 | 0.558 | \
# | 10000 |    10 | 0.856 | 0.851 |  \
# | 10000 |   100 | 0.865 | 0.860*|   } Best C = 100-1000
# | 10000 |  1000 | 0.873 | 0.860*|  /
# | 10000 | 10000 | 0.880 | 0.852 | /
# | 15000 |     1 | 0.548 | 0.558 | \
# | 15000 |    10 | 0.858 | 0.856 |  \
# | 15000 |   100 | 0.865 | 0.861*|   } Best C = 100
# | 15000 |  1000 | 0.871 | 0.857 |  /
# | 15000 | 10000 | 0.875 | 0.852 | /
#
# Best value: 0.863
# Interpret: 	Small C => Bias, Large C => Variance
#             Set C to value that maximises Test Accuracy
# Advice: 		Vary C around 100 and train with more data


### Run3: Narrow range of values C (base 2), additional training data
# m_train = trans_features_train.shape[0]
# m_test = trans_features_test.shape[0]
# print('train size = %d \ntest size = %d' % (m_train, m_test))
# print('|     C | train |  test |')
# for C in [20,50,100,200,500,1000]:
# 	clf = SVC(C = C)
# 	clf.fit(trans_features_train, labels_train)
# 	train_score = clf.score(trans_features_train, labels_train)
# 	test_score = clf.score(trans_features_test, labels_test)
# 	print('| %5d | %0.3f | %0.3f |' % (C, train_score, test_score))
# 
# train size = 30174 
# test size = 4000
# Training curve:
# |     C | train |  test |
# |    20 | 0.858 | 0.863 |
# |    50 | 0.860 | 0.865 |
# |   100 | 0.862 | 0.866*|
# |   200 | 0.864 | 0.866*|
# |   500 | 0.866 | 0.865 |
# |  1000 | 0.867 | 0.865 |


### Best performance on test set: 86 % classification accuracy

clf = SVC(C = 200)
clf.fit(trans_features_train, labels_train)
train_score = clf.score(trans_features_train, labels_train)
test_score = clf.score(trans_features_test, labels_test)
print('| %0.3f | %0.3f |' % (train_score, test_score))



### Suggested improvements:
# 1. Broader search grid to vary additional SVC parameters
#    most notably 'kernel' and 'gamma' (or 'degree' in case of 'poly')
#    use GridSearchCV()
# 
# 2. Vary the feature dimensionality as well in SelectKBest()
#    use Pipeline()
# 

exit()
