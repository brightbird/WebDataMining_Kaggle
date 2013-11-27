# 11.27 

import os
import csv
import nltk
import numpy as np
from sklearn.feature_extraction.text import *
from sklearn.feature_selection import SelectKBest, SelectPercentile, chi2, f_classif
from sklearn.linear_model import *
from sklearn.naive_bayes import *
from sklearn import svm
from scipy.sparse import * 
from scipy.io import *
from nltk.tokenize import *
from nltk.stem.snowball import *
from stemming.porter2 import stem

def cleaned_text(text):
	return text

CORPUS_SIZE_ITEMS = ['entire', 'small']
CORPUS_SIZE = 0 		# 0 for entire, 1 for small 
VECTORIZER = 1 			# 0 for CountVectorizer, 1 for TfidfVectorizer
K_FOR_BEST = 2000
SELECT_PERCENTILE = 30
SELECTOR = 1 			# 0 for K-select, 1 for precentile-select

cur_dir = os.getcwd()
if (CORPUS_SIZE == 1):
	test_csv = file(cur_dir + "/../data/small_test.csv")
	train_csv = file(cur_dir + "/../data/small_train.csv")
else:
	test_csv = file(cur_dir + "/../data/test.csv")
	train_csv = file(cur_dir + "/../data/train.csv")

#################################
#  	  Get Corpus From CSV 	    #
#################################

train_corpus = []
test_corpus = []

# get train_tweets from csv to train_corpus[ ]
cnt = 0
train_reader = csv.reader(train_csv)
for tweet in train_reader:
	text = unicode(tweet[1] + " " + tweet[2] + " " + tweet[3], 'ascii', 'ignore')
	# text = cleaned_text(text)
	train_corpus.append(text)
	cnt += 1

# delete header
del train_corpus[0]
train_csv.close()

# get test_tweets from csv to test_corpus[]
cnt = 0
test_reader = csv.reader(test_csv)
for tweet in test_reader:
	text = unicode(tweet[1] + " " + tweet[2] + " " + tweet[3], 'ascii', 'ignore')
	# text = cleaned_text(text)
	test_corpus.append(text)
	cnt += 1

# delete header
del test_corpus[0]
test_csv.close()

#################################
# 		Feature Exraction 		#
#################################

# get x_train, x_test from train_corpus, test_corpus
print "start extraction"
entire_corpus = train_corpus + test_corpus
if (VECTORIZER == 0):
	vectorizer = CountVectorizer(min_df = 1, tokenizer = nltk.word_tokenize)
elif (VECTORIZER == 1):
	vectorizer = TfidfVectorizer(max_features=10000, strip_accents='unicode', analyzer='word', tokenizer = nltk.word_tokenize)

vectorizer.fit(train_corpus)
x_train = vectorizer.transform(train_corpus)
x_test = vectorizer.transform(test_corpus)
print "finish extraction"

#################################
# 		Feature Selection 		#
#		  and Regression 		#
#	for three groups of attrs 	#
#################################

train_len = len(train_corpus)

#################################
## 	  attributes group loop    ##

for ATTRIBUTES_GROUP in xrange(0, 3):

	print "GROUP -", ATTRIBUTES_GROUP

	if (CORPUS_SIZE == 1):
		train_csv = file(cur_dir + "/../data/small_train.csv")
	else:
		train_csv = file(cur_dir + "/../data/train.csv")
	attrs_arr = []
	time_attrs = []
	weather_attrs = []
	for i in xrange(0, train_len + 1):
		attrs_arr.append([])

	# get attitude attributes from csv
	if (ATTRIBUTES_GROUP == 0):
		index_from, index_to = 4, 9
	if (ATTRIBUTES_GROUP == 1):
		index_from, index_to = 9, 13
	if (ATTRIBUTES_GROUP == 2):
		index_from, index_to = 13, 28
	train_reader = csv.reader(train_csv)
	cnt = 0
	for tweet in train_reader:
		attr = tweet[index_from:index_to]
		attrs_arr[cnt] = attr
		cnt += 1
	train_csv.close()
	del attrs_arr[0]

	# get y_train from train_attrs
	y_train = [[float(attr) for attr in attrs] for attrs in attrs_arr]
	# chi-2 select features
	print "start feature selection"
	if (SELECTOR == 0):
		selector = SelectKBest(chi2, k = K_FOR_BEST)
	else:
		selector = SelectPercentile(score_func=chi2, percentile=SELECT_PERCENTILE)
	selector.fit(x_train, y_train)
	new_x_train = selector.transform(x_train)
	new_x_test = selector.transform(x_test)
	print "feature selection done"

	# regression
	print "start regression"
	clf = LinearRegression()
	clf = clf.fit(new_x_train, y_train)
	result = clf.predict(new_x_test)
	print "regression done"

	# build csv file
	if (ATTRIBUTES_GROUP == 0):
		result_path = cur_dir + "/../data/result/attitude_res.csv"
	if (ATTRIBUTES_GROUP == 1):
		result_path = cur_dir + "/../data/result/time_res.csv"
	if (ATTRIBUTES_GROUP == 2):
		result_path = cur_dir + "/../data/result/weather_res.csv"
	if os.path.exists(result_path):
		os.remove(result_path)
	result_csv = file(result_path, 'a')
	result_writer = csv.writer(result_csv)

	# output result to csv file
	print "start writing result"
	for item in result:
		result_writer.writerow(item)
	print "writing result done"

	result_csv.close()

## 	       loop over	       ##
#################################
