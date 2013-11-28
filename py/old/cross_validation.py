# 11.27 ver2

import os
import csv
import math
import nltk
import numpy as np
from sklearn.feature_extraction.text import *
from sklearn.feature_selection import SelectKBest, SelectPercentile, chi2, f_classif
from sklearn.linear_model import *
from sklearn.naive_bayes import *
from sklearn import svm, cross_validation
from scipy.sparse import * 
from scipy.io import *
from nltk.tokenize import *
from nltk.stem.snowball import *
from stemming.porter2 import stem
from lmfit import *

def cleaned_text(text):
	text = nltk.word_tokenize(text)
	text = " ".join([stem(x.lower()) for x in text])
	return text

CORPUS_SIZE_ITEMS = ['entire', 'small']
CORPUS_SIZE = 0 		# 0 for entire, 1 for small 
PREDICT_ATTRIBUTE_NUM = 24
VECTORIZER = 1 			# 0 for CountVectorizer, 1 for TfidfVectorizer
K_FOR_BEST = 2000
SELECT_PERCENTILE = 30
SELECTOR = 1 			# 0 for K-select, 1 for precentile-select

cur_dir = os.getcwd()
if (CORPUS_SIZE == 1):
	train_csv = file(cur_dir + "/../data/small_train.csv")
else:
	train_csv = file(cur_dir + "/../data/train.csv")

#################################
#  	  Get Corpus From CSV 	    #
#################################

train_corpus = []

# get train_tweets from csv to train_corpus[]
cnt = 0
train_reader = csv.reader(train_csv)
for tweet in train_reader:
	text = unicode(tweet[1] + " " + tweet[2] + " " + tweet[3], 'ascii', 'ignore')
	# text = cleaned_text(text)
	train_corpus.append(text)
	# print "train tweet", cnt, "to corpus[]"
	cnt += 1

# delete header
del train_corpus[0]
train_csv.close()

#################################
# 		Feature Exraction 		#
#################################

# get x_train, x_test from train_corpus, test_corpus
print "start extraction"
if (VECTORIZER == 0):
	vectorizer = CountVectorizer(min_df = 1, tokenizer = nltk.word_tokenize)
elif (VECTORIZER == 1):
	vectorizer = TfidfVectorizer(min_df = 1, tokenizer = nltk.word_tokenize)
	# vectorizer = TfidfVectorizer(ngram_range=(1, 5), analyzer="word", binary=False, min_df=3)

vectorizer.fit(train_corpus)
x_train = vectorizer.transform(train_corpus)
print "finish extraction"

#################################
# 		Feature Selection 		#
#################################

# # get feature names
attribute_names = ["ATTR:I can not tell attitude"
,"ATTR:Negative"
,"ATTR:Neutral / author is just sharing information"
,"ATTR:Positive"
,"ATTR:Tweet not related to weather condition"  
,"ATTR:current (same day) weather"
,"ATTR:future (forecast)"
,"ATTR:I can not tell time"
,"ATTR:past weather"
,"ATTR:clouds"
,"ATTR:cold"
,"ATTR:dry"
,"ATTR:hot"
,"ATTR:humid"
,"ATTR:hurricane"
,"ATTR:I can not tell weather"
,"ATTR:ice"
,"ATTR:other"
,"ATTR:rain"
,"ATTR:snow"
,"ATTR:storms"
,"ATTR:sun"
,"ATTR:tornado"
,"ATTR:wind"
]
for CURRENT_ATTRIBUTE in xrange(0, PREDICT_ATTRIBUTE_NUM):

	if (CORPUS_SIZE == 1):
		train_csv = file(cur_dir + "/../data/small_train.csv")
	else:
		train_csv = file(cur_dir + "/../data/train.csv")

	# get CURRENT ATTRIBUTE train_attrs from csv
	train_attrs = []
	train_reader = csv.reader(train_csv)
	cnt = 0
	for tweet in train_reader:
		attr = tweet[CURRENT_ATTRIBUTE + 4]
		train_attrs.append(attr)
		cnt += 1

	del train_attrs[0]

	# get y_train from train_attrs
	y_train = [[float(attr)] for attr in train_attrs]

	# chi-2 select features
	print "start feature selection"
	if (SELECTOR == 0):
		selector = SelectKBest(chi2, k = K_FOR_BEST)
	else:
		selector = SelectPercentile(score_func=chi2, percentile=SELECT_PERCENTILE)
	selector.fit(x_train, y_train)
	new_x_train = selector.transform(x_train)
	print "feature selection done"

	# convert y_train to svm-fit shape
	y_train = [attr[0] for attr in y_train]

	new_x_train, new_x_test, new_y_train, new_y_test = cross_validation.train_test_split(new_x_train, y_train, test_size=0.4, random_state=0)

	# regression
	# clf = svm.SVR(kernel='rbf', degree=3, gamma=1.9, coef0=0.0, tol=0.001, \
	# 	C=0.13, epsilon=0.1, shrinking=True, probability=False, cache_size=700, \
	# 	verbose=False, max_iter=-1, random_state=None)
	clf = LinearRegression()
	clf = clf.fit(new_x_train, new_y_train)

	# cross validation
	score = clf.score(new_x_test, new_y_test)
	print "score :", score

	# my validation
	predict_y_test = clf.predict(new_x_test)
	RMSE = 0
	n = len(new_y_test)
	for i in xrange(0, n):
		RMSE += (new_y_test[i] - predict_y_test[i]) ** 2
	RMSE /= n
	RMSE = math.sqrt(RMSE)
	print "RMSE :", RMSE

	train_csv.close()
