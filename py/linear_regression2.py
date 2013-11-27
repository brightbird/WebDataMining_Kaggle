# 11.27 ver 2 - 0.18270

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

# get test_tweets from csv to test_corpus[]
cnt = 0
test_reader = csv.reader(test_csv)
for tweet in test_reader:
	text = unicode(tweet[1] + " " + tweet[2] + " " + tweet[3], 'ascii', 'ignore')
	# text = cleaned_text(text)
	test_corpus.append(text)
	# print "test tweet", cnt, "to corpus[]"
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
	vectorizer = TfidfVectorizer(min_df = 1, tokenizer = nltk.word_tokenize)
	# vectorizer = TfidfVectorizer(ngram_range=(1, 5), analyzer="word", binary=False, min_df=3)

vectorizer.fit(train_corpus)
x_train = vectorizer.transform(train_corpus)
x_test = vectorizer.transform(test_corpus)
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

	print "CURRENT_ATTRIBUTE :", attribute_names[CURRENT_ATTRIBUTE]

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
	new_x_test = selector.transform(x_test)
	print "feature selection done"

	# convert y_train to svm-fit shape
	y_train = [attr[0] for attr in y_train]

	# linear regression
	print "start regression"
	clf = LinearRegression()
	clf = clf.fit(new_x_train, y_train)
	result = clf.predict(new_x_test)
	print "regression done"

	for item in result:
		if (item > 0):
			print item

	# build csv file
	result_path = cur_dir + "/../data/result/res_" + str(CURRENT_ATTRIBUTE) + ".csv"
	if os.path.exists(result_path):
		os.remove(result_path)
	result_csv = file(result_path, 'a')
	result_writer = csv.writer(result_csv)

	# output result to csv file
	print "start writing result"
	for item in result:
		result_writer.writerow([item])
	print "writing result done"

	result_csv.close()
