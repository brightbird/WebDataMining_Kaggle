# 11.21 ver 1

import os
import csv
import nltk
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2, f_classif
from scipy.sparse import * 
from nltk.tokenize import *
from nltk.stem.snowball import *
from stemming.porter2 import stem

def cleaned_text(text):
	print text
	text = nltk.word_tokenize(text)
	text = " ".join([stem(x.lower()) for x in text])
	print text
	return text

cur_dir = os.getcwd()
test_csv = file(cur_dir + "/../data/small_test.csv")
train_csv = file(cur_dir + "/../data/small_train.csv")
# test_csv = file(cur_dir + "/../data/test.csv")
# train_csv = file(cur_dir + "/../data/train.csv")
# dst_train_csv = file(cur_dir + "/../data/result/train_result.csv", "a")
# dst_test_csv = file(cur_dir + "/../data/result/test_result.csv", "a")

train_corpus = []
train_attrs = []
test_corpus = []

# get train_tweets, their attributes from csv to train_corpus[], train_attrs[]
cnt = 0
train_reader = csv.reader(train_csv)
for tweet in train_reader:
	# tweet content + state + city
	text = unicode(tweet[1] + " " + tweet[2] + " " + tweet[3], 'ascii', 'ignore')
	text = cleaned_text(text)
	attr = []
	for i in xrange(4, 5):
		attr.append(tweet[i])
	train_corpus.append(text)
	train_attrs.append(attr)
	print "train tweet", cnt, "to corpus[]"
	cnt += 1

# delete header
del train_corpus[0]
del train_attrs[0]

# get test_tweets from csv to test_corpus[]
cnt = 0
test_reader = csv.reader(test_csv)
for tweet in test_reader:
	text = unicode(tweet[1] + " " + tweet[2] + " " + tweet[3], 'ascii', 'ignore')
	text = cleaned_text(text)
	test_corpus.append(text)
	print "test tweet", cnt, "to corpus[]"
	cnt += 1

# delete header
del test_corpus[0]

train_csv.close()
test_csv.close()

# get x_train, x_test from train_corpus, test_corpus
print "start extraction"
entire_corpus = train_corpus + test_corpus
vectorizer = TfidfVectorizer(min_df = 1, tokenizer = nltk.word_tokenize)
vectorizer.fit(train_corpus)
x_train = vectorizer.transform(train_corpus)
x_test = vectorizer.transform(test_corpus)
print "finish extraction"

# get y_train from train_attrs
y_train = [[float(x) for x in attr] for attr in train_attrs]

# # get feature names
# feature_names = [('FEATURE:' + str(i)) for i in xrange(0, 2000)]
# feature_names += ["ATTR:I can not tell attitude"
# ,"ATTR:Negative"
# ,"ATTR:Neutral / author is just sharing information"
# ,"ATTR:Positive"
# ,"ATTR:Tweet not related to weather condition"  
# # ,"ATTR:current (same day) weather"
# # ,"ATTR:future (forecast)"
# # ,"ATTR:I can not tell time"
# # ,"ATTR:past weather"
# # ,"ATTR:clouds"
# # ,"ATTR:cold"
# # ,"ATTR:dry"
# # ,"ATTR:hot"
# # ,"ATTR:humid"
# # ,"ATTR:hurricane"
# # ,"ATTR:I can not tell weather"
# # ,"ATTR:ice"
# # ,"ATTR:other"
# # ,"ATTR:rain"
# # ,"ATTR:snow"
# # ,"ATTR:storms"
# # ,"ATTR:sun"
# # ,"ATTR:tornado"
# # ,"ATTR:wind"
# ]
# feature_names = [x.encode('utf-8') for x in feature_names]

# chi-2 select features
selector = SelectKBest(chi2, k = 2000)
selector.fit(x_train, y_train)
x_train = selector.transform(x_train)
x_test = selector.transform(x_test)

# # export to csv 
# # write extracted[0...train_len] and train_attrs[] to csv
# train_len = len(train_corpus)
# train_result_writer = csv.writer(dst_train_csv)
# train_result_writer.writerow(feature_names)
# for i in xrange(0, train_len):
# 	vector = x_train[i].toarray().tolist()[0] + y_train[i]
# 	train_result_writer.writerow(vector)
# 	print "train entry", i, "to csv"

# # write extracted[0...test_len] to csv
# test_len = len(test_corpus)
# test_result_writer = csv.writer(dst_test_csv)
# test_result_writer.writerow(feature_names)
# for i in xrange(0, test_len):
# 	vector = x_test[i].toarray().tolist()[0]
# 	test_result_writer.writerow(vector)
# 	print "test entry", i, "to csv"

# print stat 
print "x_train", x_train
print "x_test", x_test