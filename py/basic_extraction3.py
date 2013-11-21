# -*- coding: utf-8 -*-
# 11.20 ver 3

import csv
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.snowball import *
from nltk.tokenize import *

# test_csv = file("/Users/Zhao/codes/eclipse/WebDataMining/data/small_test.csv")
# train_csv = file("/Users/Zhao/codes/eclipse/WebDataMining/data/small_train.csv")
test_csv = file("/Users/Zhao/codes/eclipse/WebDataMining/data/test.csv")
train_csv = file("/Users/Zhao/codes/eclipse/WebDataMining/data/train.csv")

cnt = 0
train_corpus = []
train_attrs = []
test_corpus = []
# get train_tweets, their attributes from csv to train_corpus[], train_attrs[]
train_reader = csv.reader(train_csv)
for tweet in train_reader:
	# tweet content + state + city
	text = unicode(tweet[1] + " " + tweet[2] + " " + tweet[3], 'ascii', 'ignore')
	# text = WhitespaceTokenizer().tokenize(text)
	# text = " ".join([EnglishStemmer().stem(x) for x in text])
	attr = []
	for i in xrange(4, 28):
		attr.append(tweet[i])
	train_corpus.append(text)
	train_attrs.append(attr)
	print "train tweet", cnt, "to corpus[]"
	cnt += 1

cnt = 0
# get test_tweets from csv to test_corpus[]
test_reader = csv.reader(test_csv)
for tweet in test_reader:
	text = tweet[1] + " " + tweet[2] + " " + tweet[3]
	test_corpus.append(text)
	print "test tweet", cnt, "to corpus[]"
	cnt += 1

# extract corpus[] to extracted[]
print "start extraction"
corpus = train_corpus + test_corpus
vectorizer = TfidfVectorizer(min_df=1)
extracted = vectorizer.fit_transform(corpus) 
print "finish extraction"

# get feature names
feature_names = vectorizer.get_feature_names()
feature_names += ["ATTR:I can not tell attitude"
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
,"ATTR:wind"]
feature_names = [x.encode('utf-8') for x in feature_names]

# # write extracted[0...train_len] and train_attrs[] to csv
# train_len = len(train_corpus)
# train_result_writer = csv.writer(dst_train_csv)
# train_result_writer.writerow(feature_names)
# for i in xrange(0, train_len):
# 	vector = extracted[i].toarray().tolist() + train_attrs[i]
# 	train_result_writer.writerow(vector)
# 	print "train entry", i, "to csv"

# # write extracted[0...test_len] to csv
# test_len = len(test_corpus)
# test_result_writer = csv.writer(dst_test_csv)
# test_result_writer.writerow(feature_names)
# for i in xrange(0, test_len):
# 	vector = extracted[i].toarray().tolist()
# 	test_result_writer.writerow(vector)
# 	print "test entry", i, "to csv"

test_csv.close()
train_csv.close()
print "done"