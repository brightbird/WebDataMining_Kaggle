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
	text = nltk.word_tokenize(text)
	text = " ".join([stem(x.lower()) for x in text])
	return text

cur_dir = os.getcwd()
test_csv = file(cur_dir + "/../data/small_test.csv")
train_csv = file(cur_dir + "/../data/small_train.csv")
# test_csv = file(cur_dir + "/../data/test.csv")
# train_csv = file(cur_dir + "/../data/train.csv")

#################################
#  	  Get Corpus From CSV 	    #
#################################

train_corpus = []
test_corpus = []

# get train_tweets from csv to train_corpus[]
cnt = 0
train_reader = csv.reader(train_csv)
for tweet in train_reader:
	# tweet content + state + city
	text = unicode(tweet[1] + " " + tweet[2] + " " + tweet[3], 'ascii', 'ignore')
	# text = cleaned_text(text)
	train_corpus.append(text)
	print "train tweet", cnt, "to corpus[]"
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
	print "test tweet", cnt, "to corpus[]"
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
vectorizer = TfidfVectorizer(min_df = 1, tokenizer = nltk.word_tokenize)
vectorizer.fit(train_corpus)
x_train = vectorizer.transform(train_corpus)
x_test = vectorizer.transform(test_corpus)
print "finish extraction"

#################################
# 		Feature Selection 		#
#################################

k_for_bestK = 10

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
for CURRENT_ATTRIBUTE in xrange(0, 5):

	# get CURRENT ATTRIBUTE train_attrs from csv
	train_attrs = []
	# train_csv = file(cur_dir + "/../data/train.csv")
	train_csv = file(cur_dir + "/../data/small_train.csv")
	train_reader = csv.reader(train_csv)
	cnt = 0
	for tweet in train_reader:
		attr = tweet[CURRENT_ATTRIBUTE + 4]
		train_attrs.append(attr)
		# print CURRENT_ATTRIBUTE, "- train attr", cnt, "to attrs[]"
		cnt += 1

	del train_attrs[0]

	# get y_train from train_attrs
	y_train = [[float(attr)] for attr in train_attrs]

	print x_train.getnnz()
	print len(y_train)

	# chi-2 select features
	selector = SelectKBest(chi2, k = k_for_bestK)
	selector.fit(x_train, y_train)
	new_x_train = selector.transform(x_train)
	new_x_test = selector.transform(x_test)

	# display best 100 words
	best_words = []
	words = vectorizer.get_feature_names()
	feature_value_pairs = []
	cnt = 0
	for item in selector.scores_:
		feature_value_pairs.append((cnt, item))
		cnt += 1
	feature_value_pairs = sorted(feature_value_pairs, key = lambda pair : -pair[1])
	for item in feature_value_pairs[:100]:
		best_words.append(words[item[0]])
	print attribute_names[CURRENT_ATTRIBUTE]
	print best_words

	# build csv file
	dst_train_path = cur_dir + "/../data/result/train" + str(CURRENT_ATTRIBUTE) + ".csv"
	dst_test_path = cur_dir + "/../data/result/test" + str(CURRENT_ATTRIBUTE) + ".csv"
	if os.path.exists(dst_train_path):
		os.remove(dst_train_path)
	if os.path.exists(dst_test_path):
		os.remove(dst_test_path)
	dst_train_csv = file(dst_train_path, 'a')
	dst_test_csv = file(dst_test_path, 'a')

	# get header for new_x_test csv
	header = [x for x in xrange(0, k_for_bestK)]

	# write new_x_test to csv
	test_len = len(test_corpus)
	test_result_writer = csv.writer(dst_test_csv)
	test_result_writer.writerow(header)
	for i in xrange(0, test_len):
		vector = new_x_test[i].toarray().tolist()[0]
		test_result_writer.writerow(vector)
		# print "test entry", i, "to csv"

	# get header for new_x_train csv
	header.append(attribute_names[CURRENT_ATTRIBUTE])

	# write new_x_train and new_y_train to csv
	train_len = len(train_corpus)
	train_result_writer = csv.writer(dst_train_csv)
	train_result_writer.writerow(header)
	for i in xrange(0, train_len):
		vector = new_x_train[i].toarray().tolist()[0] + y_train[i]
		train_result_writer.writerow(vector)
		# print "train entry", i, "to csv"

	train_csv.close()
	dst_train_csv.close()
	dst_test_csv.close()

	# print stat `
	print "new_x_train", new_x_train
	print "new_x_test", new_x_test
