# 11.21 ver 1

import os
import csv
import nltk
import numpy as np
from sklearn.feature_extraction.text import *
from sklearn.feature_selection import SelectKBest, chi2, f_classif
from scipy.sparse import * 
from nltk.tokenize import *
from nltk.stem.snowball import *
from stemming.porter2 import stem
from scipy.io import *

################################
## do your text cleaning here ##

def cleaned_text(text):
	text = nltk.word_tokenize(text)
	text = " ".join([stem(x.lower()) for x in text])
	return text

################################

CORPUS_SIZE_ITEMS = ['entire', 'small']
CORPUS_SIZE = 1 		# 0 for entire, 1 for small 
PREDICT_ATTRIBUTE_NUM = 15
VECTORIZER = 1 			# 0 for CountVectorizer, 1 for TfidfVectorizer

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
	text = cleaned_text(text)
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
	text = cleaned_text(text)
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
if (VECTORIZER == 0):
	vectorizer = CountVectorizer(min_df = 1, tokenizer = nltk.word_tokenize)
elif (VECTORIZER == 1):
	vectorizer = TfidfVectorizer(min_df = 1, tokenizer = nltk.word_tokenize)
vectorizer.fit(train_corpus)
x_train = vectorizer.transform(train_corpus)
x_test = vectorizer.transform(test_corpus)
print "finish extraction"

#################################
# 		Feature Selection 		#
#################################

k_for_bestK = 2000

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
		# print CURRENT_ATTRIBUTE, "- train attr", cnt, "to attrs[]"
		cnt += 1

	del train_attrs[0]

	# get y_train from train_attrs
	y_train = [[float(attr)] for attr in train_attrs]

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
		
	f=open(cur_dir + '/../zhou_test/attribute' + str(CURRENT_ATTRIBUTE + 1) + '.txt','w')
	f.write(attribute_names[CURRENT_ATTRIBUTE])
	f.write("\n")
	f.write("\n")
	for i in best_words:
		f.write(i + "\n")
	f.close()