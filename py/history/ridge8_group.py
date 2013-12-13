# 11.28 ver 1

import os
import pandas
import nltk
import re
import numpy as np
from sklearn.feature_extraction.text import *
from sklearn.linear_model import *

########################
## 		SETTINGS	  ##
########################

extra = " you posit are negat i etc me as at be between both is you was for he she that the it of state my and locat you to weather be degre am it mph weather weather a all to "

#################################
#  	  get content from CSV 	    #
#################################
print "get content from csv"

cur_dir = os.getcwd()
test_path = cur_dir + "/../data/mytest2.csv"
train_path= cur_dir + "/../data/mytrain2.csv"

train_content = pandas.read_csv(train_path)
test_content = pandas.read_csv(test_path)
train_len = len(train_content)
test_len = len(test_content)

for i in xrange(0, train_len):
	train_content['tweet'][i] = str(train_content['tweet'][i]) + " " + str(train_content['state'][i]) + " " + str(train_content['location'][i]) + extra
for i in xrange(0, test_len):
	test_content['tweet'][i] = str(test_content['tweet'][i]) + " " + str(test_content['state'][i]) + " " + str(test_content['location'][i]) + extra

train_tweets = train_content['tweet']
train_attitude = train_content.ix[:,4:9]
train_time = train_content.ix[:,9:13]
train_weather = train_content.ix[:,13:28]
train_attributes = train_content.ix[:,4:28]
test_tweets = test_content['tweet']

#################################
# 		Feature Exraction 		#
#################################
print "feature extraction"

vectorizer = TfidfVectorizer(ngram_range=(1,3), strip_accents='unicode', analyzer='word')
vectorizer.fit(train_tweets)
x_train = vectorizer.transform(train_tweets)
x_test = vectorizer.transform(test_tweets)

#################################
#			Regression			#
#################################
print "regression"

clf = Ridge (alpha = 0.65)

y_train = np.array(train_attitude)
clf.fit(x_train, y_train)
y_test_attitude = clf.predict(x_test)

y_train = np.array(train_time)
clf.fit(x_train, y_train)
y_test_time = clf.predict(x_test)

y_train = np.array(train_weather)
clf.fit(x_train, y_train)
y_test_weather = clf.predict(x_test)

y_test = np.hstack((y_test_attitude, y_test_time, y_test_weather))

#################################
#		  Normalization			#
#################################
print "normalization"

for i in xrange(len(y_test)):
	for j in xrange(len(y_test[i])):
		if y_test[i][j] <= 0.01:
			y_test[i][j] = 0
		elif y_test[i][j] >= 0.99:
			y_test[i][j] = 1
	# normalize attitude
	summary = 0
	for j in xrange(0, 5):
		summary += y_test[i][j]
	if (summary != 0):
		for j in xrange(0, 5):
			y_test[i][j] /= summary
	# normalize time
	summary = 0
	for j in xrange(5, 9):
		summary += y_test[i][j]
	if (summary != 0):
		for j in xrange(5, 9):
			y_test[i][j] /= summary

#################################
#		write to csv			#
#################################
print "write back to csv"
prediction = np.array(np.hstack([np.matrix(test_content['id']).T, y_test])) 
col = '%i,' + '%f,'*23 + '%f'
np.savetxt(cur_dir + "/../data/result/prediction.csv", prediction,col, delimiter=',')
