# 11.28 ver 1
# new templete group
# kaggle : 0.17528, RMSE : 0.17920
# new templete group 2
# kaggle : , RMSE : 0.21230
# ridge 1
# kaggle : , RMSE : 0.16409
# ridge 2
# kaggle : , RMSE : 0.15752

import os
import pandas
import nltk
import math
import re
import numpy as np
from sklearn.feature_extraction.text import *
from sklearn.feature_selection import SelectPercentile, chi2
from sklearn.linear_model import *
from sklearn.svm import *
from sklearn import cross_validation

########################
## 		SETTINGS	  ##
########################

extra = " you posit are negat i etc me as at be between both is you was for he she that the it of state my and locat you to weather be degre am it mph weather weather a all to "

#################################
#  	  get content from CSV 	    #
#################################
print "get content from csv"

cur_dir = os.getcwd()
train_path= cur_dir + "/../data/train.csv"

train_content = pandas.read_csv(train_path)
train_len = len(train_content)

for i in xrange(0, train_len):
	train_content['tweet'][i] = str(train_content['tweet'][i]) + " " + str(train_content['state'][i]) + " " + str(train_content['location'][i]) + extra

train_tweets = train_content['tweet']
train_attitude = train_content.ix[:,4:9]
train_time = train_content.ix[:,9:13]
train_weather = train_content.ix[:,13:28]
train_attributes = train_content.ix[:,4:28]

#################################
# 		Feature Exraction 		#
#################################
print "feature extraction"

vectorizer = TfidfVectorizer(ngram_range=(1,3), strip_accents='unicode', analyzer='word')
vectorizer.fit(train_tweets)
x_train = vectorizer.transform(train_tweets)

#################################
#			Regression			#
#################################
print "regression"

y_train = np.array(train_attributes)
x_train, x_test, y_train, y_test = cross_validation.train_test_split(x_train, y_train, test_size=0.4, random_state=0)

# clf = LinearRegression()
clf = Ridge (alpha = 0.55)

this_y_train = np.array([item[:5] for item in y_train])
this_x_train = x_train
this_x_test = x_test
clf.fit(this_x_train, this_y_train)
y_test_attitude = clf.predict(this_x_test)

this_y_train = np.array([item[5:9] for item in y_train])
this_x_train = x_train
clf.fit(this_x_train, this_y_train)
y_test_time = clf.predict(this_x_test)

this_y_train = np.array([item[9:24] for item in y_train])
this_x_train = x_train
clf.fit(this_x_train, this_y_train)
y_test_weather = clf.predict(this_x_test)

prediction = np.hstack((y_test_attitude, y_test_time, y_test_weather))

#################################
#		  Normalization			#
#################################
print "normalization"

for i in xrange(len(y_test)):
	for j in xrange(24):
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
#			score				#
#################################

RMSE = np.sqrt(np.sum(np.array(prediction-y_test)**2)/ (x_test.shape[0]*24.0))
print "RMSE :", RMSE
