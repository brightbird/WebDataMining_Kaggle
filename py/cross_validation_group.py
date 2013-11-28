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
from sklearn import cross_validation

########################
## 		SETTINGS	  ##
########################

CORPUS_SIZE = 0 		# 0 for entire, 1 for small 

#################################
#  	  get content from CSV 	    #
#################################
print "get content from csv"

cur_dir = os.getcwd()
if (CORPUS_SIZE == 1):
	train_path = cur_dir + "/../data/small_train.csv"
else:
	train_path= cur_dir + "/../data/train.csv"

train_content = pandas.read_csv(train_path)
train_len = len(train_content)

for i in xrange(0, train_len):
	train_content['tweet'][i] = re.sub("http\S*|@\S*|{link}|RT\s*@\S*", "",train_content['tweet'][i])
	if (isinstance(train_content['state'][i], basestring) == False):
		train_content['state'][i] = ""
	if (isinstance(train_content['location'][i], basestring) == False):
		train_content['location'][i] = ""

train_tweets = train_content['tweet']
train_location = train_content['state'] + " " + train_content['location']
train_attitude = train_content.ix[:,4:9]
train_time = train_content.ix[:,9:13]
train_weather = train_content.ix[:,13:28]
train_attributes = train_content.ix[:,4:28]

#################################
# 		Feature Exraction 		#
#################################
print "feature extraction"

train_tweets = train_tweets + " " + train_location

vectorizer = TfidfVectorizer(max_features=10000, strip_accents='unicode', analyzer='word')
vectorizer.fit(train_tweets)
x_train = vectorizer.transform(train_tweets)

#################################
#			Regression			#
#################################
print "regression"

y_train = np.array(train_attributes)
x_train, x_test, y_train, y_test = cross_validation.train_test_split(x_train, y_train, test_size=0.4, random_state=0)

# clf = LinearRegression()
clf = Ridge (alpha = 1.85)

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
#			Normalize			#
#################################
print "normalization"

length = x_test.shape[0]
temp = []
for i in xrange(0, length):
	temp.append([])
	vector = prediction[i]
 	for j in xrange(0, 24):
 		num = vector[j]
 		if (num > 1):
 			temp[i].append(1)
 		elif (num >= 0.05):
 			temp[i].append(num)
 		else:
 			temp[i].append(0)

for i in xrange(0, length):
 	summary = 0
 	for j in xrange(0, 5):
 		summary += temp[i][j]
 	if (summary != 0):
	 	for j in xrange(0, 5):
	 		temp[i][j] /= summary
 	summary = 0
 	for j in xrange(5, 9):
 		summary += temp[i][j]
 	if (summary != 0):
	 	for j in xrange(5, 9):
	 		temp[i][j] /= summary

prediction = temp

#################################
#			score				#
#################################

RMSE = np.sqrt(np.sum(np.array(prediction-y_test)**2)/ (x_test.shape[0]*24.0))
print "RMSE :", RMSE
