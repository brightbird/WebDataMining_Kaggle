# 11.28 ver 1
# new templete regression
# kaggle : 0.17528, RMSE : 0.17920
# new templete regression 2
# kaggle : 0.23332, RMSE : 0.22970
# ridge 1
# kaggle : 0.16405, RMSE : 0.16414 


import os
import pandas
import nltk
import re
import threading
import numpy as np
from sklearn.feature_extraction.text import *
from sklearn.feature_selection import SelectPercentile, chi2
from sklearn.linear_model import *
from sklearn import cross_validation

########################
## 		SETTINGS	  ##
########################

CORPUS_SIZE = 0 		# 0 for entire, 1 for small 
extra = " you posit are negat is you was the it of state my and locat you to weather be degre am it mph weather weather a all to "

#################################
#  	  get content from CSV 	    #
#################################
print "get content from csv"

cur_dir = os.getcwd()
if (CORPUS_SIZE == 1):
	train_path = cur_dir + "/../data/small_train.csv"
else:
	train_path= cur_dir + "/../data/trainStem.csv"

train_content = pandas.read_csv(train_path)
train_len = len(train_content)

for i in xrange(0, train_len):
	train_content['tweet'][i] = str(train_content['tweet'][i]) + " " + str(train_content['state'][i]) + " " + str(train_content['location'][i]) + extra

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

vectorizer = TfidfVectorizer(ngram_range = (1, 2), strip_accents='unicode', analyzer='word')
vectorizer.fit(train_tweets)
raw_x_train = vectorizer.transform(train_tweets)
raw_y_train = np.array(train_attributes)

best_a = 0
best_rmse = 1
for i in xrange(0, 50):
	ALPHA = 1 + i * 0.03

	#################################
	#			Regression			#
	#################################
	print "regression"

	x_train, x_test, y_train, y_test = cross_validation.train_test_split(raw_x_train, raw_y_train, test_size=0.4, random_state=0)
	# clf = LinearRegression()
	clf = Ridge (alpha = ALPHA)
	clf.fit(x_train, y_train)
	prediction = clf.predict(x_test)

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

	RMSE = np.sqrt(np.sum(np.array(np.array(prediction)-y_test)**2)/ (x_test.shape[0]*24.0))
	print "RMSE :", RMSE, "alpha :", ALPHA
	if (RMSE < best_rmse):
		best_rmse = RMSE
		best_a = ALPHA
	print "best rmse :", best_rmse, "best a :", best_a