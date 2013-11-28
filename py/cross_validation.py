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
import numpy as np
from sklearn.feature_extraction.text import *
from sklearn.feature_selection import SelectPercentile, chi2
from sklearn.linear_model import *
from sklearn import cross_validation

########################
## 		SETTINGS	  ##
########################

CORPUS_SIZE = 0 		# 0 for entire, 1 for small 
SELECT_PERCENTILE = 30
SELECTION = 0 			# 0 for off, 1 for on

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

vectorizer = TfidfVectorizer(max_features=10000, strip_accents='unicode', analyzer='word')
vectorizer.fit(train_tweets)
x_train = vectorizer.transform(train_tweets)
y_train = np.array(train_attributes)

#################################
# 		Feature Selection 		#
#################################
if (SELECTION == 1):
	print "feature selection"

	selector = SelectPercentile(score_func=chi2, percentile=SELECT_PERCENTILE)
	selector.fit(x_train, y_train.tolist())
	x_train = selector.transform(x_train)

#################################
#			Regression			#
#################################
print "regression"

x_train, x_test, y_train, y_test = cross_validation.train_test_split(x_train, y_train, test_size=0.4, random_state=0)
# clf = LinearRegression()
clf = Ridge (alpha = 1.85)
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
 		if (num > 0):
 			temp[i].append(1)
 		elif (num >= 0.05):
 			temp[i].append(num)
 		else:
 			temp[i].append(0)

for i in xrange(0, length):
 	summary = 0
 	for j in xrange(0, 5):
 		summary += temp[i][j]
 	for j in xrange(0, 5):
 		temp[i][j] /= summary
 	summary = 0
 	for j in xrange(5, 9):
 		summary += temp[i][j]
 	for j in xrange(5, 9):
 		temp[i][j] /= summary
 	summary = 0
 	for j in xrange(9, 24):
 		summary += temp[i][j]
 	for j in xrange(9, 24):
 		temp[i][j] /= summary

prediction = temp

#################################
#			score				#
#################################

RMSE = np.sqrt(np.sum(np.array(np.array(prediction)-y_test)**2)/ (x_test.shape[0]*24.0))
print "RMSE :", RMSE
