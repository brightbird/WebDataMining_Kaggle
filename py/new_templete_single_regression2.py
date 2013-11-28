# 11.28 ver 1

import os
import pandas
import nltk
import numpy as np
from sklearn.feature_extraction.text import *
from sklearn.feature_selection import SelectPercentile, chi2
from sklearn.linear_model import *

########################
## 		SETTINGS	  ##
########################

CORPUS_SIZE = 1 		# 0 for entire, 1 for small 
SELECT_PERCENTILE = 30

#################################
#  	  get content from CSV 	    #
#################################
print "get content from csv"

cur_dir = os.getcwd()
if (CORPUS_SIZE == 1):
	test_path = cur_dir + "/../data/small_test.csv"
	train_path = cur_dir + "/../data/small_train.csv"
else:
	test_path = cur_dir + "/../data/test.csv"
	train_path= cur_dir + "/../data/train.csv"


train_content = pandas.read_csv(train_path)
test_content = pandas.read_csv(test_path)
train_len = len(train_content)
test_len = len(test_content)

train_tweets = train_content['tweet']
train_location = train_content['state'] + " " + train_content['location']
train_attitude = train_content.ix[:,4:9]
train_time = train_content.ix[:,9:13]
train_weather = train_content.ix[:,13:28]
train_attributes = train_content.ix[:,4:28]
test_tweets = test_content['tweet']
test_location = test_content['state'] + " " + test_content['location']

#################################
# 		Feature Exraction 		#
#################################
print "feature extraction"

vectorizer = TfidfVectorizer(max_features=10000, strip_accents='unicode', analyzer='word')
vectorizer.fit(train_tweets)
x_train = vectorizer.transform(train_tweets)
x_test = vectorizer.transform(test_tweets)

#################################
#			Regression			#
#################################
print "regression"

clf = LinearRegression()
selector = SelectPercentile(score_func=chi2, percentile=SELECT_PERCENTILE)

y_test_arr = []
for i in xrange(0, 24):
	this_y = np.array(train_attributes.ix[:,i:i+1])
	selector.fit(x_train, this_y.tolist())
	this_x_train = selector.transform(x_train)
	this_x_test = selector.transform(x_test)
	clf.fit(this_x_train, this_y)
	y_test_arr.append(clf.predict(this_x_test))

length = x_test.shape[0]
y_test = []
for i in xrange(0, length):
	y_test.append([])
	for j in xrange(0, 24):
		y_test[i].append(y_test_arr[j][i][0])

y_test = np.array(y_test)

#################################
#		write to csv			#
#################################
print "write back to csv"
prediction = np.array(np.hstack([np.matrix(test_content['id']).T, y_test])) 
col = '%i,' + '%f,'*23 + '%f'
np.savetxt(cur_dir + "/../data/result/prediction.csv", prediction,col, delimiter=',')
