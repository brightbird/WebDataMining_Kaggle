# 11.28 ver 1 - 0.17528 normalized
# the same with once regression

import os
import pandas
import nltk
import numpy as np
from sklearn.feature_extraction.text import *
from sklearn.linear_model import *

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
	test_path = cur_dir + "/../data/small_test.csv"
	train_path = cur_dir + "/../data/small_train.csv"
else:
	test_path = cur_dir + "/../data/test.csv"
	train_path= cur_dir + "/../data/train.csv"

train_content = pandas.read_csv(train_path)
test_content = pandas.read_csv(test_path)
train_len = len(train_content)
test_len = len(test_content)

for i in xrange(0, train_len):
	train_content['tweet'][i] = re.sub("http\S*|@\S*|{link}|RT\s*@\S*", "",train_content['tweet'][i])
	if (isinstance(train_content['state'][i], basestring) == False):
		train_content['state'][i] = ""
	if (isinstance(train_content['location'][i], basestring) == False):
		train_content['location'][i] = ""
for i in xrange(0, test_len):
	test_content['tweet'][i] = re.sub("http\S*|@\S*|{link}|RT\s*@\S*", "",test_content['tweet'][i])
	if (isinstance(test_content['state'][i], basestring) == False):
		test_content['state'][i] = ""
	if (isinstance(test_content['location'][i], basestring) == False):
		test_content['location'][i] = ""

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

vectorizer = TfidfVectorizer(max_features=4000, strip_accents='unicode', analyzer='word')
vectorizer.fit(train_tweets)
x_train = vectorizer.transform(train_tweets)
x_test = vectorizer.transform(test_tweets)

#################################
#			Regression			#
#################################
print "regression"

clf = LinearRegression()

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
#		write to csv			#
#################################
print "write back to csv"
prediction = np.array(np.hstack([np.matrix(test_content['id']).T, y_test])) 
col = '%i,' + '%f,'*23 + '%f'
np.savetxt(cur_dir + "/../data/result/prediction.csv", prediction,col, delimiter=',')
