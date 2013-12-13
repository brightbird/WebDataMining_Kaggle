import os
import re
import pandas
import threading
from sklearn.feature_extraction.text import *

########################
## 		SETTINGS	  ##
########################

CORPUS_SIZE = 0 		# 0 for entire, 1 for small 
FEATURE_NUM = 10000

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
train_attributes = train_content.ix[:,4:28]
test_tweets = test_content['tweet']
test_location = test_content['state'] + " " + test_content['location']

#################################
# 		Feature Exraction 		#
#################################
print "feature extraction"

vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=FEATURE_NUM, strip_accents='unicode', analyzer='word')
vectorizer.fit(train_tweets)
x_train = vectorizer.transform(train_tweets)
x_test = vectorizer.transform(test_tweets)

#################################
#			Convert				#
#################################

train_feature_pair_res = []
for VECTOR_INDEX in xrange(0, train_len):
	print VECTOR_INDEX
	train_feature_pair_res.append([])
	vector = x_train[VECTOR_INDEX].toarray()[0]
	for FEATURE_INDEX in xrange(0, FEATURE_NUM):
		if (vector[FEATURE_INDEX] != 0):
			train_feature_pair_res[VECTOR_INDEX].append((str(FEATURE_INDEX + 1), str("%0.4f"%vector[FEATURE_INDEX])))

# for ATTRIBUTE_NUM in xrange(0, 24):
def write_attr_file(ATTRIBUTE_NUM):
	train_attr_res = []
	for VECTOR_INDEX in xrange(0, train_len):
		train_attr_res.append(train_attributes.ix[VECTOR_INDEX][ATTRIBUTE_NUM])
	dst_path = cur_dir + "/../libsvm-3.17/data/train_attr_" + str(ATTRIBUTE_NUM)
	if os.path.exists(dst_path):
		os.remove(dst_path)
	dst_file = file(dst_path, "a")
	for VECTOR_INDEX in xrange(0, train_len):
		dst_file.write(str(train_attr_res[VECTOR_INDEX]))
		for item in train_feature_pair_res[VECTOR_INDEX]:
			dst_file.write("".join([" ", item[0], ":", item[1]]))
		dst_file.write("\n")
	dst_file.close()

threads = []
for ATTRIBUTE_NUM in xrange(0, 24):
	threads.append(threading.Thread(target=write_attr_file, args=(ATTRIBUTE_NUM,)))
for ATTRIBUTE_NUM in xrange(0, 24):
	threads[ATTRIBUTE_NUM].start()
for ATTRIBUTE_NUM in xrange(0, 24):
	threads[ATTRIBUTE_NUM].join()

test_res = []
for VECTOR_INDEX in xrange(0, test_len):
	test_res.append([0])
	vector = x_test[VECTOR_INDEX].toarray()[0]
	for FEATURE_INDEX in xrange(0, FEATURE_NUM):
		test_res[VECTOR_INDEX].append(vector[FEATURE_INDEX])
dst_path = cur_dir + "/../libsvm-3.17/data/test_attr_" + str(0)
if os.path.exists(dst_path):
	os.remove(dst_path)
dst_file = file(dst_path, "a")
for vector in test_res:
	dst_file.write(str(vector[0]))
	for FEATURE_INDEX in xrange(0, FEATURE_NUM):
		if (vector[FEATURE_INDEX + 1] != 0):
			dst_file.write("".join([" ", str(FEATURE_INDEX + 1), ":", str("%0.4f"%vector[FEATURE_INDEX + 1])]))
	dst_file.write("\n")
dst_file.close()
