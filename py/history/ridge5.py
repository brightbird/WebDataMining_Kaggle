import pandas as p
from sklearn import linear_model
import numpy as np
from sklearn import cross_validation
from sklearn import svm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
import re
import threading

extraction_method=0


paths = ['../data/mytrain.csv', '../data/mytest.csv']	
t = p.read_csv(paths[0])
t2 = p.read_csv(paths[1])
extra2 = " you posit are negat is you was the it of state my and locat you to weather be degre am it mph weather weather a all to "

for i in range(len(t['tweet'])):
	t['tweet'][i] = str(t['tweet'][i]) + " " + str(t['state'][i]) + " " + str(t['location'][i]) + extra2


for i in range(len(t2['tweet'])):
	t2['tweet'][i] = str(t2['tweet'][i]) + " " + str(t2['state'][i]) + " " + str(t2['location'][i]) + extra2


print "start extraction"
# delete the max_feature, score is higher
if extraction_method==0:
	print "Tfidf......"
	tfidf = TfidfVectorizer(strip_accents='unicode', analyzer='word',  ngram_range=(1,2))
	tfidf.fit(t['tweet'])
	X = tfidf.transform(t['tweet'])
	test = tfidf.transform(t2['tweet'])
	y = np.array(t.ix[:,4:])
else:
	print "Count......"
	countmethod = CountVectorizer(strip_accents='unicode', analyzer='word', lowercase=True)
	countmethod.fit(t['tweet'])
	X = countmethod.transform(t['tweet'])
	test = countmethod.transform(t2['tweet'])
	y = np.array(t.ix[:,4:])


print "extraction done"

print "start fit"
clf = linear_model.Ridge (alpha = 1.0)
clf.fit(X,y)
print "fit done"
print "start prediction"
test_prediction = clf.predict(test)
for i in xrange(len(test_prediction)):
	for j in xrange(len(test_prediction[i])):
		if test_prediction[i][j] <= 0.05:
			test_prediction[i][j] = 0
		elif test_prediction[i][j] >= 0.95:
			test_prediction[i][j] = 1

	# normalize attitude
	summary = 0
	for j in xrange(0, 5):
		summary += test_prediction[i][j]
	if (summary != 0):
		for j in xrange(0, 5):
			test_prediction[i][j] /= summary
	# normalize time
	summary = 0
	for j in xrange(5, 9):
		summary += test_prediction[i][j]
	if (summary != 0):
		for j in xrange(5, 9):
			test_prediction[i][j] /= summary
print "prediction done"


coname = ['id','s1','s2','s3','s4','s5','w1','w2','w3','w4','k1','k2','k3','k4','k5','k6','k7','k8','k9','k10',
	'k11','k12','k13','k14','k15']

first = np.matrix(coname)
print "start writing"
prediction = np.array(np.hstack([np.matrix(t2['id']).T, test_prediction])) 
col = '%i,' + '%f,'*23 + '%f'

np.savetxt('../data/myresult50.csv', prediction ,col, delimiter=',')
print "writing done"