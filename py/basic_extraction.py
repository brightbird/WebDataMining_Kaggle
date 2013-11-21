# 11.16 ver 1

import csv
from sklearn.feature_extraction.text import TfidfVectorizer

testcsv = file("/Users/Zhao/codes/eclipse/WebDataMining/data/test.csv")
traincsv = file("/Users/Zhao/codes/eclipse/WebDataMining/data/small_train.csv")
dstcsv = file("/Users/Zhao/codes/eclipse/WebDataMining/data/result/small_train_result.csv", "a")

# get tweets, attributes from csv to corpus[], attrs[]
corpus = []
attrs = []
feature_names = []
trainreader = csv.reader(traincsv)
for tweet in trainreader:
	# tweet content + state + city
	attr = []
	text = tweet[1] + " " + tweet[2] + " " + tweet[3]
	for i in xrange(4, 28):
		attr.append(tweet[i])
	corpus.append(text)
	attrs.append(attr)

# extract corpus[] to extracted[]
vectorizer = TfidfVectorizer(min_df=1)
extracted = vectorizer.fit_transform(corpus) 
extracted = extracted.toarray()
extracted = extracted.tolist()

# get feature names
feature_names = vectorizer.get_feature_names()
feature_names += ["ATTR:I can not tell attitude"
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
,"ATTR:wind"]

# write extracted[] and attrs[] to csv
num = len(attrs)
resultwriter = csv.writer(dstcsv)
resultwriter.writerow(feature_names)
for i in xrange(1, num):
	vector = extracted[i - 1] + attrs[i - 1]
	resultwriter.writerow(vector)

testcsv.close()
traincsv.close()
dstcsv.close()

print "done"