import csv
from sklearn.feature_extraction.text import TfidfVectorizer

testcsv = file("/Users/Zhao/codes/eclipse/WebDataMining/data/test.csv")
traincsv = file("/Users/Zhao/codes/eclipse/WebDataMining/data/small_train.csv")

corpus = []

# get tweets from csv to corpus[]
trainreader = csv.reader(traincsv)
for tweet in trainreader:
	# tweet content + state + city
	gathered = tweet[1] + " " + tweet[2] + " " + tweet[3]
	corpus.append(gathered)

# extract 
vectorizer = TfidfVectorizer(min_df=1)
result = vectorizer.fit_transform(corpus)

# take a glimpse at the first glimpse of result
print result[0]
