import re, collections
import pandas as p
import nltk
from nltk.tokenize import *
from stemming.porter2 import stem

def words(text): return re.findall('[a-z]+', text.lower()) 

def train(features):
    model = collections.defaultdict(lambda: 1)
    for f in features:
        model[f] += 1
    return model

NWORDS = train(words(file('big.txt').read()))

alphabet = 'abcdefghijklmnopqrstuvwxyz'

def edits1(word):
   splits     = [(word[:i], word[i:]) for i in range(len(word) + 1)]
   deletes    = [a + b[1:] for a, b in splits if b]
   transposes = [a + b[1] + b[0] + b[2:] for a, b in splits if len(b)>1]
   replaces   = [a + c + b[1:] for a, b in splits for c in alphabet if b]
   inserts    = [a + c + b     for a, b in splits for c in alphabet]
   return set(deletes + transposes + replaces + inserts)

def known_edits2(word):
    return set(e2 for e1 in edits1(word) for e2 in edits1(e1) if e2 in NWORDS)

def known(words): return set(w for w in words if w in NWORDS)

def correct(word):
    candidates = known([word]) or known(edits1(word)) or known_edits2(word) or [word]
    return max(candidates, key=NWORDS.get)


paths = ['../data/mytrain2.csv', '../data/mytest2.csv']
t0 = p.read_csv(paths[0])
t1 = p.read_csv(paths[1])

for i in range(len(t1['tweet'])):
	text = nltk.word_tokenize(t1['tweet'][i])
	t1['tweet'][i] = ""
	for x in text:
		flag = 1
		while (flag==1 and x != ''):
			if (x.endswith('.')):
				x = x[:-1]
			elif (x.endswith(',')):
				x = x[:-1]
			elif (x.endswith('!')):
				x = x[:-1]
			elif (x.endswith('?')):
				x = x[:-1]
			elif (x.endswith('/')):
				x = x[:-1]
			elif (x.endswith('?')):
				x = x[:-1]
			elif (x.endswith(':')):
				x = x[:-1]
			elif (x.endswith('\\')):
				x = x[:-1]
			else:
				flag = 0
		if x.isalpha():
			x = correct(x.lower())
		if (x != 's'):
			t1['tweet'][i] = t1['tweet'][i] + correct(x) + " "
	# print t1['tweet'][i]
	
print "test done"

for i in range(len(t0['tweet'])):
	text = nltk.word_tokenize(t0['tweet'][i])
	t0['tweet'][i] = ""
	for x in text:
		flag = 1
		while (flag==1 and x != ''):
			if (x.endswith('.')):
				x = x[:-1]
			elif (x.endswith(',')):
				x = x[:-1]
			elif (x.endswith('!')):
				x = x[:-1]
			elif (x.endswith('?')):
				x = x[:-1]
			elif (x.endswith('/')):
				x = x[:-1]
			elif (x.endswith('?')):
				x = x[:-1]
			elif (x.endswith(':')):
				x = x[:-1]
			elif (x.endswith('\\')):
				x = x[:-1]
			else:
				flag = 0
		if x.isalpha():
			x = correct(x.lower())
		if (x != 's'):
			t0['tweet'][i] = t0['tweet'][i] + correct(x) + " "
	
t0.to_csv('../data/mytrain3.csv', index=False)
t1.to_csv('../data/mytest3.csv', index=False)