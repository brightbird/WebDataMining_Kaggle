import nltk
import pandas as p
import re
from nltk.tokenize import *
from stemming.porter2 import stem

paths = ['../data/train.csv', '../data/test.csv']
t0 = p.read_csv(paths[0])
t1 = p.read_csv(paths[1])
	
#for i in range(len(t0['tweet'])):
#	t0['tweet'][i] = t0['tweet'][i].translate(None, punctuation).lower()
#for i in range(len(t1['tweet'])):	
#	t1['tweet'][i] = t1['tweet'][i].translate(None, punctuation).lower()

for i in range(len(t1['tweet'])):
	t1['tweet'][i] = re.sub(r'\{\w*\}', '', t1['tweet'][i])#{link}{pic}
	t1['tweet'][i] = re.sub(r'[a-zA-z]+://[^\s]*', '', t1['tweet'][i])#url
	t1['tweet'][i] = re.sub(r'\w+([-+.]\w+)*@\w+([-.]\w+)*\.\w+([-.]\w+)*', '', t1['tweet'][i])#mail
	t1['tweet'][i] = re.sub(r'\bRT\b', '', t1['tweet'][i])#RT
	# t1['tweet'][i] = re.sub(r'[^\x00-\xff]+', '', t1['tweet'][i])#not en
	t1['tweet'][i] = re.sub(r'#', '', t1['tweet'][i])#\# may double the remain string(because it's a topic about the tweet 
	t1['tweet'][i] = re.sub(r'@\w+:?', '', t1['tweet'][i])#\@someone
	t1['tweet'][i] = re.sub(r'\d+\.\d+\.\d+\.\d+', '', t1['tweet'][i])#domain
	t1['tweet'][i] = re.sub(r'(=|(:[\-o0]?))\(+', ' sad ', t1['tweet'][i])# :( :-( =(
	t1['tweet'][i] = re.sub(r'(=|(:[\-o0]?))\)+', ' smile ', t1['tweet'][i])# :) :-) =)
	t1['tweet'][i] = re.sub(r'\^_\^', ' smile ', t1['tweet'][i])# ^_^
	t1['tweet'][i] = re.sub(r'(=|(:\-?))D+', ' excited ', t1['tweet'][i])# :D :-D =D
	t1['tweet'][i] = re.sub(r'\-(\.|_)+\-', ' annoyed ', t1['tweet'][i])# -___- -.-
	t1['tweet'][i] = re.sub(r'(o[_\.]+[O0])|([O0][_\.]+o)', ' WTF ', t1['tweet'][i])# o_0 0__o
	t1['tweet'][i] = re.sub(r'(0_0)|(O_O)', ' surprised ', t1['tweet'][i])# 0_0
	t1['tweet'][i] = re.sub(r':\-?(o|O)', ' shock ', t1['tweet'][i])# :O
	t1['tweet'][i] = re.sub(r':\-?/', ' frustrated ', t1['tweet'][i])# :/
	t1['tweet'][i] = re.sub(r'T_T', ' cry ', t1['tweet'][i])# T_T
	t1['tweet'][i] = re.sub(r'[xX]_[xX]', ' dead ', t1['tweet'][i])# x_x
	t1['tweet'][i] = re.sub(r':[pP]', ' laugh ', t1['tweet'][i])# :p
	t1['tweet'][i] = re.sub(r'[xX]D', ' LOL ', t1['tweet'][i])# LOL
	t1['tweet'][i] = re.sub(r'w/', ' with ', t1['tweet'][i])# w/
	t1['tweet'][i] = re.sub(r'[bB]4', ' before ', t1['tweet'][i])# b4 -> before
	t1['tweet'][i] = re.sub(r'\b(U|u)\b', 'you', t1['tweet'][i])# u U
	t1['tweet'][i] = re.sub(r'&\w+;', ' ', t1['tweet'][i])# html flag
	t1['tweet'][i] = re.sub(r'[!\?,]+', ' ', t1['tweet'][i])# !?,
	t1['tweet'][i] = re.sub(r'\-', ' ', t1['tweet'][i])# :
	t1['tweet'][i] = re.sub(r'\.{2,}', ' ', t1['tweet'][i])# ....
	t1['tweet'][i] = re.sub(r'\s[:|/\\]+\s', ' ', t1['tweet'][i])#  : \ / |
	t1['tweet'][i] = re.sub(r'"', ' ', t1['tweet'][i])#  "
	# text = nltk.word_tokenize(t1['tweet'][i])
	# t1['tweet'][i] = " ".join([stem(x.lower()) for x in text])#
	
	
	
for i in range(len(t0['tweet'])):
	t0['tweet'][i] = re.sub(r'\{\w*\}', '', t0['tweet'][i])#{link}{pic}
	t0['tweet'][i] = re.sub(r'[a-zA-z]+://[^\s]*', '', t0['tweet'][i])#url
	t0['tweet'][i] = re.sub(r'\w+([-+.]\w+)*@\w+([-.]\w+)*\.\w+([-.]\w+)*', '', t0['tweet'][i])#mail
	t0['tweet'][i] = re.sub(r'\bRT\b', '', t0['tweet'][i])#RT
	# t0['tweet'][i] = re.sub(r'[^\x00-\xff]+', '', t0['tweet'][i])#not en
	t0['tweet'][i] = re.sub(r'#', '', t0['tweet'][i])#\# may double the remain string(because it's a topic about the tweet 
	t0['tweet'][i] = re.sub(r'@\w+:?', '', t0['tweet'][i])#\@someone
	t0['tweet'][i] = re.sub(r'\d+\.\d+\.\d+\.\d+', '', t0['tweet'][i])#domain
	t0['tweet'][i] = re.sub(r'(=|(:[\-o0]?))\(+', ' sad ', t0['tweet'][i])# :( :-( =(
	t0['tweet'][i] = re.sub(r'(=|(:[\-o0]?))\)+', ' smile ', t0['tweet'][i])# :) :-) =)
	t0['tweet'][i] = re.sub(r'\^_\^', ' smile ', t0['tweet'][i])# ^_^
	t0['tweet'][i] = re.sub(r'(=|(:\-?))D+', ' excited ', t0['tweet'][i])# :D :-D =D
	t0['tweet'][i] = re.sub(r'\-(\.|_)+\-', ' annoyed ', t0['tweet'][i])# -___- -.-
	t0['tweet'][i] = re.sub(r'(o[_\.]+[O0])|([O0][_\.]+o)', ' WTF ', t0['tweet'][i])# o_0 0__o
	t0['tweet'][i] = re.sub(r'(0_0)|(O_O)', ' surprised ', t0['tweet'][i])# 0_0
	t0['tweet'][i] = re.sub(r':\-?(o|O)', ' shock ', t0['tweet'][i])# :O
	t0['tweet'][i] = re.sub(r':\-?/', ' frustrated ', t0['tweet'][i])# :/
	t0['tweet'][i] = re.sub(r'T_T', ' cry ', t0['tweet'][i])# T_T
	t0['tweet'][i] = re.sub(r'[xX]_[xX]', ' dead ', t0['tweet'][i])# x_x
	t0['tweet'][i] = re.sub(r':[pP]', ' laugh ', t0['tweet'][i])# :p
	t0['tweet'][i] = re.sub(r'[xX]D', ' LOL ', t0['tweet'][i])# LOL
	t0['tweet'][i] = re.sub(r'w/', ' with ', t0['tweet'][i])# w/
	t0['tweet'][i] = re.sub(r'[bB]4', ' before ', t0['tweet'][i])# b4 -> before
	t0['tweet'][i] = re.sub(r'\b(U|u)\b', 'you', t0['tweet'][i])# u U
	t0['tweet'][i] = re.sub(r'&\w+;', ' ', t0['tweet'][i])# html flag
	t0['tweet'][i] = re.sub(r'[!\?,]+', ' ', t0['tweet'][i])# !?,
	t0['tweet'][i] = re.sub(r'\-', ' ', t0['tweet'][i])# :
	t0['tweet'][i] = re.sub(r'\.{2,}', ' ', t0['tweet'][i])# ....
	t0['tweet'][i] = re.sub(r'\s[:|/\\]+\s', ' ', t0['tweet'][i])#  : \ / |
	t0['tweet'][i] = re.sub(r'"', ' ', t0['tweet'][i])#  "
	# text = nltk.word_tokenize(t0['tweet'][i])
	# t0['tweet'][i] = " ".join([stem(x.lower()) for x in text])#
	
	
t0.to_csv('../data/mytrain2.csv', index=False)
t1.to_csv('../data/mytest2.csv', index=False)