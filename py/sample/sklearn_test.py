#!/usr/bin/env python
# coding=utf-8

import os
import sys

import numpy as np
from sklearn.datasets import load_files
from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction.text import  CountVectorizer
from sklearn.naive_bayes import MultinomialNB

def text_classifly(dataset_dir_name):
    # 加载数据集，切分数据集80%训练，20%测试
    movie_reviews = load_files(dataset_dir_name)  
    doc_terms_train, doc_terms_test, doc_class_train, doc_class_test = train_test_split(movie_reviews.data, movie_reviews.target, test_size = 0.2)
    
    #BOOL型特征下的向量空间模型，注意，测试样本调用的是transform接口
    count_vec = CountVectorizer(binary = True)     
    doc_train_bool = count_vec.fit_transform(doc_terms_train)
    doc_test_bool = count_vec.transform(doc_terms_test)
    
    #调用MultinomialNB分类器
    clf = MultinomialNB().fit(doc_train_bool, doc_class_train)
    doc_class_predicted = clf.predict(doc_test_bool)
    
    print 'Accuracy: ', np.mean(doc_class_predicted == doc_class_test)
    
    
if __name__ == '__main__':
    dataset_dir_name = sys.argv[1]
    text_classifly(dataset_dir_name)