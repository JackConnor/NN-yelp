from django.shortcuts import render
from django.http import HttpResponse
from django.http import JsonResponse
from django.shortcuts import render

from django.http import HttpResponse

from sklearn.externals import joblib
import pandas as pd
import numpy as np
import random
import datetime
import nltk
import numpy as np
from nltk.stem import WordNetLemmatizer
from sklearn.linear_model import LogisticRegression
from bs4 import BeautifulSoup
from random import randint


wordnet_lemmatizer = WordNetLemmatizer()
model = joblib.load('./ml_models/first-model.pkl')
stopwords = set(w.rstrip() for w in open('./data_files/stopwords.txt'))

data = pd.read_csv('./data_files/yelp_small.csv').values[:1000]
np.random.shuffle(data)

def index(request):
    # word_map = {}
    # word_map_index = 0
    # for review in data:
    #     review_text_tokens = my_tokenizer(review[5])
    #     for token in review_text_tokens:
    #         if token not in word_map:
    #             word_map[token] = word_map_index
    #             if word_map_index % 10000 == 0:
    #                 print('word map index at: ' + str(word_map_index))
    #             word_map_index += 1

    word_map = np.load('./yelppredictor/ml_models/wordmap.npy').item()
    train = []
    reviewCount = 0
    for review in data:
        if reviewCount % 10 == 0:
            print('another ten reviews: ' + str(reviewCount))
        reviewCount += 1
        reviewText = review[5]
        answerLabel = review[3]
        review_text_tokens = my_tokenizer(reviewText)

        reviewVec = np.zeros(len(word_map) + 1)
        currInd = 0
        for token in review_text_tokens:
            map_ind = word_map[token]
            reviewVec[map_ind] += 1
            currInd += 1
        reviewVec[-1] = answerLabel
        train.append(reviewVec)
    #
    # train = np.array(train)
    # X = train[:, :-1]
    # Y = train[:, -1]
    # print('Got our XY')
    #
    # Xtrain1 = X[:-15000,]
    # Ytrain1 = Y[:-15000,]
    # Xtest1 = X[-15000:,]
    # Ytest1 = Y[-15000:,]
    # print('about to model')
    # model = LogisticRegression()
    # model.fit(Xtrain1, Ytrain1)
    # print('Classification rate: ' + str(model.score(Xtest1, Ytest1)))
    train = np.array(train)
    ind = randint(0, len(train))
    print('ind: ' + str(ind))
    singleReview = train[ind, :-1]
    singleLabel = train[ind, -1]

    print('prediction: ' + str(model.predict([singleReview])))
    print('actual: ' + str(data[ind][3]))
    print('text: ' + str(data[ind][5]))
    return HttpResponse('Actual: ' + str(data[ind][3]) + '. Predicted: ' + str(int(model.predict([singleReview])[0])) + '. Text: ' + str(data[ind][5]))

def my_tokenizer(s):
    s = s.lower()
    tokens = nltk.tokenize.word_tokenize(s)
    tokens = [t for t in tokens if len(t) > 2]
    tokens = [wordnet_lemmatizer.lemmatize(t) for t in tokens]
    tokens = [t for t in tokens if t not in stopwords]
    return tokens

def one_hot(num, numClasses):
    oh = np.zeros(numClasses)
    oh[num - 1] = 1
    return oh
