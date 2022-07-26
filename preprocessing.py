import re
import pandas as pd
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from collections import OrderedDict
import tweepy
import csv
import sys
import numpy as np
import datetime
import string, emoji
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist

list_stopwords = stopwords.words('indonesian')
normalizad_word_dict = {}
factory = StemmerFactory()
stemmer = factory.create_stemmer()
normalizad_word = pd.read_excel('E:/TA analisis/Dashboard/uploads/normalisasi.xlsx')

# Preprocessing
def lower(data):
    # Case Folding
    return data.lower()

def remove_punctuation(data):
    # Happy Emoticons
    emoticons_happy = set([
    ':-)', ':)', ';)', ':o)', ':]', ':3', ':c)', ':>', '=]', '8)', '=)', ':}',
    ':^)', ':-D', ':D', ':d', '8-D', '8D', 'x-D', 'xD', 'X-D', 'XD', '=-D', '=D',
    '=-3', '=3', ':-))', ":'-)", ":')", ':*', ':^*', '>:P', ':-P', ':P', 'X-P',
    'x-p', 'xp', 'XP', ':-p', ':p', '=p', ':-b', ':b', '>:)', '>;)', '>:-)',
    '<3'
    ])
 
    # Sad Emoticons
    emoticons_sad = set([
    ':L', ':-/', '>:/', ':S', '>:[', ':@', ':-(', ':[', ':-||', '=L', ':<',
    ':-[', ':-<', '=\\', '=/', '>:(', ':(', '>.<', ":'-(", ":'(", ':\\', ':-c',
    ':c', ':{', '>:\\', ';('
    ])
 
    # All emoticons (happy + sad)
    emoticons = emoticons_happy.union(emoticons_sad)

    data = ' '.join([word for word in data.split() if word not in emoticons])
    data = re.sub(r"b'\@[\w]*", ' ', data)
    data = re.sub(r'\@[\w]+', ' ', data)
    data = re.sub(r"b'[\w]*", ' ', data)
    data = re.sub(r'\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*', ' ', data)
    data = re.sub(r'https\:.*$', " ", data)
    data = re.sub(r'[^\w\s]+', ' ', data)
    data = re.sub(r'[0-9]+', ' ', data)
    data = re.sub(r'\$\w*', ' ', data)
    data = data.lower()
    return data

def remove_stopwords(data):
    list_stopwords = (["b", "xef", "x", "xa","n", "xe", "xf","xb", "xad", "xd","xxxxxxx",
                  "xba", "xc", "k", "xcche", "xd xd xaa xd xd xd xd", "m", "t", "xbb", "f",
                  "xbf","xbd","xbc","xab", "xaa", "c"])
    list_stopwords = set(list_stopwords)
    data = ' '.join([word for word in data.split() if word not in list_stopwords])
    return data

def normalized_term(data):
    for index, row in normalizad_word.iterrows():
        if row[0] not in normalizad_word_dict:
            normalizad_word_dict[row[0]] = row[1]

    data = ' '.join([normalizad_word_dict[term] if term in normalizad_word_dict else term for term in data.split()])
    return data

def stem_text(data):
    data = ' '.join([stemmer.stem(word) for word in data.split()])
    return data

def preprocess_data(data):
    data = remove_punctuation(data)
    data = remove_stopwords(data)
    data = normalized_term(data)
    data = stem_text(data)
    return 
