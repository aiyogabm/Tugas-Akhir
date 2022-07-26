from flask import Flask, render_template, jsonify, json, session, \
request, redirect, url_for
from werkzeug.utils import secure_filename
from nltk.sentiment import SentimentAnalyzer
from preprocessing import lower, remove_punctuation, remove_stopwords, normalized_term, stem_text, preprocess_data
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, recall_score, precision_score, confusion_matrix, accuracy_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
import pickle
from datetime import datetime
import pandas as pd
import numpy as np
import pymysql
import time
import nltk
import csv
import os
import io
import base64
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import matplotlib.pyplot as plt
import seaborn as sns

application = Flask(__name__)
application.config['UPLOAD_FOLDER'] = 'E:\\TA analisis\\Dashboard\\uploads'
application.config['SECRET_KEY'] = '1234567890!@#$%^&*()'
application.config['MAX_CONTENT_PATH'] = 10000000

#<-------------------------------------------- DASHBOARD --------------------------------------->
@application.route('/')
def dashboard():
    #Naive Bayes Visualisasi
    data_nb = pd.read_excel('E:/TA analisis/Dashboard/uploads/hasil_nb_full.xlsx')
    data_nb.dropna(axis=0)
    positif_nb = data_nb.Kelas.value_counts().Positif
    negatif_nb = data_nb.Kelas.value_counts().Negatif
    netral_nb = data_nb.Kelas.value_counts().Netral


    #Decision Tree Visualisasi
    data_dc = pd.read_excel('E:/TA analisis/Dashboard/uploads/hasil_dc_full.xlsx')
    data_dc.dropna(axis=0)
    positif_dc = data_dc.Kelas.value_counts().positif
    negatif_dc = data_dc.Kelas.value_counts().negatif
    netral_dc = data_dc.Kelas.value_counts().netral

    return render_template('index.html', positif_nb=positif_nb, negatif_nb=negatif_nb, netral_nb=netral_nb, positif_dc=positif_dc, negatif_dc=negatif_dc, netral_dc=netral_dc)

@application.route('/datatweet', methods=['GET', 'POST'])
def data_tweet():
    data = []
    if request.method == "POST":
        uploaded_file = request.files['filename'] 
        file = os.path.join(application.config['UPLOAD_FOLDER'], uploaded_file.filename)
        uploaded_file.save(file)
        with open(file) as file:
            csv_file = csv.reader(file)
            for row in csv_file:
                data.append(row)
        print(type(data))
    return render_template('datatweet.html', data=data)

#<-------------------------------------------- PREPROCESSING DATA --------------------------------------->

@application.route('/preprocessing', methods=['GET', 'POST'])
def preprocessing():
    data = pd.read_csv('E:/TA analisis/Dashboard/uploads/pln1-8.csv', names=['tanggal', 'tweet'], encoding='latin-1')
    data.dropna()
    data.drop(['tanggal'], axis=1, inplace=True)
    data.drop_duplicates(['tweet'], keep=False, inplace=True)

    data['tweet'] = data['tweet'].map(lambda x: lower(x))
    data['tweet'] = data['tweet'].map(lambda x: remove_punctuation(x))
    data['tweet'] = data['tweet'].map(lambda x: normalized_term(x))
    data['tweet'] = data['tweet'].map(lambda x: remove_stopwords(x))
    data['tweet'] = data['tweet'].map(lambda x: stem_text(x))

    data.to_csv('E:/TA analisis/Dashboard/uploads/hasil_preprocessing.csv', index = False, header=True)

    return render_template('preprocessing.html',tables=[data.to_html()])

@application.route('/pelabelan', methods=['GET', 'POST'])
def pelabelan_manual():
    data_pelabelan = pd.read_excel('E:/TA analisis/Dashboard/uploads/Training.xlsx')
    data_pelabelan.dropna()
    data_pelabelan.drop(['Unnamed: 0'], axis=1, inplace=True)

    return render_template('pelabelan_manual.html', data_pelabelan=[data_pelabelan.to_html(index=False, justify='center', classes=['table-striped', 'table-bordered', 'table-hover', 'table-condensed', 'dt-responsive'], table_id='dataTable')])

#TRAINING
def data(data_testing):
    data_testing = pd.read_excel('E:/TA analisis/Dashboard/uploads/Training.xlsx')
    X = data_testing['tweet']
    y = data_testing['Label']


    X_train, X_test, y_train, y_test = train_test_split(data_testing['tweet'], data_testing['Label'], test_size=0.1, random_state=100)
    return X_train, X_test, y_train, y_test

#NAIVE BAYES
@application.route('/training', methods=['GET', 'POST'])
def tfidf():
    data_testing = pd.read_excel('E:/TA analisis/Dashboard/uploads/Training.xlsx')
    data_testing.dropna(axis=0)
    netral, negatif, positif = data_testing['Label'].value_counts()
    total = positif + negatif + netral

    X_train, X_test, y_train, y_test = data(data_testing)


    vectorizer = TfidfVectorizer()

    X_train = vectorizer.fit_transform(data_testing['tweet'])
    X_test = vectorizer.transform(X_test)

    #Pipeline
    bow_transformer = CountVectorizer().fit(data_testing['tweet'])
    messages_bow = bow_transformer.transform(data_testing['tweet'])

    tfidf_transformer = TfidfTransformer().fit(messages_bow)
    messages_tfidf = tfidf_transformer.transform(messages_bow)

    pipeline = Pipeline([
        ('bow', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('classifier', MultinomialNB())
        ])

    X = np.asarray(data_testing['tweet'])
    pipeline = pipeline.fit(X, np.asarray(data_testing['Label']))
    file_data = pickle.dump(pipeline, open('E:/TA analisis/Dashboard/uploads/vectorizer.pickle', 'wb'))

    return render_template ('Training.html', X_train=X_train, X_test=X_test, total=total, positif=positif, negatif=negatif, netral=netral, pipeline=pipeline)

@application.route('/klasifikasi_nb', methods=['GET', 'POST'])
def klasifikasi_nb():
    import pickle
    vect = pickle.load(open('E:/TA analisis/Dashboard/uploads/vectorizer.pickle', 'rb'))
    klasifikasi = pd.read_excel('E:/TA analisis/Dashboard/uploads/Text_Preprocessing_1-8.xlsx')
    klasifikasi.dropna()
    klasifikasi.drop(['Unnamed: 0'], axis=1, inplace=True)
    klasifikasi = klasifikasi['tweet'].fillna(' ')
    prediction = vect.predict(klasifikasi)

    result = []

    for i in range(len(prediction)):
        if(prediction[i]== 1):
            sentiment = 'Positif'
        elif(prediction[i]== 0):
            sentiment = 'Netral'
        else:
            sentiment = 'Negatif'

        result.append({'tweet':klasifikasi[i],'Label':prediction[i],'Kelas':sentiment})

    klasifikasi = pd.DataFrame(result)
    klasifikasi.to_excel('E:/TA analisis/Dashboard/uploads/hasil_nb_full.xlsx', index = False, header=True)
    
    #Akurasi dan F1 score
    dataset = pd.read_excel('E:/TA analisis/Dashboard/uploads/hasil_nb_full.xlsx')
    
    X = dataset['tweet']
    Y = dataset['Label']
    X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=42)

    vect = TfidfVectorizer(analyzer = "word",min_df=0.0004,max_df=0.115, ngram_range=(1,3))
    vect.fit(X_train) 
    X_train_dtm = vect.transform(X_train)
    X_test_dtm = vect.transform(X_test)

    nbmodel = MultinomialNB(alpha=0.1)
    nbmodel = nbmodel.fit(X_train_dtm,Y_train)
    Y_pred = nbmodel.predict(X_test_dtm)

    akurasi = accuracy_score(Y_test,Y_pred) * 100
    f1 = f1_score(Y_test, Y_pred, average='weighted') * 100
    P = precision_score(Y_test,Y_pred, average='weighted') * 100
    R = recall_score(Y_test,Y_pred, average='weighted') * 100


    return render_template('klasifikasi_nb.html', klasifikasi=[klasifikasi.to_html(index=False, justify='center', classes=['table-striped', 'table-bordered', 'table-hover', 'table-condensed', 'dt-responsive'], table_id='dataTable')], akurasi=akurasi, f1_score=f1, precision=P, recall=R)


#DECISION TREE
@application.route('/training_dc', methods=['GET', 'POST'])
def tfidf_dc():
    data_testing = pd.read_excel('E:/TA analisis/Dashboard/uploads/Training.xlsx')
    data_testing.dropna(axis=0)
    netral, negatif, positif = data_testing['Label'].value_counts()
    total = positif + negatif + netral

    X_train, X_test, y_train, y_test = data(data_testing)
    vectorizer = TfidfVectorizer()
    X_train = vectorizer.fit_transform(data_testing['tweet'])
    X_test = vectorizer.transform(X_test)

    #Pipeline
    bow_transformer = CountVectorizer().fit(data_testing['tweet'])
    messages_bow = bow_transformer.transform(data_testing['tweet'])
    tfidf_transformer = TfidfTransformer().fit(messages_bow)
    messages_tfidf = tfidf_transformer.transform(messages_bow)

    pipeline = Pipeline([
        ('bow', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('classifier', DecisionTreeClassifier())
        ])

    X = np.asarray(data_testing['tweet'])
    pipeline = pipeline.fit(X, np.asarray(data_testing['Label']))
    file_data = pickle.dump(pipeline, open('E:/TA analisis/Dashboard/uploads/training_dc.pickle', 'wb'))

    return render_template ('training_dc.html', X_train=X_train, X_test=X_test, total=total, positif=positif, negatif=negatif, netral=netral, pipeline=pipeline)


@application.route('/klasifikasi_dc', methods=['GET', 'POST'])
def klasifikasi_dc():
    import pickle
    training = pickle.load(open('E:/TA analisis/Dashboard/uploads/training_dc.pickle', 'rb'))

    klasifikasi = pd.read_excel('E:/TA analisis/Dashboard/uploads/Text_Preprocessing_1-8.xlsx')
    klasifikasi.dropna()
    klasifikasi.drop(['Unnamed: 0'], axis=1, inplace=True)
    klasifikasi = klasifikasi['tweet']
    klasifikasi = klasifikasi.fillna(' ')
    prediction = training.predict(np.asarray(klasifikasi))

    result = []

    for i in range(len(prediction)):
        if(prediction[i]==1):
            sentiment = 'positif'
        elif(prediction[i]==0):
            sentiment = 'netral'
        else:
            sentiment = 'negatif'

        result.append({'tweet':klasifikasi[i],'Label':prediction[i],'Kelas':sentiment})

    klasifikasi = pd.DataFrame(result)
    klasifikasi.to_excel('E:/TA analisis/Dashboard/uploads/hasil_dc_full.xlsx', index = False, header=True)

    #Akurasi dan F1 score
    dataset = pd.read_excel('E:/TA analisis/Dashboard/uploads/hasil_dc_full.xlsx')
    
    X = dataset['tweet']
    Y = dataset['Label']
    X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=42)

    vect = TfidfVectorizer(analyzer = "word",min_df=0.0004,max_df=0.115, ngram_range=(1,3))
    vect.fit(X_train) 
    X_train_dtm = vect.transform(X_train)
    X_test_dtm = vect.transform(X_test)

    dcmodel = DecisionTreeClassifier(random_state=0)
    dcmodel = dcmodel.fit(X_train_dtm,Y_train)
    Y_pred = dcmodel.predict(X_test_dtm)

    akurasi = accuracy_score(Y_test,Y_pred) * 100
    f1 = f1_score(Y_test, Y_pred, average='weighted') * 100
    P = precision_score(Y_test,Y_pred, average='weighted') * 100
    R = recall_score(Y_test,Y_pred, average='weighted') * 100


    return render_template('klasifikasi_dc.html', klasifikasi=[klasifikasi.to_html(index=False, justify='center', classes=['table-striped', 'table-bordered', 'table-hover', 'table-condensed', 'dt-responsive'], table_id='dataTable')], akurasi=akurasi, f1_score=f1, precision=P, recall=R)


if __name__ == '__main__':
    application.run(debug=True)