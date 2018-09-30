#!/usr/bin/env python

import string
import pandas as pd
from nltk.corpus import stopwords
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report, confusion_matrix


def text_processor(text):
    msg = [c for c in text if c not in string.punctuation]
    msg = "".join(msg)
    return [word for word in msg.split() if word.lower() not in stopwords.words('english')]


df = pd.read_csv('SMSSpamCollection', delimiter='\t', names=['label', 'feature'])
print(df.info())

feature_train, feature_test, label_train, label_test = train_test_split(df['feature'], df['label'], test_size=.1, random_state=101)

pipleine = Pipeline([
    ('bow', CountVectorizer(analyzer=text_processor)),
    ('tfdif', TfidfTransformer()),
    ('Classfier', SVC(C=1000, gamma=1e-3))])

pipleine.fit(feature_train, label_train)
pred = pipleine.predict(feature_test)

print(classification_report(label_test, pred))

with open('confusion_matrix_svc.txt', 'w') as rn:
    print(confusion_matrix(label_test, pred), file=rn)
