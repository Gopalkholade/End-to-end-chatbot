import os
import nltk
import ssl
import streamlit as st
import random
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


ssl._create_default_https_context = ssl._create_unverified_context
nltk.data.path.append(os.path.abspath("nltk_data"))
nltk.download('punkt')

with open('.\\resources\\intent.pkl', 'rb') as f:
    intents = pickle.loads(f.read())

# create the vectorizer and the classifier
vectorizer = TfidfVectorizer()
clf = LogisticRegression(random_state=0, max_iter=10000)

# preprocess the data 
tags = []
patterns = []
for intent in intents:
    for pattern in intent['patterns']:
        tags.append(intent['tag'])
        patterns.append(pattern)

# training the model
x = vectorizer.fit_transform(patterns)
y=tags
clf.fit(x,y)

def chatbot(input_text):
    input_text = vectorizer.transform([input_text])
    tag = clf.predict(input_text)[0]
    for intent in intents:
        if intent['tag']==tag:
            response = random.choice(intent['responses'])
            return response

with open('.\\resources\\clf.pkl', 'wb') as f:
    f.write(pickle.dumps(clf))

with open('.\\resources\\vec.pkl', 'wb') as f:
    f.write(pickle.dumps(vectorizer))
