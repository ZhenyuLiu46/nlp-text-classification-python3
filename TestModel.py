# Loading the data set - training data.
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.datasets import fetch_20newsgroups

# You can check the target names (categories) and some data files by following commands.
twenty_train = fetch_20newsgroups(subset='train', shuffle=True)

twenty_train.target_names
print("\n".join(twenty_train.data[0].split("\n")[:4]))

# TEST_Extracting features from text files, using sklearn CountVectorizer
# an example
corpus = [
    'This is the first document.',
    'This document is the second document.',
    'And this is the third one.',
    'Is this the first document?'
]
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(corpus)
print(vectorizer.get_feature_names())
print(X.toarray())
X.shape

# Extracting features from text files, using sklearn CountVectorizer
# using bags of words model
# output: first# number of targets/inputs, second# number of words
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(twenty_train.data)
X_train_counts.shape
print(X_train_counts.shape)

# TF-IDF, how important a word is to a document in a collection or corpus
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
X_train_tfidf.shape
print(X_train_tfidf.shape)

# twenty_train target
print(twenty_train.target.size)

# Machine Learning Algorithms,
# Using Naive Bayes (NB) classifier on training data
clf = MultinomialNB().fit(X_train_tfidf, twenty_train.target)

# Building a pipeline: We can write less code and do all of the above, by building a pipeline as follows:
# The names ‘vect’ , ‘tfidf’ and ‘clf’ are arbitrary but will be used later.
# We will be using the 'text_clf' going forward.
text_clf = Pipeline([('vect', CountVectorizer()),
                     ('tfidf', TfidfTransformer()), ('clf', MultinomialNB())])
text_clf = text_clf.fit(twenty_train.data, twenty_train.target)

# Performance of NB Classifier
twenty_test = fetch_20newsgroups(subset='test', shuffle=True)
predicted = text_clf.predict(twenty_test.data)
np.mean(predicted == twenty_test.target)
