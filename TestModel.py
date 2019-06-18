# Loading the data set - training data.
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import SGDClassifier
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

# Using Support Vector Machines and calculate performace, here use SGDC
text_clf_svm = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()),
                         ('clf-svm', SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, max_iter=5, random_state=42))])
text_clf_svm = text_clf_svm.fit(twenty_train.data, twenty_train.target)
predicted_svm = text_clf_svm.predict(twenty_test.data)
np.mean(predicted_svm == twenty_test.target)

# Grid Search
# Here, we are creating a list of parameters for which we would like to do performance tuning.
# All the parameters name start with the classifier name (remember the arbitrary name we gave).
# E.g. vect__ngram_range; here we are telling to use unigram and bigrams and choose the one which is optimal.
# use_idf Enable inverse-document-frequency reweighting.
# clf_alpha Constant that multiplies the regularization term.
parameters = {'vect__ngram_range': [(1, 1), (1, 2)], 'tfidf__use_idf': (
    True, False), 'clf__alpha': (1e-2, 1e-3)}

# Next, we create an instance of the grid search by passing the classifier, parameters
# and n_jobs=-1 which tells to use multiple cores from user machine.
# first use naive
gs_clf = GridSearchCV(text_clf, parameters, n_jobs=-1)
gs_clf = gs_clf.fit(twenty_train.data, twenty_train.target)

# To see the best mean score and the params, run the following code

print(gs_clf.best_score_)
gs_clf.best_params_

# Output for above should be: The accuracy has now increased to ~90.6% for the NB classifier.
# and the corresponding parameters are {‘clf__alpha’: 0.01, ‘tfidf__use_idf’: True, ‘vect__ngram_range’: (1, 2)}.

# similarly doing grid search for SVM
parameters_svm = {'vect__ngram_range': [(1, 1), (1, 2)], 'tfidf__use_idf': (
    True, False), 'clf-svm__alpha': (1e-2, 1e-3)}
gs_clf_svm = GridSearchCV(text_clf_svm, parameters_svm, n_jobs=-1)
gs_clf_svm = gs_clf_svm.fit(twenty_train.data, twenty_train.target)

print(gs_clf_svm.best_score_)
gs_clf_svm.best_params_
