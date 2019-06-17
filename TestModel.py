# Loading the data set - training data.
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
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(twenty_train.data)
X_train_counts.shape

# TF-IDF, how important a word is to a document in a collection or corpus
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
X_train_tfidf.shape
