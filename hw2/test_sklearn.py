from sklearn.feature_extraction.text import CountVectorizer
corpus = [
     'This is the first document.',
     'This document is the second document.',
     'And this is the third one.',
     'Is this the first document?']
vectorizer = CountVectorizer()

# https://datascience.stackexchange.com/questions/12321/difference-between-fit-and-fit-transform-in-scikit-learn-models
# To center the data (make it have zero mean and unit standard error), you subtract 
# the mean and then divide the result by the standard deviation.
# You do that on the training set of data. But then you have to apply the same
# transformation to your testing set (e.g. in cross-validation), or to newly obtained
# examples before forecast. But you have to use the same two parameters μ and σ (values)
# that you used for centering the training set.
#
# Hence, every sklearn's transform's fit() just calculates the parameters
# (e.g. μ and σ in case of StandardScaler) and saves them as an internal objects state.
# Afterwards, you can call its transform() method to apply the transformation to a particular set of examples.
#
# fit_transform() joins these two steps and is used for the initial fitting of parameters on the training set x, but it also returns a transformed x′. Internally, it just calls first fit() and then transform() on the same data.

X = vectorizer.fit_transform(corpus)
print(vectorizer.get_feature_names())
print(X.toarray())


print('~'*100)
#==============================================================================
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
le.fit([1, 2, 2, 6])
print(le.classes_)
print(le.transform([1, 1, 2, 6]))
print(le.inverse_transform([0, 0, 1, 2]))
