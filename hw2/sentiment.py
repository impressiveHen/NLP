#!/bin/python

import numpy as np

def read_files(tarfname):
    """Read the training and development data from the sentiment tar file.
    The returned object contains various fields that store sentiment data, such as:

    train_data,dev_data: array of documents (array of words)
    train_fnames,dev_fnames: list of filenames of the doccuments (same length as data)
    train_labels,dev_labels: the true string label for each document (same length as data)

    The data is also preprocessed for use with scikit-learn, as:

    count_vec: CountVectorizer used to process the data (for reapplication on new data)
    trainX,devX: array of vectors representing Bags of Words, i.e. documents processed through the vectorizer
    le: LabelEncoder, i.e. a mapper from string labels to ints (stored for reapplication)
    target_labels: List of labels (same order as used in le)
    trainy,devy: array of int labels, one for each document
    """
    import tarfile
    tar = tarfile.open(tarfname, "r:gz")
    trainname = "train.tsv"
    devname = "dev.tsv"
    for member in tar.getmembers():
        if 'train.tsv' in member.name:
            trainname = member.name
        elif 'dev.tsv' in member.name:
            devname = member.name


    class Data: pass
    sentiment = Data()
    print("-- train data")
    sentiment.train_data, sentiment.train_labels = read_tsv(tar,trainname)
    print(len(sentiment.train_data))

    print("-- dev data")
    sentiment.dev_data, sentiment.dev_labels = read_tsv(tar, devname)
    print(len(sentiment.dev_data))
    print("-- transforming data and labels")
    # Convert a collection of text documents to a matrix of token counts
    from sklearn.feature_extraction.text import CountVectorizer
    sentiment.count_vect = CountVectorizer()
    sentiment.trainX = sentiment.count_vect.fit_transform(sentiment.train_data)
    sentiment.devX = sentiment.count_vect.transform(sentiment.dev_data)
    from sklearn import preprocessing
    # Encode labels with value between 0 and n_classes-1.
    sentiment.le = preprocessing.LabelEncoder()
    sentiment.le.fit(sentiment.train_labels)
    sentiment.target_labels = sentiment.le.classes_
    sentiment.trainy = sentiment.le.transform(sentiment.train_labels)
    sentiment.devy = sentiment.le.transform(sentiment.dev_labels)
    tar.close()
    return sentiment


def new_read_files(tarfname,badwords=None,min_df=1,max_df=1.0,ngram_range=(1,1)):

    import tarfile
    tar = tarfile.open(tarfname, "r:gz")
    trainname = "train.tsv"
    devname = "dev.tsv"
    for member in tar.getmembers():
        if 'train.tsv' in member.name:
            trainname = member.name
        elif 'dev.tsv' in member.name:
            devname = member.name
    class Data: pass
    sentiment = Data()
    print("-- train data")
    sentiment.train_data, sentiment.train_labels = read_tsv(tar,trainname)
    print(len(sentiment.train_data))

    print("-- dev data")
    sentiment.dev_data, sentiment.dev_labels = read_tsv(tar, devname)
    print(len(sentiment.dev_data))
    print("-- transforming data and labels")
    # Convert a collection of text documents to a matrix of token counts
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.feature_extraction.text import TfidfTransformer
    from sklearn.feature_extraction.text import TfidfVectorizer
    # stop_words=badwords,min_df=1,max_df=0.4
    sentiment.count_vect = CountVectorizer(stop_words=badwords,min_df=min_df,max_df=max_df,ngram_range=ngram_range)
    sentiment.trainX = sentiment.count_vect.fit_transform(sentiment.train_data)
    sentiment.devX = sentiment.count_vect.transform(sentiment.dev_data)

    # sentiment.tfidf_transformer = TfidfTransformer()
    # sentiment.trainX = sentiment.tfidf_transformer.fit_transform(sentiment.trainX)
    # sentiment.devX = sentiment.tfidf_transformer.transform(sentiment.devX)

    from sklearn import preprocessing
    # Encode labels with value between 0 and n_classes-1.
    sentiment.le = preprocessing.LabelEncoder()
    sentiment.le.fit(sentiment.train_labels)
    sentiment.target_labels = sentiment.le.classes_
    sentiment.trainy = sentiment.le.transform(sentiment.train_labels)
    sentiment.devy = sentiment.le.transform(sentiment.dev_labels)
    tar.close()
    return sentiment


def add_train(sentiment,cls,new_sentence,new_label):
    sentiment.train_data = sentiment.train_data + new_sentence
    sentiment.train_labels = sentiment.train_labels + list(sentiment.le.inverse_transform(new_label))


    from sklearn.feature_extraction.text import CountVectorizer
    #from sklearn.feature_extraction.text import TfidfTransformer

    sentiment.trainX = sentiment.count_vect.fit_transform(sentiment.train_data)
    sentiment.devX = sentiment.count_vect.transform(sentiment.dev_data)

    # sentiment.tfidf_transformer = TfidfTransformer()
    # sentiment.trainX = sentiment.tfidf_transformer.fit_transform(sentiment.trainX)
    # sentiment.devX = sentiment.tfidf_transformer.transform(sentiment.devX)

    from sklearn import preprocessing
    # Encode labels with value between 0 and n_classes-1.
    sentiment.le.fit(sentiment.train_labels)
    sentiment.trainy = sentiment.le.transform(sentiment.train_labels)

    #sentiment.devy = sentiment.le.transform(sentiment.dev_labels)
    new_cls = classify.train_classifier(new_sentiment.trainX, new_sentiment.trainy)
    return new_cls


def read_unlabeled(tarfname, sentiment):
    """Reads the unlabeled data.

    The returned object contains three fields that represent the unlabeled data.

    data: documents, represented as sequence of words
    fnames: list of filenames, one for each document
    X: bag of word vector for each document, using the sentiment.vectorizer
    """
    import tarfile
    tar = tarfile.open(tarfname, "r:gz")
    class Data: pass
    unlabeled = Data()
    unlabeled.data = []

    unlabeledname = "unlabeled.tsv"
    for member in tar.getmembers():
        if 'unlabeled.tsv' in member.name:
            unlabeledname = member.name

    print(unlabeledname)
    tf = tar.extractfile(unlabeledname)
    for line in tf:
        line = line.decode("utf-8")
        text = line.strip()
        unlabeled.data.append(text)

    unlabeled.X = sentiment.count_vect.transform(unlabeled.data)
    print(unlabeled.X.shape)
    tar.close()
    return unlabeled

def read_tsv(tar, fname):
    member = tar.getmember(fname)
    print(member.name)
    tf = tar.extractfile(member)
    data = []
    labels = []
    for line in tf:
        line = line.decode("utf-8")
        (label,text) = line.strip().split("\t")
        labels.append(label)
        data.append(text)
    return data, labels

def write_pred_kaggle_file(unlabeled, cls, outfname, sentiment):
    """Writes the predictions in Kaggle format.

    Given the unlabeled object, classifier, outputfilename, and the sentiment object,
    this function write sthe predictions of the classifier on the unlabeled data and
    writes it to the outputfilename. The sentiment object is required to ensure
    consistent label names.
    """
    yp = cls.predict(unlabeled.X)
    labels = sentiment.le.inverse_transform(yp)
    f = open(outfname, 'w')
    f.write("ID,LABEL\n")
    for i in range(len(unlabeled.data)):
        f.write(str(i+1))
        f.write(",")
        f.write(labels[i])
        f.write("\n")
    f.close()


def write_gold_kaggle_file(tsvfile, outfname):
    """Writes the output Kaggle file of the truth.

    You will not be able to run this code, since the tsvfile is not
    accessible to you (it is the test labels).
    """
    f = open(outfname, 'w')
    f.write("ID,LABEL\n")
    i = 0
    with open(tsvfile, 'r') as tf:
        for line in tf:
            (label,review) = line.strip().split("\t")
            i += 1
            f.write(str(i))
            f.write(",")
            f.write(label)
            f.write("\n")
    f.close()

def write_basic_kaggle_file(tsvfile, outfname):
    """Writes the output Kaggle file of the naive baseline.

    This baseline predicts POSITIVE for all the instances.
    """
    f = open(outfname, 'w')
    f.write("ID,LABEL\n")
    i = 0
    with open(tsvfile, 'r') as tf:
        for line in tf:
            (label,review) = line.strip().split("\t")
            i += 1
            f.write(str(i))
            f.write(",")
            f.write("POSITIVE")
            f.write("\n")
    f.close()

def bad_words(k,tarfname):
    sentiment = read_files(tarfname)
    cls = classify.train_classifier(sentiment.trainX, sentiment.trainy)
    coefficients=cls.coef_[0]
    top_k =np.argsort(coefficients)[-k:]
    bad_words = []
    bottom_k =np.argsort(coefficients)[:k]
    for i in range(len(sentiment.count_vect.get_feature_names())):
        if i in top_k or i in bottom_k:
            continue
        else:
            bad_words.append(sentiment.count_vect.get_feature_names()[i])
    return bad_words


def predict_unlabel(unlabeled, cls, batch):
    predict = cls.predict(unlabeled.X)
    predict_prob = cls.predict_proba(unlabeled.X)
    diff = abs(predict_prob[:,0]-predict_prob[:,1])
    max_index = np.argsort(-diff)[:batch]
    predict = np.array(predict)[max_index].tolist()
    max_sentences = np.array(unlabeled.data)[max_index].tolist()
    unlabeled.data = np.delete(np.array(unlabeled.data),max_index).tolist()
    return max_sentences, predict


# nearest neighbor implementation ---- too slow
# def predict_unlabel(unlabeled, cls, batch, sentiment, threshold):
#     predict = cls.predict(unlabeled.X)
#     predict_prob = cls.predict_proba(unlabeled.X)
#     trainXArr = sentiment.trainX.toarray()
#     diff = abs(predict_prob[:,0]-predict_prob[:,1])
#     max_predict = list()
#     max_sentences = list()
#     max_index =np.argsort(-diff)[:batch]
#     testXArr = unlabeled.X[max_index].toarray()
#
#     for i in range(batch):
#         dist = np.sum(np.abs(trainXArr-testXArr[i]),axis=1)
#         if np.any(dist<threshold):
#             max_predict.append(predict[max_index[i]])
#             max_sentences.append(unlabeled.data[max_index[i]])
#     unlabeled.data = np.delete(np.array(unlabeled.data),max_index).tolist()
#
#     return max_sentences, max_predict

def nearest_neighbor(sentiment,max_sentences,predict,threshold):
    testX = sentiment.count_vect.transform(max_sentences)
    print(testX.shape)
    far = list()
    for i,tX in enumerate(testX):
        diff = np.sum(np.abs(sentiment.trainX.toarray() - tX.toarray()) ,axis=1)
        if not np.any(diff<threshold):
            far.append(i)
    testX = np.delete(testX,far)
    predict = np.delete(np.array(predict),far).tolist()
    max_sentences = sentiment.count_vect.inverse_transform(testX)
    print(len(max_sentences))
    return max_sentences, predict

def train_classifier1(X, y):
    """Train a classifier using the given training data.

    Trains logistic regression on the input data with default parameters.
    """
    from sklearn.linear_model import LogisticRegression
    # ,intercept_scaling=5
    cls = LogisticRegression(random_state=0,solver='saga', max_iter=10000, multi_class='ovr', intercept_scaling=2)
	#cls = LogisticRegression(random_state=0, solver='sag', max_iter=10000, penalty='l2')
    cls.fit(X, y)
    print('hi')
    return cls



if __name__ == "__main__":

    # import warnings filter
    from warnings import simplefilter
    import classify
    # ignore all future warnings
    simplefilter(action='ignore', category=FutureWarning)

    print("Reading data")
    tarfname = "data/sentiment.tar.gz"
    badwords = bad_words(3750,tarfname)
    # , min_df = 1, max_df = 0.31

    new_sentiment = new_read_files(tarfname,max_df=0.31, badwords=badwords, ngram_range=(1,2))
    new_cls = train_classifier1(new_sentiment.trainX, new_sentiment.trainy)
    classify.evaluate(new_sentiment.devX, new_sentiment.devy, new_cls, 'dev')

    # trainCoeff = dict(zip(new_sentiment.count_vect.get_feature_names(), new_cls.coef_.flatten().tolist()))

    unlabeled = read_unlabeled(tarfname,new_sentiment)
    for i,sentence in enumerate(range(15)):
        max_sentence, predict = predict_unlabel(unlabeled, new_cls, 500)
        new_cls = add_train(new_sentiment,new_cls, max_sentence, predict)
        unlabeled.X = new_sentiment.count_vect.transform(unlabeled.data)
        classify.evaluate(new_sentiment.devX, new_sentiment.devy, new_cls, 'dev')
        print(i)

    # testCoeff = dict(zip(new_sentiment.count_vect.get_feature_names(), new_cls.coef_.flatten().tolist()))

    # diffCoeff = dict()
    # change = list()
    #
    # for k in testCoeff:
    #     if k in trainCoeff:
    #         diffCoeff[k] = abs(testCoeff[k]-trainCoeff[k])
    #         if(diffCoeff[k]>0.005):
    #             change.append(k)
    # print(change)

    print("\nReading unlabeled data")
    unlabeled = read_unlabeled(tarfname, new_sentiment)
    print("Writing predictions to a file")
    write_pred_kaggle_file(unlabeled, new_cls, "data/sentiment-pred.csv", new_sentiment)


# if __name__ == "__main__":
#     print("Reading data")
#     tarfname = "data/sentiment.tar.gz"
#     sentiment = read_files(tarfname)
#     print("\nTraining classifier")
#     import classify
#     cls = classify.train_classifier(sentiment.trainX, sentiment.trainy)
#     print("\nEvaluating")
#     classify.evaluate(sentiment.trainX, sentiment.trainy, cls, 'train')
#     classify.evaluate(sentiment.devX, sentiment.devy, cls, 'dev')
#
#     print("\nReading unlabeled data")
#     unlabeled = read_unlabeled(tarfname, sentiment)
#     print("Writing predictions to a file")
#     write_pred_kaggle_file(unlabeled, cls, "data/sentiment-pred.csv", sentiment)
#====================================================================================================================
    #write_basic_kaggle_file("data/sentiment-unlabeled.tsv", "data/sentiment-basic.csv")

    # You can't run this since you do not have the true labels
    # print "Writing gold file"
    # write_gold_kaggle_file("data/sentiment-unlabeled.tsv", "data/sentiment-gold.csv")
