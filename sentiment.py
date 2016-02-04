import sys
import numpy as np
import collections
import sklearn.naive_bayes as naive_bayes
import sklearn.linear_model as linear_model
import nltk
import random
random.seed(0)
from gensim.models.doc2vec import LabeledSentence, Doc2Vec
#nltk.download("stopwords")          # Download the stop words from nltk


# User input path to the train-pos.txt, train-neg.txt, test-pos.txt, and test-neg.txt datasets
if len(sys.argv) != 3:
    print "python sentiment.py <path_to_data> <0|1>"
    print "0 = NLP, 1 = Doc2Vec"
    exit(1)
path_to_data = sys.argv[1]
method = int(sys.argv[2])


def main():
    train_pos, train_neg, test_pos, test_neg = load_data(path_to_data)
    
    if method == 0:
        train_pos_vec, train_neg_vec, test_pos_vec, test_neg_vec = feature_vecs_NLP(train_pos, train_neg, test_pos, test_neg)
        nb_model, lr_model = build_models_NLP(train_pos_vec, train_neg_vec)
    if method == 1:
        train_pos_vec, train_neg_vec, test_pos_vec, test_neg_vec = feature_vecs_DOC(train_pos, train_neg, test_pos, test_neg)
        nb_model, lr_model = build_models_DOC(train_pos_vec, train_neg_vec)
    print "Naive Bayes"
    print "-----------"
    evaluate_model(nb_model, test_pos_vec, test_neg_vec, True)
    print ""
    print "Logistic Regression"
    print "-------------------"
    evaluate_model(lr_model, test_pos_vec, test_neg_vec, True)


def load_data(path_to_dir):
    """
    Loads the train and test set into four different lists.
    """
    train_pos = []
    train_neg = []
    test_pos = []
    test_neg = []
    with open(path_to_dir+"train-pos.txt", "r") as f:
        for i, line in enumerate(f):
            words = [w.lower() for w in line.strip().split() if len(w)>=3]
            train_pos.append(words)
    with open(path_to_dir+"train-neg.txt", "r") as f:
        for line in f:
            words = [w.lower() for w in line.strip().split() if len(w)>=3]
            train_neg.append(words)
    with open(path_to_dir+"test-pos.txt", "r") as f:
        for line in f:
            words = [w.lower() for w in line.strip().split() if len(w)>=3]
            test_pos.append(words)
    with open(path_to_dir+"test-neg.txt", "r") as f:
        for line in f:
            words = [w.lower() for w in line.strip().split() if len(w)>=3]
            test_neg.append(words)

    return train_pos, train_neg, test_pos, test_neg

# Function to remove stopwords and duplicate words in a single sentence
# Returns a dictionary with number of each words as values.
def create_word_map(text_pool, stopwords):
    word_map = {}
    for sentence in text_pool:
        temp = []
        for word in sentence:
            if word not in temp and word not in stopwords:
                temp.append(word)
                if word not in word_map:
                    word_map[word] = 1
                else:
                    word_map[word] += 1
    return word_map

# Function to make a feature vector from feature list
def make_feature_vector(original_list, feature_list):
    temp_list = []
    for w in original_list:
        l_list = []
        for wor in feature_list:
            if wor in w:
                l_list.append(1)
            else:
                l_list.append(0)
        temp_list.append(l_list)
    return temp_list


def feature_vecs_NLP(train_pos, train_neg, test_pos, test_neg):
    """
    Returns the feature vectors for all text in the train and test datasets.
    """
    # English stopwords from nltk
    stopwords = set(nltk.corpus.stopwords.words('english'))
    # Determine a list of words that will be used as features.
    # This list should have the following properties:
    #   (1) Contains no stop words
    #   (2) Is in at least 1% of the positive texts or 1% of the negative texts
    #   (3) Is in at least twice as many postive texts as negative texts, or vice-versa.
    # YOUR CODE HERE

    # Taking out duplicate words in a sentence and stopwords
    positive_words = create_word_map(train_pos+test_pos, stopwords)
    negative_words = create_word_map(train_neg+test_neg, stopwords)

    positive_words_fin = []
    negative_words_fin = []

    # Taking out words with less than 1% number and less than two times other sentiment
    for word, num in positive_words.iteritems():
        flag = 0
        if word in negative_words:
            if num >= 2 * negative_words[word]:
                flag = 1
        else:
            flag = 1
        if num >= 0.01 * len(train_pos + test_pos) and flag == 1:
            positive_words_fin.append(word)

    for word, num in negative_words.iteritems():
        flag = 0
        if word in positive_words:
            if num >= 2 * positive_words[word]:
                flag = 1
        else:
            flag = 1
        if num >= 0.01 * len(train_neg + test_neg) and flag == 1:
            negative_words_fin.append(word)

    # Calling the feature vector method
    final_feature_list = positive_words_fin + negative_words_fin
    train_pos_vec = make_feature_vector(train_pos, final_feature_list)
    train_neg_vec = make_feature_vector(train_neg, final_feature_list)
    test_pos_vec = make_feature_vector(test_pos, final_feature_list)
    test_neg_vec = make_feature_vector(test_neg, final_feature_list)

    # Return the four feature vectors
    return train_pos_vec, train_neg_vec, test_pos_vec, test_neg_vec


# Function to Generate labelled objects and return list of it
def label_sentence(data, tag_label):
    num = 0
    object_list = []
    for line in data:
        tagss = tag_label + "_" + str(num)
        object_list.append(LabeledSentence(words=line, tags=[tagss]))
        num += 1
    return object_list


# Function to generate feature vector given the model and tags
def extract_feature_vector(model, tot_lines, label):
    feature_vec = []

    for line in range(tot_lines):
        taggs = label + "_" + str(line)
        feature_vec.append(model.docvecs[taggs])
    return feature_vec


def feature_vecs_DOC(train_pos, train_neg, test_pos, test_neg):
    """
    Returns the feature vectors for all text in the train and test datasets.
    """
    # Doc2Vec requires LabeledSentence objects as input.
    # Turn the datasets from lists of words to lists of LabeledSentence objects.
    # YOUR CODE HERE

    labeled_train_pos = label_sentence(train_pos, "train_pos")
    labeled_train_neg = label_sentence(train_neg, "train_neg")
    labeled_test_pos = label_sentence(test_pos, "test_pos")
    labeled_test_neg = label_sentence(test_neg, "test_neg")


    # Initialize model
    model = Doc2Vec(min_count=1, window=10, size=100, sample=1e-4, negative=5, workers=4)
    sentences = labeled_train_pos + labeled_train_neg + labeled_test_pos + labeled_test_neg
    model.build_vocab(sentences)

    # Train the model
    # This may take a bit to run 
    for i in range(5):
        print "Training iteration %d" % (i)
        random.shuffle(sentences)
        model.train(sentences)

    # Use the docvecs function to extract the feature vectors for the training and test data
    # YOUR CODE HERE

    train_pos_vec = extract_feature_vector(model, len(train_pos), "train_pos")
    train_neg_vec = extract_feature_vector(model, len(train_neg), "train_neg")
    test_pos_vec = extract_feature_vector(model, len(test_pos), "test_pos")
    test_neg_vec = extract_feature_vector(model, len(test_neg), "test_neg")

    # Return the four feature vectors
    return train_pos_vec, train_neg_vec, test_pos_vec, test_neg_vec


def build_models_NLP(train_pos_vec, train_neg_vec):
    """
    Returns a BernoulliNB and LosticRegression Model that are fit to the training data.
    """
    Y = ["pos"]*len(train_pos_vec) + ["neg"]*len(train_neg_vec)

    # Use sklearn's BernoulliNB and LogisticRegression functions to fit two models to the training data.
    # For BernoulliNB, use alpha=1.0 and binarize=None
    # For LogisticRegression, pass no parameters
    # YOUR CODE HERE

    X = np.array(train_pos_vec + train_neg_vec)

    # Model Fitting
    nb_model = naive_bayes.BernoulliNB(alpha=1.0, binarize=None)
    nb_model.fit(X, Y)

    lr_model = linear_model.LogisticRegression()
    lr_model.fit(X, Y)

    return nb_model, lr_model


def build_models_DOC(train_pos_vec, train_neg_vec):
    """
    Returns a GaussianNB and LosticRegression Model that are fit to the training data.
    """
    Y = ["pos"]*len(train_pos_vec) + ["neg"]*len(train_neg_vec)

    # Use sklearn's GaussianNB and LogisticRegression functions to fit two models to the training data.
    # For LogisticRegression, pass no parameters
    # YOUR CODE HERE
    X = np.array(train_pos_vec + train_neg_vec)

    # Model Fitting
    nb_model = naive_bayes.GaussianNB()
    nb_model.fit(X, Y)

    lr_model = linear_model.LogisticRegression()
    lr_model.fit(X, Y)
    
    return nb_model, lr_model



def evaluate_model(model, test_pos_vec, test_neg_vec, print_confusion=False):
    """
    Prints the confusion matrix and accuracy of the model.
    """
    # Use the predict function and calculate the true/false positives and true/false negative.
    # YOUR CODE HERE

    pos_prediction = model.predict(test_pos_vec)
    neg_prediction = model.predict(test_neg_vec)

    tp, tn, fp, fn = 0, 0, 0, 0

    # Calculation of True Positives and False Negatives
    for word in pos_prediction:
        if word == 'pos':
            tp += 1
        else:
            fn += 1

    # Calculation of True Negatives and False Positives.
    for word1 in neg_prediction:
        if word1 == 'neg':
            tn += 1
        else:
            fp += 1

    accuracy = float(tn+tp)/float(tn+tp+fn+fp)
    if print_confusion:
        print "predicted:\tpos\tneg"
        print "actual:"
        print "pos\t\t%d\t%d" % (tp, fn)
        print "neg\t\t%d\t%d" % (fp, tn)
    print "accuracy: %f" % (accuracy)


if __name__ == "__main__":
    main()
