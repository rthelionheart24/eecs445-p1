"""EECS 445 - Winter 2022.

Project 1
"""

from locale import normalize
import pandas as pd
import numpy as np
import itertools
import string

from sklearn.svm import SVC, LinearSVC
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics
from sklearn.feature_extraction.text import TfidfVectorizer
from matplotlib import pyplot as plt


from helper import *

import warnings
from sklearn.exceptions import ConvergenceWarning

warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=ConvergenceWarning)

np.random.seed(445)


def extract_word(input_string):
    """Preprocess review into list of tokens.

    Convert input string to lowercase, replace punctuation with spaces, and split along whitespace.
    Return the resulting array.

    E.g.
    > extract_word("I love EECS 445. It's my favorite course!")
    > ["i", "love", "eecs", "445", "it", "s", "my", "favorite", "course"]

    Input:
        input_string: text for a single review
    Returns:
        a list of words, extracted and preprocessed according to the directions
        above.
    """
    for r in string.punctuation:
        input_string = input_string.replace(r, " ")

    return input_string.lower().split()


def extract_dictionary(df):
    """Map words to index.

    Reads a pandas dataframe, and returns a dictionary of distinct words
    mapping from each distinct word to its index (ordered by when it was
    found).

    E.g., with input:
        | text                          | label | ... |
        | It was the best of times.     |  1    | ... |
        | It was the blurst of times.   | -1    | ... |

    The output should be a dictionary of indices ordered by first occurence in
    the entire dataset:
        {
           it: 0,
           was: 1,
           the: 2,
           best: 3,
           of: 4,
           times: 5,
           blurst: 6
        }
    The index should be autoincrementing, starting at 0.

    Input:
        df: dataframe/output of load_data()
    Returns:
        a dictionary mapping words to an index
    """
    word_dict = {}
    for t in df["text"]:
        unique_list = extract_word(t)
        for w in unique_list:
            if w not in word_dict:
                word_dict[w] = len(word_dict)

    return word_dict


def generate_feature_matrix(df, word_dict):
    """Create matrix of feature vectors for dataset.

    Reads a dataframe and the dictionary of unique words to generate a matrix
    of {1, 0} feature vectors for each review.  Use the word_dict to find the
    correct index to set to 1 for each place in the feature vector. The
    resulting feature matrix should be of dimension (# of reviews, # of words
    in dictionary).

    Input:
        df: dataframe that has the text and labels
        word_dict: dictionary of words mapping to indices
    Returns:
        a numpy matrix of dimension (# of reviews, # of words in dictionary)
    """
    number_of_reviews = df.shape[0]
    number_of_words = len(word_dict)
    feature_matrix = np.zeros((number_of_reviews, number_of_words))
    for i in range(number_of_reviews):
        unique_list = extract_word(df["text"][i])
        for w in unique_list:
            if w in word_dict.keys():
                feature_matrix[i][word_dict[w]] = 1
    return feature_matrix


def performance(y_true, y_pred, metric="accuracy"):
    """Calculate performance metrics.

    Performance metrics are evaluated on the true labels y_true versus the
    predicted labels y_pred.

    Input:
        y_true: (n,) array containing known labels
        y_pred: (n,) array containing predicted scores
        metric: string specifying the performance metric (default='accuracy'
                 other options: 'f1-score', 'auroc', 'precision', 'sensitivity',
                 and 'specificity')
    Returns:
        the performance as an np.float64
    """
    # This is an optional but very useful function to implement.
    # See the sklearn.metrics documentation for pointers on how to implement
    # the requested metrics

    if (metric == "accuracy"):
        return metrics.accuracy_score(y_true, y_pred, normalize=True)
    elif (metric == "f1-score"):
        return metrics.f1_score(y_true, y_pred)
    elif (metric == "auroc"):
        return metrics.roc_auc_score(y_true, y_pred)
    elif (metric == "precision"):
        return metrics.precision_score(y_true, y_pred)
    elif (metric == "sensitivity"):
        matrix = metrics.confusion_matrix(
            y_true, y_pred, labels=[1, -1])
        return matrix[0][0]/(matrix[0][0]+matrix[0][1])
    elif (metric == "specificity"):
        matrix = metrics.confusion_matrix(
            y_true, y_pred, labels=[1, -1])
        return matrix[1][1]/(matrix[1][1]+matrix[1][0])


def cv_performance(clf, X, y, k=5, metric="accuracy"):
    """Split data into k folds and run cross-validation.

    Splits the data X and the labels y into k-folds and runs k-fold
    cross-validation: for each fold i in 1...k, trains a classifier on
    all the data except the ith fold, and tests on the ith fold.
    Calculates and returns the k-fold cross-validation performance metric for
    classifier clf by averaging the performance across folds.
    Input:
        clf: an instance of SVC()
        X: (n,d) array of feature vectors, where n is the number of examples
           and d is the number of features
        y: (n,) array of binary labels {1,-1}
        k: an int specifying the number of folds (default=5)
        metric: string specifying the performance metric (default='accuracy'
             other options: 'f1-score', 'auroc', 'precision', 'sensitivity',
             and 'specificity')
    Returns:
        average 'test' performance across the k folds as np.float64
    """
    # HINT: You may find the StratifiedKFold from sklearn.model_selection
    # to be useful
    skf = StratifiedKFold(n_splits=k, shuffle=False)

    # Put the performance of the model on each fold in the scores array
    scores = []
    for train, test in skf.split(X, y):
        clf.fit(X[train], y[train])
        if (metric == "auroc"):
            y_pred = clf.decision_function(X[test])
        else:
            y_pred = clf.predict(X[test])
        p = performance(y[test], y_pred, metric)
        scores.append(p)
    # print("AVG: ", np.mean(scores))
    return np.array(scores).mean()


def select_param_linear(
    X, y, k=5, metric="accuracy", C_range=[], loss="hinge", penalty="l2", dual=True
):
    """Search for hyperparameters of linear SVM with best k-fold CV performance.

    Sweeps different settings for the hyperparameter of a linear-kernel SVM,
    calculating the k-fold CV performance for each setting on X, y.
    Input:
        X: (n,d) array of feature vectors, where n is the number of examples
        and d is the number of features
        y: (n,) array of binary labels {1,-1}
        k: int specifying the number of folds (default=5)
        metric: string specifying the performance metric (default='accuracy',
             other options: 'f1-score', 'auroc', 'precision', 'sensitivity',
             and 'specificity')
        C_range: an array with C values to be searched over
        loss: string specifying the loss function used (default="hinge",
             other option of "squared_hinge")
        penalty: string specifying the penalty type used (default="l2",
             other option of "l1")
        dual: boolean specifying whether to use the dual formulation of the
             linear SVM (set True for penalty "l2" and False for penalty "l1"ß)
    Returns:
        the parameter value for a linear-kernel SVM that maximizes the
        average 5-fold CV performance.
    """
    # TODO: Implement this function
    # HINT: You should be using your cv_performance function here
    # to evaluate the performance of each SVM
    scores = []
    for c in C_range:
        clf = LinearSVC(C=c, penalty=penalty, dual=dual,
                        loss=loss, random_state=445)
        score = cv_performance(clf, X, y, k, metric)
        scores.append(score)
    # print("Metric: " + metric)
    # print("Best c: " + str(C_range[scores.index(max(scores))]))
    # print("Best CV Score: " + str(max(scores)))
    return C_range[scores.index(max(scores))]


def plot_weight(X, y, penalty, C_range, loss, dual):
    """Create a plot of the L0 norm learned by a classifier for each C in C_range.

    Input:
        X: (n,d) array of feature vectors, where n is the number of examples
        and d is the number of features
        y: (n,) array of binary labels {1,-1}
        penalty: penalty to be forwarded to the LinearSVC constructor
        C_range: list of C values to train a classifier on
        loss: loss function to be forwarded to the LinearSVC constructor
        dual: whether to solve the dual or primal optimization problem, to be
            forwarded to the LinearSVC constructor
    Returns: None
        Saves a plot of the L0 norms to the filesystem.
    """
    norm0 = []
    # TODO: Implement this part of the function
    # Here, for each value of c in C_range, you should
    # append to norm0 the L0-norm of the theta vector that is learned
    # when fitting an L2- or L1-penalty, degree=1 SVM to the data (X, y)
    for c in C_range:
        clf = LinearSVC(C=c, penalty=penalty, dual=dual,
                        loss=loss, random_state=445)
        clf.fit(X, y)
        norm0.append(np.count_nonzero(clf.coef_))

    plt.plot(C_range, norm0)
    plt.xscale("log")
    plt.legend(["L0-norm"])
    plt.xlabel("Value of C")
    plt.ylabel("Norm of theta")
    plt.title("Norm-" + penalty + "_penalty.png")
    plt.savefig("Norm-" + penalty + "_penalty.png")
    plt.close()


def select_param_quadratic(X, y, k=5, metric="accuracy", param_range=[]):
    """Search for hyperparameters of quadratic SVM with best k-fold CV performance.

    Sweeps different settings for the hyperparameters of an quadratic-kernel SVM,
    calculating the k-fold CV performance for each setting on X, y.
    Input:
        X: (n,d) array of feature vectors, where n is the number of examples
           and d is the number of features
        y: (n,) array of binary labels {1,-1}
        k: an int specifying the number of folds (default=5)
        metric: string specifying the performance metric (default='accuracy'
                 other options: 'f1-score', 'auroc', 'precision', 'sensitivity',
                 and 'specificity')
        param_range: a (num_param, 2)-sized array containing the
            parameter values to search over. The first column should
            represent the values for C, and the second column should
            represent the values for r. Each row of this array thus
            represents a pair of parameters to be tried together.
    Returns:
        The parameter values for a quadratic-kernel SVM that maximize
        the average 5-fold CV performance as a pair (C,r)
    """
    # Hint: This will be very similar to select_param_linear, except
    # the type of SVM model you are using will be different...
    scores = []
    for p in param_range:
        clf = SVC(kernel="poly", degree=2,
                  C=p[0], coef0=p[1], gamma="auto", random_state=445)
        score = cv_performance(clf, X, y, k, metric)
        scores.append(score)
        print("CV Score: " + str(p[0]) + ", " + str(p[1]) + " " + str(score))
    best_score = param_range[scores.index(max(scores))]
    best_C_val, best_r_val = best_score[0], best_score[1]
    print("Metric: " + metric)
    print("Best c, r: " + str(best_C_val) + ", " + str(best_r_val))
    print("Best CV Score: " + str(max(scores)))
    return best_C_val, best_r_val


def main():
    # Read binary data
    # NOTE: READING IN THE DATA WILL NOT WORK UNTIL YOU HAVE FINISHED
    #       IMPLEMENTING generate_feature_matrix AND extract_dictionary
    X_train, Y_train, X_test, Y_test, dictionary_binary = get_split_binary_data(
        fname="data/dataset.csv"
    )
    IMB_features, IMB_labels, IMB_test_features, IMB_test_labels = get_imbalanced_data(
        dictionary_binary, fname="data/dataset.csv"
    )

    # TODO: Questions 3, 4, 5
    # # * Pass debug
    # # 3A
    # print(extract_word("BEST book ever! It\'s great"))
    # # * Pass debug
    # # 3B
    # print(len(dictionary_binary))
    # # * Pass debug
    # # 3C
    # avg_non_zero_feature = np.sum(X_train > 0, axis=1).mean()
    # print(avg_non_zero_feature)
    # num_apperances = np.sum(X_train, axis=0)
    # print([k for k, v in dictionary_binary.items()
    #       if v == np.argmax(num_apperances)])

    # #* Pass debug
    # # 4.1b
    # print("Picking the best C value using accuracy as the metrics: ")
    # print(select_param_linear(X_train, Y_train, k=5, metric="accuracy",
    #       C_range=[1e-3, 1e-2, 1e-1, 1e0, 1e+1, 1e+2, 1e+3]))
    # print("============================================================")
    # print("Picking the best C value using f1-score as the metrics: ")
    # print(select_param_linear(X_train, Y_train, k=5, metric="f1-score",
    #       C_range=[1e-3, 1e-2, 1e-1, 1e0, 1e+1, 1e+2, 1e+3]))
    # print("============================================================")
    # print("Picking the best C value using auroc as the metrics: ")
    # print(select_param_linear(X_train, Y_train, k=5, metric="auroc",
    #       C_range=[1e-3, 1e-2, 1e-1, 1e0, 1e+1, 1e+2, 1e+3]))
    # print("============================================================")
    # print("Picking the best C value using precision as the metrics: ")
    # print(select_param_linear(X_train, Y_train, k=5, metric="precision",
    #                           C_range=[1e-3, 1e-2, 1e-1, 1e0, 1e+1, 1e+2, 1e+3]))
    # print("============================================================")
    # print("Picking the best C value using sensitivity as the metrics: ")
    # print(select_param_linear(X_train, Y_train, k=5, metric="sensitivity",
    #       C_range=[1e-3, 1e-2, 1e-1, 1e0, 1e+1, 1e+2, 1e+3]))
    # print("============================================================")
    # print("Picking the best C value using specificity as the metrics: ")
    # print(select_param_linear(X_train, Y_train, k=5, metric="specificity",
    #                           C_range=[1e-3, 1e-2, 1e-1, 1e0, 1e+1, 1e+2, 1e+3]))
    # print("============================================================")

    # #* Pass debug
    # # 4.1c
    # y_pred = LinearSVC(C=1, loss="hinge", penalty="l2", random_state=445).fit(
    #     X_train, Y_train).predict(X_test)
    # perf = performance(Y_test, y_pred, metric="accuracy")
    # print(perf)

    # y_pred = LinearSVC(C=1, loss="hinge", penalty="l2", random_state=445).fit(
    #     X_train, Y_train).predict(X_test)
    # perf = performance(Y_test, y_pred, metric="f1-score")
    # print(perf)

    # y_pred = LinearSVC(C=0.1, loss="hinge", penalty="l2", random_state=445).fit(
    #     X_train, Y_train).decision_function(X_test)
    # perf = performance(Y_test, y_pred, metric="auroc")
    # print(perf)

    # y_pred = LinearSVC(C=0.01, loss="hinge", penalty="l2", random_state=445).fit(
    #     X_train, Y_train).predict(X_test)
    # perf = performance(Y_test, y_pred, metric="precision")
    # print(perf)

    # y_pred = LinearSVC(C=1, loss="hinge", penalty="l2", random_state=445).fit(
    #     X_train, Y_train).predict(X_test)
    # perf = performance(Y_test, y_pred, metric="sensitivity")
    # print(perf)
    # perf = performance(Y_test, y_pred, metric="specificity")
    # print(perf)

    # #* Pass debug
    # # 4.1d
    # plot_weight(X_train, Y_train, penalty="l2", C_range=[
    #             1e-3, 1e-2, 1e-1, 1e0, 1e+1, 1e+2, 1e+3], dual=True, loss='hinge')

    # #* Pass debug
    # # 4.1e
    # clf = LinearSVC(C=0.1, penalty="l2", dual=True, loss="hinge",
    #                 random_state=445)
    # clf.fit(X_train, Y_train)
    # coef_vs_words = np.concatenate(
    #     [np.sort(clf.coef_)[0][:5], np.sort(clf.coef_)[0][len(clf.coef_[0])-5:]])
    # idx_coef_vs_words = np.concatenate(
    #     [np.argsort(clf.coef_)[0][:5], np.argsort(clf.coef_)[0][len(clf.coef_[0])-5:]])
    # words = []
    # for i in idx_coef_vs_words:
    #     words.append(list(dictionary_binary.keys())[
    #                  list(dictionary_binary.values()).index(i)])

    # plt.bar(words, coef_vs_words)
    # plt.xlabel("Words")
    # plt.ylabel("Coefficient")
    # plt.xticks(fontsize=5)
    # plt.title("Coefficient_vs_word.png")
    # plt.savefig("Coefficient_vs_word.png")
    # plt.close()

    # #* Pass debug
    # # 4.2a
    # best_c = select_param_linear(X_train, Y_train, k=5, metric="auroc", C_range=[
    #                              1e-3, 1e-2, 1e-1, 1e0], loss="squared_hinge", penalty="l1", dual=False)
    # cv_score = cv_performance(LinearSVC(C=best_c, penalty="l1", dual=False,
    #                                     loss="squared_hinge", random_state=445), X_test, Y_test, k=5, metric="auroc")

    # score = performance(Y_test, LinearSVC(C=best_c, penalty="l1", dual=False, loss="squared_hinge").fit(
    #     X_train, Y_train).decision_function(X_test), metric="auroc")
    # print("Best C value: ", best_c)
    # print("CV score: ", cv_score)
    # print("Score: ", score)

    # #* Pass debug
    # # 4.2b
    # plot_weight(X_train, Y_train, penalty="l1", C_range=[
    #             1e-3, 1e-2, 1e-1, 1e0, 1e+1, 1e+2, 1e+3], dual=False, loss='squared_hinge')

    # #* Pass debug
    # # 4.3a
    # print("Grid Search")
    # C_range = [1e-2, 1e-1, 1e0, 1e+1, 1e+2, 1e+3]
    # combination_C_r = []
    # for c in C_range:
    #     for r in C_range:
    #         combination_C_r.append([c, r])

    # best_c, best_r = select_param_quadratic(
    #     X_train, Y_train, k=5, metric="auroc", param_range=combination_C_r)

    # print("Test performance: ", performance(Y_test, SVC(C=best_c, coef0=best_r, kernel='poly',
    #                                                     degree=2, gamma='auto').fit(X_train, Y_train).decision_function(X_test), metric="auroc"))

    # print("Random Search")
    # combination_C_r = []
    # for i in range(25):
    #     element = [10**(np.random.uniform(-2, 3)), 10 **
    #                (np.random.uniform(-2, 3))]
    #     combination_C_r.append(element)
    # best_c, best_r = select_param_quadratic(
    #     X_train, Y_train, k=5, metric="auroc", param_range=combination_C_r)

    # print("Test performance: ", performance(Y_test, SVC(C=best_c, coef0=best_r, kernel='poly',
    #                                                     degree=2, gamma='auto').fit(X_train, Y_train).decision_function(X_test), metric="auroc"))

    # # 5.1c
    # clf = LinearSVC(C=0.01, loss="hinge", penalty="l2",
    #                 class_weight={-1: 1, 1: 10}, random_state=445).fit(X_train, Y_train)

    # print(performance(Y_test, clf.predict(X_test), metric="accuracy"))
    # print(performance(Y_test, clf.predict(X_test), metric="f1-score"))
    # print(performance(Y_test, clf.decision_function(X_test), metric="auroc"))
    # print(performance(Y_test, clf.predict(X_test), metric="precision"))
    # print(performance(Y_test, clf.predict(X_test), metric="sensitivity"))
    # print(performance(Y_test, clf.predict(X_test), metric="specificity"))

    # # 5.2
    # clf = LinearSVC(C=0.01, loss="hinge", penalty="l2",
    #                 class_weight={-1: 1, 1: 10}, random_state=445).fit(IMB_features, IMB_labels)

    # print(performance(IMB_test_labels, clf.predict(IMB_test_features), metric="accuracy"))
    # print(performance(IMB_test_labels, clf.predict(IMB_test_features), metric="f1-score"))
    # print(performance(IMB_test_labels, clf.decision_function(IMB_test_features), metric="auroc"))
    # print(performance(IMB_test_labels, clf.predict(IMB_test_features), metric="precision"))
    # print(performance(IMB_test_labels, clf.predict(
    #     IMB_test_features), metric="sensitivity"))
    # print(performance(IMB_test_labels, clf.predict(IMB_test_features), metric="specificity"))

    # 5.3a
    # neg = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    # pos = neg
    # comb = []
    # perf = []
    # for n in neg:
    #     for p in pos:
    #         weight = {-1: n, 1: p}
    #         clf = LinearSVC(C=0.01, penalty="l2", loss="hinge", class_weight=weight,
    #                         random_state=445).fit(IMB_features, IMB_labels)
    #         print(p, n,  performance(IMB_test_labels,
    #               clf.decision_function(IMB_test_features), metric="auroc"))
    #         comb.append([p, n])
    #         perf.append(performance(IMB_test_labels,
    #                                 clf.decision_function(IMB_test_features), metric="auroc"))
    # print(comb[perf.index(max(perf))])
    # print(max(perf))

    # clf = LinearSVC(C=0.01, penalty="l2", loss="hinge", class_weight={-1: 10, 1: 3},
    #                 random_state=445).f   it(IMB_features, IMB_labels).fit(IMB_features, IMB_labels)

    # print(performance(IMB_test_labels, clf.predict(
    #     IMB_test_features), metric="accuracy"))
    # print(performance(IMB_test_labels, clf.predict(
    #     IMB_test_features), metric="f1-score"))
    # print(performance(IMB_test_labels, clf.decision_function(
    #     IMB_test_features), metric="auroc"))
    # print(performance(IMB_test_labels, clf.predict(
    #     IMB_test_features), metric="precision"))
    # print(performance(IMB_test_labels, clf.predict(
    #     IMB_test_features), metric="sensitivity"))
    # print(performance(IMB_test_labels, clf.predict(
    #     IMB_test_features), metric="specificity"))

    # # 5.4
    # metrics=["Accuracy", "F1-score", "AUROC", "Precision", "Sensitivity", "Specificity"]
    # n1p1 = [0.7995079950799509, 0.8885850991114149,
    #         0.9484568192543652, 0.7995079950799509, 1.0, 0.0]
    # n10p3 = [0.8597785977859779, 0.9043624161073825, 0.9657291175082586,
    #          0.9944649446494465, 0.8292307692307692, 0.9815950920245399]
    
    # plt.plot(metrics, n1p1, label="W_n=1, W_p=1")
    # plt.plot(metrics, n10p3, label="W_n=10, W_p=3")
    # plt.xlabel("Metrics")
    # plt.ylabel("Performance")
    # plt.legend()
    # plt.savefig("ROC.png")
    
    # Read multiclass data
    # TODO: Question 6: Apply a classifier to heldout features, and then use
    #       generate_challenge_labels to print the predicted labels

    (multiclass_features,
     multiclass_labels,
     multiclass_dictionary) = get_multiclass_training_data()

    heldout_features = get_heldout_reviews(multiclass_dictionary)
    
    print(multiclass_features)
    
    df = pd.read_csv()

    vectorizer = TfidfVectorizer(strip_accents="unicode")
    X=vectorizer.fit_transform(multiclass_features)

    print()

if __name__ == "__main__":
    main()
