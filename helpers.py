from sklearn import metrics


def evaluate_and_print(true, pred):
    print("Classification report SKLearn GNB:\n%s\n"
    % (metrics.classification_report(true, pred)))
    print("Confusion matrix SKLearn GNB:\n%s" % metrics.confusion_matrix(true, pred))

def evaluate_and_print_full(true, pred):
    print("Classification report SKLearn GNB:\n%s\n"
    % (metrics.classification_report(true, pred)))
    print("Confusion matrix SKLearn GNB:\n%s" % metrics.confusion_matrix(true, pred))
    print(f'Completeness score: {metrics.completeness_score(true, pred)}')
    print(f'Homogeneity score: {metrics.homogeneity_score(true, pred)}')
    print(f'Adjusted mutual info score {metrics.adjusted_mutual_info_score(true, pred)}')

def unsupervised_scores(true, pred):
    print(f'Completeness score: {metrics.completeness_score(true, pred)}')
    print(f'Homogeneity score: {metrics.homogeneity_score(true, pred)}')
    print(f'Adjusted mutual info score {metrics.adjusted_mutual_info_score(true, pred)}')