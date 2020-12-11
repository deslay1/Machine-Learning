from sklearn import datasets, metrics
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
import MNIST
import datasets
import helpers

def main():

  # ---------------------SKLEARN DIGITS---------------------
  train_features, test_features, train_labels, test_labels = datasets.sklearn_digits()
  print(train_features[0])
  print(train_features.shape)
  '''
  Classification report SKLearn GNB:
              precision    recall  f1-score   support

           0       1.00      1.00      1.00        45
           1       0.74      0.88      0.81        52
           2       0.96      0.49      0.65        53
           3       0.66      0.85      0.74        54
           4       0.95      0.75      0.84        48
           5       0.98      0.89      0.94        57
           6       0.95      0.98      0.97        60
           7       0.79      0.98      0.87        53
           8       0.61      0.84      0.70        61
           9       0.97      0.58      0.73        57

    accuracy                           0.82       540
    macro avg       0.86      0.83      0.82       540
    weighted avg       0.86      0.82      0.82       540


  Confusion matrix SKLearn GNB:
  [[45  0  0  0  0  0  0  0  0  0]
  [ 0 46  0  0  0  0  0  0  6  0]
  [ 0  6 26  5  0  0  0  0 16  0]
  [ 0  0  0 46  0  0  0  1  6  1]
  [ 0  3  0  0 36  0  2  7  0  0]
  [ 0  1  0  2  0 51  1  2  0  0]
  [ 0  0  1  0  0  0 59  0  0  0]
  [ 0  0  0  0  1  0  0 52  0  0]
  [ 0  5  0  3  0  1  0  1 51  0]
  [ 0  1  0 14  1  0  0  3  5 33]]
  '''

  # ---------------------SKLEARN DIGITS SUMMARIZED---------------------
  #train_features, test_features, train_labels, test_labels = datasets.sklearn_digits_summarized()
  '''
  Classification report SKLearn GNB:
              precision    recall  f1-score   support

           0       0.98      1.00      0.99        45
           1       0.97      0.63      0.77        52
           2       0.95      0.70      0.80        53
           3       0.96      0.48      0.64        54
           4       0.93      0.88      0.90        48
           5       0.98      0.75      0.85        57
           6       0.89      0.98      0.94        60
           7       0.82      0.96      0.89        53
           8       0.54      1.00      0.70        61
           9       0.75      0.82      0.78        57

    accuracy                           0.82       540
    macro avg       0.88      0.82      0.83       540
    weighted avg       0.87      0.82      0.82       540


  Confusion matrix SKLearn GNB:
  [[45  0  0  0  0  0  0  0  0  0]
  [ 0 33  0  0  1  1  3  1 10  3]
  [ 1  1 37  0  0  0  0  0 13  1]
  [ 0  0  0 26  0  0  0  1 16 11]
  [ 0  0  0  0 42  0  2  3  1  0]
  [ 0  0  0  1  1 43  2  3  6  1]
  [ 0  0  1  0  0  0 59  0  0  0]
  [ 0  0  0  0  1  0  0 51  1  0]
  [ 0  0  0  0  0  0  0  0 61  0]
  [ 0  0  1  0  0  0  0  3  6 47]]
 '''

  # ---------------------MNIST LIGHT DIGITS---------------------
  #mnist = MNIST.MNISTData('MNIST_Light/*/*.png')
  #train_features, test_features, train_labels, test_labels = mnist.get_data()
  #mnist.visualize_random()
  '''
  Classification report SKLearn GNB:
              precision    recall  f1-score   support

           0       0.54      0.94      0.69       164
           1       0.71      0.94      0.81       152
           2       0.83      0.50      0.62       155
           3       0.83      0.53      0.65       154
           4       0.75      0.31      0.44       143
           5       0.67      0.16      0.25       141
           6       0.81      0.85      0.83       143
           7       0.83      0.82      0.83       158
           8       0.41      0.64      0.50       132
           9       0.60      0.84      0.70       158

    accuracy                           0.66      1500
    macro avg       0.70      0.65      0.63      1500
    weighted avg       0.70      0.66      0.64      1500


  Confusion matrix SKLearn GNB:
  [[154   0   6   0   1   1   0   0   1   1]
  [  1 143   1   0   0   1   0   1   3   2]
  [ 11   6  77  10   2   1  19   1  27   1]
  [ 32  11   5  82   0   0   2   3  12   7]
  [ 10   1   2   2  45   2   6   9  24  42]
  [ 55   7   0   2   2  22   1   0  47   5]
  [ 10   7   0   0   0   2 122   0   2   0]
  [  2   1   1   2   3   0   0 130   3  16]
  [  5  23   1   1   0   4   1   0  84  13]
  [  3   1   0   0   7   0   0  12   2 133]]
  '''

  gnb = GaussianNB()
  gnb.fit(train_features, train_labels)
  y_pred = gnb.predict(test_features)

  helpers.evaluate_and_print(test_labels, y_pred)

  #mnist.visualize_wrong_class(y_pred, 8)

if __name__ == "__main__": main()