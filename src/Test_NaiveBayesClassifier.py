import numpy as np
import csv
import unittest
from NaiveBayesClassifier import NaiveBayesClassifier


reader = csv.reader(open("C:/Users/frmim/Downloads/training_classification_regression_2015.csv"), delimiter=',');
#data = np.asarray(list(reader), dtype=np.float)
data = np.asarray(list(reader))
data = np.delete(data, 0, axis=0)

training_data = np.array(data[:, 1], dtype=float)
result_data = data[:, 12]

X_train = training_data[:100];
Y_train = result_data[:100];

X_test = training_data[100:];
Y_test = result_data[100:];

classifier = NaiveBayesClassifier()

classifier.fit(X_train, Y_train)

print classifier.u1, classifier.o1;
print classifier.u2, classifier.o2;

prediction = classifier.predict(X_test)

accuracy = np.sum(prediction == Y_test)/float(prediction.size);

print accuracy;

##########################################

training_data = np.array(data[:, 2], dtype=float)
result_data = data[:, 12]

X_train = training_data[:100];
Y_train = result_data[:100];

X_test = training_data[100:];
Y_test = result_data[100:];

classifier = NaiveBayesClassifier()

classifier.fit(X_train, Y_train)

print classifier.u1, classifier.o1;
print classifier.u2, classifier.o2;

prediction = classifier.predict(X_test)

accuracy = np.sum(prediction == Y_test)/float(prediction.size);

print accuracy;

###########################################

training_data = np.array(data[:, 1:12], dtype=float)
result_data = data[:, 12]

X_train = training_data[:100];
Y_train = result_data[:100];

X_test = training_data[100:];
Y_test = result_data[100:];

classifier = NaiveBayesClassifier()

classifier.fit(X_train, Y_train)

print classifier.u1, classifier.o1;
print classifier.u2, classifier.o2;

prediction = classifier.predict(X_test)

accuracy = np.sum(prediction == Y_test)/float(prediction.size);

print accuracy;