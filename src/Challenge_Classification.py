import numpy as np
import csv
import unittest
from NaiveBayesClassifier import NaiveBayesClassifier


reader = csv.reader(open("../data/training_classification_regression_2015.csv"), delimiter=',');
data = np.asarray(list(reader))
print data[0]
data = np.delete(data, 0, axis=0)

training_data = np.array(data[:, 0:11], dtype=float)
result_data = data[:, 12]

classifier = NaiveBayesClassifier()

classifier.fit(training_data, result_data)

print classifier.u1, classifier.o1;
print classifier.u2, classifier.o2;

prediction = classifier.predict(training_data)

accuracy = np.sum(prediction == result_data)/float(prediction.size);

print accuracy;

reader = csv.reader(open("C:/Users/frmim/Downloads/challenge_public_test_classification_regression_2015.csv"), delimiter=',');
data = np.asarray(list(reader))
print data[0]
data = np.delete(data, 0, axis=0)

challenge_data = np.array(data[:, 1:12], dtype=float)

prediction = classifier.predict(challenge_data)
print prediction

output_type = np.row_stack((["id", "type"], np.column_stack((data[:,0], prediction))));

writer = csv.writer(open("../data/prediction.csv", "wb"), delimiter=",");
writer.writerows(output_type);