import numpy as np
import csv
from sklearn import neighbors

reader_training = csv.reader(open("../data/training_classification_regression_2015.csv"), delimiter=',')
data_training = np.asarray(list(reader_training))
data_training = np.delete(data_training, 0, axis=0)
x_train = np.array(data_training[:, 0:11:1], dtype=float)
r_train = np.array(data_training[:, 11], dtype=float)

reader_challenge = csv.reader(open("../data/challenge_public_test_classification_regression_2015.csv"), delimiter=',')
data_challenge = np.asarray(list(reader_challenge))
data_challenge = np.delete(data_challenge, 0, axis=0)
x_test = np.array(data_challenge[:, 1:12:1], dtype=float)

k=39
w='distance'
regressor = neighbors.KNeighborsRegressor(k, weights=w) 
prediction = regressor.fit(x_train, r_train).predict(x_test)
prediction = np.asarray(np.round(prediction), np.int)

output = np.row_stack((["id", "quality"], np.column_stack((data_challenge[:,0], prediction))))
writer = csv.writer(open("../data/prediction_kNNRegressor.csv", "wb"), delimiter=",")
writer.writerows(output)


#x_train = x[0:4000:1, :]
#r_train = r[0:4000:1]
#x_test = x[4000:5001:1, :]
#r_test = r[4000:5001:1]

#k_neighbors = [39]
#best_accurracy = 0

#for k in k_neighbors:
#	for w in ['uniform', 'distance']:
		#print 'k =', k, 'w =', w
#		regressor = neighbors.KNeighborsRegressor(k, weights=w) 
#		prediction = regressor.fit(x_train, r_train).predict(x_test)
#		prediction = np.asarray(np.round(prediction), np.int)
#		accurracy = np.sum(prediction == r_test) / float(prediction.size)
		#if accurracy > best_accurracy:
		#	best_accurracy = accurracy
		#	Kopt = k
		#	Wopt = w
#		print accurracy
	
#print 'Kopt =', Kopt, 'Wopt =', Wopt
#print best_accurracy
