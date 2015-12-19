import numpy as np
import csv
from LinearRegressor import LinearRegressor

reader = csv.reader(open("../data/training_classification_regression_2015.csv"), delimiter=',')
data = np.asarray(list(reader))
data = np.delete(data, 0, axis=0)

x = np.array(data[:, 0:11:1], dtype=float)
r = np.array(data[:, 11], dtype=float)
r = r.reshape((np.shape(r)[0],1))
#x = np.delete(data, 12, axis=1)
#x = np.delete(x, 11, axis=1)

regressor = LinearRegressor()

x_train = x[0:4000:1, :]
r_train = r[0:4000:1]
regressor.fit(x_train, r_train)

##############################


x_test = x[4000:5001:1, :]
r_test = r[4000:5001:1]

regressor.predict(x_test)

m = regressor.g
#print np.shape(m)
print m[0:50:1, 0]

e = np.subtract(regressor.g, r_test)
#Et = np.sum(np.dot(e, e))
s = np.shape(e)
Et = np.sum(np.dot(e.reshape(s[0]), e.reshape(s[0]))) / float(2*1000)

accurracy = np.sum(np.array(m, dtype=np.int) == r_test) / float(np.shape(r_test)[0])

print "Et", Et
print accurracy
