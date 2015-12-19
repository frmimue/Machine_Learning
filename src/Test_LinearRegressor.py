import numpy as np
import csv
from LinearRegressor import LinearRegressor

reader = csv.reader(open("training.csv"), delimiter=',')
data = np.asarray(list(reader))
data = np.delete(data, 0, axis=0)

x = data[:, 0:11:1]
r = data[:, 11]
#x = np.delete(data, 12, axis=1)
#x = np.delete(x, 11, axis=1)

regressor = LinearRegressor()

x_train = x[0:3000:1, :]
r_train = r[0:3000:1]
regressor.fit(x_train, r_train)

##############################


x_test = x[3000:5001:1, :]
r_test = r[3000:5001:1]

regressor.predict(x_test)

e = np.subtract(regressor.g, r_test)
Et = np.sum(np.dot(e, e))

print Et

