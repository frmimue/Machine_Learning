import numpy as np

class NaiveBayesClassifier:
    
    def fit(self, X, y):

        labels = np.unique(y)
        self.labels = labels

        T1 = X[y == labels[0]]
        T2 = X[y == labels[1]]

		#calculate mean
        u1 = np.sum(T1, axis=0)/float(T1.shape[0])
        u2 = np.sum(T2, axis=0)/float(T2.shape[0])

		#calculate variance
        o1 = np.sum((T1 - u1) ** 2, axis=0)/float(T1.shape[0])
        o2 = np.sum((T2 - u2) ** 2, axis=0)/float(T2.shape[0])

        self.u1 = u1
        self.u2 = u2
        self.o1 = o1
        self.o2 = o2

    def predict(self, X):
        probability_negative = 1.0/(np.sqrt(2.0 * np.pi * self.o1)) * np.exp(-((X - self.u1) ** 2 / (2.0 * self.o1)))
        probability_positive = 1.0/(np.sqrt(2.0 * np.pi * self.o2)) * np.exp(-((X - self.u2) ** 2 / (2.0 * self.o2)))

        if probability_negative.ndim > 1:
            probability_negative = np.prod(probability_negative, axis=1)
            probability_positive = np.prod(probability_positive, axis=1)

        y = self.labels[(probability_positive > probability_negative).astype(int)];

        return y

