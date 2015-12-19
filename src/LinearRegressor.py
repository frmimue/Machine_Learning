#!/usr/bin/python

import numpy as np

class LinearRegressor(object):

	def __init__():

	def fit(self, x, r):
		alfa = 0.1
		Ec = float('inf')
		x0 = np.ones(5000) 
		W = np.ones(10, 1)
		
		x = np.insert(x, [0], x0, axis=1)	
	
		while 1: 
			h = np.dot(x, W)		
			e = np.subtract(h, r)

			Ep = Ec
			Ec = np.sum(np.dot(e, e))

			if Ec > Ep:
				break

			f = np.dot(e.T, x)
			W = np.subtract(W, multiply(alfa, f))

		self.Wopt = W
		
	def predict(self, x):

		x0 = np.ones(5000) 
		x = np.insert(x, [0], x0, axis=1)	
		
		h = np.dot(x, self.Wopt)

		self.g = np.around(h)		
		
