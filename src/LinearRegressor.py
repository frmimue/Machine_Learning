import numpy as np

class LinearRegressor:

	def fit(self, x, y):
		
		alfa = 0.00001
		epsilon = 0.000001
		Ec = float('inf')
		dim = np.shape(x)
		x0 = np.ones((dim[0],1)) 

		# initial value of the parameter vector W
		W = np.array([[-53], [-0.08], [1], [-0.003], [-0.04], [0.8], [-0.005], [0.002], [62], [-0.55], [-0.6], [-0.25]])

		# adding x0=1 for all sample vectors
		x = np.insert(x, [0], x0, axis=1)	
	
		while 1:
			# g = xW = (W.T)x 
			g = np.dot(x, W)
			# e = h - y
			e = np.subtract(g, np.array(y, dtype=np.float))

			Ep = Ec
			Ec = np.sum(np.square(e)) / float(2*dim[0])
			if (Ep - Ec) < epsilon:
				break
			
			# f = ((e.T)x)/N 
			f = np.dot(e.T, x)
			f = np.divide(f, float(dim[0]))
			
			# W(k+1) = W(k) - alpha * f.T
			W = np.subtract(W, np.multiply(alfa, f.T))

		self.Wopt = W
		
	def predict(self, x):
		dim = np.shape(x)
		x0 = np.ones((dim[0],1)) 
		x = np.insert(x, [0], x0, axis=1)	
	
		# g(x) = xW = (W.T)x	
		g = np.dot(x, self.Wopt)

		self.g = np.around(g)
		
		return self.g		

