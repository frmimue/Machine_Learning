import numpy as np
import sys

class LinearRegressor:

	def fit(self, x, r):
		alfa = 0.0001
		epsilon = 0.00001
		#lfa = 0.2
		Ec = float('inf')
		dim = np.shape(x)
		x0 = np.ones((dim[0],1)) 
		#W = np.zeros((dim[1]+1, 1))
		W = np.array([[-53], [-0.08], [1], [-0.003], [-0.04], [0.8], [-0.005], [0.002], [62], [-0.55], [-0.6], [-0.25]])
		#print W
		#sys.exit(1)

		x = np.insert(x, [0], x0, axis=1)	
		#print "x", np.shape(x)
		#print x[0:10:1, 0:2:1]
	
		while 1: 
			h = np.dot(x, W)
			#print h[0:50:1,0]
			#print x[0:10:1,0:12:1]
			#for i in range(3000):
			#	if h[i,0] != np.dot(W.reshape(12), x[i,:]):
			#		print "deu ruim!", h[i,0], "!=", np.dot(W.reshape(12), x[i,:])
			#sys.exit(1)
			#print "x", np.sum(x[0, :])
			#print "w", W
			#print "h", h[0,0]
			e = np.subtract(h, np.array(r, dtype=np.float))
			#print h[0:10:1,0] 
			#print r[0:10:1,0] 
			#print e[0:10:1,0]
			#sys.exit(1)
			#print "r", np.shape(r)
			#print "h", np.shape(h)
			#print "e", np.shape(e)

			Ep = Ec
			Ec = np.sum(np.square(e)) / float(2*dim[0])
			print "Ec", Ec
			#sys.exit(1)
			if Ep - Ec < epsilon:
				break

			f = np.dot(e.T, x)
			f = np.divide(dim[0], f)
			#print "eT", np.shape(e)
			#print "x", np.shape(x[:,0])
			#print "f", f[0,0]
			#print "eTx", np.sum(np.multiply(e.reshape(3000),x[:,0].reshape(3000)))
			#print "f", np.shape(f)
			#print "w", W
			W = np.subtract(W, np.multiply(alfa, f.T))
			#print W
			#print f.T
			#sys.exit(1)

		self.Wopt = W
		
	def predict(self, x):
		dim = np.shape(x)
		x0 = np.ones((dim[0],1)) 
		x = np.insert(x, [0], x0, axis=1)	
		
		h = np.dot(x, self.Wopt)

		self.g = np.around(h)		

