import numpy as np

#first index of pts, mean, and cov
#get pdf of 1 point evaluated across a list of mean 
# and covariances
#first index of mean and cov should be number of distributions
def mvn(x, mean, cov, cov_inv):
  y = x - mean;
  y1 = np.einsum('ij,ijk->ik', y, cov_inv) 
  y2 = sum(y1*y, axis=1)
  y3 = np.np.exp(-0.5*y2)
  y4 = 1/np.sqrt((2*np.pi)**len(x)*np.linalg.det(cov))
  return y3*y4
  
