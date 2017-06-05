#! /usr/bin/env python3
import numpy as np

from find_cars import mvn


mu = np.array([[1,0],[0,1]])
S = np.array([[[1,0],[0,2]],[[3,0],[0,2]]])
x = np.array([[1,0],[0,1]])
print(mvn(x,mu,S,np.linalg.inv(S)))

