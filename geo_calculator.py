#! /usr/bin/env python3
import sys
sys.path.append("/Users/marcuspan/code/darkflow")
import numpy as np
import pdb
import cv2
from matplotlib import pyplot as plt

#script to calculate homography between world and image


# Data:
px_coords = np.array([
  (182,282), (761, 267), (588,290), 
  (775, 325), (790, 343), (777,362), 
  (451, 463), (176, 324), (218, 488),
  (783, 278), (175, 307)])

#lat, long points from google map
world_coords_ang = np.radians(np.array([
  (45.521156, -73.565452),
  (45.521315, -73.565707), 
  (45.521299, -73.565620), 
  (45.521367, -73.565643), 
  (45.521380, -73.565631), 
  (45.521388, -73.565614),
  (45.521385, -73.565499), 
  (45.521232, -73.565438),
  (45.521371, -73.565442),
  (45.521333, -73.565700),
  (45.521204, -73.565438)
]))

#distance between 2 points in google maps
#(dist, index1, index2)
d_world = np.array([(15.06, 7,8), (24.56, 9, 10), (26.84, 0,1)])


def wrapToPi(angles_rad):
  return np.mod(angles_rad+np.pi, 2.0 * np.pi) - np.pi

#get radius of earth from distance,lat,long
#dist = sqrt(sq(R cos(lat)d_long) + sq(R d_lat))
diff = world_coords_ang[d_world[:, 2].astype(int), :] - \
  world_coords_ang[d_world[:, 1].astype(int), :]
diff = wrapToPi(diff)
dist_ang = np.sqrt(diff[:,0]**2 + 
  (np.cos(world_coords_ang[d_world[:,1].astype(int), 0])**2) * (diff[:,1]**2) )
R_arr = d_world[:,0]/dist_ang
R = np.linalg.lstsq(np.array([dist_ang]).T, np.array([d_world[:,0]]).T)
R = R[0][0][0]


#get world coords, with origin at point(origin_i)
origin_i = 10
N = len(px_coords)
diff = world_coords_ang - world_coords_ang[origin_i,:] 
world_coords = -np.array([R*np.cos(world_coords_ang[origin_i, 0])*diff[:,1],  
R*diff[:,0]]).T

#find homography and reproject to check
H,mask = cv2.findHomography(world_coords, px_coords)
world_coords_H = np.linalg.inv(H).dot(np.vstack((px_coords.T, np.ones(len(px_coords)))))
world_coords_H = world_coords_H[0:2, :]/world_coords_H[2,:]



#plot
ind = list(range(N))
col = ['b', 'g', 'r', 'c', 'm', 'k', 'g', 'b', 'b', 'b']
plt.figure(1)
cap = cv2.VideoCapture('video/sherbrooke_shorter.mov')
ret,frame = cap.read()
cap.release()
plt.imshow(frame)
plt.scatter(px_coords[ind,0], px_coords[ind,1], c=col)

plt.figure(2)
plt.scatter(world_coords[ind,0], world_coords[ind,1], label='world', c=col)

plt.scatter(world_coords_H[0,ind], world_coords_H[1,ind], label='world H', c=col)
plt.legend()
plt.show()

np.save('homography1.npy', H)
#homography from dataset:
h1 =  -0.093906
h2 =  -0.593566
h3 =  92.340814
h4 =  -0.100614
h5 =  -0.414678
h6 =  117.043125
h7 =  -0.001519
h8 =  -0.005823
h9 =  1.0
Hdata = np.array([[h1, h2, h3], 
  [h4, h5, h6],
  [h7, h8, h9]])

pdb.set_trace()



