#! /usr/bin/env python3

import sys
sys.path.append("/Users/marcuspan/code/darkflow")
from darkflow.net.build import TFNet
import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import multivariate_normal
#from utils import mvn
import pdb
import json
import matplotlib.cm as cm

save_boxes = 0;
nFramesToSave = 600;
clim = (0, 798)
rlim = (140, 500)
measurement_prob_th = 1e-6
load_net = 0
lenPlot = 300;
colors = cm.rainbow(np.random.permutation(256))

H = np.load('homography.npy')
Hinv = np.linalg.inv(H)

boxes_list = []
options = {"model": "./cfg/yolo.cfg", "load": "./yolo.weights", "threshold": 0.1}
if (load_net):
  tfnet = TFNet(options)
else:
  boxes_list= np.load('boxes_list_600.npy')
  
cap = cv2.VideoCapture('video/sherbrooke_shorter.mov')
if not cap.isOpened():
  print("video closed")

ret,frame_large = cap.read()
frame = frame_large[rlim[0]:rlim[1], clim[0]:clim[1]]

boxes = boxes_list[0]

lowerX = []
lowerY = []
for b in boxes:
  #print(b)
  if b['label'] == 'car':
    cv2.rectangle(frame, (b['topleft']['x'],b['topleft']['y']),
      (b['bottomright']['x'],b['bottomright']['y']),(255,0,0),1)

    #get lower x and y points
    lowerX.append((b['topleft']['x'], b['bottomright']['x']))
    lowerY.append(b['bottomright']['y'])
  
nPts = len(lowerY)

lane1 = np.array([[321,44],[317,83],[360,148]])
lane2 = np.array([[350,44],[356,83],[448,152]])
lane3 = np.array([[378,49],[399,83],[528,148]])
lane4 = np.array([[174,171],[355,157],[528,148]])
laneC = np.vstack((lane1,lane2,lane3,lane4)) + np.array([clim[0], rlim[0]])
laneC = np.vstack((laneC.T, np.ones((1,len(laneC)))))
laneW = Hinv.dot(laneC)
laneW = laneW / laneW[2,:]
laneW = laneW.T
laneC = laneC.T

#get 3D points of cars
if nPts > 0:
  px = np.vstack((np.mean(lowerX, axis=1) + clim[0], np.array(lowerY) + rlim[0], 
    np.ones(nPts)))
  pw = Hinv.dot(px)
  px = px[0:2,:].T 
  pw = pw[0:2,:] / pw[2,:]
  pw = pw.T


plt.figure(1)
im_ax = plt.gca()
plt.figure(2)
w_ax = plt.gca()
im_ax.imshow(frame)

col = colors[range(nPts)]
#plot lanes
for p in range(4):
  start = p*3
  ind = range(p*3, p*3+3)
  im_ax.plot(laneC[ind,0]-clim[0], laneC[ind,1]-rlim[0])
  w_ax.plot(laneW[ind,0], laneW[ind,1])


  
print(laneW)

im_ax.scatter(px[:,0]-clim[0], px[:,1]-rlim[0],
  c=col, marker = 'x', s=50)
w_ax.scatter(pw[:,0], pw[:,1], marker='x', c=col,)

im_ax.set_xlabel('x position (px)')
im_ax.set_ylabel('y position (px)')
w_ax.set_xlabel('x position (m)')
w_ax.set_ylabel('y position (m)')

# pdb.set_trace()
plt.show(block=True)
plt.pause(0.1)


cap.release()

if save_boxes: 
  np.save('boxes_list_%d' % nFramesToSave, boxes_list)

