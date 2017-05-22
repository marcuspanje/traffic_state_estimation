#! /usr/bin/env python3

import sys
sys.path.append("/Users/marcuspan/code/darkflow")
from darkflow.net.build import TFNet
import cv2
import numpy as np
from matplotlib import pyplot as plt
import pdb


#parameters

clim = (0, 798)
rlim = (140, 540)



def calibrate_homography():
  #find 4 corners of a land divider to calculate homography
  #pixel points before cropping
  dest = np.array([[970, 256], [476, 256], [496, 268], [501, 268]])
  #after cropping
  dest = dest - np.array([clim[0], rlim[0]])
  #world points, centered at top-right corner 
  src = np.array([[-0.1, 0], [0,0],[-0.1,1.5], [0, 1.5]])
  #dest = H*src
  H,mask = cv2.findHomography(src, dest);
  print(H)
  xp = np.linalg.inv(H).dot(np.vstack((dest.T, (1,1,1,1))))
  print(xp)
  #print('x: {0:.2f}, y:{1:.2f}'.format(xp[0]/xp[2], xp[1]/xp[2]))
  return H
  

if __name__ == "__main__":

  Hinv = np.linalg.inv(calibrate_homography())

  options = {"model": "./cfg/yolo.cfg", "load": "./yolo.weights", "threshold": 0.1}
  tfnet = TFNet(options)
  cap = cv2.VideoCapture('video/sherbrooke_short.mov')
  if not cap.isOpened():
    print("video closed")
  
  fr = 0;
  while (cap.isOpened()): 
    fr = fr + 1;
    ret,frame = cap.read()
    #crop picture to region of interest
    frame = frame[rlim[0]:rlim[1], clim[0]:clim[1],:]

    boxes = tfnet.return_predict(frame)
  
    '''
    lowerX = np.mean([(b['topleft']['x'], b['bottomright']['x']) 
      for b in boxes if b['label'] == 'car'], axis=1)
    lowerY = np.array([b['bottomright']['y'] 
      for b in boxes if b['label'] == 'car'])
    pxPoints = np.hstack((lowerX, lowerY, np.ones(len(lowerX))); 
    '''

    lowerX = []
    lowerY = []
    for b in boxes:
      print(b)
      if b['label'] == 'car':
        #draw bounding box
        cv2.rectangle(frame, (b['topleft']['x'],b['topleft']['y']),
          (b['bottomright']['x'],b['bottomright']['y']),5)

        #get lower x and y points
        lowerX.append((b['topleft']['x'], b['bottomright']['x']))
        lowerY.append(b['bottomright']['y'])
      
    f1 = plt.figure()
    plt.imshow(frame)

    #get 3D points of cars
    if len(lowerY) > 0:
      px = np.vstack((np.mean(lowerX, axis=1), lowerY, np.ones(len(lowerY))))
      pw = Hinv.dot(px)
      #pdb.set_trace()
      pw = pw / pw[2,:]
      f2 = plt.figure()
      plt.scatter(pw[0,:], pw[1,:])

      
    #plt.show(block=False)
    plt.show()
    plt.pause(0.001)


  cap.release()
  cv2.destroyAllWindows()

