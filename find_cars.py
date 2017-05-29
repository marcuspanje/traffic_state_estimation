#! /usr/bin/env python3

import sys
sys.path.append("/Users/marcuspan/code/darkflow")
from darkflow.net.build import TFNet
import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import multivariate_normal
from utils import mvn
import pdb



clim = (0, 798)
rlim = (140, 540)
measurement_prob_th = 0.5

if __name__ == "__main__":

  H = np.load('homography.npy')
  Hinv = np.linalg.inv(H)

  options = {"model": "./cfg/yolo.cfg", "load": "./yolo.weights", "threshold": 0.1}
  tfnet = TFNet(options)
  cap = cv2.VideoCapture('video/sherbrooke_shorter.mov')
  if not cap.isOpened():
    print("video closed")
  
  fr = 0;
  initialized = 0
  memory = 2;

  #kalman filter variables
  x_est = [] 
  nStates = 6 #[xt,yt,xt-1,yt-1, vxt, vyt]
  nMeas = 2
  fps = 30
  dt = 1/fps

  A = np.array([
    [1,0,0,0,dt,0],
    [0,1,0,0,0,dt],
    [1,0,0,0,0,0],
    [0,1,0,0,0,0],
    [dt/2,0,-dt/2,0,0,0],
    [0,dt/2,0,-dt/2,0,0]])
  
  Q = (dt**2)*np.eye(nStates)

  C = np.array([[1,0,0,0,0,0],[0,1,0,0,0,0]])

  x_est = np.array([])
  sig_est = np.array([])

  while (cap.isOpened()): 
    ret,frame = cap.read()

    
    if fr == 1:
      f2 = plt.figure()
      plt.imshow(frame)
      plt.show()
      
    #crop picture to region of interest
    frame = frame[rlim[0]:rlim[1], clim[0]:clim[1],:]

    #get bounding boxes around cars from nnet
    boxes = tfnet.return_predict(frame)

    lowerX = []
    lowerY = []
    confidence = []
    for b in boxes:
      print(b)
      if b['label'] == 'car':
        #draw bounding box
        cv2.rectangle(frame, (b['topleft']['x'],b['topleft']['y']),
          (b['bottomright']['x'],b['bottomright']['y']),5)

        #get lower x and y points
        lowerX.append((b['topleft']['x'], b['bottomright']['x']))
        lowerY.append(b['bottomright']['y'])
        conf.append(b['confidence'])
      

    f1 = plt.figure(1)
    plt.subplot(1,2,1)
    plt.imshow(frame)
    im_ax = plt.gca()
    nPts = len(lowerY)

    #get 3D points of cars
    if nPts > 0:
      px = np.vstack((np.mean(lowerX, axis=1) + clim[0], np.array(lowerY) + rlim[0], 
        np.ones(nPts)))
      pw = Hinv.dot(px)
      pw = pw[0:2,:] / pw[2,:]
      plt.subplot(1,2,2)
      plt.scatter(pw[0,:], pw[1,:])

    #kalman filter tracking

    #predict
    if len(x_est) > 0:
      #forward step
      x_est = (A.dot(x_est.T)).T
      sig_est = np.matmul(np.matmul(A, sig_pred), A.T) + Q

    else:
      #initialize filter to measuements
      x_pred = np.reshape(np.vstack((pw, pw, zeros(2, nPts))).T, 
        (nPts, nStates, 1))
      #set the variance to log(1/confidence)
      sig_pred = np.log(1/np.array(confidence))[:, np.newaxis, np.newaxis]*np.eye(nStates)
      x_est = np.copy(x_pred)
      sig_est = np.copy(sig_pred)

  #update
    R = np.log(1/np.array(confidence))[:, np.newaxis, np.newaxis]*np.eye(nMeas)
    S = np.matmul(np.matmul(C, sig_pred), C.T) + R
    Rmarked = [False]*nPts

    #for all points, get most likely measurement and update KF
    for p in range(len(x_est)):
      
      S = C.dot(sig_pred[p].dot(C.T)) + R
      S_inv = np.linalg.inv(S)
      #calculates meas likelihood for a single point, across all measurements
      probs = mvn(x_pred[:, 0:2], pw.T, S, S_inv)
      yi = np.argmax(probs) 

      if probs[yi] > measurement_prob_th:
        #finally update KF
        Rmarked[yi] = True
        y = pw[yi]
        K = sig_pred[p].dot(C.T).dot(S_inv[yi])
        x_est[p] = x_pred[p] + K.dot(pw[yi] - C.dot(x_pred[p]))
        sig_est[p] = (np.eye(nStates) - K.dot(C)).dot(sig_pred[p]) 
      
      
    #for all unmarked measurements, create new state
    Rmarked_inv = np.invert(Rmarked)
    x_new = np.tile(pw[Rmarked_inv, :], (1,2))
    x_new = np.hstack((x_new, np.zeros(len(x_new), 2)))
    x_est = np.vstack((x_est, x_new))
    sig_new = confidence[Rmarked_inv, np.newaxis, np.newaxis]*np.eye(nStates)
    sig_est = np.vstack((sig_est, sig_new))

    #reproject onto picture 
    px_est = H.dot(np.vstack((x_est[:, 0:2].T, np.ones(len(x_est)))))
    px_est = px_est[0:2, :]/px_est[2,:]

    #delete states out of bounds
    valid = np.logical_and(
      np.logical_and(px_est[0,:] > rlim[0], px_est[0,:] < rlim[1]),  
      np.logical_and(px_est[1,:] > clim[0], px_est[1,:] < clim[1]))  

    x_est = x_est[valid, :]
    sig_est = sig_est[valid, :, :]
  
    im_ax.scatter(px_est[0,valid], px_est[1,valid])


    fr = fr + 1;
    plt.show(block=False)
    plt.pause(0.001)


  cap.release()
  #cv2.destroyAllWindows()

