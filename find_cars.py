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
measurement_prob_th = 1e-10
load_net = 0
lenPlot = 100;
colors = cm.rainbow(np.random.permutation(256))
#font = cv.InitFont(cv.CV_FONT_HERSHEY_SIMPLEX, 1, 1, 0, 3, 8) #Creates a font

def mvn(x, mean, cov, cov_inv):
  y = x - mean;
  y1 = np.einsum('ij,ijk->ik', y, cov_inv) 
  y2 = np.sum(y1*y, axis=1)
  if (y2 < 0).any():
    print(cov)
    print(cov_inv)
    #print(y)
  y3 = np.exp(-0.5*y2)
  y4 = 1/np.sqrt((2*np.pi)**len(x)*np.linalg.det(cov))
  return y3*y4
  
def associate_data(M1):
  M = M1.copy()
  Mi = np.argsort(M,axis=0) 
  nM = len(M1)
  nS = len(M1[0])
  M_max = -1*np.zeros(nM,dtype=int)
  S_max = -1*np.ones(nS,dtype=int)
  for m in range(nM):
    #only consider states that maximize meas
    
    maxS = np.argmax(M[m,:])
    if (M[m,maxS] > measurement_prob_th):
      maxM = S_max[maxS]
      if (maxM > -1) :
        #the ML state has been associated w another measurement
        #check to see which is more likely
        if M[m,maxS] > M[maxM,maxS]: 
          M_max[m] = maxS
          S_max[maxS] = m
          M_max[maxM] = -1
      else:
        M_max[m] = maxS
        S_max[maxS] = m
        
  return M_max, S_max

    
Ma = np.array([[0.1, 0.2, 0.3], [0.2,0.1,0.5]])
Mb = np.array([[0.1,0.2],[0.4,0.5],[0.2,0.3]])
Mc = np.array([[0.3,0.2,0.1],[0.2,0.3,0.1],[0.1,0.1,0.3]])
Mar, Sar = associate_data(Ma)
Mbr, Sbr = associate_data(Mb)
Mcr, Scr = associate_data(Mc)
print(Mar)
print(Sar)
print(Mbr)
print(Sbr)
print(Mcr)
print(Scr)

    
  

if __name__ == "__main__":

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
  
  fr = -1;

  #kalman filter variables
  x_est = [] 
  nStates = 10 #[xt,yt,xt-1,yt-1, vxt, vyt]
  nMeas = 2
  fps = 30
  dt = 1/fps
  idt = 1/dt

  
  '''
  A = np.array([
    [1,0,0,0,dt,0,0,0,0.5*dt**2,0],
    [0,1,0,0,0,dt,0,0,0,0.5*dt**2],
    [1,0,0,0,0,0,0,0,0,0],
    [0,1,0,0,0,0,0,0,0,0],
    [idt,0,-idt,0,0,0,0,0,0,0],
    [0,idt,0,-idt,0,0,0,0,0,0],
    [0,0,0,0,1,0,0,0,0,0],
    [0,0,0,0,0,1,0,0,0,0],
    [0,0,0,0,idt,0,-idt,0,0,0],
    [0,0,0,0,0,idt,0,-idt,0,0]])
  '''
   
  A = np.array([
    [1,0,0,0,dt,0],
    [0,1,0,0,0,dt],
    [1,0,0,0,0,0],
    [0,1,0,0,0,0],
    [idt,0,-idt,0,0,0],
    [0,idt,0,-idt,0,0]])
  

  nStates = len(A)
  
  Q = (dt**2)*10*np.eye(nStates)
  
  #C = np.array([[1,0,0,0,0,0,0,0,0,0],[0,1,0,0,0,0,0,0,0,0]])
  C = np.array([[1,0,0,0,0,0],[0,1,0,0,0,0]])

  x_est = np.array([])
  sig_est = np.array([])

  id_all = np.array([], dtype=int); 
  id_cur = np.array([], dtype=int); 
  x_est_plot = np.array([]).reshape(0,2,lenPlot);
  x_meas_plot = np.array([]).reshape(0,2,lenPlot);
  valid_t = np.array([],dtype=bool).reshape(0,2,lenPlot);#axis 0 is meas, axis 1 is estimate



  while (cap.isOpened()): 
    ret,frame_large = cap.read()
    fr = fr+1

    if fr >= lenPlot:
      break;
      
    print(fr)
      
    #crop picture to region of interest
    frame = frame_large[rlim[0]:rlim[1], clim[0]:clim[1],:]

    #get bounding boxes around cars from nnet
    if load_net:
      boxes = tfnet.return_predict(frame)
    else:
      boxes = boxes_list[fr]
    
    if save_boxes:
      boxes_list.append(boxes);
      if fr >= nFramesToSave:
        break;
      else:
        continue
    
    plt.figure(1)
    plt.subplot(2,1,1)
    im_ax = plt.gca()
    plt.subplot(2,1,2)
    w_ax = plt.gca()
   
    lowerX = []
    lowerY = []
    confidence = []
    for b in boxes:
      #print(b)
      if b['label'] == 'car':
        cv2.rectangle(frame, (b['topleft']['x'],b['topleft']['y']),
          (b['bottomright']['x'],b['bottomright']['y']),(255,0,0),1)

        #get lower x and y points
        lowerX.append((b['topleft']['x'], b['bottomright']['x']))
        lowerY.append(b['bottomright']['y'])
        confidence.append(b['confidence'])
      
    confidence = np.array(confidence)



    nPts = len(lowerY)

    #get 3D points of cars
    if nPts > 0:
      px = np.vstack((np.mean(lowerX, axis=1) + clim[0], np.array(lowerY) + rlim[0], 
        np.ones(nPts)))
      pw = Hinv.dot(px)
      px = px[0:2,:].T 
      pw = pw[0:2,:] / pw[2,:]
      pw = pw.T

    #kalman filter tracking

    #predict
    if len(x_est) > 0:
      #forward step
      x_pred = A.dot(x_est.T).T
      sig_pred = np.matmul(np.matmul(A, sig_est), A.T) + Q

    else:
      #initialize filter to measuements
      x_pred = np.vstack((pw.T, pw.T, np.zeros((nStates-4, nPts)))).T
      #set the variance to log(1/confidence)
      sig_pred = np.minimum(np.log(1/confidence),5)[:, np.newaxis, np.newaxis]*np.eye(nStates)
      x_est = np.copy(x_pred)
      sig_est = np.copy(sig_pred)

      id_cur = np.array(range(len(id_all), len(id_all)+len(x_pred)), dtype=int)
      id_all = np.concatenate((id_all, id_cur))
      x_meas_plot = np.vstack((x_meas_plot, np.zeros((nPts, 2, lenPlot))))
      x_est_plot = np.vstack((x_est_plot, np.zeros((nPts, 2, lenPlot))))
      valid_t = np.vstack((valid_t, np.zeros((nPts,2,lenPlot), dtype=bool)))

  #update
    R = np.log(1/confidence)[:, np.newaxis, np.newaxis]*np.eye(nMeas)
    #S = np.matmul(np.matmul(C, sig_pred), C.T) + R
    Rmarked = np.array([False]*nPts, dtype=bool)
    meas_id = np.zeros(nPts, dtype=int)

    #fill up measurement probability matrix, M
    #M_ij = p(measurement_i | state_j)
    #for all points, get most likely measurement and update KF

    M = np.zeros((nPts, len(x_pred)))
    #print('confidence')
    #print(np.log(1/confidence))
    for p in range(len(x_est)):
      S = C.dot(sig_pred[p].dot(C.T)) + R
      S_inv = np.minimum(np.linalg.inv(S), 1000000)
      M[:,p] = mvn(pw,x_pred[p, 0:2], S, S_inv)

    #print(M)
    M_max, S_max = associate_data(M)
    Rmarked[M_max > -1] = True
    #print(S_max)
    for p in range(len(x_est)):
      yi = S_max[p]
      if yi > -1: 
        Rmarked[yi] = True
        y = pw[yi]
        K = sig_pred[p].dot(C.T).dot(S_inv[yi])
        x_est[p] = x_pred[p] + K.dot(pw[yi] - C.dot(x_pred[p]))
        sig_est[p] = (np.eye(nStates) - K.dot(C)).dot(sig_pred[p]) 
      
        x_meas_plot[id_cur[p],:,fr] = y
        x_est_plot[id_cur[p],:,fr] = x_est[p][0:2]
        valid_t[id_cur[p],0,fr] = True #valid measurement
        meas_id[yi] = id_cur[p]

        
        cv2.putText(frame, '%d' % id_cur[p], (int(px[yi,0]-clim[0]), int(px[yi,1]-rlim[0]+40)), 
          0,0.8, (0,0,255), 1,1)

    #pdb.set_trace()
    sig_est = np.maximum(sig_est, 0.001)
      
    #for all unmarked measurements, create new state
    Rmarked_inv = np.invert(Rmarked)
    x_new = np.tile(pw[Rmarked_inv, :], (1,2))
    x_new = np.hstack((x_new, np.zeros((len(x_new), nStates-4))))
    x_est = np.vstack((x_est, x_new))
    x_pred = np.vstack((x_pred, x_new))
    sig_new = np.minimum(np.log(1/confidence)[Rmarked_inv, np.newaxis, np.newaxis], 5)*np.eye(nStates)
    sig_est = np.vstack((sig_est, sig_new))

    id_new = np.array(range(len(id_all), len(id_all) + len(x_new)),dtype=int)
    meas_id[Rmarked_inv] = id_new
    id_cur = np.concatenate((id_cur, id_new))
    id_all = np.concatenate((id_all, id_new))
    x_meas_plot_new = np.zeros((len(x_new), 2, lenPlot))
    x_meas_plot_new[:,:,fr] = x_new[:,0:2];
    x_meas_plot = np.vstack((x_meas_plot, x_meas_plot_new))

    x_est_plot = np.vstack((x_est_plot, x_meas_plot_new))

    valid_t_new = np.zeros((len(x_new),2,lenPlot),dtype=bool);
    valid_t = np.vstack((valid_t, valid_t_new))

    #reproject onto picture 
    px_est = H.dot(np.vstack((x_est[:, 0:2].T, np.ones(len(x_est)))))
    px_est = (px_est[0:2, :]/px_est[2,:]).T
    
    px_pred = H.dot(np.vstack((x_pred[:, 0:2].T, np.ones(len(x_pred)))))
    px_pred = (px_pred[0:2, :]/px_pred[2,:]).T

    #delete states out of bounds
    valid = np.logical_and(
      np.logical_and(px_est[:,0] > clim[0], px_est[:,0] < clim[1]),  
      np.logical_and(px_est[:,1] > rlim[0], px_est[:,1] < rlim[1]))  
  
    #valid = np.logical_and(valid, sig_est[:,0,0] < 1.5)

    #print(valid)
    px_est = px_est[valid,:]
    px_pred = px_pred[valid,:]
    x_est = x_est[valid, :]
    sig_est = sig_est[valid, :, :]
    #print(sig_est[:,0,0])
    id_cur = id_cur[valid]
    valid_t[id_cur, 1,fr] = True; #valid state
    '''
    print('x_est')
    print(x_est)
    print('sig_est')
    print(sig_est)
    '''

    #pdb.set_trace()
    im_ax.clear()
    im_ax.imshow(frame)

    im_ax.scatter(px_est[:,0]-clim[0], px_est[:,1]-rlim[0],
      c=colors[np.mod(id_cur, 256)],marker='x', s=50)
    im_ax.scatter(px_pred[:,0]-clim[0], px_pred[:,1]-rlim[0],
      c=colors[np.mod(id_cur, 256)],marker='<', s=50)
    im_ax.scatter(px[:,0]-clim[0], px[:,1]-rlim[0],
      c=colors[np.mod(meas_id, 256)], marker = '^', s=50)
    w_ax.scatter(x_est[:,0], x_est[:,1], marker='x', c=colors[np.mod(id_cur, 256)])
    w_ax.scatter(pw[:,0], pw[:,1], marker='^', c=colors[np.mod(meas_id, 256)],)
  
   # pdb.set_trace()
    plt.show(block=False)
    plt.pause(0.1)


  cap.release()

  if save_boxes: 
    np.save('boxes_list_%d' % nFramesToSave, boxes_list)

  nTracksPlot = 5
  f,ax = plt.subplots()
  for p in [6,7,8]:
    ax.scatter(x_meas_plot[p,0,valid_t[p,0,:]], x_meas_plot[p,1,valid_t[p,0,:]], color='C%d'%p, marker='x')
    ax.scatter(x_est_plot[p,0,valid_t[p,1,:]], x_est_plot[p,1,valid_t[p,1,:]], color='C%d'%p, marker='o')

  plt.show()

  plt.pause(1.0)
  #pdb.set_trace()
  print('n cars: %d' % len(id_all))
    
    
  #cv2.destroyAllWindows()

