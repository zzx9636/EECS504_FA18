# -*- coding: utf-8 -*-
"""
Created on Wed Sep 19 12:27:59 2018

@author: nadha
"""
# F18 EECS 504 HW1p1 Homography Estimation
import numpy as np
import matplotlib.pyplot as plt
import os

import eta.core.image as etai

def get_correspondences(img1, img2, n):
    '''
    Function to pick corresponding points from two images and save as .npy file
    Args:
	img1: Input image 1
	img1: Input image 2
	n   : Number of corresponding points 
   '''
    
    correspondence_pts = []
    fig = plt.figure()
    ax = fig.add_subplot(111)
    coords = []
    def onclick(event):
        global ix, iy
        ix, iy = event.xdata, event.ydata
        print("The current point is: ")
        print (ix, iy)
        
        coords.append((ix, iy))

        if len(coords) == n:
            fig.canvas.mpl_disconnect(cid)
            plt.close()
        return coords

    ax.imshow(img1)
    cid = fig.canvas.mpl_connect('button_press_event', onclick)
    plt.show()
    correspondence_pts.append(coords)
    coords = []
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.imshow(img2)
    cid = fig.canvas.mpl_connect('button_press_event', onclick)
    plt.show()
    correspondence_pts.append(coords)
    
    np.save('football_pts_'+str(n)+'.npy',correspondence_pts)

def main(n):
    '''
    This function will find the homography matrix and then use it to find corresponding marker in football image 2
    '''
    # reading the images
    img1 = etai.read('football1.jpg')
    img2 = etai.read('football2.jpg')

    filepath = 'football_pts_'+str(n)+'.npy'
    # get n corresponding points
    if not os.path.exists(filepath):
        get_correspondences(img1, img2,n)
    
    correspondence_pts = np.load(filepath)
    
    XY1 = correspondence_pts[0]
    XY2 = correspondence_pts[1]
    # plotting the Fooball image 1 with marker 33
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.imshow(img1)
    u=[1210,1701]
    v=[126,939]
    ax.plot(u, v,color='yellow')
    ax.set(title='Football image 1')
    plt.xlim(0,img1.shape[1])
    plt.ylim(0,img1.shape[0])
    plt.gca().invert_yaxis()
    plt.show()
    #------------------------------------------------
    # FILL YOUR CODE HERE
    # Your code should estimate the homogrphy and draw the  
    # corresponding yellow line in the second image.

    yVec = np.ones((3*n)) # vector y
    aMat = np.zeros((3*n,9)) # Matrix A
    for i in range(n): 
        # form matrix a and y for LR
        yVec[3*i] = XY2[i, 0]
        yVec[3*i+1] = XY2[i, 1]
        aMat[3*i, 0:2] = XY1[i, :]
        aMat[3*i, 2]=1
        aMat[3*i+1, 3:5] = XY1[i, :]
        aMat[3*i+1, 5]=1
        aMat[3*i+2, 6:8] = XY1[i, :]
        aMat[3*i+2, 8]=1
    
    #solve for x*=argmin||Ax-y||_2
    xVec = np.matmul(np.matmul(np.linalg.inv((np.matmul(aMat.T,aMat))),aMat.T),yVec)

    #Form homogenous transformation matrix 
    tMat=np.zeros((3,3))
    tMat[0,:]=xVec[0:3]
    tMat[1,:]=xVec[3:6]
    tMat[2,:]=xVec[6:9]
    
    line1MatHomo = np.array([[u[0],u[1]],[v[0],v[1]],[1,1]], dtype=float)
    
    # calculate corresponding points
    line2MatHomo = np.matmul(tMat,line1MatHomo)
    
    # check if new line go out of image
    '''
    if line2MatHomo[1,1] > img2.shape[0]:
        y_diff_all = line2MatHomo[1,1]-line2MatHomo[1,0]
        x_diff_all = line2MatHomo[0,1]-line2MatHomo[0,0]
        y_diff = line2MatHomo[1,1]-img2.shape[0]
        x_diff = (x_diff_all/y_diff_all)*y_diff
        #print(x_diff)
        line2MatHomo[1,1] = img2.shape[0]
        line2MatHomo[0,1] = line2MatHomo[0,1]-x_diff
    '''
    u2Vec = line2MatHomo[0,:]
    v2Vec = line2MatHomo[1,:]
    
    # plotting the Fooball image 1 with marker 33
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111)
    ax2.imshow(img2)
    ax2.plot(u2Vec, v2Vec,color='yellow')
    ax2.set(title='Football image 2')
    ax2.set_adjustable('box-forced')
    plt.xlim(0,img2.shape[1])
    plt.ylim(0,img2.shape[0])
    plt.gca().invert_yaxis()
    plt.show()




    
    


if __name__ == "__main__": 
    
    #------------------------------------------------
    # FILL BLANK HERE
    # Specify the number of pairs of points you need.
    n = 6
    #------------------------------------------------
    main(n)
