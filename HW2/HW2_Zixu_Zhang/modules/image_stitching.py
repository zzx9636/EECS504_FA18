#!/usr/bin/env python
'''
A module for stitching two parts of an image into one whole image.

Info:
    type: eta.core.types.Module
    version: 0.1.0
'''
# pragma pylint: disable=redefined-builtin
# pragma pylint: disable=unused-wildcard-import
# pragma pylint: disable=wildcard-import
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from builtins import *
from collections import defaultdict
# pragma pylint: enable=redefined-builtin
# pragma pylint: enable=unused-wildcard-import
# pragma pylint: enable=wildcard-import

import sys

import numpy as np
import cv2
#import scipy.io as sio

from eta.core.config import Config
import eta.core.image as etai
import eta.core.module as etam


class ImageStitchingConfig(etam.BaseModuleConfig):
    '''Image stitching configuration settings.

    Attributes:
        data (DataConfig)
        parameters (ParametersConfig)
    '''

    def __init__(self, d):
        super(ImageStitchingConfig, self).__init__(d)
        self.data = self.parse_object_array(d, "data", DataConfig)
        self.parameters = self.parse_object(d, "parameters", ParametersConfig)


class DataConfig(Config):
    '''Data configuration settings.

    Inputs:
        corner_locs_1 (eta.core.types.NpzFile): An Nx2 matrix
            containing (x,y) locations of all corners in image 1,
            detected by the Harris Corner algorithm
        corner_locs_2 (eta.core.types.NpzFile): A Mx2 matrix
            containing (x,y) locations of all corners in image 2,
            detected by the Harris Corner algorithm
        image_1 (eta.core.types.Image): the first input image
        image_2 (eta.core.types.Image): the second input image
    Outputs:
        stitched_image (eta.core.types.ImageFile): The final stitched image
    '''

    def __init__(self, d):
        self.corner_locs_1 = self.parse_string(d, "corner_locs_1")
        self.corner_locs_2 = self.parse_string(d, "corner_locs_2")
        self.image_1 = self.parse_string(d, "image_1")
        self.image_2 = self.parse_string(d, "image_2")
        self.stitched_image = self.parse_string(d, "stitched_image")


class ParametersConfig(Config):
    '''Parameter configuration settings.

    Parameters:
        no_correspondence (eta.core.types.Number): [4] the number of
            points to use when computing the homography
    '''

    def __init__(self, d):
        self.no_correspondence = self.parse_number(
            d, "no_correspondence", default=4)


def _get_HOG_descriptors(corner_locs, in_img):
    '''Return a MxN matrix that contains the M-dimensional HOG feature vectors
        for all N corners.

    Args:
        corner_locs: the location of Harris corners, given as a
            Nx2 2-dimensional matrix
        in_img: the input image

    Returns:
        hog_features: a N x 3780 matrix containing HOG feature vectors for every
            detected corner
    '''
    # Setting parameters
    win_size = (64,128)
    blockSize = (16,16)
    blockStride = (8,8)
    cellSize = (8,8)
    nbins = 9
    derivAperture = 1
    winSigma = 4.
    histogramNormType = 0
    L2HysThreshold = 2.0000000000000001e-01
    gammaCorrection = 0
    nlevels = 64

    # Initializing descriptor
    hog = cv2.HOGDescriptor(win_size,blockSize,blockStride,cellSize,nbins,derivAperture,winSigma,
                            histogramNormType,L2HysThreshold,gammaCorrection,nlevels)

    # Setting compute parameters
    win_stride = (8,8)
    padding = (8,8)
    new_locations = []

    # Gathering all corner locations
    # NOTE: This will not work until you successfully implement the Harris
    #       Corner Detector in 'modules/harris.py'.
    for i in range(corner_locs.shape[0]):
        new_locations.append((int(corner_locs[i][0]), int(corner_locs[i][1])))
    N = len(new_locations)

    # Computing HOG feature vectors for all corners and concatenating them
    # together
    hog_descrp = hog.compute(in_img, win_stride, padding, new_locations)
    feat_size = int((((win_size[0] / 8) - 1) * ((win_size[1] / 8) - 1)) * 36)
    hog_features = np.asarray(hog_descrp)

    # Reshaping as a N x 3780 array
    hog_features = np.reshape(hog_descrp,(N,feat_size))

    return hog_features


def _match_keypoints(hog_features_1, hog_features_2, img1_corners, img2_corners, img1, img2):
    '''Match the HOG features of the two images and return a list of matched
    keypoints.

    Args:
        hog_features_1: the HOG features for the first image
        hog_features_2: the HOG features for the second image
        img1_corners: the corners detected in the first image, from which the
            HOG features were computed
        img2_corners: the corners detected in the second image, from which the
            HOG feautures were computed
        img1: the first image, in case you want to visualize the matches.
        img2: the second image, in case you want to visualize the matches

    Returns:
       img1_matched_points: a list of corner locations in the first image that
            match with those in the second image
       img2_matched_points: a list of corner locations in the second image that
            match with those in the first image
    '''
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

    matches = bf.match(hog_features_1,hog_features_2)

    # Sort them in the order of their distance.
    matches = sorted(matches, key = lambda x:x.distance)
    img1_matched_pts = []
    img2_matched_pts = []
    for match in matches:
        img1_matched_pts.append(img1_corners[match.queryIdx])
        img2_matched_pts.append(img2_corners[match.trainIdx])

    '''
    # Draw the first 20 matches (the blue dots are the matches)
    out_img = img1.copy()
    out_img_2 = img2.copy()
    for i in range(11):
        cv2.circle(out_img,
                   (img1_matched_pts[i][0], img1_matched_pts[i][1]),
                   4,
                   (0, 0, 255),
                   -1)
    for i in range(11):
        cv2.circle(out_img_2,
                   (img2_matched_pts[i][0], img2_matched_pts[i][1]),
                   4,
                   (0, 0, 255),
                   -1)
    # The images below will be stored in your current working directory
    etai.write(out_img, "out1.png")
    etai.write(out_img_2, "out2.png")
    '''
    return img1_matched_pts, img2_matched_pts


def _get_homography(img1_keypoints, img2_keypoints):
    '''Calculate the homography matrix that relates the first image with the
    second image, using the matched keypoints.

    Args:
        img1_keypoints: a list of matched keypoints in image 1. The number
            of keypoints is indicated by the parameter, 'no_correspondence'.
        img2_keypoints: a list of matched keypoints in image 2. The number
            of keypoints is indicated by the parameter, 'no_correspondence'.

    Returns:
        homog_matrix: the homography matrix that relates image 1 and image 2
    '''
    homog_matrix = np.zeros((3,3))
    num_corres = len(img1_keypoints)
    yVec = np.ones((2*num_corres)) # vector y
    aMat = np.zeros((2*num_corres,8)) # Matrix A
    for i in range(num_corres): 
        # form matrix a and y for LR
        yVec[2*i] = img1_keypoints[i][0]
        yVec[2*i+1] = img1_keypoints[i][1]
        aMat[2*i, 0:2] = img2_keypoints[i]
        aMat[2*i, 2]=1
        aMat[2*i,6:8] = -img1_keypoints[i][0]*img2_keypoints[i]
        aMat[2*i+1, 3:5] = img2_keypoints[i]
        aMat[2*i+1, 5]=1
        aMat[2*i+1,6:8] = -img1_keypoints[i][1]*img2_keypoints[i]
        

    #solve for x*=argmin||Ax-y||_2
    xVec = np.matmul(np.matmul(np.linalg.inv((np.matmul(aMat.T,aMat))),aMat.T),yVec)
    homog_matrix[0,:] = xVec[0:3]
    homog_matrix[1,:]=xVec[3:6]
    homog_matrix[2,0:2]=xVec[6:8]
    homog_matrix[2,2] = 1
    return homog_matrix

def _bilinear_interpolation(x, y, pixel_mat, x1, y1, x2, y2):
    x_diff = np.array([[x2-x, x-x1]])
    y_diff = np.array([[y2-y],[y-y1]])
    if len(pixel_mat.shape)==3: #rgb case
        interpolation = np.zeros((1,1,3))
        for i in range(3):
            interpolation[:,:,i] = np.matmul(np.matmul(x_diff, pixel_mat[:,:,i]),y_diff)/((x2-x1)*(y2-y1))
    else:
        interpolation = np.matmul(np.matmul(x_diff, pixel_mat),y_diff)/((x2-x1)*(y2-y1))
    return interpolation

def _overlap(img1, img2, homog_matrix):
    '''Applies a homography transformation to img2 to stitch img1 and img2
    togther.

    Args:
        img1: the first image
        img2: the second image
        homog_matrix: the homography matrix

    Returns:
        stitched_image: the final stitched image
    '''                    
    if len(img1.shape)>2: #RGB img
        img1_height, img1_width, num_ch1 = img1.shape
        img2_height, img2_width, _ = img2.shape
    else: #grayscale img
        num_ch1 =1
        img1_height, img1_width = img1.shape
        img2_height, img2_width = img2.shape

    #create max/min pixel location matrix in homogenous coordindates
    locMat = np.array([[0, 0, img2_width-1, img1_width-1], [0, img1_height-1, 0, img1_height-1], [1,1,1,1]])
    #apply homography to the img2's pixel location
    locMat_trans_homo = np.matmul(homog_matrix, locMat)
    locMat_trans = np.round(locMat_trans_homo[0:2,:]/locMat_trans_homo[2,:])
    T_inv = np.linalg.inv(homog_matrix)
    #find the size of new image
    min_x = int(min(0, np.amin(locMat_trans[0])))
    min_y = int(min(0, np.amin(locMat_trans[1])))
    max_x = int(max(img1_width-1, np.amax(locMat_trans[0])))
    max_y = int(max(img1_height-1, np.amax(locMat_trans[1])))
    
    new_img_height = max_y-min_y+1
    new_img_width = max_x-min_x+1
    if num_ch1 ==1: #grayscale
        out_img = np.zeros((new_img_height, new_img_width))
        # map img1 to output img
        out_img[(-min_y):(img1_height-min_y),(-min_x):(img1_width-min_x)] = img1
        # map img2 to stitched img
        for i in range(new_img_height):
            for j in range(new_img_width):
                temp_x_p = np.array([[j+min_x, i+min_y, 1]]).T
                temp_x = np.matmul( T_inv , temp_x_p)
                x_int = temp_x[0,0]/temp_x[2,0]
                y_int = temp_x[1,0]/temp_x[2,0]
                if 0<=x_int and x_int<img2_width-1 and 0<=y_int and y_int<img2_height-1:
                    x1_int = np.floor(x_int)
                    x2_int = x1_int+1
                    y1_int = np.floor(y_int)
                    y2_int = y1_int+1
                    pixel_mat = img2[int(y1_int):int(y2_int)+1 , int(x1_int):int(x2_int)+1]
                    interpolation = _bilinear_interpolation(x_int,y_int,pixel_mat,x1_int,y1_int,x2_int,y2_int)
                    out_img[i,j] = interpolation.astype(int)
    else: #RGB case
        out_img = 255*np.ones((new_img_height, new_img_width, num_ch1))
        # map img1 to output img
        out_img[(-min_y):(img1_height-min_y),(-min_x):(img1_width-min_x), :] = img1
        # map img2 to stitched img
        
        for i in range(new_img_height):
            for j in range(new_img_width):
                temp_x_p = np.array([[j+min_x, i+min_y, 1]]).T
                temp_x = np.matmul( T_inv , temp_x_p)
                x_int = temp_x[0,0]/temp_x[2,0]
                y_int = temp_x[1,0]/temp_x[2,0]
                if 0<=x_int and x_int<img2_width-1 and 0<=y_int and y_int<img2_height-1:
                    x1_int = np.floor(x_int)
                    x2_int = x1_int+1
                    y1_int = np.floor(y_int)
                    y2_int = y1_int+1
                    pixel_mat = img2[int(y1_int):int(y2_int)+1 , int(x1_int):int(x2_int)+1,:]
                    interpolation = _bilinear_interpolation(x_int,y_int,pixel_mat,x1_int,y1_int,x2_int,y2_int)
                    out_img[i,j,:] = interpolation.astype(int)
    return out_img


def _stitch_images(image_stitching_config):
    for data in image_stitching_config.data:
        # Load the corner locations
        img1_corners = np.load(data.corner_locs_1)["corner_locations"]
        img2_corners = np.load(data.corner_locs_2)["corner_locations"]
    
        # Read in the input images
        img1 = etai.read(data.image_1)
        img2 = etai.read(data.image_2)

        # Compute HOG feature vectors for every detected corner
        hog_features_1 = _get_HOG_descriptors(img1_corners, img1)
        hog_features_2 = _get_HOG_descriptors(img2_corners, img2)

        # Match the feature vectors
        img_1_pts, img_2_pts = _match_keypoints(hog_features_1, hog_features_2,
                                    img1_corners, img2_corners, img1, img2)

        # Tune this parameter in "requests/image_stitching_request.json"
        # to specify the number of corresponding points to use when computing
        # the homography matrix
        no_correspondence = image_stitching_config.parameters.no_correspondence

        # Compute the homography matrix that relates image 1 and image 2
        H = _get_homography(img_1_pts[1:no_correspondence+1], img_2_pts[1:no_correspondence+1])

        # Stitching the images by applying the homography matrix to image 2
        final_img = _overlap(img1, img2, H)

        # Write the final stitched image
        etai.write(final_img, data.stitched_image)


def run(config_path, pipeline_config_path=None):
    '''Run the Image Stitching module.

    Args:
        config_path: path to a ImageStitchingConfig file
        pipeline_config_path: optional path to a PipelineConfig file
    '''
    image_stitching_config= ImageStitchingConfig.from_json(config_path)
    etam.setup(image_stitching_config,
               pipeline_config_path=pipeline_config_path)
    _stitch_images(image_stitching_config)


if __name__ == "__main__":
    run(*sys.argv[1:])
