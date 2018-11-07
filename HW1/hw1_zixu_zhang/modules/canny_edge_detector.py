#!/usr/bin/env python
'''
A module for determining the edges of an image using the Canny Edge Detector.

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
# pragma pylint: enable=redefined-builtin
# pragma pylint: enable=unused-wildcard-import
# pragma pylint: enable=wildcard-import

import sys

import numpy as np
#import scipy.io as sio

from eta.core.config import Config
import eta.core.image as etai
import eta.core.module as etam


# CONSTANTS
STRONG = 10
WEAK = 5
SUPPRESSED = 0


class CannyEdgeConfig(etam.BaseModuleConfig):
    '''CannyEdge configuration settings.

    Attributes:
        data (DataConfig)
        parameters (ParametersConfig)
    '''

    def __init__(self, d):
        super(CannyEdgeConfig, self).__init__(d)
        self.data = self.parse_object_array(d, "data", DataConfig)
        self.parameters = self.parse_object(d, "parameters", ParametersConfig)


class DataConfig(Config):
    '''Data configuration settings.

    Inputs:
        input_image (eta.core.types.Image): the input image
        sobel_horizontal_result (eta.core.types.NpzFile): The result of
            convolving the original image with the "sobel_horizontal" kernel.
            This will give the value of Gx (the gradient in the x
            direction).
        sobel_vertical_result (eta.core.types.NpzFile): The result of
            convolving the original image with the "sobel_vertical" kernel.
            This will give the value of Gy (the gradient in the y
            direction).

    Outputs:
        image_edges (eta.core.types.ImageFile): A new image displaying
            the edges of the original image.
        gradient_orientation (eta.core.types.ImageFile): [None] An image
            displaying the gradient orientation for each pixel in the
            input image
        gradient_intensity (eta.core.types.ImageFile): [None] An image
            displaying the gradient intensity for each pixel in the input
            image
    '''

    def __init__(self, d):
        self.input_image = self.parse_string(d, "input_image")
        self.sobel_horizontal_result = self.parse_string(
            d, "sobel_horizontal_result")
        self.sobel_vertical_result = self.parse_string(
            d, "sobel_vertical_result")
        self.image_edges = self.parse_string(d, "image_edges")
        self.gradient_orientation = self.parse_string(
            d, "gradient_orientation", default=None)
        self.gradient_intensity = self.parse_string(
            d, "gradient_intensity", default=None)


class ParametersConfig(Config):
    '''Parameter configuration settings.

    Parameters:
        high_threshold (eta.core.types.Number): The upper threshold to use
            during double thresholding
        low_threshold (eta.core.types.Number): The lower threshold to use
            during double thresholding
    '''

    def __init__(self, d):
        self.high_threshold = self.parse_number(d, "high_threshold")
        self.low_threshold = self.parse_number(d, "low_threshold")


def _create_intensity_orientation_matrices(Gx, Gy):
    '''Creates two matrices: one for intensity and one for orientation.
    The intensity at each pixel is defined as sqrt(Gx^2 + Gy^2) and
    the orientation of each pixel is defined as arctan(Gy/Gx).

    Args:
        Gx: the result of convolving with the "sobel_horizontal" kernel
        Gy: the result of convolving with the "sobel_vertical" kernel

    Returns:
        (g_intensity, orientation): a tuple with the first element as
            the intensity matrix and the second element as the
            orientation matrix.
    '''
    
    g_intensity = np.sqrt(np.square(Gx)+np.square(Gy))
    orientation = np.arctan2(Gy,Gx)
    return g_intensity, orientation

def _choose_orientation_mode( theta ):
    '''
    mode 0: -pi/8 < theta <=  pi/8 U theta > 7/8pi U theta<= -7/8pi check horiz
    mode 1:  pi/8 < theta <= 3pi/8 U -7pi/8 < theta <= -5pi/8 check 1,3 quad
    mode 2: 3pi/8 < theta <= 5pi/8 U -5pi/8 < theta <= -3pi/8 check vertical
    mode 3: 5pi/8 < theta <= 7pi/8 U -3pi/8 < theta <= -pi/8 check 2,4 quad
    '''
    mode = 0

    if (-np.pi/8 < theta and theta <= np.pi/8) or theta > 7*np.pi/8 or theta <= -7*np.pi/8:
        mode = 0 
    elif (np.pi/8 < theta and theta <= 3*np.pi/8) or (-7*np.pi/8 < theta and theta <= -5*np.pi/8):
        mode = 1
    elif (3*np.pi/8 < theta and theta <= 5*np.pi/8) or (-5*np.pi/8 < theta and theta <= -3*np.pi/8):
        mode = 2
    elif (5*np.pi/8 < theta and theta <= 7*np.pi/8) or (-3*np.pi/8 < theta and theta <= -np.pi/8):
        mode = 3
    return mode


def _non_maximum_suppression(g_intensity, orientation, input_image):
    '''Performs non-maximum suppression. If a pixel is not a local maximum
    (not bigger than it's neighbors with the same orientation), then
    suppress that pixel.

    Args:
        g_intensity: the gradient intensity of each pixel
        orientation: the gradient orientation of each pixel
        input_image: the input image

    Returns:
        g_sup: the gradient intensity of each pixel, with some intensities
            suppressed to 0 if the corresponding pixel was not a local
            maximum
    '''
    input_H = input_image.shape[0]
    input_W = input_image.shape[1]
    
    outputImg = np.zeros((input_H,input_W))

    kernel_H_2 = int((input_H-g_intensity.shape[0])/2) # y of pixel corresponding to gradient(0,0)
    kernel_W_2 = int((input_W-g_intensity.shape[1])/2) # x of pixel corresponding to gradient(0,0)

    for i in range(g_intensity.shape[0]): #vertical(y)
        for j in range(g_intensity.shape[1]): #horizontal(x)
            if(g_intensity[i, j]>0):
                cur_mode = _choose_orientation_mode(orientation[i,j])
                if cur_mode == 0:
                    prev_x = j-1
                    prev_y = i
                    next_x = j+1
                    next_y = i
                elif cur_mode == 1:
                    prev_x = j-1
                    prev_y = i-1
                    next_x = j+1
                    next_y = i+1
                elif cur_mode == 2:
                    prev_x = j
                    prev_y = i-1
                    next_x = j
                    next_y = i+1
                elif cur_mode == 3:
                    prev_x = j-1
                    prev_y = i+1
                    next_x = j+1
                    next_y = i-1
                cur_bool = True
                if prev_x>=0 and prev_x<g_intensity.shape[1] and prev_y>=0 and prev_y<g_intensity.shape[0]:
                    if(g_intensity[prev_y,prev_x]>g_intensity[i,j]):
                        cur_bool=False
                if next_x>=0 and next_x<g_intensity.shape[1] and next_y>=0 and next_y<g_intensity.shape[0]:
                    if(g_intensity[next_y,next_x]>g_intensity[i,j]):
                        cur_bool=False
                if cur_bool:
                    outputImg[i+kernel_H_2,j+kernel_W_2] = g_intensity[i, j]    
    return outputImg


def _double_thresholding(g_suppressed, low_threshold, high_threshold):
    '''Performs a double threhold. All pixels with gradient intensity larger
    than 'high_threshold' are considered strong edges, all pixels with gradient
    intensity in between 'high_threshold' and 'low_threshold' are considered
    weak edges, and all pixels with gradient intensity smaller than
    'low_threshold' are suppressed to 0.

    Args:
        g_suppressed: the gradient intensities of all pixels, after
            non-maxiumum suppression
        low_threshold: the lower threshold in double thresholding
        high_threshold: the higher threshold in double thresholding

    Returns:
        g_thresholded: the result of double thresholding
    '''
    g_thresholded = np.zeros((g_suppressed.shape[0],g_suppressed.shape[1]))

    for i in range(g_thresholded.shape[0]):
        for j in range(g_thresholded.shape[1]):
            if g_suppressed[i,j]<low_threshold:
                g_thresholded[i,j]=0
            elif g_suppressed[i,j]>high_threshold:
                g_thresholded[i,j]=high_threshold
            else:
                g_thresholded[i,j]=low_threshold
    return g_thresholded


def _hysteresis(g_thresholded, low_threshold, high_threshold):
    '''Performs hysteresis. If a weak pixel is connected to a strong pixel,
    then the weak pixel is marked as strong. Otherwise, it is suppressed.
    The result will be an image with only strong pixels.

    Args:
        g_thresholded: the result of double thresholding

    Returns:
        g_strong: an image with only strong edges
    '''
    g_strong = np.zeros((g_thresholded.shape[0],g_thresholded.shape[1]))
    for i in range(1, g_thresholded.shape[0]-1):
        for j in range(1, g_thresholded.shape[1]-1):
            if g_thresholded[i,j]==high_threshold:
                g_strong[i,j]=high_threshold
            elif g_thresholded[i,j]==low_threshold:
                if (g_thresholded[i,j-1]==high_threshold or g_thresholded[i,j+1]==high_threshold 
                    or g_thresholded[i+1,j]==high_threshold or g_thresholded[i-1,j]==high_threshold 
                    or g_thresholded[i+1,j+1]==high_threshold or g_thresholded[i+1,j-1]==high_threshold
                    or g_thresholded[i-1,j+1]==high_threshold or g_thresholded[i-1,j-1]==high_threshold):
                    g_strong[i,j] = high_threshold
                else:
                    g_strong[i,j]=0
    return g_strong
    


def _perform_canny_edge_detection(canny_edge_config):
    for data in canny_edge_config.data:
        in_img = etai.read(data.input_image)
        sobel_horiz = np.load(data.sobel_horizontal_result)["filtered_matrix"]
        sobel_vert = np.load(data.sobel_vertical_result)["filtered_matrix"]
        (g_intensity, orientation) = _create_intensity_orientation_matrices(
                                        sobel_horiz,
                                        sobel_vert)
        if data.gradient_intensity is not None:
            etai.write(g_intensity, data.gradient_intensity)
        if data.gradient_orientation is not None:
            etai.write(orientation, data.gradient_orientation)
        np.save('out/gradient_orientation.npy',orientation)
        etai.write(g_intensity, 'out/g_intensity.jpg')
        g_suppressed = _non_maximum_suppression(g_intensity, orientation,
                                                in_img)
        etai.write(g_suppressed, 'out/g_suppressed.jpg')
        g_thresholded = _double_thresholding(
                            g_suppressed,
                            canny_edge_config.parameters.low_threshold,
                            canny_edge_config.parameters.high_threshold)
        g_strong = _hysteresis(g_thresholded,
                        canny_edge_config.parameters.low_threshold,
                        canny_edge_config.parameters.high_threshold)
        g_strong = g_strong.astype(int)
        #sio.savemat('/home/zixu/canny.mat',dict([('in_img',in_img),('sobel_horiz',sobel_horiz),('sobel_vert',sobel_vert),('g_intensity',g_intensity),('orientation',orientation),('g_suppressed',g_suppressed),('g_thresholded',g_thresholded),('g_strong',g_strong)]))
        etai.write(g_strong, data.image_edges)
        #etai.write(g_intensity, data.gradient_intensity)
        


def run(config_path, pipeline_config_path=None):
    '''Run the canny edge detector module.

    Args:
        config_path: path to a ConvolutionConfig file
        pipeline_config_path: optional path to a PipelineConfig file
    '''
    canny_edge_config = CannyEdgeConfig.from_json(config_path)
    etam.setup(canny_edge_config, pipeline_config_path=pipeline_config_path)
    _perform_canny_edge_detection(canny_edge_config)


if __name__ == "__main__":
    run(*sys.argv[1:])
