#!/usr/bin/env python
'''
A module for determining the locations of corners in an image using
the Harris Corner Detection algorithm.

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
from scipy import signal
from scipy import io

from eta.core.config import Config, ConfigError
import eta.core.image as etai
import eta.core.module as etam
import eta.core.utils as etau


class HarrisCornerConfig(etam.BaseModuleConfig):
    '''Harris corner configuration settings.

    Attributes:
        data (DataConfig)
        parameters (ParametersConfig)
    '''

    def __init__(self, d):
        super(HarrisCornerConfig, self).__init__(d)
        self.data = self.parse_object_array(d, "data", DataConfig)
        self.parameters = self.parse_object(d, "parameters", ParametersConfig)
        self._validate()

    def _validate(self):
        '''Ensures the right inputs are given to the module before
        actual computation begins.
        '''
        for data in self.data:
            if ((data.corners_img_after_sup or data.corners_img_before_sup)
                    and not data.input_image):
                raise ConfigError(
                    "ERROR! Original image required for visualization.")


class DataConfig(Config):
    '''Data configuration settings.

    Inputs:
        input_image (eta.core.types.Image): [None] The input image
        sobel_horizontal_result (eta.core.types.NpzFile): The result of
            convolving the original image with the "sobel_horizontal" kernel.
            This will give the value of Gx (the gradient in the x
            direction).
        sobel_vertical_result (eta.core.types.NpzFile): The result of
            convolving the original image with the "sobel_vertical" kernel.
            This will give the value of Gy (the gradient in the y
            direction).
    Outputs:
        corner_locations (eta.core.types.NpzFile): [None] The location of every
            corner detected by the Harris Corner algorithm, sorted by
            confidence
        corners_img_before_sup (eta.core.types.ImageFile): [None] The corner
            response image before suppression
        corners_img_after_sup (eta.core.types.ImageFile): [None] The corner
            response image after suppression
    '''

    def __init__(self, d):
        self.input_image = self.parse_string(
            d, "input_image", default=None)
        self.sobel_horizontal_result = self.parse_string(
            d, "sobel_horizontal_result")
        self.sobel_vertical_result = self.parse_string(
            d, "sobel_vertical_result")
        self.corner_locations = self.parse_string(
            d, "corner_locations", default=None)
        self.corners_img_before_sup = self.parse_string(
            d, "corners_img_before_sup", default=None)
        self.corners_img_after_sup = self.parse_string(
            d, "corners_img_after_sup", default=None)
        self.corners_response_img = self.parse_string(
            d, "corners_response", default=None)


class ParametersConfig(Config):
    '''Parameter configuration settings.

    Parameters:
        window_half_size (eta.core.types.Number): [7] The half-size of the window
        threshold (eta.core.types.Number): [0.001] The lower-bound threshold for
            corner "confidence".
        non_max_suppression_radius (eta.core.types.Number): [2] The
            circular radius used when applying non-maximum suppression to the
            corner response.
    '''

    def __init__(self, d):
        self.window_half_size = self.parse_number(
            d, "window_half_size", default=7)
        self.threshold = self.parse_number(
            d, "threshold", default=0.001)
        self.non_max_radius = self.parse_number(
            d, "non_max_suppression_radius", default=2)


def _get_harris_corner(Gx, Gy, window_half_size, threshold):
    '''Performs Harris corner detection. The output will be a matrix, the
    same size as Gx and Gy, where every element (pixel) contains the
    corner strength at that location. The corner strength is defined to
    be the minimum eigenvalue of the structure tensor at that pixel
    location.

    Args:
        Gx: The x-derivative of the image, which is the output of convolving
            with the horizontal sobel kernel
        Gy: the y-derivative of the image, which is the output of convolving
            with the vertical sobel kernel
        window_half_size: the half-size of the window to consider when computing
            the structure tensor
        threshold: the lower-bound threshold to use when choosing corners

    Returns:
        corner_response_matrix: a matrix, the same size as Gx and Gy, where
            every element (pixel) contains the corner strength at that
            location.
        corner_location_matrix: an Nx2 matrix, where every row is the location
            of a corner whose response is greater than the threshold. The matrix
            should also be sorted by corner strength, such that the corner
            with the highest corner strength is put in row 0 and the corner
            with the lowest corner strength is put in row N-1.
    '''
    height, width = Gx.shape
    corner_location_matrix = np.array([[],[]]).T #initialize corner_location_matrix 0x2
    corner_response_matrix = np.zeros([height,width])
    #full_response_matrix = np.zeros([height,width]) # for debug
    Gx2 = np.multiply(Gx,Gx) #Ix^2(y,x)
    Gy2 = np.multiply(Gy,Gy) #Iy^2(y,x)
    Gxy = np.multiply(Gx,Gy) #I_xy(y,x)
    kernel_sum = np.ones([window_half_size*2+1,window_half_size*2+1])
    sum_Gx2 = signal.convolve2d(Gx2, kernel_sum, mode='valid')
    sum_Gy2 = signal.convolve2d(Gy2, kernel_sum, mode='valid')
    sum_Gxy = signal.convolve2d(Gxy, kernel_sum, mode='valid')
    for i in range(sum_Gx2.shape[0]):
        for j in range(sum_Gx2.shape[1]):
            structureTensor = np.array([[sum_Gx2[i,j], sum_Gxy[i,j]],[sum_Gxy[i, j], sum_Gy2[i, j]]])
            eigVal = np.linalg.eigvals(structureTensor)
            minEig = min(eigVal)
            #full_response_matrix[i+window_half_size, j+window_half_size] = minEig
            if minEig > threshold:
                corner_response_matrix[i+window_half_size, j+window_half_size] = minEig
                cur_loc = np.array([[j+window_half_size,i+window_half_size]])
                corner_location_matrix = np.append(corner_location_matrix, cur_loc, axis=0)
    return corner_response_matrix, corner_location_matrix


def non_max_suppression(corner_response, radius):
    """Finds corners in a given response map.

    This method uses a circular region to define the non-maxima suppression
    area. For example, let c1 be a corner representing a peak in the Harris
    response map, any corners in the area determined by the circle of radius
    'radius' centered in c1 should not be returned in the peaks array.
    Make sure you account for duplicate and overlapping points.

    Args:
        corner_response: floating-point response map,
            e.g. output from the Harris detector.
        radius: radius of circular region for non-maximal suppression.

    Returns:
        corners_sup: peaks found in response map R, each row must be defined
            as [x, y]. Array size must be N x 2, where N are the number of
            points found.
    """
    r_map1 = np.copy(corner_response)
    data_max = np.zeros(r_map1.shape)
    ind = np.nonzero(r_map1)

    for n in range(len(ind[0])):
        i = ind[0][n]
        j = ind[1][n]
        frame = r_map1[max((i-radius), 0):min((i+radius),r_map1.shape[0]),
                       max((j-radius),0):min((j+radius),r_map1.shape[1])]
        if r_map1 [i,j]< np.max(frame): # not local max
            data_max [i,j]  = 0
        elif np.max(data_max[
                max((i-radius), 0):min((i+radius),data_max.shape[0]),
                max((j-radius),0):min((j+radius),data_max.shape[1])])>0:
            # Tie, and already as a max t
            data_max [i,j]  = 0
        else:
            data_max [i,j] = r_map1[i, j]

    col_ind, row_ind = np.nonzero(data_max)
    # You can use the distance as a conditional measure to merge the points.
    # Average of the x and y coordinates of the close points to merge.
    corners = []
    for i in range(len(row_ind)):
        corners.append((row_ind[i], col_ind[i]))
    corners = tuple(corners)

    return np.array(corners)


def _visualize_corners(in_img, corner_locations):
    '''Creates an image that shows the detected corners.

    Args:
        in_img: the original image
        corner_locations: the locations of each detected corner in the
            image, stored in an Nx2 matrix

    Returns:
        out_img: an image with the detected corners colored in red

    '''
    if etai.is_gray(in_img):
        out_img = etai.gray_to_rgb(in_img)
    else:
        out_img = in_img.copy()
    for i in range(corner_locations.shape[0]):
        cv2.circle(out_img,
                   (int(corner_locations[i][0]), int(corner_locations[i][1])),
                   2,
                   (255,0,0),
                   -1)
    return out_img


def _find_corners(harris_corner_config):
    for data in harris_corner_config.data:
        sobel_horiz = np.load(data.sobel_horizontal_result)["filtered_matrix"]
        sobel_vert = np.load(data.sobel_vertical_result)["filtered_matrix"]
        corner_response, corner_locations = _get_harris_corner(
                                sobel_horiz, sobel_vert,
                                harris_corner_config.parameters.window_half_size,
                                harris_corner_config.parameters.threshold)
        corner_locs_after_sup = non_max_suppression(
            corner_response, harris_corner_config.parameters.non_max_radius)
        if data.corner_locations:
            etau.ensure_basedir(data.corner_locations)
            np.savez(data.corner_locations,
                     corner_locations=corner_locs_after_sup)
        if data.corners_img_before_sup or data.corners_img_after_sup:
            in_img = etai.read(data.input_image)
            if data.corners_response_img:
                etai.write( corner_response*255, data.corners_response_img)
            if data.corners_img_before_sup:
                corners_viz_before_sup = _visualize_corners(in_img,
                    corner_locations)
                etai.write(corners_viz_before_sup, data.corners_img_before_sup)
            if data.corners_img_after_sup:
                corners_viz_after_sup = _visualize_corners(in_img,
                    corner_locs_after_sup)
                etai.write(corners_viz_after_sup, data.corners_img_after_sup)


def run(config_path, pipeline_config_path=None):
    '''Run the Harris Corner module.

    Args:
        config_path: path to a HarrisCornerConfig file
        pipeline_config_path: optional path to a PipelineConfig file
    '''
    harris_corner_config = HarrisCornerConfig.from_json(config_path)
    etam.setup(harris_corner_config, pipeline_config_path=pipeline_config_path)
    _find_corners(harris_corner_config)


if __name__ == "__main__":
    run(*sys.argv[1:])
