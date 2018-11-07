#!/usr/bin/env python
'''
A module for finding the line segments of an image, given the output
of Canny Edge Detection.

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
#import cv2 as cv2
#from scipy import signal
from collections import defaultdict



from eta.core.config import Config, ConfigError
import eta.core.image as etai
import eta.core.module as etam
import eta.core.serial as etas


class FindSegmentsConfig(etam.BaseModuleConfig):
    '''Find line segments module configuration settings.

    Attributes:
        data (DataConfig)
    '''

    def __init__(self, d):
        super(FindSegmentsConfig, self).__init__(d)
        self.data = self.parse_object_array(d, "data", DataConfig)


class DataConfig(Config):
    '''Data configuration settings.

    Inputs:
        input_image (eta.core.types.Image): The input image
        canny_edge_output (eta.core.types.Image): The output of canny
            edge detection
        gradient_intensity (eta.core.types.Image): The gradient intensity
            for each pixel in the input image
        gradient_orientation (eta.core.types.Image): The gradient orientation
            for each pixel in the input image

    Outputs:
        line_segments (eta.core.types.JSONFile): A list of coordinate tuples,
            specifying the start and  of each line segment in the image
    '''

    def __init__(self, d):
        self.input_image = self.parse_string(d, "input_image")
        #self.canny_edge_output = self.parse_string(d, "canny_edge_output")
        #self.gradient_intensity = self.parse_string(d, "gradient_intensity")
        #self.gradient_orientation = self.parse_string(
        #    d, "gradient_orientation")
        self.line_segments = self.parse_string(d, "line_segments")


def _hough_vote(input_img,gradient_orientation,angle_step):
    angle_set = np.arange(0,180,angle_step) #set of angles in accumlator with step 1 deg
    height, width = input_img.shape # width x, height y
    #print(height)
    #print(width)
    #print(gradient_orientation.shape)
    diag_length = np.ceil(np.sqrt(height*height+width*width))
    r_set = np.arange(-width,diag_length,1) #set of radius in accumlator with step 1 pixel
    cos_set = np.cos(np.deg2rad(angle_set))
    sin_set = np.sin(np.deg2rad(angle_set))

    #build accumlator i: angle j: radius
    accumMat = np.zeros((len(angle_set),len(r_set)))

    for x in range(width):
        for y in range(height):
            if input_img[y,x]>40: #non edges are black
                grad_theta = np.rad2deg(gradient_orientation[y+1,x+1])
                if (grad_theta<0):
                    grad_theta+=180
                grad_theta=(np.round(grad_theta/angle_step)*angle_step)
                if(grad_theta == 180):
                    grad_theta=0
                theta_idx = np.where(angle_set == grad_theta)[0][0]
                if (theta_idx>5 and theta_idx<85) or (theta_idx>95 and theta_idx<175):
                    start_idx = theta_idx-5
                    end_idx = theta_idx+6
                    if start_idx<0:
                        start_idx=0
                    if end_idx>len(angle_set):
                        end_idx=len(angle_set)
                    r_cur = np.round(x*cos_set[start_idx:end_idx]+y*sin_set[start_idx:end_idx])
                    for i in range(start_idx,end_idx):
                        r_idx = np.where(r_set == r_cur[i-start_idx])[0][0]
                        accumMat[i, r_idx]+=1
                else:
                    r_cur = np.round(x*cos_set[theta_idx]+y*sin_set[theta_idx])
                    r_idx = np.where(r_set == r_cur)[0][0]
                    accumMat[theta_idx, r_idx]+=1
    return accumMat, r_set, angle_set

def _non_max_supress(accumMat, threashold): #perform non max supression for accumator
    sup_accum=accumMat
    h, w = sup_accum.shape
    for i in range(h):
        for j in range(w):
            cur=sup_accum[i,j]
            if i==0 and j==0:
                if cur<accumMat[i,j+1] or cur<accumMat[i+1,j] or cur<accumMat[i+1,j+1]:
                    sup_accum[i,j]=0
                
            elif i==1 and j==w-1:
                if cur<accumMat[i,j-1] or cur<accumMat[i+1,j] or cur<accumMat[i+1,j-1]:
                    sup_accum[i,j]=0
                
            elif i==h-1 and j==1:
                if cur<accumMat[i,j+1] or cur<accumMat[i-1,j] or cur<accumMat[i-1,j+1]:
                    sup_accum[i,j]=0
                
            elif i==h-1 and j==w-1:
                if cur<accumMat[i,j-1] or cur<accumMat[i-1,j] or cur<accumMat[i-1,j-1]:
                    sup_accum[i,j]=0
                
            elif i==1:
                if cur<accumMat[i,j+1] or cur<accumMat[i,j-1] or cur<accumMat[i+1,j] or cur<accumMat[i+1,j+1] or cur<accumMat[i+1,j-1]:
                        sup_accum[i,j]=0
            
            elif i==h-1:
                if cur<accumMat[i,j+1] or cur<accumMat[i,j-1] or cur<accumMat[i-1,j] or cur<accumMat[i-1,j+1] or cur<accumMat[i-1,j-1]:
                    sup_accum[i,j]=0
                
            elif j==1:
                if cur<accumMat[i,j+1] or  cur<accumMat[i+1,j] or cur<accumMat[i-1,j] or cur<accumMat[i+1,j+1] or cur<accumMat[i-1,j+1]: 
                    sup_accum[i,j]=0
                
            elif j==w-1:
                if  cur<accumMat[i,j-1]or cur<accumMat[i+1,j] or cur<accumMat[i-1,j] or cur<accumMat[i+1,j-1] or  cur<accumMat[i-1,j-1]:
                    sup_accum[i,j]=0
                
            else:
                if cur<accumMat[i,j+1] or cur<accumMat[i,j-1]or cur<accumMat[i+1,j] or cur<accumMat[i-1,j] or cur<accumMat[i+1,j+1] or cur<accumMat[i+1,j-1] or cur<accumMat[i-1,j+1] or cur<accumMat[i-1,j-1]:
                    sup_accum[i,j]=0

            #looking for line that voted above threashold
            line_list=np.where(sup_accum>=threashold) #first tuble angle idx second radius idx
            return line_list, sup_accum
             
    
def _remove_repeat(line_list_idx, r_set, angle_set, sup_accum):
    line_list=np.array([angle_set[line_list_idx[0]],r_set[line_list_idx[1]],sup_accum[line_list_idx[0],line_list_idx[1]]]).T
    line_removed = np.array([[],[],[]]).T
    #print(line_removed)
    idx=0 
    while idx<len(line_list_idx[0]):
        ref_angle=line_list[idx,0]
        ref_r = line_list[idx,1]
        j=idx+1
        if ref_angle!=90 and ref_angle!=0:
            while j<len(line_list_idx[0]) and (line_list[j,0]-ref_angle)<=7 and np.absolute(line_list[j,1]-ref_r)<=100:
                j=j+1
        if (j-idx)<=2:
            if max(line_list[idx:j,2])>50:
                line_removed=np.append(line_removed,line_list[idx:j,:],axis=0)
        else :
            temp=line_list[idx:j,:]
            if max(temp[:,2])>50:
                '''
                temp=temp[temp[:,2].argsort()[::-1],]
                idx = np.where(temp[:,2]>=temp[1,2])[0] #pick two with largest score
                temp=temp[idx,:]
                r_diff=np.array([np.absolute(temp[:,1]-temp[0,1])]) #if repeated score, pick the two to maxmize distance
                temp=np.append(temp,r_diff.T,axis=1)
                #print(temp)
                to_be_append=np.array([temp[0,0:3],temp[-1,0:3]])
                line_removed=np.append(line_removed,to_be_append,axis=0)
                '''
                idx_max = np.argmax(temp[:,2]) # find the most voted 
                to_be_append_1 = temp[idx_max,:]
                idx_dis = np.where(np.absolute(temp[:,1]-temp[idx_max,1])>=5) #find the second most voted that is at least 5 pixel away from the most voted one
                temp = temp[idx_dis]
                temp=temp[temp[:,2].argsort()[::-1],] 
                to_be_append = np.array([to_be_append_1,temp[0,:]])
                line_removed=np.append(line_removed,to_be_append,axis=0)
        idx=j
    #print(line_removed)
    return line_removed

def _find_Start_End(input_img,theta, radius):
    #print(theta)
    #print(radius)
    height, width = input_img.shape
    if(theta == 90): #horizontal case
        x_set = np.arange(1,width-1,1)
        y_set = radius*np.ones([len(x_set)])
    elif theta == 0: # vertical case
        y_set = np.arange(1,height-1,1)
        x_set = radius*np.ones([len(y_set)])
    else:
        y_set = np.arange(1,height-1,1)
        x_set = np.round((radius - np.sin(np.deg2rad(theta))*y_set)/np.cos(np.deg2rad(theta)))
    
    Tracked = False #bool if we track the line
    Start=[] #list of start point
    End=[] #list of end point
    for i in range(len(y_set)):
        x_cur=np.int(x_set[i])
        y_cur=np.int(y_set[i])
        if x_cur>0 and x_cur<(width-1):
            Track_pixel = False
            if theta ==0 or theta ==90:
                if input_img[y_cur,x_cur]>50:
                    Track_pixel=True
            elif input_img[y_cur,x_cur]>50 or input_img[y_cur-1,x_cur]>50 or input_img[y_cur+1,x_cur]>50 or input_img[y_cur,x_cur-1]>50 or input_img[y_cur,x_cur+1]>50:
                Track_pixel=True
            if Tracked==False and Track_pixel==True:
                if len(Start)==0 or (i-End[-1])>3:
                    Start.append(i)
                    Tracked=True
                else:
                    End.pop()
                    Tracked=True
            elif Track_pixel==False and Tracked==True:
                if (i-Start[-1])>10: # check if crossing other line
                    End.append(i-1)
                    Tracked=False
                elif len(Start)>len(End):
                    Tracked=False
                    Start.pop()

    line_segment = np.array([x_set[Start],y_set[Start],x_set[End],y_set[End]]).T
    #print(line_segment)
    return line_segment
    


def _hough_line_seg(input_img,gradient_orientation):
    
    smooth_input=input_img
    #smooth_input=signal.convolve2d(input_img,Gaussian_K,'same')
    #smooth_input=np.uint8(smooth_input)
    accumMat, r_set, angle_set = _hough_vote(smooth_input,gradient_orientation,1)
    line_list, sup_accum = _non_max_supress(accumMat, 40)
    line_removed = _remove_repeat(line_list, r_set, angle_set, sup_accum)
    full_segment = np.array([[],[],[],[]]).T
    for cur_lin in line_removed:
        cur_segment = _find_Start_End(smooth_input,cur_lin[0],cur_lin[1])
        full_segment = np.append(full_segment,cur_segment,axis=0)

    #sio.savemat('/home/zixu/hough.mat',dict([('accumMat',accumMat),('angle_set',angle_set),('r_set',r_set),('input_img',smooth_input),('sup_accum',sup_accum)]))
    #sio.savemat('/home/zixu/hough.mat',dict([('full_segment',full_segment),('input_img',input_img)]))
    return full_segment




def _find_line_segments(find_segments_config):
    for data in find_segments_config.data:
        in_img = etai.read(data.input_image)
        gradient_orientation = np.load("out/gradient_orientation.npy")
        full_segment = _hough_line_seg(in_img,gradient_orientation)
        #print(data)
        temp_dict_list=[]
        temp = defaultdict()
        for i in range(full_segment.shape[0]):
            temp = defaultdict()
            temp["No"]=i+1
            temp["coordinates"]=[(full_segment[i,0],full_segment[i,1]),(full_segment[i,2],full_segment[i,3])]
            temp_dict_list.append(temp)
        segment_out = defaultdict(lambda: defaultdict())
        segment_out["Line_segments"]=temp_dict_list
        
        
        
        
        etas.write_json(segment_out, data.line_segments)
        
    


def run(config_path, pipeline_config_path=None):
    '''Run this module to find line segments in the image.

    Args:
        config_path: path to a ConvolutionConfig file
        pipeline_config_path: optional path to a PipelineConfig file
    '''
    #print(config_path)
    find_segments_config = FindSegmentsConfig.from_json(config_path)
    etam.setup(find_segments_config, pipeline_config_path=pipeline_config_path)
    _find_line_segments(find_segments_config)


if __name__ == "__main__":
    run(*sys.argv[1:])
