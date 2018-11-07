#!/usr/bin/env python
'''
A module for classifying the SVHN (Street View House Number) dataset
using an eigenbasis.

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

import cv2
import gzip
import matplotlib.pyplot as plt
import numpy as np
import os
import struct
import sys

from dig_struct import *

from eta.core.config import Config, ConfigError
import eta.core.image as etai
import eta.core.module as etam
import eta.core.serial as etas
import scipy.io as sio


import matplotlib.pyplot as plt


class SVHNClassificationConfig(etam.BaseModuleConfig):
    '''SVHN Classification configuration settings.

    Attributes:
        data (DataConfig)
    '''

    def __init__(self, d):
        super(SVHNClassificationConfig, self).__init__(d)
        self.data = self.parse_object_array(d, "data", DataConfig)
        self.parameters = self.parse_object(d, "parameters", ParametersConfig)


class DataConfig(Config):
    '''Data configuration settings.

    Inputs:
        svhn_test (eta.core.types.File): the path of the tar.gz file
            containing all test images for the SVHN dataset and the
            file "digitStruct.mat".
        mnist_train_images (eta.core.types.File): the path of the training
            images for the MNIST dataset
        mnist_train_labels (eta.core.types.File): the path of the training
            labels for the MNIST dataset
        mnist_test_images (eta.core.types.File): the path of the test images
            for the MNIST dataset
        mnist_test_labels (eta.core.types.File): the path of the test labels
            for the MNIST dataset

    Outputs:
        error_rate_file (eta.core.types.JSONFile): the JSON file that will
            hold the error rates computed for the MNIST test set and the
            SVHN test set
    '''

    def __init__(self, d):
        self.svhn_test_path = self.parse_string(d, "svhn_test")
        self.mnist_train_images_path = self.parse_string(d, "mnist_train_images")
        self.mnist_train_labels_path = self.parse_string(d, "mnist_train_labels")
        self.mnist_test_images_path = self.parse_string(d, "mnist_test_images")
        self.mnist_test_labels_path = self.parse_string(d, "mnist_test_labels")
        self.error_rate_file = self.parse_string(d, "error_rate_file")

class ParametersConfig(Config):
    '''Parameter configuration settings.

    Parameters:
        no_correspondence (eta.core.types.Number): [4] the number of
            points to use when computing the homography
    '''

    def __init__(self, d):
        self.num_PCA = self.parse_number(d, "num_PCA", default=20)
        self.num_neighbor = self.parse_number(d, "num_neighbor", default=1)


def read_idx(mnist_filename):
    '''Reads both the MNIST images and labels.

    Args:
        mnist_filename: the path of the MNIST file

    Returns:
        data_as_array: a numpy array corresponding to the data within the
            MNIST file. For example, for MNIST images, the output is a
            (n, 28, 28) numpy array, where n is the number of images.
    '''
    with gzip.open(mnist_filename) as f:
        zero, data_type, dims = struct.unpack('>HBB', f.read(4))
        shape = tuple(struct.unpack('>I',
                      f.read(4))[0] for d in range(dims))
        data_as_array = np.fromstring(f.read(), dtype=np.uint8).reshape(shape)
        return data_as_array


def Get_PCA(train_data, num_basis):
    '''
    get first num_basis of principle components in train_data
    Args:
        train_data: input data
        num_basis: take first num_basis principle basis
    '''
    print("\n****** Start calculating PCA ******")
    num_data, dim_data_h, dim_data_w=train_data.shape
    train_data_3D = train_data.astype(np.float)
    data_class_2DMat = np.reshape(train_data_3D,(num_data,dim_data_h*dim_data_w))
    #make data to be centered at 0
    data_mean = np.mean(data_class_2DMat,axis=0)
    train_data_shifted = data_class_2DMat-data_mean
    #Use svd to get PCA of the data
    U,S,Vh = np.linalg.svd(train_data_shifted,full_matrices=False)
    #Unit vector of principle direction
    PCA_dir = Vh.T
    # US is the score of each data
    PCA_socre = np.matmul(U,np.diag(S))
    # extract first num_basis basis.
    PCA_dir_out = PCA_dir[:,:num_basis]
    PCA_socre_out = PCA_socre[:,:num_basis]
    
    print("****** Successfully obtain first "+str(num_basis)+" PCA ******")
    return PCA_dir_out, PCA_socre_out, data_mean

def kNN(img2classify, train_PCA, train_PCA_socre, train_mean, train_label, num_neighbor):
    '''
    k-nearest neighboor
    Args:
        img2classify: [28*28] data to be label
    '''
    
    img2data = np.reshape(img2classify, [1,train_PCA.shape[0]])
    img2data_shifted = img2data-train_mean
    dataScore = np.matmul(img2data_shifted, train_PCA)
    #get Eculidian distance between data and all train sample
    diff  = train_PCA_socre - dataScore
    Ecu_dis = np.linalg.norm(diff, axis=1)
    sort_dis_idx = np.argsort(Ecu_dis)
    #build historgram of k nearest neighbor
    k_hist = np.zeros(10)
    for i in range(num_neighbor):
        cur_label = train_label[sort_dis_idx[i]]
        k_hist[cur_label]+=1
    data_label = np.argmax(k_hist)
    return data_label

def test_MNIST(train_PCA, train_mean, train_PCA_socre, train_label, test_img, test_label, num_neighbor, num_to_test):
    print("\n****** Start testing MNIST dataset with "+str(num_neighbor)+" nearest neighbors ******")
    num_test_data = test_img.shape[0]
    test_img_3D = test_img.astype(np.float)
    test_output_class = np.zeros(num_test_data)
    test_output_TF = np.ones(num_test_data)
    num_false_class = 0
    if num_to_test==-1:
        num_to_test = num_test_data
    #test all images
    for i in range(num_to_test):
        cur_class = kNN(test_img_3D[i,:,:], train_PCA, train_PCA_socre, train_mean, train_label, num_neighbor)
        test_output_class[i] = cur_class
        if cur_class != test_label[i]:
            test_output_TF[i]=0
            num_false_class+=1
    Error_rate = num_false_class/num_to_test
    print("****** Finish testing MNIST dataset. Totoal "+str(num_to_test)+" images tested. "+
        str(num_to_test-num_false_class)+" images are labeled correctly. Error rate = "+str(Error_rate*100)+"% ******")
    return Error_rate, test_output_class, test_output_TF

def test_MNIST_sample(train_PCA, train_mean, train_PCA_socre, train_label, test_img, test_label, num_neighbor):
    print("\n****** Start testing MNIST dataset with "+str(num_neighbor)+" nearest neighbors ******")
    num_test_data = test_img.shape[0]
    test_img_3D = test_img.astype(np.float)
    false_sample = False
    correct_sample = False
    #find one correct and one 
    for i in range(num_test_data):
        cur_class = kNN(test_img_3D[i,:,:], train_PCA, train_PCA_socre, train_mean, train_label, num_neighbor)
        if cur_class != test_label[i] and false_sample==False:
            F_img = test_img[i,:,:]
            F_label = [cur_class,test_label[i]]
            false_sample = True
            
        elif cur_class == test_label[i] and correct_sample==False:
            T_img = test_img[i,:,:]
            T_label = [cur_class,test_label[i]]
            correct_sample=True
        
        if correct_sample and false_sample:
            break
    
    _, axs = plt.subplots(ncols=2, nrows=1)
    axs[0].imshow(T_img, cmap='gray')
    axs[0].set_title("True Class: "+str(T_label[1])+" ,Classified as "+str(T_label[0]))
    axs[1].imshow(F_img, cmap='gray')
    axs[1].set_title("True Class: "+str(F_label[1])+" ,Classified as "+str(F_label[0]))
    plt.show()
        
def resize_SVHN_img(input_img):
    height, width = input_img.shape
    bg_color = input_img[-1,-1]
    '''
    # padding the image to make it square
    if width > height:
        img_padding = np.zeros((width,width),dtype=np.uint8)
        y_start = int(width/2-height/2)
        y_end = int(y_start+height)
        img_padding[y_start:y_end,:]=input_img
    else:
        img_padding = np.zeros((height,height),dtype=np.uint8)
        x_start = int(height/2-width/2)
        x_end = int(x_start+width)
        img_padding[:,x_start:x_end] = input_img
        '''
    output_img = etai.resize(input_img, width=28, height=28, interpolation=cv2.INTER_AREA)
    if bg_color>100:
        output_img = 255 - output_img
    '''
    #plot for debug
    _, axs = plt.subplots(ncols=3, nrows=1)
    axs[0].matshow(input_img, cmap='gray')
    axs[1].matshow(img_padding, cmap='gray')
    axs[2].matshow(output_img, cmap='gray')
    plt.show()
    '''
    return output_img

def test_SVHN(train_PCA, train_mean, train_PCA_socre, train_label, test_box, test_data_path, num_neighbor, num_to_test):
    print("\n****** Start testing SVHN dataset with "+str(num_neighbor)+" nearest neighbors ******")
    Img_data_all = test_box.getAllDigitStructure_ByDigit()
    print("****** Loaded "+str(len(Img_data_all))+" images from SVHN dataset ******")
    #initialize some values for analysis
    num_total_test = 0
    num_false_class = 0
    test_histogram =[]
    test_false_histogram=[]
    if num_to_test == -1:
        num_to_test = len(Img_data_all)

    for cur_img_data in Img_data_all[:num_to_test]:
        cur_img_gray = etai.read(test_data_path+"/"+cur_img_data['filename'], flag=cv2.IMREAD_GRAYSCALE)
        height, width = cur_img_gray.shape
        cur_num_bbox = len(cur_img_data['boxes'])
        num_total_test+=cur_num_bbox
        for j in range(cur_num_bbox):
            #check bounding box does not lay outside of image
            cur_box = cur_img_data['boxes'][j]
            x_idx_1 = int(max(cur_box['left'],0))
            x_idx_2 = int(min(cur_box['left']+cur_box['width'],width))
            y_idx_1 = int(max(cur_box['top'],0))
            y_idx_2 = int(min(cur_box['top']+cur_box['height'],height))
            #extract bounded image and resize to 28*28
            box_img = cur_img_gray[y_idx_1:y_idx_2, x_idx_1:x_idx_2]
            box_img_resize = resize_SVHN_img(box_img)
            #test with knn
            box_class = kNN(box_img_resize, train_PCA, train_PCA_socre, train_mean, train_label, num_neighbor)
            if box_class == 0:
                box_class = 10
            #save histogram
            cur_box_hist = cur_box
            cur_box_hist['filename'] = cur_img_data['filename']
            cur_box_hist['classify'] = box_class
            test_histogram.append(cur_box_hist)
            if box_class!=cur_box['label']:
                num_false_class+=1
                test_false_histogram.append(cur_box_hist)
        
    Error_rate = num_false_class/num_total_test
    print("****** Finish testing SVHN dataset. Totoal "+str(num_total_test)+" images tested. "+
        str(num_total_test-num_false_class)+" images are labeled correctly. Error rate = "+str(Error_rate*100)+"% ******")
    return Error_rate, test_histogram, test_false_histogram


def test_SVHN_sample(train_PCA, train_mean, train_PCA_socre, train_label, test_box, test_data_path, num_neighbor):
    print("\n****** Start testing SVHN dataset with "+str(num_neighbor)+" nearest neighbors ******")
    Img_data_all = test_box.getAllDigitStructure_ByDigit()
    print("****** Loaded "+str(len(Img_data_all))+" images from SVHN dataset ******")
    false_sample = False
    correct_sample = False
    for cur_img_data in Img_data_all:
        cur_img_gray = etai.read(test_data_path+"/"+cur_img_data['filename'], flag=cv2.IMREAD_GRAYSCALE)
        height, width = cur_img_gray.shape
        cur_num_bbox = len(cur_img_data['boxes'])
        for j in range(cur_num_bbox):
            #check bounding box does not lay outside of image
            cur_box = cur_img_data['boxes'][j]
            x_idx_1 = int(max(cur_box['left'],0))
            x_idx_2 = int(min(cur_box['left']+cur_box['width'],width))
            y_idx_1 = int(max(cur_box['top'],0))
            y_idx_2 = int(min(cur_box['top']+cur_box['height'],height))
            #extract bounded image and resize to 28*28
            box_img = cur_img_gray[y_idx_1:y_idx_2, x_idx_1:x_idx_2]
            box_img_resize = resize_SVHN_img(box_img)
            #test with knn
            cur_class = kNN(box_img_resize, train_PCA, train_PCA_socre, train_mean, train_label, num_neighbor)
            if cur_class == 0:
                cur_class = 10
            if cur_class != cur_box['label'] and false_sample==False:
                F_img = box_img
                F_label = [cur_class,int(cur_box['label'])]
                false_sample = True
            elif cur_class == cur_box['label'] and correct_sample==False:
                T_img = box_img
                T_label = [cur_class,int(cur_box['label'])]
                correct_sample=True
            if correct_sample and false_sample:
                break
        if correct_sample and false_sample:
                break
    
    _, axs = plt.subplots(ncols=2, nrows=1)
    axs[0].imshow(T_img,cmap='gray')
    axs[0].set_title("True Class: "+str(T_label[1])+" ,Classified as "+str(T_label[0]))
    axs[1].imshow(F_img,cmap='gray')
    axs[1].set_title("True Class: "+str(F_label[1])+" ,Classified as "+str(F_label[0]))
    plt.show()
               
    

def visualize_PCA(vis_PCA):
    _, axs = plt.subplots(ncols=5, nrows=2)
    for i in range(2):
        for j in range(5):
            idx = i*5+j
            ax = axs[i,j]
            cur_PCA_2D = np.reshape(vis_PCA[:,idx],(28,28))
            ax.matshow(cur_PCA_2D,cmap='rainbow')
            ax.set_title("# "+str(idx+1)+" PCA")
            ax.axis('off')
    plt.show()


def run(config_path, pipeline_config_path=None):
    '''Run the SVHN Classification Module.

    Args:
        config_path: path to a ConvolutionConfig file
        pipeline_config_path: optional path to a PipelineConfig file
    '''
    svhn_config = SVHNClassificationConfig.from_json(config_path)
    etam.setup(svhn_config, pipeline_config_path=pipeline_config_path)
    for data in svhn_config.data:
        # Read all the MNIST data as numpy arrays
        mnist_train_images = read_idx(data.mnist_train_images_path)
        mnist_train_labels = read_idx(data.mnist_train_labels_path)
        mnist_test_images = read_idx(data.mnist_test_images_path)
        mnist_test_labels = read_idx(data.mnist_test_labels_path)
        
        # Read the digitStruct.mat from the SVHN test folder
        base_svhn_path = data.svhn_test_path
        dsf = DigitStructFile(base_svhn_path + "/digitStruct.mat")
       
        # training
        PCA_dir_train, PCA_socre_train, mean_train = Get_PCA(mnist_train_images, svhn_config.parameters.num_PCA)
        
        # testing
        svhn_error_rate, _, _ = test_SVHN(PCA_dir_train, mean_train, PCA_socre_train, mnist_train_labels, 
            dsf, base_svhn_path, svhn_config.parameters.num_neighbor, 300 )
        
        mnist_error_rate,_,_ = test_MNIST(PCA_dir_train, mean_train, PCA_socre_train, mnist_train_labels, 
            mnist_test_images, mnist_test_labels, svhn_config.parameters.num_neighbor, -1)

        '''
        # visualize
        PCA_vis, _, _ = Get_PCA(mnist_train_images, 10)
        visualize_PCA(PCA_vis)

        # get example
        test_MNIST_sample(PCA_dir_train, mean_train, PCA_socre_train, mnist_train_labels, 
            mnist_test_images, mnist_test_labels, svhn_config.parameters.num_neighbor)
        test_SVHN_sample(PCA_dir_train, mean_train, PCA_socre_train, mnist_train_labels, 
            dsf, base_svhn_path, svhn_config.parameters.num_neighbor)
        '''
        
        error_rate_dic = defaultdict(lambda: defaultdict())
        error_rate_dic["error_rates"]["mnist_error_rate"] = mnist_error_rate
        error_rate_dic["error_rates"]["svhn_error_rate"] = svhn_error_rate
        etas.write_json(error_rate_dic, data.error_rate_file)


if __name__ == "__main__":
    run(*sys.argv[1:])
