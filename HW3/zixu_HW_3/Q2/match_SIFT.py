from image_crawler import image_retrieval_json, get_queries, FlickrAPI
import cv2
import urllib
import numpy as np
import argparse
import sys, os
import glob
import json
from matplotlib import pyplot as plt


"""
    Modify this section to reflect your data and specific search
    1. APIKey and Secret, this is got from flicker official website
"""
flickrAPIKey = "b653e65cf5ffd83d7584e5c860627ae8"  # API key
flickrSecret = "2df09d4260333f44"                  # shared "secret"
desired_photos = 250

ref_img = cv2.imread('./hw3_img.jpg',cv2.IMREAD_GRAYSCALE)
ref_img = cv2.resize(ref_img, (600,1060), interpolation=cv2.INTER_AREA)

sift = cv2.xfeatures2d.SIFT_create()
kp_ref, des_ref = sift.detectAndCompute(ref_img, None)
'''
img1 = cv2.drawKeypoints(ref_img, kp_ref, None, color=(0,255,0), flags=0)
plt.imshow(img1), plt.show()
'''


def parse_args():
    parser = argparse.ArgumentParser(description='Flicker crawler')
    parser.add_argument('--query', dest='query_list', help='list to be queried',
                        type=str, required=True)
    parser.add_argument('--json', dest='json_loc', help='list to be queried',
                        type=str, required=False, default="")
    args = parser.parse_args()
    return args

def retrive_json(save_diactory):
    files_list = glob.glob(save_diactory+'/*.json')
    return files_list


def match_img_sift(input_img, draw_bool):
    kp2, des2 = sift.detectAndCompute(input_img,None)
    img2 = cv2.drawKeypoints(input_img, kp2, None, color=(0,255,0), flags=0)
    '''
    use FLANN to find match
    '''
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks=50)

    flann = cv2.FlannBasedMatcher(index_params,search_params)
    matches = flann.knnMatch(des2, des_ref, k=2)
    
    matchesMask = [[0, 0] for i in range(len(matches))]
    num_match = 0
    
    '''
    use ratio test to find appropriate match
    '''
    for i, (m, n) in enumerate(matches):
        if m.distance < 0.65*n.distance and m.distance<200:
            '''
            ref_kp = kp_ref[m.trainIdx].pt
            test_kp = kp2[m.queryIdx].pt
            print(ref_kp)
            print(test_kp)
            print(m.distance)
            print('\n')
            '''
            matchesMask[i] = [1, 0]
            num_match += 1
    '''
    debug
    '''
    if draw_bool:
        draw_params = dict(matchColor=(0, 255, 0),
                        singlePointColor=(255, 0, 0),
                        matchesMask=matchesMask,
                        flags=0)
        print(str(num_match)+' matches\n')
        img3 = cv2.drawMatchesKnn(img2, kp2, ref_img, kp_ref,  matches, None, **draw_params)
        plt.imshow(img3),plt.show()
    return num_match




def retrive_img(json_file_location, draw_bool):
    '''
    1. read image list from json files
    '''
    ##print(json_file_location)
    with open(json_file_location) as json_data:
        dict_list = json.load(json_data)
    max_match = 0
    '''
    for each image download online, find number of match with reference image
    '''
    for img_dict in dict_list:
        cur_url = 'http://farm1.staticflickr.com/'+img_dict['server']+'/'+img_dict['id']+'_'+img_dict['secret']+'.jpg'
        cur_img = url_to_image(cur_url)
        cur_match = match_img_sift(cur_img, draw_bool)
        max_match = max(cur_match,max_match)
        '''
        #show images
        cv2.imshow("Image", cur_img)
        cv2.waitKey(5)
        '''
    '''
    return the max number of match among all searched images
    '''
    print("Max match is "+str(max_match))
    return max_match


def retrive_img_idx(json_file_location):
    ## for debug purpose
    print(json_file_location)
    with open(json_file_location) as json_data:
        dict_list = json.load(json_data)
    img_dict = dict_list[2]
    cur_url = 'http://farm1.staticflickr.com/'+img_dict['server']+'/'+img_dict['id']+'_'+img_dict['secret']+'.jpg'
    cur_img = url_to_image(cur_url)
    
    return cur_img


def url_to_image(url):
	# download the image, convert it to a NumPy array, and then read
	# it into OpenCV format
	resp = urllib.urlopen(url)
	image = np.asarray(bytearray(resp.read()), dtype="uint8")
	image = cv2.imdecode(image, cv2.IMREAD_GRAYSCALE)
 
	# return the image
	return image


if __name__ == '__main__':
    args = parse_args()
    pos_queries, num_queries = get_queries(args.query_list)
    print( 'positive queries:  ')
    #print( pos_queries)
    print( 'num_queries = ' + str(num_queries))

    flicker_api = FlickrAPI(flickrAPIKey, flickrSecret)
    '''
    for current_tag in range(0, num_queries):
        image_retrieval_json(flicker_api, pos_queries[current_tag]+' facade','list_json')
    '''
    json_list = retrive_json('./list_json') 
    max_match = retrive_img('/home/extra_disk/GoogleDrive/UMich Academic/Fall 2018/EECS504/HW3/Flicrawler/query_imgs/list_json/Orvieto_Cathedral_facade.json', False)
    print(max_match)
