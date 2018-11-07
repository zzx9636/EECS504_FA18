#!/usr/bin/python

# Image querying script based on the code by Tamara Berg,James Hays and Haozhi Qi
# Modified by Zixu Zhang for EECS 504
#

import sys, os
import socket
import time
import argparse
import json
from flickrapi2 import FlickrAPI

socket.setdefaulttimeout(30)
# 30 second time out on sockets before they throw
# an exception.  I've been having trouble with urllib.urlopen hanging in the
# flickr API.  This will show up as exceptions.IOError.
# The time out needs to be pretty long, it seems, because the flickr servers can be slow
# to respond to our big searches.

"""
    Modify this section to reflect your data and specific search
    1. APIKey and Secret, this is got from flicker official website
"""
flickrAPIKey = "b653e65cf5ffd83d7584e5c860627ae8"  # API key
flickrSecret = "2df09d4260333f44"                  # shared "secret"
desired_photos = 250


def parse_args():
    parser = argparse.ArgumentParser(description='Flicker crawler')
    parser.add_argument('--query', dest='query_list', help='list to be queried',
                        type=str, required=True)
    args = parser.parse_args()
    return args


def get_queries(query_list):
    query_file = open(query_list, 'r')
    # aggregate all of the positive and negative queries together.
    pos_queries = []
    num_queries = 0
    for line in query_file:
        if line[0] != '#' and len(line) > 2:
            pos_queries += [line[0:len(line)-1]]
            num_queries += 1
    query_file.close()
    return pos_queries, num_queries


def search_from_current(flicker_api,query_string):
    # number of seconds to skip per query
    time_skip = 62899200*10 #20 years
    current_time = int(time.time())
    threshold_time = current_time - time_skip

    rsp = flicker_api.photos_search(api_key=flickrAPIKey,
                ispublic="1",
                media="photos",
                per_page="50",
                page="1",
                sort="relevance",
                text=query_string,
                min_upload_date=str(threshold_time),
                max_upload_date=str(current_time))
    # we want to catch these failures somehow and keep going.
    time.sleep(1)
    flicker_api.testFailure(rsp)
    total_images = rsp.photos[0]['total']
    print('num_imgs: ' + total_images + '\n')
    return threshold_time, current_time, total_images, rsp

def image_retrieval_json(flicker_api, query_string, save_path):
    print('Search Photos for '+query_string)
    [_, _, total_images, rsp] = search_from_current(flicker_api, query_string)
    total_images=int(total_images)
    if total_images >= 120:
        out_dict = []
        total_images_queried = 0
        if getattr(rsp, 'photos', None):
            if getattr(rsp.photos[0], 'photo', None):
                for b in rsp.photos[0].photo:
                    if b is not None:
                        total_images_queried+=1
                        cur_dict ={}
                        cur_dict["num_img"] = total_images_queried
                        cur_dict["server"] = b['server']
                        cur_dict["id"] = b['id']
                        cur_dict["secret"] = b['secret']
                        out_dict.append(cur_dict)
        if os.path.isdir(save_path)==False:
            os.mkdir(save_path)
        church_name = query_string.replace(" ", "_")
        with open(save_path+'/'+church_name+'.json', 'w') as fp:
            json.dump(out_dict, fp)


if __name__ == '__main__':
    args = parse_args()
    pos_queries, num_queries = get_queries(args.query_list)
    print( 'positive queries:  ')
    print( pos_queries)
    print( 'num_queries = ' + str(num_queries))

    flicker_api = FlickrAPI(flickrAPIKey, flickrSecret)

    for current_tag in range(0, num_queries):
        image_retrieval_json(flicker_api, pos_queries[current_tag]+' facade','list_json')