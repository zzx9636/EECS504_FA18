#!/usr/bin/env python
'''
For debug 
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

base_svhn_path = "../data/test"
dsf = DigitStructFile(base_svhn_path + "/digitStruct.mat")
Img_data_all = dsf.getAllDigitStructure_ByDigit()
cur_img_data=Img_data_all[345]
cur_img_gray = etai.read(base_svhn_path+"/"+cur_img_data['filename'], flag=cv2.IMREAD_GRAYSCALE)
cv2.imshow('img1',cur_img_gray)
cv2.waitKey()