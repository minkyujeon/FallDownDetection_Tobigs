import os
import cv2
import sys
import math
import time
import numpy as np
import matplotlib.pyplot as plt
from Crop import crop

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

from torch import np
from torch.autograd import Variable

from utils import *
from pose_estimation import *
from scipy.ndimage.filters import gaussian_filter

from V2I import video2image
import PIL
from PIL import Image

import tensorflow as tf
import pandas as pd
import pickle
import datetime

% matplotlib
inline
% config
InlineBackend.figure_format = 'retina'



def crop(t):
    global all_black
    for i in range(np.shape(t)[0]):
        if False in (t[i, :] == 0):
            up = i
            break
        if i == np.shape(t)[0] - 1:
            all_black += 1
            return (t)
    for i in reversed(range(np.shape(t)[0])):
        if False in (t[i, :] == 0):
            down = i
            break
    for i in range(np.shape(t)[0]):
        if False in (t[:, i] == 0):
            left = i
            break
    for i in reversed(range(np.shape(t)[0])):
        if False in (t[:, i] == 0):
            right = i
            break

    wid = right - left
    hei = down - up
    center_x = (left + right) // 2
    center_y = (up + down) // 2

    if wid > hei:
        half_len = wid // 2
    else:
        half_len = hei // 2
    a = center_y - half_len  # up
    b = center_y + half_len  # down
    c = center_x - half_len  # left
    d = center_x + half_len  # right
    if a < 0:
        a = 0
    if c < 0:
        c = 0
    return (t[a:b, c:d])


height =512
width =512
seq_len=30
ms = 200

video_wd ='D:\\tobigs2\\kakao\\Trainset\\'
model_saving_wd = "D:\\tobigs2\\kakao\\Model 2\\Model Saved\\"

print(datetime.datetime.now())
data=np.zeros(128*128*3*30).reshape([1,30,128,128,3])
label = []
count = []
i=0
for path, dir, files in os.walk(video_wd):
    if path == video_wd + 'No_Fall':
        idx = 0 # no _fall
    else :
        idx= 1 # Fall
    for file in files:
        wd_file = path + '\\' + file
        cnt, clips = video2image(wd = video_wd,
                                video = wd_file,
                                width =width,
                                height = height,
                                ms= ms,
                                seq_len = seq_len,
                                standard = 50)
        one_video=np.zeros(128*128*3).reshape([1,128,128,3])
        for asdf in np.array(clips).astype(np.float32) : # 30 phtos
            scale_param = [1.5,2.0]
            paf_info, heatmap_info = get_paf_and_heatmap(model_pose, asdf, scale_param)
            peaks = extract_heatmap_info(heatmap_info)
            sp_k, con_all = extract_paf_info(asdf, paf_info, peaks)
            subsets, candidates = get_subsets(con_all, sp_k, peaks)
            subsets, img_points = draw_key_point(subsets, peaks,asdf)
            img_canvas = link_key_point(asdf, candidates, subsets)
            result = (img_canvas-asdf)
            result = cv2.resize(result,(128,128))
            one_video = np.concatenate([one_video, result.reshape([1,128,128,3])], axis=0)
        one_video = one_video[1:]
        data = np.concatenate([data, one_video.reshape([1,30,128,128,3])], axis=0)
        i+=1
        label.append(idx)
        count.append(cnt)
        print('{} videos done.' . format(i))
label = np.array(label)
count = np.array(count)
count[count>seq_len] = seq_len
data = data[1:]
ch_to_idx = {'Fall':1, 'No_Fall':0}
print(datetime.datetime.now())

np.shape(data)

# Croping only human-existing part
resizing_size = 64
resized_data = np.zeros(30 * resizing_size * resizing_size * 3).reshape([1, 30, resizing_size, resizing_size, 3])
for video in data:
    frames_30 = []
    for frame in video:
        frames_30.append(cv2.resize(crop(frame), (resizing_size, resizing_size)))
    resized_data = np.concatenate([resized_data, np.array(frames_30).reshape([1, 30, resizing_size, resizing_size, 3])])
resized_data = resized_data[1:]

print(datetime.datetime.now())


train_x = resized_data
train_y = label
train_count = count

print(np.shape(test_x))
print(np.shape(test_y))
print(np.shape(test_count))


total_result=[]
batch_size =200
for i in range(int(n_videos/batch_size )+1):
    trying=0 # 첫번쨰모델로만 성능을 확인할거임
    if i == int(n_videos/batch_size ): # 배치 다돌고 나머지 처리할떄
        batch_test_x = test_x[i*batch_size:]
        batch_label = label[i*batch_size:]
        batch_count = count[i*batch_size:]
    else:
        batch_train_x = test_x[i*batch_size:(i+1)*batch_size]
        batch_label = label[i*batch_size:(i+1)*batch_size]
        batch_count = count[i*batch_size:(i+1)*batch_size]
    tf.reset_default_graph()
    test_graph = tf.Graph()
    with tf.Session(graph=test_graph) as sess:
        loader = tf.train.import_meta_graph(model_saving_wd + str(trying) + 'th try-' + 'Model.ckpt.meta')
        loader.restore(sess, model_saving_wd + str(trying) + 'th try-'  +  'Model.ckpt')

        LSTM_x = test_graph.get_tensor_by_name(name='LSTM_x:0')
        LSTM_y = test_graph.get_tensor_by_name(name='LSTM_y:0')
        LSTM_count = test_graph.get_tensor_by_name(name='LSTM_count:0')
        pred= test_graph.get_tensor_by_name(name='pred:0')
        acc= test_graph.get_tensor_by_name(name='acc:0')
        #L_2= loaded_graph.get_tensor_by_name(name='L_2:0')

        print('test accuracy : ', sess.run(acc, feed_dict={LSTM_x:batch_test_x,
                                                           LSTM_y:batch_label,
                                                           LSTM_count:batch_count}))
        prediction =  sess.run(pred,  feed_dict={LSTM_x:batch_test_x,
                                                 LSTM_y:batch_label,
                                                 LSTM_count:batch_count})

        total_result += list(prediction)
print(pd.crosstab(total_result, test_y))