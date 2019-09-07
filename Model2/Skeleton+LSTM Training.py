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

plt.imshow(resized_data[30][15])
plt.show()


# train_x = resized_data.reshape([n_videos, seq_len, -1])
train_x = resized_data
train_y = label
train_count = count

print(np.shape(train_x))
print(np.shape(train_y))
print(np.shape(train_count))

### LSTM Graph
n_param = height * width * 3
hidden_dim = 64
n_stack = 5
n_class = len(ch_to_idx)
n_epoch = 5000
batch_size = 100
lr = 1e-04
n_batches = int(n_videos / batch_size)  # 20개짜리 데이터 뱃치셋을 넣는거임

tf.reset_default_graph()
train_graph = tf.Graph()
# Building the graph
with train_graph.as_default():
    LSTM_x = tf.placeholder(tf.float32, shape=[None, seq_len, 64*64*3], name='LSTM_x')
    LSTM_y = tf.placeholder(tf.int32, shape=[None], name='LSTM_y')
    LSTM_count = tf.placeholder(tf.int32, shape=[None], name='LSTM_count')

    fw_cell = tf.nn.rnn_cell.MultiRNNCell(
        [tf.nn.rnn_cell.LSTMCell(num_units=hidden_dim, state_is_tuple=True) for _ in range(n_stack)],
        state_is_tuple=True)
    # bw_cell = tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.LSTMCell(num_units= hidden_dim, state_is_tuple=True) for _ in range(n_stack)], state_is_tuple=True)
    outputs, state = tf.nn.dynamic_rnn(fw_cell, dtype=tf.float32, inputs=LSTM_x,
                                       sequence_length=LSTM_count)  # batch로 하면 여기서 걸리네 ... n_frame_list를 뱃치를 만든 인덱스에 맞게 그때그때 바꿀수도 없고
    outputs = tf.concat(outputs, 2)

    last_index = tf.shape(outputs)[1] - 1  # -1 붙여줘야 !
    output_rs = tf.transpose(outputs, [1, 0, 2])  # Treshape the output to [sequence_length,batch_size,num_units]
    last_states = tf.nn.embedding_lookup(output_rs, last_index)  # Last state of all batches
    last_states = tf.identity(last_states, 'last_states')

    LSTM_w_1 = tf.get_variable(shape=[hidden_dim, 16], name='LSTM_w_1')
    LSTM_b_1 = tf.get_variable(shape=[16], name='LSTM_b_1')
    LSTM_w_2 = tf.get_variable(shape=[16, n_class], name='LSTM_w_2')  # 0 ,1 (안쓰러짐 ,쓰러짐)
    LSTM_b_2 = tf.get_variable(shape=[n_class], name='LSTM_b_2')
    # param_list = [ LSTM_w_1, LSTM_b_1, LSTM_w_2, LSTM_b_2]
    L_1 = tf.add(tf.matmul(last_states, LSTM_w_1), LSTM_b_1)  # 굿
    L_1 = tf.nn.relu(L_1)
    L_1 = tf.nn.dropout(L_1, 0.5)
    L_1 = tf.contrib.layers.batch_norm(L_1)
    L_2 = tf.add(tf.matmul(L_1, LSTM_w_2), LSTM_b_2)
    L_2 = tf.identity(L_2, name="L_2")
    loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits=L_2, labels=tf.one_hot(LSTM_y, depth=n_class)))
    train = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)

    pred = tf.cast(tf.argmax(L_2, axis=1), tf.float32, name='pred')
    acc = tf.reduce_mean(tf.cast(tf.equal(pred, tf.cast(LSTM_y, tf.float32)), dtype=tf.float32))
    acc = tf.identity(acc, name='acc')
# saver = tf.train.Saver(tf.global_variables())

# LSTM Training
print(datetime.datetime.now())
with tf.Session(graph=train_graph) as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(n_epoch):
        batch_idx_start = 0
        total_idx = np.arange(n_videos);
        np.random.shuffle(total_idx)
        for i in range(n_batches):
            batch_idx_end = batch_idx_start + batch_size
            batch_idx = total_idx[batch_idx_start:batch_idx_end]
            batch_x = train_x[batch_idx]
            batch_y = train_y[batch_idx]
            batch_count = train_count[batch_idx]
            sess.run(train, feed_dict={LSTM_x: batch_x,
                                       LSTM_y: batch_y,
                                       LSTM_count: batch_count})
            batch_idx_start = batch_idx_end
        if epoch % 10 == 0:
            print('--------------------------------', 'Epoch: ', epoch, '--------------------------------')

            print('Loss : ', sess.run(loss, feed_dict={LSTM_x: batch_x,
                                                       LSTM_y: batch_y,
                                                       LSTM_count: batch_count}))  # per epoch
            print(pd.crosstab(sess.run(pred, feed_dict={LSTM_x: batch_x,
                                                        LSTM_count: batch_count}),
                              batch_y))
            print('Accuracy : ', sess.run(acc, feed_dict={LSTM_x: batch_x,
                                                          LSTM_y: batch_y,
                                                          LSTM_count: batch_count}))
            saver = tf.train.Saver(var_list=tf.global_variables())
            saver.save(sess, save_path=model_saving_wd + str(trying) + 'th try-' + 'Model.ckpt')

print(datetime.datetime.now())
