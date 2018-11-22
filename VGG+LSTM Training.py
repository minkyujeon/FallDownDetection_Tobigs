import cv2
from os import chdir
import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
from V2I import video2image
import datetime
from moviepy.editor import *
import pandas as pd
import datetime
import pickle

from keras.models import load_model

my_vgg = load_model('finetuned_vgg.h5')

### Reading in data
height = 128
width = 128
seq_len = 30
ms = 200

video_wd = 'D:\\tobigs2\\kakao\\Trainset\\'
model_saving_wd = 'D:\\tobigs2\\kakao\\Model 1\\5. Not Ensemble\\Model_Saved\\'

print(datetime.datetime.now())
data = np.zeros(128 * 128 * 3 * 30).reshape([1, 30, 128, 128, 3])
label = []
count = []
i = 0
for path, dir, files in os.walk(video_wd):
    if path == video_wd + 'No_Fall':
        idx = 0  # no _fall
    else:
        idx = 1  # Fall
    for file in files:
        wd_file = path + '\\' + file
        cnt, clips = video2image(wd=video_wd,
                                 video=wd_file,
                                 width=width,
                                 height=height,
                                 ms=ms,
                                 seq_len=seq_len,
                                 standard=50)
        data = np.concatenate([data, np.array(clips).reshape([1, 30, 128, 128, 3])], axis=0)
        label.append(idx)
        count.append(cnt)
        i += 1
        print('{} videos done.'.format(i))
label = np.array(label)
count = np.array(count)
count[count > seq_len] = seq_len
data = data[1:]
ch_to_idx = {'Fall': 1, 'No_Fall': 0}
n_videos = len(data)
n_class = len(ch_to_idx)
print(datetime.datetime.now())
print(np.shape(data))
print(np.shape(count))
print(np.shape(label))

data_ftmap = np.zeros(30 * 4 * 4 * 512).reshape([1, 30, 4 * 4 * 512])
for i in range(len(data)):
    data_ftmap = np.concatenate([data_ftmap, my_vgg.predict(data[i]).reshape(1, seq_len, -1)], axis=0)
    if i % 300 == 0:
        print(i)

data_ftmap = data_ftmap[1:]
print(np.shape(data_ftmap))
print(np.shape(count))
print(np.shape(label))

idx = np.arange(len(data));
np.random.shuffle(idx)

train_x = data_ftmap[idx]
train_y = label[idx]
train_count = count[idx]

print(np.shape(train_x))
print(np.shape(train_y))
print(np.shape(train_count))

### LSTM Graph
n_param = height / 32 * width / 32 * 512
hidden_dim = 128
n_stack = 3
n_class = len(ch_to_idx)
n_epoch = 70
batch_size = 50
lr = 1e-04
n_batches = int(n_videos / batch_size)

tf.reset_default_graph()
train_graph = tf.Graph()

# Building the graph
with train_graph.as_default():
    LSTM_x = tf.placeholder(tf.float32, shape=[None, seq_len, n_param], name='LSTM_x')
    LSTM_y = tf.placeholder(tf.int32, shape=[None], name='LSTM_y')
    LSTM_count = tf.placeholder(tf.int32, shape=[None], name='LSTM_count')

    fw_cell = tf.nn.rnn_cell.MultiRNNCell(
        [tf.nn.rnn_cell.LSTMCell(num_units=hidden_dim, state_is_tuple=True) for _ in range(n_stack)],
        state_is_tuple=True)
    outputs, state = tf.nn.dynamic_rnn(fw_cell, dtype=tf.float32, inputs=LSTM_x,
                                       sequence_length=LSTM_count)

    LSTM_w_1 = tf.get_variable(shape=[hidden_dim, 64], name='LSTM_w_1')
    LSTM_b_1 = tf.get_variable(shape=[64], name='LSTM_b_1')
    LSTM_w_2 = tf.get_variable(shape=[64, n_class], name='LSTM_w_2')  # 0 ,1
    LSTM_b_2 = tf.get_variable(shape=[n_class], name='LSTM_b_2')
    # param_list = [ LSTM_w_1, LSTM_b_1, LSTM_w_2, LSTM_b_2]
    # saver = tf.train.Saver()
    L_1 = tf.add(tf.matmul(state[0].h, LSTM_w_1), LSTM_b_1)  # 굿
    L_1 = tf.nn.elu(L_1)
    L_1 = tf.nn.dropout(L_1, 0.5)
    L_1 = tf.contrib.layers.batch_norm(L_1)
    L_2 = tf.add(tf.matmul(L_1, LSTM_w_2), LSTM_b_2)
    L_2 = tf.identity(L_2, name="L_2")
    loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits=L_2, labels=tf.one_hot(LSTM_y, depth=n_class)))
    train = tf.train.RMSPropOptimizer(learning_rate=lr).minimize(loss)

    pred = tf.cast(tf.argmax(L_2, axis=1), tf.float32, name='pred')
    acc = tf.reduce_mean(tf.cast(tf.equal(pred, tf.cast(LSTM_y, tf.float32)), dtype=tf.float32))
    acc = tf.identity(acc, name='acc')

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
        print('--------------------------------', 'Epoch: ', epoch, '--------------------------------')

        print('Loss : ', sess.run(loss, feed_dict={LSTM_x: batch_x,
                                                   LSTM_y: batch_y,
                                                   LSTM_count: batch_count}))  # per epoch
        print(pd.crosstab(sess.run(pred, feed_dict={LSTM_x: batch_x,
                                                    LSTM_y: batch_y,
                                                    LSTM_count: batch_count}),
                          batch_y))
        print('Accuracy : ', sess.run(acc, feed_dict={LSTM_x: batch_x,
                                                      LSTM_y: batch_y,
                                                      LSTM_count: batch_count}))
        saver = tf.train.Saver(var_list=tf.global_variables())
        saver.save(sess, save_path=model_saving_wd + str(smp_num) + '_Model.ckpt')

print(datetime.datetime.now())