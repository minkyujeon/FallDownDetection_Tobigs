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

n_videos = len(data)

print(np.shape(data))
print(np.shape(count))
print(np.shape(label))

idx = np.arange(len(data));
np.random.shuffle(idx)

train_x = data[idx]
train_y = label[idx]
train_count = count[idx]

print(np.shape(train_x))
print(np.shape(train_y))
print(np.shape(train_count))

# End to End ( CNN + LSTM )

n_param = height * width * 3
hidden_dim = 96
n_stack = 3
n_class = len(ch_to_idx)
n_epoch = 30
batch_size = 14
lr = 1e-05
n_batches = int(len(train_x) / batch_size)

tf.reset_default_graph()
train_graph = tf.Graph()

# Building the graph
with train_graph.as_default():
    LSTM_x = tf.placeholder(tf.float32, shape=[None, seq_len, height, width, 3], name='LSTM_x')
    LSTM_y = tf.placeholder(tf.int32, shape=[None], name='LSTM_y')
    LSTM_count = tf.placeholder(tf.int32, shape=[None], name='LSTM_count')
    is_training = tf.placeholder(tf.bool, name='is_training')
    dropout_ratio = tf.placeholder(tf.float32, name='dropout_ratio')

    w1 = tf.Variable(tf.random_normal(shape=[3, 3, 3, 32], stddev=0.01), name="W1")  # shape !!!!!!!!!!
    CNN_x = tf.reshape(LSTM_x, shape=[-1, height, width, 3])
    L1 = tf.nn.conv2d(input=CNN_x, filter=w1, strides=[1, 1, 1, 1], padding='SAME', name="L1")
    L1 = tf.nn.relu(L1)
    L1 = tf.nn.dropout(L1, dropout_ratio)
    L1 = tf.contrib.layers.batch_norm(L1, is_training=is_training)  # batch normal, dropout, layer
    L1 = tf.nn.max_pool(L1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    w1_1 = tf.Variable(tf.random_normal(shape=[3, 3, 32, 32], stddev=0.01), name="WW")  # shape !!!!!!!!!!
    L2 = tf.nn.conv2d(input=L1, filter=w1_1, strides=[1, 1, 1, 1], padding='SAME', name="L2")
    L2 = tf.nn.relu(L2)
    L2 = tf.nn.dropout(L2, dropout_ratio)
    L2 = tf.contrib.layers.batch_norm(L2, is_training=is_training)
    L2 = tf.nn.max_pool(L2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    L2_flat = tf.reshape(L2, shape=[-1, int(height / 4 * width / 4 * 32)])

    w2 = tf.get_variable(shape=[height / 4 * width / 4 * 32, 5], initializer=tf.contrib.layers.xavier_initializer(),
                         name="w2")
    b = tf.Variable(tf.random_normal(shape=[5]))
    logits = tf.matmul(L2_flat, w2) + b
    logits = tf.reshape(logits, shape=[-1, seq_len, 5])
    logits = tf.identity(logits, 'logits')

    fw_cell = tf.nn.rnn_cell.MultiRNNCell(
        [tf.nn.rnn_cell.LSTMCell(num_units=hidden_dim, state_is_tuple=True) for _ in range(n_stack)],
        state_is_tuple=True)
    outputs, state = tf.nn.dynamic_rnn(fw_cell, dtype=tf.float32, inputs=logits,
                                       sequence_length=LSTM_count)
    outputs = tf.reshape(outputs, shape=[-1, seq_len * hidden_dim])

    LSTM_w_1 = tf.get_variable(shape=[seq_len * hidden_dim, 256], name='LSTM_w_1')  # 전체아웃풋을 함 넣어볼까 ..
    LSTM_b_1 = tf.get_variable(shape=[256], name='LSTM_b_1')
    LSTM_w_2 = tf.get_variable(shape=[256, n_class], name='LSTM_w_2')  # 0 ,1
    LSTM_b_2 = tf.get_variable(shape=[n_class], name='LSTM_b_2')
    L_1 = tf.add(tf.matmul(outputs, LSTM_w_1), LSTM_b_1)  # 굿
    L_1 = tf.nn.relu(L_1)
    L_1 = tf.nn.dropout(L_1, dropout_ratio)  # tf.layers.dropout() >> this function has training parameter.
    L_1 = tf.contrib.layers.batch_norm(L_1, is_training=is_training)
    L_2 = tf.add(tf.matmul(L_1, LSTM_w_2), LSTM_b_2)
    L_2 = tf.identity(L_2, name="L_2")
    weights = tf.ones(shape=[batch_size, seq_len])
   loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits=L_2,
                                                                 labels=tf.reshape(tf.one_hot(LSTM_y, depth=n_class),
                                                                                   [-1, 2])))
    train = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)

    pred = tf.cast(tf.argmax(L_2, axis=1), tf.float32, name='pred')
    acc = tf.reduce_mean(tf.cast(tf.equal(pred, tf.cast(LSTM_y, tf.float32)), dtype=tf.float32))
    acc = tf.identity(acc, name='acc')

# LSTM training
for trying in range(1):  # Adjust the number if you want ensemble !
    # smp_idx = np.random.choice(len(data),len(data),replace=True)
    smp_idx = np.arange(len(train_x))
    print('ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ', trying, ' th MDOEL ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ')
    print(datetime.datetime.now())
    with tf.Session(graph=train_graph) as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(n_epoch):
            batch_idx_start = 0
            total_idx = np.arange(len(train_x));
            np.random.shuffle(total_idx)
            for i in range(n_batches):
                batch_idx_end = batch_idx_start + batch_size
                batch_idx = total_idx[batch_idx_start:batch_idx_end]
                batch_x = train_x[smp_idx][batch_idx]
                batch_y = train_y[smp_idx][batch_idx]
                batch_count = train_count[smp_idx][batch_idx]
                sess.run(train, feed_dict={LSTM_x: batch_x,
                                           LSTM_y: batch_y,
                                           LSTM_count: batch_count,
                                           is_training: True,
                                           dropout_ratio: 0.5})
                batch_idx_start = batch_idx_end

            print('--------------------------------', 'Epoch: ', epoch, '--------------------------------')

            print('Loss : ', sess.run(loss, feed_dict={LSTM_x: batch_x,
                                                       LSTM_y: batch_y,
                                                       LSTM_count: batch_count,
                                                       is_training: True,
                                                       dropout_ratio: 0.5}))  # per epoch
            print(pd.crosstab(sess.run(pred, feed_dict={LSTM_x: batch_x,
                                                        LSTM_count: batch_count,
                                                        is_training: True,
                                                        dropout_ratio: 0.5}),
                              batch_y))
            print('Accuracy : ', sess.run(acc, feed_dict={LSTM_x: batch_x,
                                                          LSTM_y: batch_y,
                                                          LSTM_count: batch_count,
                                                          is_training: True,
                                                          dropout_ratio: 0.5}))
            saver = tf.train.Saver(var_list=tf.global_variables())
            saver.save(sess, save_path=model_saving_wd + str(trying) + 'th try-' + 'Model.ckpt')

    print(datetime.datetime.now())