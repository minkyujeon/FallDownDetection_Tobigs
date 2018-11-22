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

video_wd = 'D:\\tobigs2\\kakao\\Testset\\'
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

test_x = data
test_y = label
test_count = count

print(np.shape(test_x))
print(np.shape(test_y))
print(np.shape(test_count))

ch_to_idx = {'Fall': 1, 'No_Fall': 0}
n_videos = len(data)
n_class = len(ch_to_idx)
model_saving_wd = 'D:\\tobigs2\\kakao\\Model 1\\5. Not Ensemble\\Model_Saved\\'

total_result = []
batch_size = 10
trying = 0  # 1st Model

tf.reset_default_graph()
test_graph = tf.Graph()
with tf.Session(graph=test_graph) as sess:
    loader = tf.train.import_meta_graph(model_saving_wd + str(trying) + 'th try-' + 'Model.ckpt.meta')
    loader.restore(sess, model_saving_wd + str(trying) + 'th try-' + 'Model.ckpt')

    LSTM_x = test_graph.get_tensor_by_name(name='LSTM_x:0')
    LSTM_y = test_graph.get_tensor_by_name(name='LSTM_y:0')
    LSTM_count = test_graph.get_tensor_by_name(name='LSTM_count:0')
    pred = test_graph.get_tensor_by_name(name='pred:0')
    acc = test_graph.get_tensor_by_name(name='acc:0')

    for i in range(int(len(test_x) / batch_size) + 1):
        if i == int(n_videos / batch_size):
            batch_test_x = test_x[i * batch_size:]
            batch_label = test_y[i * batch_size:]
            batch_count = test_count[i * batch_size:]

        else:
            batch_test_x = test_x[i * batch_size:(i + 1) * batch_size]
            batch_label = test_y[i * batch_size:(i + 1) * batch_size]
            batch_count = test_count[i * batch_size:(i + 1) * batch_size]

        print('test accuracy : ', sess.run(acc, feed_dict={LSTM_x: batch_test_x,
                                                           LSTM_y: batch_label,
                                                           LSTM_count: batch_count}))
        prediction = sess.run(pred, feed_dict={LSTM_x: batch_test_x,
                                               LSTM_count: batch_count})
        print(batch_label)
        print(batch_count)

        total_result += list(prediction)

print(pd.crosstab(np.array(total_result), test_y))