import os
from os import chdir
import cv2
import numpy as np

def video2image(wd, video, width, height, ms, seq_len, standard):
    # wd : working directory
    # name : image's name
    # video : video
    # width : width when saving as image
    # height : height when saving as image
    # ms : 1/1000 second when to save as image
    clips=[]

    chdir(wd)
    cap = cv2.VideoCapture(video)
    success, frame = cap.read()
    new_frame = cv2.resize(frame, (width, height))
    new_frame = new_frame.tolist()
    count = 1
    rate = ms
    cap.set(cv2.CAP_PROP_POS_MSEC, rate)
    clips.append(new_frame)
    while (success == True and count <= standard):
        rate += ms
        count += 1
        cap.set(cv2.CAP_PROP_POS_MSEC, rate)
        clips.append(new_frame)
        success, frame = cap.read()
        if frame is not None:
            new_frame = cv2.resize(frame, (width, height))
            new_frame = new_frame.tolist()
    if count < seq_len:
        clips = clips + [np.zeros([width,height,3])]*(seq_len-count)
    else:
        clips = clips[:seq_len]
    return(count, clips)