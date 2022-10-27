
from datetime import datetime
import os
from re import S
import shutil
import argparse
import cv2
import numpy as np
import glob
import Searching
import args_parser
from threading import *
import sys
import av


#args.demo = 'image'
_frames = 0
def ArgParse():
    try:
        tmp_args = args_parser.make_parser().parse_args()
        try:
            
            tmp_args.frames = tmp_args.frames.split(",")
            
        except:
          tmp_args.frames = "all"
        
        tmp_args.ckpt = '../YOLOX/assets/yolox_s.pth'
        tmp_args.exp_file = '../YOLOX/exps/default/yolox_s.py'
        tmp_args.path = "../YOLOX/assets/"
        tmp_args.save_result = True
        return tmp_args
    except:
        print("Arguments Error")
        raise SystemExit
    


def SplitVideo(video):
    # try:
        current_video = av.open(video)
        frames_list = []
        for frame in current_video.decode():
            
            arr_frame = frame.to_ndarray(format='bgr24')
            frames_list += [arr_frame]
            # print(frames_list)
            
        return frames_list
    # except:
    
    #     VideoError()

            


def VideoError():
    print("Video Error")
    raise SystemExit

def FrameProcessing(video):
   return Searching.main_search(Searching.get_exp(args.exp_file), args, video)
      
                

def WriteVideo(video_frames, type='not_all'):
    
    out = cv2.VideoWriter('output_video.avi',cv2.VideoWriter_fourcc(*'DIVX'), 22, (1920, 1080))

    for frame in video_frames:
        
        
        out.write(frame)
        
    
        
    out.release()
    

def Start():

    frame_list = []

    global args 
    args = ArgParse()
    if (args.frames != "all"):
        for param in args.frames:
            param = int(param)
            frame_list.append(param)


    
    splited_video = SplitVideo(args.video)
    processed_video = FrameProcessing(splited_video)

    if (args.frames != "all"):
        WriteVideo(splited_video)
    else: 
        WriteVideo(processed_video, 'all')  
    

Start()

#/app/resources/develop_streem.ts
