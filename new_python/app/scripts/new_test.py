
from datetime import datetime
from multiprocessing.dummy import Array
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
import time
from datetime import datetime, timedelta
from PIL import Image
from io import BytesIO
from av.video.frame import VideoFrame
import av.datasets
from av.video.stream import VideoStream

#args.demo = 'image'

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
            print(frame)
            arr_frame = frame.to_ndarray(format='rgb24')
            frames_list += [arr_frame]
            print(frames_list)
            # for f in frames_list:
            #     if type == False:
            #         os.system(f"python ../YOLOX/tools/Searching.py \
            #         image -f ../YOLOX/exps/default/yolos_s.py -c ../YOLOX/assets/yolox_x.pth \
            #         --path test/ --conf 0.25 --nms 0.45 --tsize 640 \
            #         --save_result --device [cpu/gpu]")
            #     else:
            #         for frame in frames_list:
            #             if os.path.exists(f"processed_frames/frame_{(frame)[type == True]}.jpg"):
            #                 continue
            #             else:
            #                 os.system(f"python Searching.py \
            #             image -f ../YOLOX/exps/default/yolox_s.py -c ../YOLOX/assets/yolox_x.pth \
            #             --path test/frame_{(frame)[type == True]}.jpg --conf 0.25 --nms 0.45 --tsize 640 \
            #             --save_result --device [cpu/gpu]")
            
        return frames_list

# def ClearDir():
#     shutil.rmtree("test/")
#     os.mkdir("test/")

# def SplitVideo(video):
#     # try:
#         current_video = av.open(video)
#         frames_list = []
#         for frame in current_video.decode():
#             frames_list.append(frame.to_image())
            
#             im = bytes(frame)
#             im2arr = np.array(im) # im2arr.shape: height x width x channel
#             arr2im = Image.fromarray(im2arr)
#             print(arr2im)
#         return frames_list
#     # except:
#     #     VideoError()

# def SaveFrames(frames_list, splited_video):
#     global sv_frames
#     global i
#     global time_save_frames

#     if (frames_list != "all"):
       
#         for frame in frames_list:
#             try:
#                 t1 = time.time()
                
#                 splited_video[frame].to_image().save("test/frame_%d.jpg" % splited_video[frame].index)
#                 sv_frames +=1
                
#                 t2 = time.time()
#                 # print("%s секунд" % (t2-t1))
#                 time_save_frames = float(("%s" % (t2 - t1)))
                
#                 save_frames_speed()
#                 i += 1
#             except: 
#                 print("frames saved")
#                 continue
            

#     else:
        
#         for frame in av.open(splited_video).decode():
#             t1 = time.time()
            
#             frame.to_image().save("test/frame_%d.jpg" % frame.index)
#             sv_frames +=1
            
#             t2 = time.time() 
#             # print("%s секунд" % (t2-t1))
#             time_save_frames = float(("%s" % (t2 - t1)))
            

#             save_frames_speed()

# def VideoError():
#     print("Video Error")
#     raise SystemExit

# def SearchObjects(type, frames):
#     if type == False:
#         os.system(f"python ../YOLOX/tools/Searching.py \
#         image -f ../YOLOX/exps/default/yolos_s.py -c ../YOLOX/assets/yolox_x.pth \
#         --path test/ --conf 0.25 --nms 0.45 --tsize 640 \
#         --save_result --device [cpu/gpu]")
#     else:
#         for frame in frames:
#             if os.path.exists(f"processed_frames/frame_{(frame, frame.index)[type == True]}.jpg"):
#                 continue
#             else:
#                 os.system(f"python Searching.py \
#             image -f ../YOLOX/exps/default/yolox_s.py -c ../YOLOX/assets/yolox_x.pth \
#             --path test/frame_{(frame, frame.index)[type == True]}.jpg --conf 0.25 --nms 0.45 --tsize 640 \
#             --save_result --device [cpu/gpu]")
                

# def WriteVideo(video_frames, type='not_all'):
#     global i
#     out = cv2.VideoWriter('output_video.avi',cv2.VideoWriter_fourcc(*'DIVX'), 22, (1920, 1080))
#     i = 0
#     global _frames
#     global t_w_v
#     for frame in video_frames:
        
#         image = Searching.main_search(Searching.get_exp(args.exp_file), args, f'test/frame_{i}.jpg')
#         t1 = time.time()
        
#         out.write(image)
#         _frames += 1
#         t2 = time.time()
#         t_w_v = float(("%s" % (t2 - t1)))
#         # print(t_w_v)
        

#         write_video_speed()
#         _frames = 0
#         i += 1
       
        
#         # filename = f"processed_frames/frame_{(frame, i)[type == 'all']}.jpg"
        
#         # image = Searching.main_search(Searching.get_exp(args.exp_file), args, f'test/frame_{i}.jpg')
        
      
        
#         # img = cv2.imread(filename)
#         # print(img)
        
    
        
#     out.release()
    
# def save_frames_speed():
#     global time_save_frames
#     global i
#     save_speed = 0
#     save_speed = sv_frames / time_save_frames
#     sys.stdout.write("\r" + f'save_frames_speed: {"%.2f" % save_speed} f/s, loaded: {sv_frames}')
#     sys.stdout.flush()



# def write_video_speed():
#     global i
#     global t_w_v
#     _frames_speed = 0
    
#     # Event(target=WriteVideo).wait(1)
#     # Event().wait(1)
#     # print(_frames.size)
#     _frames_speed = _frames / t_w_v
#     sys.stdout.write("\r" + f'write_video_speed: {"%.2f" % _frames_speed} f/s, loaded: {i}')
#     sys.stdout.flush()

def Start():



    global args 
    args = ArgParse()
    SplitVideo(args.video)
#    if (args.frames != "all"):
#         for param in args.frames:
#             param = int(param)
#             frame_list.append(param)

#     # ClearDir()
    
#     splited_video = SplitVideo(args.video)
#     if (args.frames != "all"):
#          SaveFrames(frame_list, splited_video)
#     else: 
#         SaveFrames("all", args.video)
#     # upload_speed()
#     # SearchObjects(True, splited_video)
#     if (args.frames != "all"):
#         WriteVideo(frame_list)
#     else: 
#         WriteVideo(splited_video, 'all')  
    

Start()

#/app/resources/develop_streem.ts

