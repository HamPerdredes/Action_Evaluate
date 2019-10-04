import cv2 as cv
import numpy as np
import os
import sys
from pathlib import Path
from Pose.pose_visualizer import TfPoseVisualizer

file_path = Path.cwd()
out_file_path = ''
# camera resolution setting
cam_width, cam_height = 1280, 720
# input size to the model
# VGG trained in 656*368; mobilenet_thin trained in 432*368 (from tf-pose-estimation)
#input_width, input_height = 656, 368
input_width, input_height = 432,368

def choose_run_mode(cam_video,video_list,vid_index):
    global out_file_path
    if cam_video==1:#video mode
        out_file_path='out_'+str(vid_index)+'_vid.mp4'
        file_p=video_list[vid_index]
        if not os.path.isfile(file_p):#args.video):
            print("input video file",file_p,"doesn't exist")
            sys.exit(1)
        cap = cv.VideoCapture(file_p)
        if(cap.isOpened()==False):
            print('unable to open the video')
        cap.set(cv.CAP_PROP_FRAME_WIDTH, cam_width)
        cap.set(cv.CAP_PROP_FRAME_HEIGHT, cam_height)
    else:#camera mode
        out_file_path='out_cam.mp4'
        cap = cv.VideoCapture(0)#打开默认相机
        cap.set(cv.CAP_PROP_FRAME_WIDTH, cam_width)
        cap.set(cv.CAP_PROP_FRAME_HEIGHT, cam_height)
    return cap

def load_pretrain_model(model):
    dyn_graph_path = {#预训练模型所在路径
        'VGG_origin': str(file_path / "Pose/graph_models/VGG_origin/graph_opt.pb"),
        'mobilenet_thin': str(file_path / "Pose/graph_models/mobilenet_thin/graph_opt.pb")
    }
    graph_path = dyn_graph_path[model]
    if not os.path.isfile(graph_path):
        raise Exception('Graph file doesn\'t exist, path=%s' % graph_path)
    return TfPoseVisualizer(graph_path, target_size=(input_width, input_height))

def set_video_writer(cap, write_fps=15,output_path='after_process.mp4'):
    return cv.VideoWriter(output_path,
                          cv.VideoWriter_fourcc(*'mp4v'),
                          write_fps,
                          (round(cap.get(cv.CAP_PROP_FRAME_WIDTH)), round(cap.get(cv.CAP_PROP_FRAME_HEIGHT))))