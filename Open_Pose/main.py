import cv2 as cv
import argparse
import numpy as np
import time
from utils import choose_run_mode, load_pretrain_model, set_video_writer
from Pose.pose_visualizer import TfPoseVisualizer
from Action.recognizer import load_action_premodel, framewise_recognize,get_pose_dict
from Action.classifier import  classify,draw_box,suspicious_delay
import glob
import sys
import os
 
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

#导入特征提取模型
#estimator = load_pretrain_model('VGG_origin')              #VGG网络和mobilenet两种网络
estimator = load_pretrain_model('mobilenet_thin')
# 参数初始化
video_list=[]
video_len=0
realtime_fps = '0.0000'
start_time = time.time()
fps_interval = 1
fps_count = 0
run_timer = 0
frame_count = 0
cam_video=1#    cam_video=0->camera         cam_video=1->video
file_p="camera_record/*.mp4"#要读取的视频所在文件夹的路径
is_save=True
vid_index=0
#读取视频列表
if(cam_video==1):
    video_list=glob.glob(file_p)
    video_len=len(video_list)
    print(video_list)
    if(video_len<=0):
        print('No video in that file')
        sys.exit(1)
cap = choose_run_mode(cam_video,video_list,vid_index)#读取视频
#存储处理后的视频
tmp_string='after_process_'+str(vid_index)+'.mp4'
video_writer = set_video_writer(cap, write_fps=int(30.0),output_path=tmp_string)
#存储获取的关节点信息   目前只考虑单人的情况
if is_save:
    data_file='gained_data/origin_data_'+str(vid_index)+'.txt'
    f = open(data_file, 'a+')
vid_index+=1
while cv.waitKey(1) < 0:
#while True :
    has_frame, show = cap.read()
    if has_frame:
        fps_count += 1
        frame_count += 1
        # pose estimation
        humans = estimator.inference(show)  #openpose输出检测到的humans(人体关键点)
        # get pose info
        pose = TfPoseVisualizer.draw_pose_rgb(show, humans)  # return frame, joints, bboxes, xcenter 
        #返回npimg(检测结果图), joints(每一帧所有的关节点，原图尺寸下的坐标), bboxes(框出人的ROI),
        # xcenter(每个人的同一个关节点坐标,用以区分开每个人对应的bboxes), record_joints_norm(720p下的相对二维坐标,即绝对坐标除以尺寸)
        
        #存储每一帧的关节点在图像坐标系下的坐标 1280*720
        if is_save :
            joints_per_frame=np.array(pose[-1]).astype(np.str)
            f.write(' '.join(joints_per_frame))
            f.write('\n')
        height, width = show.shape[:2]
        # 显示实时FPS值
        if (time.time() - start_time) > fps_interval:
            # 计算这个interval过程中的帧数，若interval为1秒，则为FPS
            realtime_fps = fps_count / (time.time() - start_time)
            fps_count = 0  # 帧数清零
            start_time = time.time()
        #fps_label = 'FPS:{0:.2f}'.format(realtime_fps)
        fps_label='FPS:'+str(realtime_fps)
        cv.putText(show, fps_label, (width-160, 25), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

        # 显示检测到的人数
        #num_label = "Human: {0}".format(len(humans))
        num_label='Human:'+str(len(humans))
        cv.putText(show, num_label, (5, height-45), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

        # 显示目前的运行时长及总帧数
        if frame_count == 1:
            run_timer = time.time()
        run_time = time.time() - run_timer
        #time_frame_label = '[Time:{0:.2f} | Frame:{1}]'.format(run_time, frame_count)
        time_frame_label='Time:'+str(round(run_time,2))+' | Frame:'+str(frame_count)
        cv.putText(show, time_frame_label, (5, height-15), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
        video_writer.write(show)
        cv.imshow('result',show)
        #cv.waitKey(0) 
    else :
        video_writer.release()
        cap.release()
        if is_save :
            f.close()
        if cam_video==1 and vid_index<video_len :
            #video_writer.release()
            cap=choose_run_mode(cam_video,video_list,vid_index)
            tmp_string='after_process_'+str(vid_index)+'.mp4'
            video_writer = set_video_writer(cap, write_fps=int(30.0),output_path=tmp_string)
            if is_save:
                data_file='gained_data/origin_data_'+str(vid_index)+'.txt'
                f = open(data_file, 'a+')
            vid_index+=1
        else :
            break
'''
for i in range(vid_index+1) :#逐一处理视频数据
    cur_data='gained_data/origin_data_'+str(vid_index)+'.txt'
'''

    

        