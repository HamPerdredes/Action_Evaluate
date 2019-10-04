import numpy as np
import cv2 as cv

anti_a=[0]
flag_delay=[0,0]
basic_delay=[0]
#basic_list=['', 'stand', 'sit', 'walk', 'walk close', 'walk away', 'sit down', 'stand up']
basic_list=['', 'stand', 'stand', 'walk', 'walk', 'walk', 'sit down', 'stand up']#hard to define some curtain pose,therefore we set it as walk

def draw_box(frame,cur_tags,cur_box,frame_count):
    for i in cur_box:
        ID=i
        if cur_tags.get(i)==None:#则只画框和ID， 不标注运动类型
            box=cur_box[i]
            cv.rectangle(frame,(box[0]-10,box[1]-30),(box[2]+10,box[3]),(0,255,0),2)

            #cur_id='ID-'+str(i)
            cur_id='ID-'+str(ID)
            cv.putText(frame,cur_id,(box[0],box[1]-45),cv.FONT_HERSHEY_SIMPLEX,0.8,(0,255,0),3)
        else:
            box=cur_box[i]
            cur_tag=cur_tags[i]
            is_sus= cur_tag[:10]=='suspicious'
            if is_sus :
                cv.rectangle(frame,(box[0]-10,box[1]-30),(box[2]+10,box[3]),(0,0,255),2)
                cv.putText(frame, 'Caution: Someone seems suspicious in strict area', (20, 60), cv.FONT_HERSHEY_SIMPLEX,
                               1.5, (0, 0, 255), 4)
            else:
                if(cur_tags[i]=='fall_down'):
                    cv.rectangle(frame,(box[0]-10,box[1]-30),(box[2]+10,box[3]),(0,0,255),2)
                    cv.putText(frame, 'Caution: Someone has fell down', (20, 60), cv.FONT_HERSHEY_SIMPLEX,
                                1.5, (0, 0, 255), 4)
                elif(cur_tags[i]=='fight'):
                    cv.rectangle(frame,(box[0]-10,box[1]-30),(box[2]+10,box[3]),(0,0,255),2)
                    cv.putText(frame, 'Caution: People FIghting', (20, 60), cv.FONT_HERSHEY_SIMPLEX,
                                1.5, (0, 0, 255), 4)
                else:
                    cv.rectangle(frame,(box[0]-10,box[1]-30),(box[2]+10,box[3]),(0,255,0),2)
                cur_id='ID-'+str(ID)+': '+cur_tags[i]
                cv.putText(frame,cur_id,(box[0],box[1]-45),cv.FONT_HERSHEY_SIMPLEX,0.8,(0,255,0),3)
    return frame

def suspicious_delay(ROI,delay_rec,cur_box,cur_tags,cur_poses,sus_thresh):#ROI格式：[x1,y1,x2,y2]  x1/y1为左上角 x2/y2为右下角
 #遍历当前帧内每一个人的位置和行为，根据站在ROI或行走经过ROI的次数，判定是否可疑
    for i in cur_box:
        pose=cur_poses[i]
        pose=pose[1:]
        if not(pose[20]==0 and pose[21]==0) :
            if pose[20]*1280>ROI[0] and pose[20]*1280<ROI[2] and pose[21]*720>ROI[1] and pose[21]*720<ROI[3] : #show in suspicious area
                if delay_rec.get(i)==None:
                    delay_rec[i]=1
                else:
                    delay_rec[i]+=1
        elif not(pose[26]==0 and pose[27]==0) :
                if pose[26]*1280>ROI[0] and pose[26]*1280<ROI[2] and pose[27]*720>ROI[1] and pose[27]*720<ROI[3] :
                    if delay_rec.get(i)==None:
                        delay_rec[i]=1
                    else:
                        delay_rec[i]+=1
        if delay_rec.get(i)==None :
            delay_rec[i]=0
        elif delay_rec[i] > sus_thresh :
            if cur_tags.get(i)==None:
                cur_tags[i]='suspicious'
            else:
                cur_tags[i]='suspicious'+cur_tags[i]


def classify(befo_poses,cur_poses,frame_count,cur_tags):
    global basic_delay
    global basic_list
    global flag_delay
    behave_class=''
    global anti_a
    for i in cur_poses:
        if befo_poses.get(i)!=None :
            befo_pose=befo_poses[i]
        else:
            befo_pose=None
        cur_pose=cur_poses[i]
        if(cur_pose[0]!=frame_count):#当前的human在本次采样没有得到更新
            cur_tags[i]=None
            continue
        elif befo_pose==None:
            cur_tags[i]=None
            continue
        else:
            thresh=befo_pose[4]*1
            print('thresh=',thresh)
            length=(befo_pose[3]*1280-cur_pose[3]*1280)**2+(befo_pose[4]*720-cur_pose[4]*720)**2
            length=length**(0.5)
            print('length=',length)

            behave_class=''
            if(length>thresh):#dynamic
                if is_attacker(befo_pose,cur_pose,frame_count,flag_delay) :
                    flag=True
                    behave_class='fight'
                elif anti_fight(befo_pose,cur_pose,anti_a):
                    flag=True
                    behave_class='fight'
                elif fall_d(befo_pose, cur_pose) :
                    flag=True
                    behave_class='fall_down'
                else:
                    flag=basic(befo_pose, cur_pose,basic_delay)
                    if(flag in [3,4,5]):
                        behave_class=basic_list[flag]
                        flag=True
                    elif flag in [1,2] :
                        behave_class='walk'
                        flag=True
                    else:
                        flag=False
            else:#static
                if is_attacker(befo_pose,cur_pose,frame_count,flag_delay) :
                    flag=True
                    behave_class='fight'
                elif anti_fight(befo_pose,cur_pose,anti_a):
                    flag=True
                    behave_class='fight'
                elif fall_d(befo_pose, cur_pose) :
                    flag=True
                    behave_class='fall_down'
                else:
                    flag=basic(befo_pose, cur_pose,basic_delay)
                    if(flag in [1,2,6,7]):
                        behave_class=basic_list[flag]
                        flag=True
                    else:
                        flag=False
            if(flag==True):
                cur_tags[i]=behave_class
            else:
                cur_tags[i]=None
    befo_poses.update(cur_poses)




def cal_angle(x,y,o):#ox向量到oy向量的逆时针旋转角
    x=np.array(x)-np.array(o)
    y=np.array(y)-np.array(o)
    lx=np.sqrt(x.dot(x))
    ly=np.sqrt(y.dot(y))
    cos_ang=x.dot(y)/(lx*ly)
    ang=np.arccos(cos_ang)
    ang=ang*360/2/np.pi
    flag=x[0]*y[1]-x[1]*y[0]
    if(flag<0):
        ang=360-ang
    return ang
    
def fall_d(pose, pose_now):
    # 是动态
    # if static(pose, pose_now) != 1:
    if pose != None:
        # 考虑y坐标，选定的几个坐标都比之前低(point1.y,point3.y/point6.y)
        if pose[4] >= pose_now[4] and (pose[8] >= pose_now[8] and pose[14] >= pose_now[14]):
            # 区分sit，因此选择小腿部分y差比较;且考虑下降速度(此处1应是一个合适阈值，但尚未确定)
            if (pose_now[22] - pose_now[20]) <= 1 / 2 * (pose[22] - pose_now[20]) and \
                            (pose_now[28] - pose_now[26]) <= 1 / 2 * (pose[28] - pose_now[26]) :
#                    ((pose_now[4] - pose[4]) * (pose_now[0] - pose[0]) > 1 / 720):
                return bool(1)
            else:
                return bool(0)
        else:
            return bool(0)

    else:
        return bool(0)



def is_attacker(befo_pose,cur_pose,cur_flame_num,flag_delay):
#befo_pose 上一次采样的结果
#cur_pose 本次采样的结果
#cur_flame 当前帧号
#joint_check 一个长度为18的列表，表征哪些关节点有检测到，True即检测到
    joint_check=[]
    for i in range(18):
        if(cur_pose[2*i+1]==0 and cur_pose[2*i+2]==0):
            joint_check.append(False)
        else:
            joint_check.append(True)
    if(cur_pose[0]==cur_flame_num):#当前帧检测到了人
        pose=cur_pose[1:]
        for i in range(18):
            pose[2*i]=1280*pose[2*i]
            pose[2*i+1]=720*pose[2*i+1]

        if_side=False
        #分析是否侧对着
        if(joint_check[2] and joint_check[8]):
            tmp=np.linalg.norm(np.array(pose[2:4])-np.array(pose[4:6]))
            thres=8
            if(abs(pose[3]-pose[17])/tmp > thres):
                if_side=True
        elif(joint_check[2] and joint_check[11]):
            tmp=np.linalg.norm(np.array(pose[2:4])-np.array(pose[4:6]))
            thres=8
            if(abs(pose[3]-pose[23])/tmp > thres):
                if_side=True
        elif(joint_check[5] and joint_check[8]):
            tmp=np.linalg.norm(np.array(pose[2:4])-np.array(pose[10:12]))
            thres=8
            if(abs(pose[3]-pose[17])/tmp > thres):
                if_side=True
        elif(joint_check[5] and joint_check[11]):
            tmp=np.linalg.norm(np.array(pose[2:4])-np.array(pose[10:12]))
            thres=8
            if(abs(pose[3]-pose[23])/tmp > thres):
                if_side=True

        suspicious=0
        if(if_side==True):#侧身 考虑基于两方面，一个是一些关键动作的判别，一个是相对于上一阵的整体关键部位的位移情况
            butt_cent=[]#分析侧身的上下两点，以及哪个方向是侧向的方向
            neck_cent=pose[2:4]
            side=''
            if joint_check[0] and joint_check[1] :
                print('pose_0:',pose[0],'          pose_2:',pose[2])
                if pose[0]>pose[2] :
                    side='right'
                else:
                    side='left'
            if joint_check[8] and joint_check[11] :
                butt_cent=[(pose[16]+pose[22])/2,(pose[17]+pose[23])/2]
            elif joint_check[8] :
                butt_cent=pose[16:18]
            elif joint_check[11]:
                butt_cent=pose[22:24]
            elif joint_check[9] and joint_check[12] :
                butt_cent=[(pose[18]+pose[24])/2,(pose[19]+pose[25])/2]
            elif joint_check[9] :
                butt_cent=pose[18:20]
            elif joint_check[12] :
                butt_cent=pose[24:26]
            else:
                return False
            ang_thresh=160
            if(side=='left'):
                print('left')
                if  abs((butt_cent[0]-neck_cent[0])*1280) <15 :#stand straight---检测上半身
                    if joint_check[5] and joint_check[6] and joint_check[7] :
                        tmp_ang=cal_angle(pose[10:12],pose[14:16],pose[12:14])
                        if (pose[15] < pose[13] and pose[13]>pose[11] and tmp_ang<ang_thresh) :#and (pose[14]<pose[12] and pose[12]<pose[10]) :
                            suspicious+=2
                        if tmp_ang >ang_thresh and cal_angle(pose[12:14],butt_cent,neck_cent)>75 :
                            suspicious+=2
                    if joint_check[2] and joint_check[3] and joint_check[4] :
                        tmp_ang=cal_angle(pose[4:6],pose[8:10],pose[6:8])
                        if (pose[9] < pose[7] and pose[7]>pose[5] and tmp_ang<ang_thresh) :#and (pose[8]<pose[6] and pose[6]<pose[4]):
                            suspicious+=2
                        if tmp_ang>ang_thresh and cal_angle(pose[6:8],butt_cent,neck_cent)>75 :
                            suspicious+=2
                #检测下半身
                depth_thresh=5
                if joint_check[11] and joint_check[13] and joint_check[9] :
                    depth=pose[27]-pose[19]
                    if depth<depth_thresh :#and pose[26]<pose[22] and pose[22]<pose[18] :
                        suspicious+=1
                if joint_check[8] and joint_check[10] and joint_check[12] :
                    depth=pose[21]-pose[25]
                    if depth<depth_thresh :#and pose[20]<pose[16] and pose[16]<pose[24] :
                        suspicious+=1
            else:#right
                print('right')
                if  abs((butt_cent[0]-neck_cent[0])*1280 )<15 :#stand straight
                    if joint_check[5] and joint_check[6] and joint_check[7] :
                        tmp_ang=cal_angle(pose[14:16],pose[10:12],pose[12:14])
                        if (pose[15] < pose[13] and pose[13]>pose[11] and tmp_ang<ang_thresh) :#and (pose[14]>pose[12] and pose[12]>pose[10]) :
                            suspicious+=1
                        if tmp_ang >ang_thresh and cal_angle(butt_cent,pose[12:14],neck_cent)>80 :
                            suspicious+=1
                    if joint_check[2] and joint_check[3] and joint_check[4] :
                        tmp_ang=cal_angle(pose[8:10],pose[4:6],pose[6:8])
                        if (pose[9] < pose[7] and pose[7]>pose[5] and tmp_ang<ang_thresh) :#and (pose[8]>pose[6] and pose[6]>pose[4]):
                            suspicious+=1
                        if tmp_ang>ang_thresh and cal_angle(butt_cent,pose[6:8],neck_cent)>75 :
                            suspicious+=1
                #检测下半身
                depth_thresh=5
                if joint_check[11] and joint_check[13] and joint_check[9] :
                    depth=pose[27]-pose[19]
                    if depth<depth_thresh :#and pose[26]>pose[22] and pose[22]>pose[18] :
                        suspicious+=1
                if joint_check[8] and joint_check[10] and joint_check[12] :
                    depth=pose[21]-pose[25]
                    if depth<depth_thresh :#and pose[20]>pose[16] and pose[16]>pose[24] :
                        suspicious+=1
            if(suspicious>1):
                flag_delay[0]=1
                flag_delay[1]=1
                return True
            else:
                if(flag_delay[0]==1):
                    flag_delay[0]=0
                    return True
                elif(flag_delay[1]==1):
                    flag_delay[1]=0
                    return True
                else:
                    return False

def anti_fight(x,y,a):
    l=True
    if (y[7]<=y[3] or y[9]<=y[3]) and (y[13]<y[3]or y[15]<y[3]) and cal_angle(y[23:25],y[13:15],y[11:13])>30:
        if (y[7]<=x[7]or y[9]<=x[9]) and (y[13]<x[13] or y[15]<x[15]) and y[3]>x[3]+0.05:
            l=True
            a.append(y[0])
        elif (y[7]>=x[7]or y[9]>=x[9]) and y[13]>x[13]+0.05 and y[3]>x[3]+0.05:
            l=True
            a.append(y[0])
        else :
            l=False
    elif (y[7]>y[3] or y[9]>y[3]) and (y[13]>=y[3] or y[15]>=y[3]) and cal_angle(y[7:9],y[17:19],y[5:7])>30:
        if (y[7]>x[7] or y[9]>x[9]) and (y[13]>=x[13] or y[15]>=x[15]) and y[3]<y[3]-0.05:
            l=True
            a.append(y[0])
        elif y[7]<x[7]-0.05 and (y[13]<=x[13] or y[15]<=x[15]) and y[3]<x[3]-0.05:
            l=True
            a.append(y[0])
        else :
            l=False
    elif a[-1]!=0 and a[-1]>y[0]-30:
        l=True
    else :
        l=False
    return l

def basic(pose, pose_now,basic_delay):
    if pose[3]==0 and pose[4]==0 and pose_now[3]==0 and pose_now[4]==0:
        return 0
    if pose[17]==0 and pose[18]==0 and pose_now[23]==0 and pose_now[24]==0:
        return 0
    if pose[19]==0 and pose[20]==0 and pose_now[25]==0 and pose_now[26]==0:
        return 0
    
    else:
        if (pose_now[17]==0 and pose_now[18]==0) and (pose_now[23]!=0 or pose_now[24]!=0):
            pose_now[17]=pose_now[23]
            pose_now[18]=pose_now[24]
        elif (pose_now[17]!=0 or pose_now[18]!=0) and (pose_now[23]==0 or pose_now[24]==0):
            pose_now[23]=pose_now[17]
            pose_now[24]=pose_now[18]
        if (pose_now[19]==0 and pose_now[20]==0) and (pose_now[25]!=0 or pose_now[26]!=0):
            pose_now[19]=pose_now[25]
            pose_now[20]=pose_now[26]
        elif (pose_now[19]!=0 or pose_now[18]!=0) and (pose_now[25]==0 or pose_now[26]==0):
            pose_now[25]=pose_now[19]
            pose_now[26]=pose_now[20]
        
        if (pose[17]==0 and pose[18]==0) and (pose[23]!=0 or pose[24]!=0):
            pose[17]=pose[23]
            pose[18]=pose[24]
        elif (pose[17]!=0 or pose[18]!=0) and (pose[23]==0 or pose[24]==0):
            pose[23]=pose[17]
            pose[24]=pose[18]
        if (pose[19]==0 and pose[20]==0) and (pose[25]!=0 or pose[26]!=0):
            pose[19]=pose[25]
            pose[20]=pose[26]
        elif (pose[19]!=0 or pose[18]!=0) and (pose[25]==0 or pose[26]==0):
            pose[25]=pose[19]
            pose[26]=pose[20]
        
        init_x=float(pose[3]+pose[17]+pose[23])/3
        init_y=float(pose[4]+pose[18]+pose[24])/3
        end_x=float(pose_now[3]+pose_now[17]+pose_now[23])/3
        end_y=float(pose_now[4]+pose_now[18]+pose[24])/3

        init_h1=float(pose[18]+pose[24])/2-pose[4]
        end_h1=float(pose_now[18]+pose_now[24])/2-pose_now[4]
        try:
            h1=end_h1/init_h1
        except:
            h1=0.0
        init_h2=(float(pose[20]+pose[26])-float(pose[18]+pose[24]))/2
        end_h2=(float(pose_now[20]+pose_now[26])-float(pose_now[18]+pose_now[24]))/2
        try:
            h2 = end_h2 / init_h2
        except:
            h2 = 0.0
        xc = end_x - init_x
        yc = end_y - init_y
        if abs(xc) < 70. and abs(yc) < 20.:
            ty_1=float(pose_now[4])
            ty_8=float(pose_now[18]+pose_now[24])/2
            ty_9=float(pose_now[20]+pose[26])/2
            try:
                t = float(ty_8 - ty_1) / (ty_9 - ty_8)
            except:
                t = 0.0
            if h1 < 1.26 and h1 > 0.78 and h2 < 1.26 and h2 > 0.80:

                 if t < 1.73:
                    basic_delay[0]=1
                    return 1
                 else:
                     basic_delay[0]=2
                     return 2
            else:
                if t < 1.7:
                    if h1 >= 1.08:
                        basic_delay[0]=4
                        return 4

                    elif h1 < 0.92:
                        basic_delay[0]=5
                        return 5
                    else:
                        return basic_delay[0]
                else:
                    return basic_delay[0]
        elif abs(xc) < 70. and abs(yc) >= 40.:
            init_y1 = float(pose[4])
            init_y8 = float(pose[18] + pose[24]) / 2
            init_y9 = float(pose[20] + pose[26]) / 2

            end_y1 = float(pose_now[4])
            end_y8 = float(pose_now[18] + pose_now[24]) / 2
            end_y9 = float(pose_now[20] + pose_now[26]) / 2
            try:
                init_yc = float(init_y8 - init_y1) / (init_y9 - init_y8)
            except:
                init_yc = 0.0
            try:
                end_yc = float(end_y8 - end_y1) / (end_y9 - end_y8)
            except:
                end_yc = 0.0
            th_yc = 0.1
            if yc >= 25 and abs(end_yc - init_yc) >= th_yc:
                basic_delay[0]=6
                return 6
            elif yc < -20 and abs(end_yc - init_yc) >= th_yc:
                basic_delay[0]=7
                return 7
            else:
                return basic_delay[0]
        elif abs(xc) > 70. and abs(yc) < 40.:
            basic_delay[0]=3
            return 3
        else:
            return basic_delay[0]





                
            
            