import numpy as np
import matplotlib.pyplot as plt
#def cal_trend(eight_angs):#考虑每30帧对个体拟合各部件的运动趋势

#def trend_match(eight_angs):


def curve_fit(data,is_show=False):#某一部件的旋转角数据,list
    data_y=np.array(data[:])
    flag=(data_y!=-1)
    data_x=np.array([i for i in range(data_y.shape[0])])
    data_x=data_x[flag]
    data_y=data_y[flag]
    W=np.polyfit(data_x,data_y,3)
    if is_show :
        f_x=np.poly1d(W)
        print(W)
        print(f_x)
        f_x=f_x(data_x)
        plt.plot(data_x,data_y,'*',label='original')
        plt.plot(data_x,f_x,color='b',label='fitting curve')
        plt.xlabel('frame_num')
        plt.ylabel('response')
        plt.legend(loc='upper left')
        plt.title('curve fit')
        plt.show()
    return W


def cal_angle(O,A,B):
    #计算OA到OB的顺时针旋转角,OAB格式为np.array,返回角度范围0-360°
    x=A-O
    y=B-O
    x_len=np.sqrt(x.dot(x))
    y_len=np.sqrt(y.dot(y))
    cos_ang=x.dot(y)/(x_len*y_len)
    angle=np.arccos(cos_ang)
    flag=(A[0]-O[0])*(B[1]-O[1])-(B[0]-O[0])*(A[1]-O[1])
    if flag>0 :
        angle=2*np.pi-angle
    angle=angle*180/np.pi
    return angle

def trans_data(joints):#未验证是否正确
    #将关节点数据由位置转换成各部件旋转角，每个部件的数据单独放在一个列表内
    joint_2=[]  #joint后面编号对应所计算旋转角的旋转点
    joint_3=[]
    joint_5=[]
    joint_6=[]
    joint_8=[]
    joint_9=[]
    joint_11=[]
    joint_12=[]
    for i in joints:
        human=np.array(i)
        key_flag=[]#记录部件是否被检测到
        for i in range(18):#18个部件，x y依次排布
            if human[2*i]==0 and human[2*i+1]==0 :
                key_flag.append(False)
            else:
                key_flag.append(True)
        if key_flag[1] :
            if key_flag[2] and key_flag[3] :
                joint_2.append(cal_angle(human[4:6],human[2:4],human[6:8]))
            else :
                joint_2.append(-1)
            if key_flag[5] and key_flag[6] :
                joint_5.append(cal_angle(human[10:12],human[2:4],human[12:14]))
            else :
                joint_5.append(-1)
            if key_flag[8] and key_flag[9] :
                joint_8.append(cal_angle(human[16:18],human[2:4],human[18:20]))
            else :
                joint_8.append(-1)
            if key_flag[11] and key_flag[12] :
                joint_11.append(cal_angle(human[22:24],human[2:4],human[24:26]))
            else :
                joint_11.append(-1)
        else :
            joint_2.append(-1)
            joint_5.append(-1)
            joint_8.append(-1)
            joint_11.append(-1)
        if key_flag[2] and key_flag[3] and key_flag[4] :
            joint_3.append(cal_angle(human[6:8],human[4:6],human[8:10]))
        else :
            joint_3.append(-1)
        if key_flag[5] and key_flag[6] and key_flag[7] :
            joint_6.append(cal_angle(human[12:14],human[10:12],human[14:16]))
        else :
            joint_6.append(-1)
        if key_flag[8] and key_flag[9] and key_flag[10] :
            joint_9.append(cal_angle(human[18:20],human[16:18],human[20:22]))
        else :
            joint_9.append(-1)
        if key_flag[11] and key_flag[12] and key_flag[13] :
            joint_12.append(cal_angle(human[24:26],human[22:24],human[26:28]))
        else :
            joint_12.append(-1)
    return [joint_2,joint_3,joint_5,joint_6,joint_8,joint_9,joint_11,joint_12]


def load_data(file_string):
    data=[]
    with open(file_string,'r') as fp :#load data
        while True :
            lines=fp.readline()
            if not lines :
                break
            tmp=[float(i) for i in lines.split()]
            data.append(tmp)
        data=np.array(data)
    return data

#TEST
data=[-1,5,5,6,-1,8,9]
curve_fit(data,is_show=True)
