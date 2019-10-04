from tools import curve_fit,trans_data,load_data

def get_standard_template(file_string,key_frame,action_type,action_num):
    #key_frame为一个列表，存储两个相邻分解动作的最后一帧和第一帧
    data=load_data(file_string)
    ang_data=trans_data(data)
    f=open(action_type+'.txt','w')
    for i in range(len(key_frame)/2) :
        for j in range(len(ang_data)):
            cur_W=curve_fit(ang_data[j][key_frame[2*i]-action_num:key_frame[2*i]])
            

