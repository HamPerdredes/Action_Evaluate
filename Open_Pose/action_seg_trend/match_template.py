class Action_segmentor :
    def __init__(self,action_type):
        if action_type == 'Forehand_stroke' :#正手击球
            self.step_one=[1,1,1]#list存储拟合的运动趋势方程的函数
            self.step_two=[2,2,2]
