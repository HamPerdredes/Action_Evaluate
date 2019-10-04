import pandas as pd
from enum import Enum
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout,Embedding,SimpleRNN,LSTM
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
from keras.models import load_model

import matplotlib.pyplot as plt
from keras.callbacks import Callback
import itertools
from sklearn.metrics import confusion_matrix
#将main.py中注释掉的数据读取部分取消注释，获得txt文件，再转换成csv文件
#注意修改这里的Action和action_enum.py文件内的Action中与csv文件动作编号相对应
#分类器至少要求两类动作
class Actions(Enum):
    strike_car = 0
    fight = 1
    stand = 2
    # framewise_recognition.h5
    # squat = 0
    # stand = 1
    # walk = 2
    # wave = 3
'''
    # framewise_recognition_under_scene.h5
    stand = 0
    walk = 1
    operate = 2
    fall_down = 3

    # run = 4
'''

# Callback class to visialize training progress
class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = {'batch':[], 'epoch':[]}
        self.accuracy = {'batch':[], 'epoch':[]}
        self.val_loss = {'batch':[], 'epoch':[]}
        self.val_acc = {'batch':[], 'epoch':[]}

    def on_batch_end(self, batch, logs={}):
        self.losses['batch'].append(logs.get('loss'))
        self.accuracy['batch'].append(logs.get('acc'))
        self.val_loss['batch'].append(logs.get('val_loss'))
        self.val_acc['batch'].append(logs.get('val_acc'))

    def on_epoch_end(self, batch, logs={}):
        self.losses['epoch'].append(logs.get('loss'))
        self.accuracy['epoch'].append(logs.get('acc'))
        self.val_loss['epoch'].append(logs.get('val_loss'))
        self.val_acc['epoch'].append(logs.get('val_acc'))

    def loss_plot(self, loss_type):
        iters = range(len(self.losses[loss_type]))
        plt.figure()
        # acc
        plt.plot(iters, self.accuracy[loss_type], 'r', label='train acc')
        # loss
        plt.plot(iters, self.losses[loss_type], 'g', label='train loss')
        if loss_type == 'epoch':
            # val_acc
            plt.plot(iters, self.val_acc[loss_type], 'b', label='val acc')
            # val_loss
            plt.plot(iters, self.val_loss[loss_type], 'k', label='val loss')
        plt.grid(True)
        plt.xlabel(loss_type)
        plt.ylabel('acc-loss')
        plt.legend(loc="upper right")
        plt.show()


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# load data
#注意根据csv文件名修改
#raw_data = pd.read_csv('data_with_scene.csv', header=0)
raw_data = pd.read_csv('origin_data.csv', header=0)
dataset = raw_data.values
# X = dataset[:, 0:36].astype(float)
# Y = dataset[:, 36]
'''
X = dataset[0:3289, 0:36].astype(float)  # 忽略run数据
Y = dataset[0:3289, 36]
'''
#这里的0:1625指的是全部数据的行，0:36指的是对应关键点
Y = dataset[0:1816, 36]
X= dataset[0:1816,0:36]
X=np.reshape(X,[-1,6,6])
# 将类别编码为数字
# encoder = LabelEncoder()
# encoder_Y = encoder.fit_transform(Y)
# print(encoder_Y[0], encoder_Y[900], encoder_Y[1800], encoder_Y[2700])
# encoder_Y = [0]*744 + [1]*722 + [2]*815 + [3]*1008 + [4]*811
#encoder_Y = [0]*744 + [1]*722 + [2]*815 + [3]*1008
#这里0 1分别对应编号为0 1的动作在csv文件中的行数
encoder_Y = [0]*904+[1]*528+[2]*384
# one hot 编码
dummy_Y = np_utils.to_categorical(encoder_Y)

# train test split
X_train, X_test, Y_train, Y_test = train_test_split(X, dummy_Y, test_size=0.1, random_state=9)
# build keras model
#可考虑对整个神经网络进行修改
'''
model = Sequential()
model.add(Dense(units=128, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(units=64, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(units=16, activation='relu'))
model.add(BatchNormalization())
#model.add(Dense(units=4, activation='softmax'))  # units = nums of classes
#注意将此处的输出数修改为分类数
model.add(Dense(units=3, activation='softmax'))
'''
'''
model=Sequential(name='rnn')
model.add(SimpleRNN(256,dropout=0.2,input_shape=(6,6)))#without difine input shape
model.add(Dropout(0.2))
model.add(Dense(units=256,activation='relu'))
model.add(Dropout(0.2))
'''
model=Sequential()
model.add(LSTM(units=60,input_shape=(6,6)))
model.add(Dense(256,activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(128,activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(64,activation='relu'))
model.add(Dropout(0.2))
#model.add(Dense(units=4, activation='softmax'))  # units = nums of classes
#注意将此处的输出数修改为分类数
model.add(Dense(units=3, activation='softmax'))
# training
his = LossHistory()
model.compile(loss='categorical_crossentropy', optimizer=Adam(0.0001), metrics=['accuracy'])
model.fit(X_train, Y_train, batch_size=32, epochs=500, verbose=1, validation_data=(X_test, Y_test), callbacks=[his])#epochs=20
model.summary()
his.loss_plot('epoch')
#注意这里训练完的模型和train.py在同一文件价内，而main.py读取的路径在Action文件夹下
model.save('framewise_recognition.h5')#saving
# # evaluate and draw confusion matrix
# print('Test:')
# score, accuracy = model.evaluate(X_test,Y_test,batch_size=32)
# print('Test Score:{:.3}'.format(score))
# print('Test accuracy:{:.3}'.format(accuracy))
# # confusion matrix
# Y_pred = model.predict(X_test)
# cfm = confusion_matrix(np.argmax(Y_test,axis=1), np.argmax(Y_pred, axis=1))
# np.set_printoptions(precision=2)
#
# plt.figure()
# class_names = ['squat', 'stand', 'walk', 'wave']
# plot_confusion_matrix(cfm, classes=class_names, title='Confusion Matrix')
# plt.show()

# # test
# model = load_model('framewise_recognition.h5')
#
# test_input = [0.43, 0.46, 0.43, 0.52, 0.4, 0.52, 0.39, 0.61, 0.4,
#               0.67, 0.46, 0.52, 0.46, 0.61, 0.46, 0.67, 0.42, 0.67,
#               0.42, 0.81, 0.43, 0.91, 0.45, 0.67, 0.45, 0.81, 0.45,
#               0.91, 0.42, 0.44, 0.43, 0.44, 0.42, 0.46, 0.44, 0.46]
# test_np = np.array(test_input)
# test_np = test_np.reshape(-1, 36)
#
# test_np = np.array(X[1033]).reshape(-1, 36)
# if test_np.size > 0:
#     pred = np.argmax(model.predict(test_np))
#     init_label = Actions(pred).name
#     print(init_label)
