This is a non-profit project which aims to figure a proper way to assessment someone's action.
Data: https://pan.baidu.com/s/1-KERFHV97Ma9tK4bvoObmA	提取码：u8v6

工程概述：
整体上，我们考虑基于某种网络提取出人体关节点坐标，再基于各主要部件之间的旋转角，实现在分离动作时关键帧的匹配，从而实现动作的分块，再对分块后的动作片段进行归一化（基于插帧和等间隔采样），即确保其与标准动作视频长度一致，而后再通过《Learning Character-Agnostic Motion for Motion Retargeting in 2D》中提到的动作迁移算法(https://github.com/ChrisWu1997/2D-Motion-Retargeting),实现由标准动作到测试者骨架上的动作迁移，从而避免比对时骨骼差异或是拍摄视角不同而带来的干扰，随后通过逐帧匹配的方式进行评估。


目前基于OpenPose做的关键点提取，具体使用VGG net或者mobilenet可在main中进行修改；
提取出关键点后，输出各主要部件的旋转角(0~360)为txt文件，见gained_data；
而后考虑基于这些角度进行关键帧匹配；---目前的进度

当前的演示demo是在mobile net下跑出来的，VGG的预训练模型需要从外网下载(见/Pose/graph_models/VGG_origin/download.sh)，效果会比mobile net要好很多，但速度有所下降。

硬件环境：
GTX 1660TI

需要的依赖:
tensorflow/tensorflow-gpu
opencv
keras-gpu
scikit-learn
opencv 3.4.7

演示demo见(github文件大小限制)：
https://pan.baidu.com/s/1NvXU27kKsnX9sjuPEtMY3A
提取码：ifhi

