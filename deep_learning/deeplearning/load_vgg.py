# -*- coding:utf-8 -*-
# Created by ibf at 2018/11/24 0024
import scipy.io
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from scipy.misc import imread
# 1.加载vgg mat模型
# mat 路径
mat_path='./imagenet-vgg-verydeep-19.mat'

# 2.读取他每一层的权重
def _conv_layer(img,w,b):
    conv=tf.nn.conv2d(tf.cast(img,tf.float32),tf.constant(w,dtype=tf.float32),strides=[1,1,1,1],padding='SAME')
    return tf.nn.bias_add(conv,b)

def net(mat_path,input_image):
    # (1)2个 3*3-64的卷积
    # (2)2个3*3-128的卷积
    # (3)4个3*3-256的卷积
    # (4)4个3*3-512的卷积
    # (5)4个3*3-512的卷积
    # (6)2个fc-4096
    # （7）fc-1000
    # input_image 也需要做个预处理，resize成为一个（224，224,3) 图片数组
    network=scipy.io.loadmat(mat_path)
    #print(network['layers'])# 网络层，网络层中间包含了偏置量、卷积核、卷积窗口大小，fc权重。。。
    # 构建一个字典，来存储每一个权重，方便之后调用
    layers=(
        'conv1_1','relu1_1','conv1_2','relu1_2','pool1',
        'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',
        'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3', 'relu3_3', 'conv3_4', 'relu3_4','pool3',
        'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3', 'relu4_3', 'conv4_4', 'relu4_4','pool4',
        'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3', 'relu5_3', 'conv5_4', 'relu5_4','pool5'
    )
    weights=network['layers'][0]
    dense=input_image
    net={} #记录每一层计算的结果
    for i,name in enumerate(layers):
        kind=name[:4]
        # print(kind)
        # print(name)
        if kind=='conv':
            #print(weights[i][0][0][0][0])
            kernels,bias=weights[i][0][0][0][0]
            # mat_conv_net shape:[width,height,in_channels,out_channels]
            # tf_conv_net shape:[height,width,in_channels,out_channels]
            kernels=np.transpose(kernels,[1,0,2,3])
            bias=bias.reshape(-1)
            # tf conv 进行卷积计算
            dense=_conv_layer(dense,kernels,bias)
        elif kind=='relu':
            dense=tf.nn.relu(dense)
        elif kind=='pool':
            dense=tf.nn.max_pool(dense,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
        net[name]=dense
    mean = network['normalization'][0][0][0]  # 这个值代表了mean值，原始图片数组-mean实现预处理
    mean_pixel=np.mean(mean,axis=(0,1))
    print(mean_pixel)
    # (224,224,3) (3,)
    return net,mean_pixel,layers

#
# 3.查看每一层的输出结果
#图片预处理
def process(img,mean_pixel):
    return img-mean_pixel

# 还原#图片预处理
def unprocess(img,mean_pixel):
    return img+mean_pixel

with tf.Session() as sess:
    input_image=imread('./ocean.jpg').astype(np.uint8) # 读取图片
    # 图片重组
    shape = (1,input_image.shape[0],input_image.shape[1],input_image.shape[2])
    image = tf.placeholder(tf.float32, shape=shape)
    out, mean_pixel, layers = net(mat_path,image)
    imge_pro=np.array([process(input_image,mean_pixel)])
    for i,name in enumerate(layers):
        print('{}/{}:{}'.format(i+1,len(layers),name))
        features=out[name].eval(feed_dict={image:imge_pro})

        print('Type of "features" is {}'.format(type(features)))
        plt.figure(i+1,figsize=(10,5))
        plt.matshow(features[0,:,:,0],cmap=plt.cm.gray,fignum=i+1)
        plt.title(name)
        plt.colorbar()# 颜色柱状条
        plt.show()

# vgg.mat 在做图像分类、图片向量化，特征工程
# 使用vgg输出的1000个向量作为我们自己网络的输入，然后进行分类
#这样计算时间会减小，因为不需要训练太多的参数

# (1)一般使用训练好的模型,调用它的神经网络输出的向量做为自定义网络的输入
# (2)一般使用训练好的模型,调用它的fc层前一层的输出内容做为自定义全连接层的输入
# **(3) 使用训练好的模型的中间的几层的参数和输出内容，作为自定义网络的一部分进行合理镶嵌。

# facenet（镶嵌了mtcnn网络中的pnet，onet，rnet三个结构）