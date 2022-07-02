# -*- coding: utf-8 -*-
# @Time    : 2022/6/25 21:38
# @Author  : 石鑫磊
# @Site    : 
# @File    : vgg16_preprocess.py
# @Software: PyCharm 
# @Comment :
# 引入Tensorflow框架
import tensorflow as tf
# 引入keras
from tensorflow import keras
# 引入keras层结构
from tensorflow.keras import layers
# 引入Numpy数据处理模块
import numpy as np
# 引入时间模块
import time
# 引入日期模块
from datetime import datetime
# 引入图像绘制模块
import matplotlib.pyplot as plt
# 引入字体属性
from matplotlib.font_manager import FontProperties

# 系统，开发者依系统而定
font = FontProperties(fname="/Library/Fonts/Songti.ttc", size=8)

# 风格权重
style_weight = 1e-2
# 内容权重
content_weight = 1e4


def image_read(image_path):
    """图像读取
    参数:
        image_path: 图像路径
    返回:
        img: 图像矩阵数据（0.0~1.0）
    """
    # 读取图像文件
    img = tf.io.read_file(image_path)
    # 图像Base64编码
    img = tf.io.encode_base64(img)
    # 图像Base64解码
    img = tf.io.decode_base64(img)
    # 图像Base64转矩阵数据
    img = tf.io.decode_image(img)
    # 图像矩阵数据转为0.0~1.0范围
    img = tf.image.convert_image_dtype(img, dtype=tf.float32)
    # 添加数据维度
    img = img[np.newaxis, :]
    return img


def layers_name():
    """预训练卷积神经网络层
    参数:
        无
    返回:
        提取内容层
        提取风格层
        内容层数量
        风格层数量
    """
    # 提取图像内容层
    pre_contents_layers = ["block5_conv2"]
    # 提取图像风格层
    pre_styles_layers = [
        "block1_conv1",
        "block2_conv1",
        "block3_conv1",
        "block4_conv1"
    ]
    # 风格层数量
    num_style_layers = len(pre_styles_layers)
    # 内容层数量
    num_content_layers = len(pre_contents_layers)
    # 返回数据
    return pre_contents_layers, pre_styles_layers, num_style_layers, num_content_layers


def pre_vgg16(layers_name):
    """预训练神经网络提取信息
    参数:
        layers: 神经网络层
    返回:
        Model对象
    """
    vgg16 = tf.keras.applications.VGG16(include_top=False, weights="imagenet")
    vgg16.trainable = False
    outputs = [vgg16.get_layer(name).output for name in layers_name]
    model = tf.keras.Model([vgg16.input], outputs)
    return model

def pre_res_single(layers_name, outputs):
    """输出预训练神经网络图像特征
    参数:
        layers_name: 网络层
        outputs: 预训练神经网络特征
    返回:
        无
    """

    i = 0
    for name, output in zip(layers_name, outputs):
        i += 1
        plt.figure(i)
        print("网络层:", name)
        print("特征维度:", output.numpy().shape)
        # print("特征值:", output.numpy())

        for j in range(4):
            plt.subplot(2, 2, j + 1)
            plt.subplots_adjust(wspace=0.3, hspace=0.3)
            plt.imshow(output.numpy()[0][:, :, j])
            plt.title(name + "-" + str(j + 1))
        plt.savefig("./images/feature-{}.png".format(i), format="png", dpi=300)
    plt.show()


def main():
    stamp = datetime.now().strftime("%Y%m%d-%H:%M:%S")
    # 图像路径：图像内容
    image_content_path = "./images/swimming_monkey.jpg"
    # 图像路径：图像风格
    image_style_path = "./images/starry.jpg"
    # 图像内容
    img_contents = image_read(image_content_path)
    # 获取图像数据信息
    # print("图像数据维度:", img_contents.shape)
    # [0.0, 1.0]
    # print("图像数据:", img_contents)
    # [0.0, 1.0]
    img_styles = image_read(image_style_path)
    # 获取层结构以及层数量
    pre_contents_layers, pre_styles_layers, num_style_layers, num_content_layers = layers_name()
    # 预训练模型数据提取
    pre_model = pre_vgg16(pre_styles_layers)
    pre_styles = pre_model(img_styles * 255)
    # 获取图像特征
    pre_res_single(pre_styles_layers, pre_styles)
if __name__ == "__main__":
    main()



