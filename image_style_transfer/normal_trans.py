# -*- coding: utf-8 -*-
# @Time    : 2022/6/25 21:10
# @Author  : 石鑫磊
# @Site    : 
# @File    : normal_trans.py
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

# 风格权重
style_weight=1e-2
# 内容权重
content_weight=1e4
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
    print(img.shape)
    print(np.newaxis)
    img = img[np.newaxis, :]
    print(img.shape)
    return img



if __name__ == "__main__":
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
    # print("图像数据:",img_contents)
    # [0.0, 1.0]
    img_styles = image_read(image_style_path)
    # 获取层结构以及层数量
