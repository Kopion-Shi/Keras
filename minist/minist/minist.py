import tensorflow.keras as keras
from tensorflow.keras.datasets import mnist
import numpy as np
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
import cv2
import matplotlib
import os

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
def load_data():

    num_classes = 10

    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()

    print(train_images.shape, train_labels.shape)
    print(test_images.shape, test_labels.shape)

    ######
    img = np.reshape(train_images[0], (-1, 28, 28, 1))

    #####

    img_row, img_col, channel = 28, 28, 1

    mnist_input_shape = (img_row, img_col, 1)

    # 将数据维度进行处理
    train_images = train_images.reshape(train_images.shape[0], img_row, img_col, channel)
    test_images = test_images.reshape(test_images.shape[0], img_row, img_col, channel)

    train_images = train_images.astype("float32")
    test_images = test_images.astype("float32")

    ## 进行归一化处理
    train_images /= 255
    test_images /= 255

    # 将类向量，转化为类矩阵
    # 从 5 转换为 0 0 0 0 1 0 0 0 0 0 矩阵
    train_labels = keras.utils.to_categorical(train_labels, num_classes)
    test_labels = keras.utils.to_categorical(test_labels, num_classes)
    return train_images,test_images,train_labels,test_labels

"""
构造网络结构
"""
def model():
    num_classes = 10
    img_row, img_col, channel = 28, 28, 1
    mnist_input_shape = (img_row, img_col, 1)

    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),
                     activation="relu",
                     input_shape=mnist_input_shape))
    # kernalsize = 3*3 并没有改变数据维度
    model.add(Conv2D(16, kernel_size=(3, 3),
                     activation="relu"
                     ))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # 进行数据降维操作
    model.add(Flatten())  # Flatten层用来将输入“压平”，即把多维的输入一维化，
    # 常用在从卷积层到全连接层的过渡。Flatten不影响batch的大小。
    model.add(Dense(32, activation="relu"))
    # 全连接层
    model.add(Dense(num_classes, activation='softmax'))

    """
    编译网络模型,添加一些超参数
    """

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])
    return model
####




if __name__=='__main__':
    train_images,test_images,train_labels,test_labels=load_data()
    log_path= '../logs/mat'
    summary_write=tf.summary.create_file_writer(logdir=log_path)
    ####
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        tf.summary.trace_on(graph=True, profiler=True)
        model=model()
        with summary_write.as_default():
            tf.summary.trace_export(name='mat',step=0,profiler_outdir=log_path)
        model.summary()

    model.fit(train_images,
              train_labels,
              batch_size=32,
              epochs=100,
              verbose=1,
              validation_data=(test_images, test_labels),
              shuffle=True,
      )
    score = model.evaluate(test_images, test_labels, verbose=1)

    print('test loss:', score[0])
    print('test accuracy:', score[1])
