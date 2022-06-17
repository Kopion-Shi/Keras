import numpy as np

import tensorflow as tf

from keras.models import Sequential  # 采用贯序模型
from keras.layers import Input, Dense, Dropout, Activation, Conv2D, MaxPool2D, Flatten
from keras.optimizers import SGD
from keras.datasets import mnist
from keras.utils import to_categorical
from keras.callbacks import TensorBoard
import tensorflow.compat.v1 as tf
tf.disable_eager_execution()



def create_model():
    model = Sequential()
    model.add(Conv2D(32, (5, 5), activation='relu', input_shape=[28, 28, 1]))  # 第一卷积层
    model.add(Conv2D(64, (5, 5), activation='relu'))  # 第二卷积层
    model.add(MaxPool2D(pool_size=(2, 2)))  # 池化层
    model.add(Flatten())  # 平铺层
    model.add(Dropout(0.5))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))
    return model


def compile_model(model):
    model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['acc'])
    return model


def train_model(model, x_train, y_train, x_val, y_val, batch_size=128, epochs=10):
    train_loss = tf.placeholder(tf.float32, [], name='train_loss')
    train_acc = tf.placeholder(tf.float32, [], name='train_acc')
    val_loss = tf.placeholder(tf.float32, [], name='val_loss')
    val_acc = tf.placeholder(tf.float32, [], name='val_acc')

    # 可视化训练集、验证集的loss、acc、四个指标，均是标量scalers
    tf.summary.scalar("train_loss", train_loss)
    tf.summary.scalar("train_acc", train_acc)
    tf.summary.scalar("val_loss", val_loss)
    tf.summary.scalar("val_acc", val_acc)

    merge = tf.summary.merge_all()

    batches = int(len(x_train) / batch_size)  # 没一个epoch要训练多少次才能训练完样本

    with tf.Session() as sess:
        logdir = './logs'
        writer = tf.summary.FileWriter(logdir, sess.graph)

        for epoch in range(epochs):  # 用keras的train_on_batch方法进行训练
            print(F"正在训练第 {epoch + 1} 个 epoch")
            for i in range(batches):
                # 每次训练128组数据
                train_loss_, train_acc_ = model.train_on_batch(x_train[i * 128:(i + 1) * 128:1, ...],
                                                               y_train[i * 128:(i + 1) * 128:1, ...])
                # 验证集只需要每一个epoch完成之后再验证即可
            val_loss_, val_acc_ = model.test_on_batch(x_val, y_val)

            summary = sess.run(merge, feed_dict={train_loss: train_loss_, train_acc: train_acc_, val_loss: val_loss_,
                                                 val_acc: val_acc_})
            writer.add_summary(summary, global_step=epoch)


if __name__ == "__main__":
    (x_train, y_train), (x_test, y_test) = mnist.load_data()  # 数据我已经下载好了
    print(np.shape(x_train), np.shape(y_train), np.shape(x_test),
          np.shape(y_test))  # (60000, 28, 28) (60000,) (10000, 28, 28) (10000,)

    x_train = np.expand_dims(x_train, axis=3)
    x_test = np.expand_dims(x_test, axis=3)
    y_train = to_categorical(y_train, num_classes=10)
    y_test = to_categorical(y_test, num_classes=10)
    print(np.shape(x_train), np.shape(y_train), np.shape(x_test),
          np.shape(y_test))  # (60000, 28, 28, 1) (60000, 10) (10000, 28, 28, 1) (10000, 10)

    x_train_ = x_train[1:50000:1, ...]  # 重新将训练数据分成训练集50000组
    x_val_ = x_train[50000:60000:1, ...]  # 重新将训练数据分成测试集10000组
    y_train_ = y_train[1:50000:1, ...]
    y_val_ = y_train[50000:60000:1, ...]
    print(np.shape(x_train_), np.shape(y_train_), np.shape(x_val_), np.shape(y_val_), np.shape(x_test),
          np.shape(y_test))
    # (49999, 28, 28, 1) (49999, 10) (10000, 28, 28, 1) (10000, 10) (10000, 28, 28, 1) (10000, 10)

    model = create_model()  # 创建模型

    model = compile_model(model)  # 编译模型

    train_model(model, x_train_, y_train_, x_val_, y_val_)


