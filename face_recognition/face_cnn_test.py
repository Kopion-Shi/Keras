import random

import numpy as np
from tensorflow import keras
import tensorflow as tf
from keras import backend as K
from keras import layers
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.models import Sequential  # 序贯模型是函数式模型的简略版，为最简单的线性、从头到尾的结构顺序，不分叉，是多个网络层的线性堆叠。
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator  # 图片生成器，同时也可以在batch中对数据进行增强，扩充数据集大小，增强模型的泛化能力。比如进行旋转，变形，归一化等等。
from keras.utils import np_utils
from sklearn.model_selection import train_test_split  # 分割训练集和测试集
from tensorflow.keras.optimizers import SGD

from keras.callbacks import TensorBoard

from load_dataset import load_dataset, resize_image, IMAGE_SIZE

# 1创建Sequential模型，2添加所需要的神经层，3使用.compile方法确定模型训练结构，4使用.fit方法使模型与训练数据“拟合”，5.predict方法进行预测。
# Dense层：全连接层 Dropout：神经网络单元，按照一定的概率将其暂时从网络中丢弃
# activatio函数n：激活函数   Flatten层：来将输入压平，即把多维的输入一维化，常用在从卷积层到全连接层的过度
# optimizers :优化器
# SGD:随机梯度下降法;SGDM:随机梯度下降算法_动量优化;NAG牛顿加速梯度算法
# AdaGrad:自适应学习率优化算法;AdaDelta:自适应学习率优化算法;Nadam:自适应学习率优化算法
# 保存model到指定文件夹和加载load_model指定文件夹中的文件
# 用Keras写出兼容theano和tensorflow两种backend的代码，那么你必须使用抽象keras backend API来写代码

gpus = tf.config.list_physical_devices(device_type='GPU')  # 指定GPU、CPU
for gpu in gpus:
    tf.config.experimental.set_memory_growth(device=gpu, enable=True)


#     将所有 GPU 设置为仅在需要时申请显存空间


class Dataset:
    def __init__(self, path_name):
        # 训练集
        self.train_images = None
        self.train_labels = None

        # 验证集
        self.valid_images = None
        self.valid_labels = None

        # 测试集
        self.test_images = None
        self.test_labels = None

        # 数据集加载路径
        self.path_name = path_name

        # 当前库采用的维度顺序
        self.input_shape = None

    # 加载数据集并按照交叉验证的原则划分数据集并进行相关预处理工作
    def load(self, img_rows=IMAGE_SIZE, img_cols=IMAGE_SIZE,
             img_channels=3, nb_classes=2):
        # 加载数据集到内存
        images, labels = load_dataset(self.path_name)

        train_images, valid_images, train_labels, valid_labels = train_test_split(images, labels, test_size=0.3,
                                                                                  random_state=random.randint(0, 100))
        _, test_images, _, test_labels = train_test_split(images, labels, test_size=0.5,
                                                          random_state=random.randint(0, 100))
        """
        X_train,X_test, y_train, y_test =sklearn.model_selection.train_test_split(train_data,train_target,test_size=0.4, random_state=0,stratify=y_train)
        # train_data：所要划分的样本特征集

        # train_target：所要划分的样本结果

        # test_size：样本占比，如果是整数的话就是样本的数量

        # random_state：是随机数的种子。
        # 随机数种子：其实就是该组随机数的编号，在需要重复试验的时候，保证得到一组一样的随机数。比如你每次都填1，其他参数一样的情况下你得到的随机数组是一样的。但填0或不填，每次都会不一样。

        stratify是为了保持split前类的分布。比如有100个数据，80个属于A类，20个属于B类。如果train_test_split(... test_size=0.25, stratify = y_all), 那么split之后数据如下： 
        training: 75个数据，其中60个属于A类，15个属于B类。 
        testing: 25个数据，其中20个属于A类，5个属于B类。 

        用了stratify参数，training集和testing集的类的比例是 A：B= 4：1，等同于split前的比例（80：20）。通常在这种类分布不平衡的情况下会用到stratify。

        将stratify=X就是按照X中的比例分配 

        将stratify=y就是按照y中的比例分配 

        整体总结起来各个参数的设置及其类型如下：

        主要参数说明：

        *arrays：可以是列表、numpy数组、scipy稀疏矩阵或pandas的数据框

        test_size：可以为浮点、整数或None，默认为None

        ①若为浮点时，表示测试集占总样本的百分比

        ②若为整数时，表示测试样本样本数

        ③若为None时，test size自动设置成0.25

        train_size：可以为浮点、整数或None，默认为None

        ①若为浮点时，表示训练集占总样本的百分比

        ②若为整数时，表示训练样本的样本数

        ③若为None时，train_size自动被设置成0.75

        random_state：可以为整数、RandomState实例或None，默认为None

        ①若为None时，每次生成的数据都是随机，可能不一样

        ②若为整数时，每次生成的数据都相同

        stratify：可以为类似数组或None

        ①若为None时，划分出来的测试集或训练集中，其类标签的比例也是随机的

        ②若不为None时，划分出来的测试集或训练集中，其类标签的比例同输入的数组中类标签的比例相同，可以用于处理不均衡的数据集
        """

        # 当前的维度顺序如果为'channels_first'，则输入图片数据时的顺序为：channels,rows,cols，否则:rows,cols,channels
        # 这部分代码就是根据keras库要求的维度顺序重组训练数据集
        if K.image_data_format() == "channels_first":
            train_images = train_images.reshape(train_images.shape[0], img_channels, img_rows, img_cols)
            valid_images = valid_images.reshape(valid_images.shape[0], img_channels, img_rows, img_cols)
            test_images = test_images.reshape(test_images.shape[0], img_channels, img_rows, img_cols)
            self.input_shape = (img_channels, img_rows, img_cols)
        else:
            train_images = train_images.reshape(train_images.shape[0], img_rows, img_cols, img_channels)
            valid_images = valid_images.reshape(valid_images.shape[0], img_rows, img_cols, img_channels)
            test_images = test_images.reshape(test_images.shape[0], img_rows, img_cols, img_channels)
            self.input_shape = (img_rows, img_cols, img_channels)

            # 输出训练集、验证集、测试集的数量
            print(train_images.shape[0], 'train samples')
            print(valid_images.shape[0], 'valid samples')
            print(test_images.shape[0], 'test samples')

            # 我们的模型使用categorical_crossentropy作为损失函数，因此需要根据类别数量nb_classes将
            # 类别标签进行one-hot编码使其向量化，在这里我们的类别只有两种，经过转化后标签数据变为二维
            train_labels = np_utils.to_categorical(train_labels, nb_classes)
            valid_labels = np_utils.to_categorical(valid_labels, nb_classes)
            test_labels = np_utils.to_categorical(test_labels, nb_classes)

            # 像素数据浮点化以便归一化:int->浮点型
            train_images = train_images.astype('float32')
            valid_images = valid_images.astype('float32')
            test_images = test_images.astype('float32')

            # 将其归一化,图像的各像素值归一化到0~1区间
            train_images /= 255
            valid_images /= 255
            test_images /= 255

            self.train_images = train_images
            self.valid_images = valid_images
            self.test_images = test_images
            self.train_labels = train_labels
            self.valid_labels = valid_labels
            self.test_labels = test_labels


# CNN网络模型类
class Model:
    def __init__(self):
        self.model = None

        # 建立模型

    def build_model(self, dataset, nb_classes=2):
        self.model = Sequential()
        self.model.add(layers.Convolution2D(32, 3, 3, padding='same', activation='relu',
                                            input_shape=dataset.input_shape, name='conv-1'))
        # 1.filter：卷积核的个数
        # 2.kenel_size：卷积核尺寸，如果是正方形，则用一个整数表示；如果是长方形，则需要明确指明高用h表示，宽用w表示，可以用元组或者列表的形式表示两者(如：[h,w]或者{h, w})
        # 3.strides：滑动步长，默认横纵向滑动步长均为1(1,1)，也可以设置其他步长(纵向步长，横向步长)
        # 4.padding：补零策略，当padding = 'same’时，全零填充，当padding = ‘valid’，则不需要区分大小写
        # 5.data_format：输入数据的格式，值有两种channels_first(输入和输出的shape为(batch_size, channels，height, width)，即为(图片数量，通道数， 长，宽))、channels_last(默认值通道数为左最后一个)
        # 6.dalition_rate：数组或者列表，卷积核的膨胀系数(将卷积核进行形状膨胀，新的位置用0填充)
        # 新卷积核的尺寸核膨胀系数的计算公式如下：原卷积核的尺寸S，膨胀系数K，则膨胀后的卷积核尺寸为size = K*(S-1)+1
        # 7.activaton：激活函数，相当于经过卷积输出后，再经过一次激活函数(常见的激活函数有relu，softmax，selu)

        self.model.add(layers.Convolution2D(32, 3, 3, padding='same', activation='relu', name='conv-2'))

        self.model.add(layers.MaxPooling2D((2, 2), padding='same', name='max-polling-1'))
        self.model.add(Dropout(0.25))

        self.model.add(layers.Convolution2D(64, 3, 3, padding='same', activation='relu', name='conv-3'))

        self.model.add(layers.Convolution2D(32, 3, 3, padding='same', activation='relu', name='conv-4'))

        self.model.add(layers.MaxPooling2D((2, 2), padding='same', name='max-polling-2'))
        self.model.add(Dropout(0.25))

        self.model.add(Flatten())  # 13 Flatten层
        self.model.add(layers.Dense(512, activation='relu', name='fuulc-1'))  # 14 Dense层,又被称作全连接层
        self.model.add(Dropout(0.5))  # 16 Dropout层
        self.model.add(layers.Dense(nb_classes, activation='relu', name='fuulc-2'))  # 17 Dense层
        self.model.add(Activation('softmax'))  # 18 分类层，输出最终结果
        keras.utils.plot_model(self.model, r"C:\Users\Carl3\Desktop\face\fae_train.png", show_shapes=True)
        # 输出模型概况
        self.model.summary()

    # 训练模型
    def train(self, dataset, batch_size=20, nb_epoch=10, data_augmentation=True):

        sgd = SGD(learning_rate=0.01, decay=1e-6,
                  momentum=0.9, nesterov=True)  # 采用SGD+momentum的优化器进行训练，首先生成一个优化器对象
        print('11111', sgd)

        """
        lr：大或等于0的浮点数，学习率
        momentum：大或等于0的浮点数，动量参数
        decay：大或等于0的浮点数，每次更新后的学习率衰减值
        nesterov：布尔值，确定是否使用Nesterov动量
        """

        self.model.compile(loss='categorical_crossentropy',
                           optimizer=sgd,
                           metrics=['accuracy'])  # 完成实际的模型配置工作

        # 不使用数据提升，所谓的提升就是从我们提供的训练数据中利用旋转、翻转、加噪声等方法创造新的
        # 训练数据，有意识的提升训练数据规模，增加模型训练量
        if not data_augmentation:
            self.model.fit(dataset.train_images,
                           dataset.train_labels,
                           batch_size=batch_size,
                           nb_epoch=nb_epoch,
                           validation_data=(dataset.valid_images, dataset.valid_labels),
                           shuffle=True)
        # 使用实时数据提升
        else:
            # 定义数据生成器用于数据提升，其返回一个生成器对象datagen，datagen每被调用一
            # 次其生成一组数据（顺序生成），节省内存，其实就是python的数据生成器
            datagen = ImageDataGenerator(
                featurewise_center=False,  # 是否使输入数据去中心化（均值为0），
                samplewise_center=False,  # 是否使输入数据的每个样本均值为0
                featurewise_std_normalization=False,  # 是否数据标准化（输入数据除以数据集的标准差）
                samplewise_std_normalization=False,  # 是否将每个样本数据除以自身的标准差
                zca_whitening=False,  # 是否对输入数据施以ZCA白化
                rotation_range=20,  # 数据提升时图片随机转动的角度(范围为0～180)
                width_shift_range=0.2,  # 数据提升时图片水平偏移的幅度（单位为图片宽度的占比，0~1之间的浮点数）
                height_shift_range=0.2,  # 同上，只不过这里是垂直
                horizontal_flip=True,  # 是否进行随机水平翻转
                vertical_flip=False)  # 是否进行随机垂直翻转

            # 计算整个训练样本集的数量以用于特征值归一化、ZCA白化等处理
            datagen.fit(dataset.train_images)

            # 利用生成器开始训练模型
            tbCallBack = tf.compat.v1.keras.callbacks.TensorBoard(log_dir="log", histogram_freq=1, write_grads=True)
            self.model.fit(datagen.flow(dataset.train_images, dataset.train_labels,
                                        batch_size=batch_size),
                           steps_per_epoch=dataset.train_images.shape[0],
                           epochs=nb_epoch,
                           validation_data=(dataset.valid_images, dataset.valid_labels), callbacks=[tbCallBack])

    MODEL_PATH = 'model/SXL.face.model.h5'

    def save_model(self, file_path=MODEL_PATH):
        self.model.save(file_path)

    def load_model(self, file_path=MODEL_PATH):
        self.model = load_model(file_path)

    def evaluate(self, dataset):
        score = self.model.evaluate(dataset.test_images, dataset.test_labels, verbose=1)
        print("%s: %.2f%%" % (self.model.metrics_names[1], score[1] * 100))

    # 识别人脸
    def face_predict(self, image):
        # 依然是根据后端系统确定维度顺序
        if K.image_data_format() == 'channels_first' and image.shape != (1, 3, IMAGE_SIZE, IMAGE_SIZE):
            image = resize_image(image)  # 尺寸必须与训练集一致都应该是IMAGE_SIZE x IMAGE_SIZE
            image = image.reshape((1, 3, IMAGE_SIZE, IMAGE_SIZE))  # 与模型训练不同，这次只是针对1张图片进行预测
        elif K.image_data_format() == 'channels_last' and image.shape != (1, IMAGE_SIZE, IMAGE_SIZE, 3):
            image = resize_image(image)
            image = image.reshape((1, IMAGE_SIZE, IMAGE_SIZE, 3))

            # 浮点并归一化
        image = image.astype('float32')
        image /= 255

        # 给出输入属于各个类别的概率，我们是二值类别，则该函数会给出输入图像属于0和1的概率各为多少
        result = self.model.predict(image)
        print('result:', result)

        # 给出类别预测：0或者1
        result = np.argmax(result, axis=1)

        # 返回类别预测结果
        return result[0]


if __name__ == '__main__':
    dataset = Dataset('data')
    dataset.load()

    model = Model()
    model.build_model(dataset)

    # 先前添加的测试build_model()函数的代码
    model.build_model(dataset)

    # 测试训练函数的代码
    model.train(dataset)

if __name__ == '__main__':
    dataset = Dataset('data')
    dataset.load()

    model = Model()
    model.build_model(dataset)
    model.train(dataset)
    model.save_model(file_path='model/SXL.face.model.h5')

if __name__ == '__main__':
    dataset = Dataset('data')
    dataset.load()

    # 评估模型
    model = Model()
    model.load_model(file_path='model/SXL.face.model.h5')
    model.evaluate(dataset)
