import  os

import keras.losses
import tensorflow as tf
from tensorflow.keras import  datasets,layers,models



gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

class CNN():
    def __init__(self):
        model=models.Sequential()
        model.add(layers.Conv2D(32,(3,3),activation='relu',input_shape=(28,28,1),name='Conv1'))
        model.add(layers.MaxPooling2D((2,2),name='max-pooling1'))
        model.add(layers.Conv2D(64,(3,3),activation='relu',name='Conv2'))
        model.add(layers.MaxPooling2D((2,2),name='max-pooling2'))
        model.add(layers.Conv2D(64, (3, 3), activation='relu', name='Conv3'))
        model.add(layers.Flatten(name='Flatten1'))##一维压缩层
        model.add(layers.Dense(64,activation='relu',name='Dense'))
        model.add(layers.Dense(10,activation='softmax',name='softmax'))
        model.summary()#模型各层的参数状况
        self.model=model

# cnn=CNN()
class Data():
    def __init__(self):
        data_path = os.path.abspath(os.path.dirname(
            __file__)) + '\\data_set_tf2\\mnist.npz'
        (train_images,train_labels),(test_images,test_labels)=datasets.mnist.load_data(path=data_path)
        train_images=train_images.reshape((60000,28,28,1))
        test_images=test_images.reshape((10000,28,28,1))
        train_images,test_images=train_images/255.0,test_images/255.0
        self.train_images,self.train_labels=train_images,train_labels
        self.test_images,self.test_labels=test_images,test_labels

class Train():
    def __init__(self):
        self.cnn=CNN()
        self.data=Data()

    def train(self):
        check_path='./ckpt/cp-{epoch:04d}.ckpt'
        save_model_cb=tf.keras.callbacks.ModelCheckpoint(check_path,
                                                         save_weights_only=True,
                                                         verbose=1,
                                                         period=5)
        """
       
        """
        self.cnn.model.compile(optimizer='adam',
                               loss=keras.losses.sparse_categorical_crossentropy,
                               metrics=['accuracy'])
        tensorboard_callback=tf.keras.callbacks.TensorBoard(log_dir='./log',
                                                            histogram_freq=1)
        self.cnn.model.fit(self.data.train_images,self.data.train_labels,
                           epochs=20,
                           validation_data=(self.data.train_images,self.data.train_labels),
                           callbacks=[tensorboard_callback,save_model_cb])
        test_loss,test_acc=self.cnn.model.evaluate(
            self.data.test_images,self.data.test_labels
        )
        print("准确率：%.4f,共测试了%d张图片"%(test_acc,len(self.data.test_images)))
        tf.constant()

if __name__=='__main__':
    mnist=Train()
    mnist.train()