
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.keras.datasets import mnist
# 定义神经网络的神经元数目
INPUT_NODE = 784
LAYER1_NODE = 500
OUTPUT_NODE = 10

# 每次训练数据的个数
BATCH_SIZE = 100

# 衰减学习率的参数
LEARNING_RATE_BASE = 0.8
LEARNING_RATE_DECAY = 0.99

# 正则化项的系数
REGULARIATION_RATE = 0.0001

# 滑动平均的参数
TRAINING_STEPS = 1000
MOVING_AVERAGE_DECAY = 0.99

# 定义神经网络和前向传播算法
def inference(input_tensor, avg_class, weights1, biases1, weights2, biases2):
    if avg_class == None:
        with tf.name_scope('layer1'):
            layer1 = tf.nn.relu(tf.matmul(input_tensor, weights1) + biases1)
        with tf.name_scope('layer2'):
            output = tf.matmul(layer1, weights2) + biases2
    else:
        with tf.name_scope('layer1'):
            layer1 = tf.nn.relu(tf.matmul(input_tensor, avg_class.average(weights1)) + avg_class.average(biases1))
        with tf.name_scope('layer2'):
            output = tf.matmul(layer1, avg_class.average(weights2)) + avg_class.average(biases2)

    tf.summary.histogram('weights1', weights1)
    tf.summary.histogram('biases1', biases1)
    tf.summary.histogram('weights2', weights2)
    tf.summary.histogram('biases2', biases2)

    return output

def train(mnist):
    with tf.name_scope('input'):
        x = tf.placeholder(tf.float32, [None, INPUT_NODE], name='x-input')
        x_image = tf.reshape(x, [-1, 28, 28, 1])
        tf.summary.image('input', x_image, 10)
        y_ = tf.placeholder(tf.float32, [None, OUTPUT_NODE], name='y-input')

    # 定义神经网络的参数
    weights1 = tf.Variable(tf.truncated_normal([INPUT_NODE, LAYER1_NODE], stddev=0.1))
    biases1 = tf.Variable(tf.constant(0.1, shape=[LAYER1_NODE]))
    weights2 = tf.Variable(tf.truncated_normal([LAYER1_NODE, OUTPUT_NODE], stddev=0.1))
    biases2 = tf.Variable(tf.constant(0.1, shape=[OUTPUT_NODE]))

    # 计算前向传播结果
    y = inference(x, None, weights1, biases1, weights2, biases2)

    # 使用带有滑动平均的模型计算前行传播结果
    with tf.name_scope('moving_average'):
        global_step = tf.Variable(0, trainable=False)
        variable_average = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
        variables_averages_op = variable_average.apply(tf.trainable_variables())

        average_y = inference(x, variable_average, weights1, biases1, weights2, biases2)

    # 计算交叉熵和损失函数
    with tf.name_scope('loss_function'):
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))
        cross_entropy_mean = tf.reduce_mean(cross_entropy)
        regularizer = tf.contrib.layers.l2_regularizer(REGULARIATION_RATE)
        regularization = regularizer(weights1) + regularizer(weights2)
        loss = cross_entropy_mean + regularization

        tf.summary.scalar('max', tf.reduce_max(loss))

    # 使用衰减学习率
    with tf.name_scope('train_step'):
        learning_rate = tf.train.exponential_decay(
            LEARNING_RATE_BASE,
            global_step,
            mnist.train.images.shape[0] / BATCH_SIZE,
            LEARNING_RATE_DECAY
        )

        # 定义使用的优化方法
        train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)

        # 定义同时更新滑动平均值和参数的方法
        with tf.control_dependencies([train_step, variables_averages_op]):
            train_op = tf.no_op('train')

    # 定义精度的计算
    correct_prediction = tf.equal(tf.argmax(average_y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    tf.summary.histogram('accuracy', accuracy)
    summ = tf.summary.merge_all()


    # 初始化会话并开始训练
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        validate_feed = {x: mnist.validation.images, y_: mnist.validation.labels}
        test_feed = {x: mnist.test.images, y_: mnist.test.labels}

        writer = tf.summary.FileWriter('log/')
        writer.add_graph(sess.graph)

        for i in range(TRAINING_STEPS):
            # 每1000次就在验证集上测试训练的模型精度
            if i % 100 == 0:
                # 配置运行时要记录的信息
                run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                # 运行时记录运行信息的proto
                run_metadata = tf.RunMetadata()
                # 将配置信息和运行记录信息的proto传入运行过程，从而进行记录
                validate_acc, sum = sess.run([accuracy, summ], feed_dict=validate_feed, options=run_options, run_metadata=run_metadata)
                # 将节点的运行信息写入日志文件
                writer.add_run_metadata(run_metadata, 'step%03d' % i)
                writer.add_summary(sum, i)

                print('After %d training step(s), validation accuracy using average model is %g' % (i, validate_acc))

            # 用于生成下一次迭代的训练数据
            xs, ys = mnist.train.next_batch(BATCH_SIZE)
            sess.run(train_op, feed_dict={x: xs, y_: ys})

        # 验证在测试集上的准确度
        test_acc = sess.run(accuracy, feed_dict=test_feed)
        print('After %d training step(S), test accuracy using average model is %g' % (TRAINING_STEPS, test_acc))

def main(argv=None):
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    train(mnist)

if __name__ == '__main__':
    tf.app.run()