import tensorflow as tf
import CreateDataset
import os
from os import listdir
from os.path import isfile, join
import sys

numEpochs = 4
imageHeight = 28
imageWidth = 28
filterHeight = 5
filterWidth = 5
channels = 3
numInputs = imageWidth * imageHeight * channels
numOutputs = 26
batchSize = 50
learning_rate = 0.001

x = tf.placeholder(tf.float32, shape=[None, numInputs])
y = tf.placeholder(tf.float32, shape=[None, numOutputs])
keep_prob = tf.placeholder(tf.float32)


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def nn_ff(x):
    x_image = tf.reshape(x, [-1, imageHeight, imageWidth, channels])
    toLayer1 = {'weights':weight_variable([filterHeight, filterWidth, channels, 32]),
        'biases':bias_variable([32])}
    toLayer2 = {'weights':weight_variable([filterHeight, filterWidth, 32, 64]),
        'biases':bias_variable([64])}
    toLayer3 = {'weights':weight_variable([int(imageHeight/4) * int(imageWidth/4) * 64, 1024]),
        'biases':bias_variable([1024])}
    toLayer4 = {'weights':weight_variable([1024, numOutputs]),
        'biases':bias_variable([numOutputs])}

    convolution1 = conv2d(x_image, toLayer1['weights'])
    convolution1 = tf.add(convolution1, toLayer1['biases'])
    convolution1 = tf.nn.relu(convolution1)

    pool1 = max_pool_2x2(convolution1)

    convolution2 = conv2d(pool1, toLayer2['weights'])
    convolution2 = tf.add(convolution2, toLayer2['biases'])
    convolution2 = tf.nn.relu(convolution2)

    pool2 = max_pool_2x2(convolution2)
    pool2 = tf.reshape(pool2, [-1, int(imageHeight/4) * int(imageWidth/4) * 64])

    fullyConnectedLayer = tf.matmul(pool2, toLayer3['weights'])
    fullyConnectedLayer = tf.add(fullyConnectedLayer, toLayer3['biases'])
    fullyConnectedLayer = tf.nn.relu(fullyConnectedLayer)

    fullyConnectedLayerReduced = tf.nn.dropout(fullyConnectedLayer, keep_prob)

    result = tf.matmul(fullyConnectedLayerReduced, toLayer4['weights'])
    result = tf.add(result, toLayer4['biases'])

    return result

def train_nn(x):
    prediction = nn_ff(x)
    
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=prediction))
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

    ASL = CreateDataset.ASLDataset()
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        dataset = ASL.dataset
        dataset = dataset.shuffle(buffer_size=10000)
        dataset = dataset.batch(batchSize)
        dataset = dataset.repeat(numEpochs)
        iterator = dataset.make_initializable_iterator()
        
        for i in range(numEpochs):
            sess.run(iterator.initializer)
            epochLoss = 0
            for _ in range(int(ASL.size() / batchSize)):
                batch = sess.run(iterator.get_next())
                _, c = sess.run([optimizer, cost], feed_dict={x: batch[0], y: batch[1], keep_prob: 0.5})
                epochLoss += c
            print('Epoch ', i + 1, '/ ', numEpochs, ' completed - Loss=',epochLoss,'\n')

        correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.train.Saver().save(sess, join(join(os.getcwd(), "PremadeNetworks"), "ASL_NN"))


if __name__ == "__main__":
    
    train_nn(x)
