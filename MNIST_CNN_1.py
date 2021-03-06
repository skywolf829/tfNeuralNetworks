import tensorflow as tf

numEpochs = 10

numInputs = 784
numOutputs = 10

batchSize = 2
learning_rate = 0.001

x = tf.placeholder(tf.float32, shape=[None, numInputs])
y = tf.placeholder(tf.float32, shape=[None, numOutputs])
keep_prob = tf.placeholder(tf.float32)

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

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
    
    x_image = tf.reshape(x, [-1, 28, 28, 1])
    toLayer1 = {'weights':weight_variable([5, 5, 1, 32]),
        'biases':bias_variable([32])}
    toLayer2 = {'weights':weight_variable([5, 5, 32, 64]),
        'biases':bias_variable([64])}
    toLayer3 = {'weights':weight_variable([7 * 7 * 64, 1024]),
        'biases':bias_variable([1024])}
    toLayer4 = {'weights':weight_variable([1024, 10]),
        'biases':bias_variable([10])}

    convolution1 = conv2d(x_image, toLayer1['weights'])
    convolution1 = tf.add(convolution1, toLayer1['biases'])
    convolution1 = tf.nn.relu(convolution1)

    pool1 = max_pool_2x2(convolution1)

    convolution2 = conv2d(pool1, toLayer2['weights'])
    convolution2 = tf.add(convolution2, toLayer2['biases'])
    convolution2 = tf.nn.relu(convolution2)

    pool2 = max_pool_2x2(convolution2)
    pool2 = tf.reshape(pool2, [-1, 7*7*64])

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
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(numEpochs):
            epochLoss = 0
            for _ in range(int(mnist.train.num_examples / batchSize)):
                batch = mnist.train.next_batch(batchSize)
                _, c = sess.run([optimizer, cost], feed_dict={x: batch[0], y: batch[1], keep_prob: 0.5})
                epochLoss += c
            print('Epoch ', i + 1, '/ ', numEpochs, ' completed - Loss=',epochLoss,'\n')

        correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        print('test accuracy %g' % accuracy.eval(feed_dict={x: mnist.test.images, y: mnist.test.labels, keep_prob: 1.0}))

if __name__ == "__main__":
    train_nn(x)
