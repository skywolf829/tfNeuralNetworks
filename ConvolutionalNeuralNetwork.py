import tensorflow as tf
import CreateDataset
import os
from os import listdir
from os.path import isfile, join
import sys

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
def max_pool_2x2(x, kSize):
    return tf.nn.max_pool(x, ksize=[1, kSize, kSize, 1], strides=[1, kSize, kSize, 1], padding='SAME')

class ConvolutionalNeuralNetwork:
    def __init__(self, imageWidth = 28, imageHeight = 28, filterWidth = 5,
                 filterHeight = 5, channels = 3, outputs = 26, numEpochs = 5, batchSize = 50,
                 learningRate = 0.001, filter1Size = 32, filter2Size = 64, lastLayerSize = 1024,
                 kernelSize = 2):
        self.numEpochs = numEpochs
        self.batchSize = batchSize
        self.learningRate = learningRate
        self.imageWidth = imageWidth
        self.imageHeight = imageHeight
        self.channels = channels
        self.inputs = imageWidth * imageHeight * channels
        self.outputs = outputs
        self.filterWidth = filterWidth
        self.filterHeight = filterHeight
        self.filter1Size = filter1Size
        self.filter2Size = filter2Size
        self.lastLayerSize = lastLayerSize
        self.kernelSize = kernelSize

        self.x = tf.placeholder(tf.float32, shape=[None, self.inputs])
        self.y = tf.placeholder(tf.float32, shape=[None, self.outputs])
        self.keepProb = tf.placeholder(tf.float32)
    
        self.toLayer1 = {'weights':weight_variable([filterHeight, filterWidth, channels, filter1Size]),
            'biases':bias_variable([filter1Size])}
        self.toLayer2 = {'weights':weight_variable([filterHeight, filterWidth, filter1Size, filter2Size]),
            'biases':bias_variable([filter2Size])}
        self.toLayer3 = {'weights':weight_variable([int(imageHeight/(kernelSize ** 2)) *
                                                    int(imageWidth/(kernelSize ** 2)) *
                                                    filter2Size, 1024]),
                         'biases':bias_variable([lastLayerSize])}
        self.toLayer4 = {'weights':weight_variable([lastLayerSize, outputs]),
            'biases':bias_variable([outputs])}

    def feedForward(self,x):
        x_image = tf.reshape(x, [-1, self.imageHeight, self.imageWidth, self.channels])

        convolution1 = conv2d(x_image, self.toLayer1['weights'])
        convolution1 = tf.add(convolution1, self.toLayer1['biases'])
        convolution1 = tf.nn.relu(convolution1)

        pool1 = max_pool_2x2(convolution1, self.kernelSize)
        
        convolution2 = conv2d(pool1, self.toLayer2['weights'])
        convolution2 = tf.add(convolution2, self.toLayer2['biases'])
        convolution2 = tf.nn.relu(convolution2)
        
        pool2 = max_pool_2x2(convolution2, self.kernelSize)
        pool2 = tf.reshape(pool2, [-1, int(self.imageHeight/(self.kernelSize ** 2)) *
                                   int(self.imageWidth/(self.kernelSize ** 2)) *
                                   self.filter2Size])
        
        fullyConnectedLayer = tf.matmul(pool2, self.toLayer3['weights'])
        fullyConnectedLayer = tf.add(fullyConnectedLayer, self.toLayer3['biases'])
        fullyConnectedLayer = tf.nn.relu(fullyConnectedLayer)
        
        fullyConnectedLayerReduced = tf.nn.dropout(fullyConnectedLayer, self.keepProb)
        
        result = tf.matmul(fullyConnectedLayerReduced, self.toLayer4['weights'])
        result = tf.add(result, self.toLayer4['biases'])
        
        return result

    def train(self):
        prediction = self.feedForward(self.x)
        
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.y, logits=prediction))
        optimizer = tf.train.AdamOptimizer(self.learningRate).minimize(cost)
        
        ASL = CreateDataset.ASLDataset()
        
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            dataset = ASL.dataset
            dataset = dataset.shuffle(buffer_size=10000)
            dataset = dataset.batch(self.batchSize)
            dataset = dataset.repeat(self.numEpochs)
            iterator = dataset.make_initializable_iterator()
            
            for i in range(self.numEpochs):
                sess.run(iterator.initializer)
                epochLoss = 0
                for _ in range(int(ASL.size() / self.batchSize)):
                    batch = sess.run(iterator.get_next())
                    _, c = sess.run([optimizer, cost], feed_dict={self.x: batch[0], self.y: batch[1], self.keepProb: 0.5})
                    epochLoss += c
                print('Epoch ', i + 1, '/ ', self.numEpochs, ' completed - Loss=',epochLoss,'\n')
            
            correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(self.y, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            tf.train.Saver().save(sess, join(join(os.getcwd(), "PremadeNetworks"), "ASL_NN"))

if __name__ == "__main__":
    nn = ConvolutionalNeuralNetwork()
    nn.train()
