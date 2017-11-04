import tensorflow as tf
import CreateDataset
import os
from os import listdir
from os.path import isfile, join
import sys

def weight_variable(shape, n):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial, name=n)
def bias_variable(shape, n):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name=n)
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
def max_pool_2x2(x, kSize):
    return tf.nn.max_pool(x, ksize=[1, kSize, kSize, 1], strides=[1, kSize, kSize, 1], padding='SAME')

class ConvolutionalNeuralNetwork:
    def __init__(self, path = None, file=None, imageWidth = 200, imageHeight = 200, filterWidth = 5,
                 filterHeight = 5, channels = 3, outputs = 26, numEpochs = 300, batchSize = 50,
                 learningRate = 0.0001, filter1Size = 32, filter2Size = 64, lastLayerSize = 1024,
                 kernelSize = 2):
        self.sess = tf.InteractiveSession()
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

        with tf.name_scope('input'):
            self.x = tf.placeholder(tf.float32, shape=[None, self.inputs], name='x')
            self.y = tf.placeholder(tf.float32, shape=[None, self.outputs], name='y')

        self.keepProb = tf.placeholder(tf.float32, name='keepProb')

        with tf.name_scope('Layer1'):
            self.toLayer1 = {'weights':weight_variable([filterHeight, filterWidth, channels, filter1Size], 'W1'),
                            'biases':bias_variable([filter1Size], 'B1')}

        with tf.name_scope('Layer2'):
            self.toLayer2 = {'weights':weight_variable([filterHeight, filterWidth, filter1Size, filter2Size], 'W2'),
                            'biases':bias_variable([filter2Size], 'B2')}

        with tf.name_scope('Layer3'):
            self.toLayer3 = {'weights':weight_variable([int(imageHeight/(kernelSize ** 2)) *
                                                        int(imageWidth/(kernelSize ** 2)) *
                                                        filter2Size, lastLayerSize], 'W3'),
                            'biases':bias_variable([lastLayerSize], 'B3')}

        with tf.name_scope('Layer4'):
            self.toLayer4 = {'weights':weight_variable([lastLayerSize, outputs], 'W4'),
                            'biases':bias_variable([outputs], 'B4')}
        
        
        self.prediction = self.feedForward(self.x)

        with tf.name_scope('Cross_entropy'):
            self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.y, logits=self.prediction))

        with tf.name_scope('Train'):
            self.optimizer = tf.train.AdamOptimizer(self.learningRate).minimize(self.cost)

        with tf.name_scope('Accuracy'):
            self.correct_prediction = tf.equal(tf.argmax(self.prediction, 1), tf.argmax(self.y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))

        # create a summary for our cost and accuracy
        tf.summary.scalar("cost", self.cost)
        tf.summary.scalar("accuracy", self.accuracy)

        # merge all summaries into a single "operation" which we can execute in a session
        self.merged = tf.summary.merge_all()

        self.sess.run(tf.global_variables_initializer())
        
        if(path is not None and file is not None):
            saver = tf.train.Saver()
            saver.restore(self.sess, tf.train.latest_checkpoint(path))

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
        ASL = CreateDataset.ASLDataset()
        saver = tf.train.Saver(max_to_keep=1)
        
        dataset = ASL.dataset
        dataset = dataset.shuffle(buffer_size=10000)
        dataset = dataset.batch(self.batchSize)
        dataset = dataset.repeat(self.numEpochs)
        iterator = dataset.make_initializable_iterator()
            
        writer = tf.summary.FileWriter(join(join(os.getcwd(), "TrainingGraphs"), "ASL_NN"), graph=tf.get_default_graph())

        for i in range(self.numEpochs):
            self.sess.run(iterator.initializer)
            epochLoss = 0
            for batchNum in range(int(ASL.size() / self.batchSize)):
                batch = self.sess.run(iterator.get_next())
                summary, _, c = self.sess.run([self.merged, self.optimizer, self.cost], feed_dict={self.x: batch[0], self.y: batch[1], self.keepProb: 0.5})
                epochLoss += c
                writer.add_summary(summary, i * int(ASL.size() / self.batchSize) + batchNum)
                print('Batch ', batchNum + 1, ' / ', int(ASL.size() / self.batchSize), ' completed with loss ', c)
            print('Epoch ', i + 1, '/ ', self.numEpochs, ' completed - Loss=',epochLoss,'\n')
            #summary, a, c = self.sess.run([self.merged, self.accuracy, self.cost], feed_dict={self.x: batch[0], self.y: batch[1], self.keepProb: 0.5})
            #writer.add_summary(summary, i)
            saver.save(self.sess, join(join(os.getcwd(), "PremadeNetworks"), "ASL_NN"), global_step=ASL.size()*(i+1))
    
        
        testImages, testLabels = self.sess.run(iterator.get_next())
        print('Test accuracy %g' % accuracy.eval(feed_dict={self.x:testImages, self.y:testLabels, self.keepProb:1.0}))


if __name__ == "__main__":
    #nn = ConvolutionalNeuralNetwork(join(os.getcwd(), "PremadeNetworks"), "ASL_NN-18400.meta")
    nn = ConvolutionalNeuralNetwork()
    nn.train()
