import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
'''

1. Input -> weights -> hidden layer 1 activation function -> 
    weights -> hidden layer 2 activation function -> weights -> output

2. compare output to intended output

3. Optimize through minimizing cost

epoch := a full feedforward + backpropogation

'''


mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

# Number of hidden nodes in each layer
input_size = 784;
n_nodes_hl1 = 200;
n_nodes_hl2 = 100;
n_nodes_hl3 = 20;

# Number of outputs (classes)
n_classes = 10;

# Number of pieces of data to use at a time
batch_size = 100;

# x is out input, y is our output
x = tf.placeholder('float',[None,input_size])
y = tf.placeholder('float')

def neural_network_model(data):
	# Random numbers as weights 
	hidden_1_layer = {'weights':tf.Variable(tf.random_normal([input_size, n_nodes_hl1])), 'biases':tf.Variable(tf.random_normal([n_nodes_hl1]))}
	hidden_2_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])), 'biases':tf.Variable(tf.random_normal([n_nodes_hl2]))}
	hidden_3_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])), 'biases':tf.Variable(tf.random_normal([n_nodes_hl3]))}
	output_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl3, n_classes])), 'biases':tf.Variable(tf.random_normal([n_classes]))}

	# Multiply matrices, add, then do activation function	
	l1 = tf.add(tf.matmul(data, hidden_1_layer['weights']), hidden_1_layer['biases'])
	l1 = tf.nn.relu(l1)

	l2 = tf.add(tf.matmul(l1, hidden_2_layer['weights']), hidden_2_layer['biases'])
	l2 = tf.nn.relu(l2)

	l3 = tf.add(tf.matmul(l2, hidden_3_layer['weights']), hidden_3_layer['biases'])
	l3 = tf.nn.relu(l3)
    
	output = tf.add(tf.matmul(l3, output_layer['weights']), output_layer['biases'])

	return output

def train_neural_network(x):
	prediction = neural_network_model(x)
	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = prediction, labels = y))

	# Same as stochastic gradient descent
	learning_rate = 0.001
	optimizer = tf.train.AdamOptimizer().minimize(cost)

	# Number of epochs
	n_epochs = 200;

	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		# Train the network!!!
		for epoch in range(n_epochs):
			epoch_loss = 0
			for _ in range(int(mnist.train.num_examples / batch_size)):
				# This function is what I'll have to build for my own datasets
				epoch_x, epoch_y = mnist.train.next_batch(batch_size)
				_, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})
				epoch_loss += c
			print('Epoch ', epoch + 1, '/ ', n_epochs, ' completed - Loss=',epoch_loss,'\n')


		correct = tf.equal(tf.argmax(prediction,1), tf.argmax(y, 1))
		accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
		print('Accuracy: ', accuracy.eval({x:mnist.test.images, y:mnist.test.labels}))





train_neural_network(x)





