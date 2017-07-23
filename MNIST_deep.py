import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('/tmp/data/', one_hot=True)

nNodesHL1 = 600
nNodesHL2 = 900
nNodesHL3 = 300

nClasses = 10
batchSize = 100

X = tf.placeholder('float', [None, 784])
Y = tf.placeholder('float')

def neuralNetworkModel(data):

	#Hidden layer 1
	HL1 = {'weights': tf.Variable(tf.random_normal([784, nNodesHL1])),
			'biases': tf.Variable(tf.random_normal([nNodesHL1]))}

	#Hidden layer 2
	HL2 = {'weights': tf.Variable(tf.random_normal([nNodesHL1, nNodesHL2])),
			'biases': tf.Variable(tf.random_normal([nNodesHL2]))}

	#Hidden layer 3
	HL3 = {'weights': tf.Variable(tf.random_normal([nNodesHL2, nNodesHL3])),
			'biases': tf.Variable(tf.random_normal([nNodesHL3]))}

	#Output layer
	OL = {'weights': tf.Variable(tf.random_normal([nNodesHL3, nClasses])),
			'biases': tf.Variable(tf.random_normal([nClasses]))}

	#Layer activation
	L1 = tf.add(tf.matmul(data, HL1['weights']), HL1['biases'])
	L1 = tf.nn.relu(L1)
	
	L2 = tf.add(tf.matmul(L1, HL2['weights']), HL2['biases'])
	L2 = tf.nn.relu(L2)

	L3 = tf.add(tf.matmul(L2, HL3['weights']), HL3['biases'])
	L3 = tf.nn.relu(L3)

	output = tf.add(tf.matmul(L3, OL['weights']), OL['biases']) 

	return output

def trainModel(data):

	prediction = neuralNetworkModel(data)
	cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=Y))
	optimizer = tf.train.AdamOptimizer().minimize(cost)

	nEpochs = 40

	with tf.Session() as sess:

		sess.run(tf.global_variables_initializer())

		for epoch in range(nEpochs):
			epochLoss = 0

			print('Epoch', epoch + 1, 'running...')
			for  i in range(int(mnist.train.num_examples/batchSize)):

				x, y = mnist.train.next_batch(batchSize)
				i, c = sess.run([optimizer, cost], feed_dict={X: x, Y: y})

				epochLoss += c

			print('Epoch', epoch + 1, 'completed out of', nEpochs, '| Loss:', epochLoss)

		correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
		accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
		print('Accuracy:', round(accuracy.eval({X: mnist.test.images, Y: mnist.test.labels})*100, 2), '%')

trainModel(X)

