import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.contrib import rnn 
mnist = input_data.read_data_sets("/tmp/data/", one_hot = True)

nEpochs = 10
nClasses = 10
batchSize = 128
chunkSize = 28
nChunks = 28
RNNSize = 512


X = tf.placeholder('float', [None, nChunks,chunkSize])
Y = tf.placeholder('float')

def recurrent_neural_network(X):
    layer = {'weights':tf.Variable(tf.random_normal([RNNSize,nClasses])),
             'biases':tf.Variable(tf.random_normal([nClasses]))}

    X = tf.transpose(X, [1,0,2])
    X = tf.reshape(X, [-1, chunkSize])
    X = tf.split(X, nChunks, 0)

    LSTMCell = rnn.BasicLSTMCell(RNNSize)
    outputs, states = rnn.static_rnn(LSTMCell, X, dtype=tf.float32)

    output = tf.matmul(outputs[-1],layer['weights']) + layer['biases']

    return output

def train_neural_network(X):
    prediction = recurrent_neural_network(X)
    cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=Y))
    optimizer = tf.train.AdamOptimizer().minimize(cost)
    
    
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())

        for epoch in range(nEpochs):
            epochLoss = 0
            
            print('Epoch', epoch + 1, 'running...')
            for i in range(int(mnist.train.num_examples/batchSize)):
                x, y = mnist.train.next_batch(batchSize)
                x = x.reshape((batchSize,nChunks,chunkSize))

                i, c = sess.run([optimizer, cost], feed_dict={X: x, Y: y})
                epochLoss += c

            print('Epoch', epoch + 1, 'completed out of', nEpochs, '| Loss:', epochLoss)

        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))

        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('Accuracy:',accuracy.eval({X:mnist.test.images.reshape((-1, nChunks, chunkSize)), Y:mnist.test.labels}))

train_neural_network(X)