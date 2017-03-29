import tensorflow as tf

x_data = [[0.3, 1.66], [0.34,1.83],[2.03, 0.23],[5.50, 2.70],[1.99, 0.30],[0.36, 1.77],[1.50, 0.19],[0.33, 1.54],[1.90, 0.40],[6.60, 3.04],[5.94, 2.98],[0.40, 1.92],[4.97, 3.14],[4.55, 2.63],[1.66, 0.30]]
y_data = [[1, 0, 0],[1, 0, 0],[0, 1, 0],[0, 0, 1],[0, 1, 0],[1, 0, 0],[0, 1, 0],[1, 0, 0],[0, 1, 0],[0, 0, 1],[0, 0, 1],[1, 0, 0],[0, 0, 1],[0, 0, 1],[0, 1, 0]]
#x_data = [[0.3, 1.66], [0.34,1.83]]
#y_data= [[1,0,0],[0,0,1]]

X = tf.placeholder("float",[None, 2])
Y = tf.placeholder("float",[None, 3])
nb_class = 3

W = tf.Variable(tf.random_normal([2, nb_class]), name='weight')
b = tf.Variable(tf.random_normal([nb_class], name='bias'))

'''hypothesis = tf.nn.softmax(tf.matmul(X,W) + b)

cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(hypothesis), axis=1))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)
'''

# tf.nn.softmax computes softmax activations
# softmax = exp(logits) / reduce_sum(exp(logits), dim)
hypothesis = tf.nn.softmax(tf.matmul(X, W) + b)

# Cross entropy cost/loss
cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(hypothesis), axis=1))
#cost = tf.log(tf.Variable([1,1]))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(2001):
        sess.run(optimizer, feed_dict={X: x_data, Y:y_data})
        if step % 200 == 0:
            print(step, sess.run(cost,  feed_dict={X: x_data, Y: y_data}))
