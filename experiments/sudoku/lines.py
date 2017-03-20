import tensorflow as tf
import numpy as np

n = 100000

data = np.zeros((n, 9**2))
answers = np.zeros((n, 9))
# make test data
for i in range(n):
    line = np.arange(9) + 1
    np.random.shuffle(line)

    j = np.random.random_integers(0, 8)
    missing = line[j]
    line[j] = 0

    def num2class(num):
        classes = np.zeros(9)
        if num == 0:
            return classes
        classes[num - 1] = 1
        return classes

    line_classes = np.zeros((9, 9))
    for index, num in enumerate(line):
        line_classes[index] = num2class(num)

    answer = num2class(missing)

    data[i] = line_classes.reshape(81)
    answers[i] = answer


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

x = tf.placeholder(tf.float32, shape=[None, 81])

W1 = weight_variable([81, 9])
b1 = bias_variable([9])

y = tf.nn.softmax(tf.matmul(x, W1) + b1)
#
# W2 = weight_variable([81, 9])
# b2 = bias_variable([9])
#
# h1 = tf.nn.tanh(tf.matmul(h1, W2) + b2)
#
# W3 = weight_variable([9, 9])
# b3 = bias_variable([9])
#
# y = tf.nn.softmax(tf.matmul(h2, W3) + b3)

y_ = tf.placeholder(tf.float32, [None, 9])

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

train_step = tf.train.GradientDescentOptimizer(1).minimize(cross_entropy)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

for i in range(100):
    batch_xs, batch_ys = data[i*1000:(i+1)*1000], answers[i*1000:(i+1)*1000]
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

sess.run(train_step, feed_dict={x: data, y_: answers})

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(sess.run(accuracy, feed_dict={x: data, y_: answers}))
for i in range(5):
    print(np.matmul(data[i].reshape((9, 9)), np.arange(9) + 1))
print(sess.run(y, feed_dict={x: data[:5]}))
print("Layer 1", sess.run(W1), sess.run(b1))
# print("Layer 2", sess.run(W2), sess.run(b2))
