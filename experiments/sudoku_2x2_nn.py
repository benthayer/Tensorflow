from experiments.sudoku_2x2_generator import convert_to_normal, get_training_and_test_sets
import tensorflow as tf
import numpy as np


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

data = tf.placeholder(tf.float32, shape=[None, 4**3])

W1 = weight_variable([4**3, 4**3])
b1 = bias_variable([4**3])

h1 = tf.nn.softmax(tf.matmul(data, W1) + b1)

y = h1

y_ = tf.placeholder(tf.float32, [None, 4**3])

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)


training_puzzles, training_solutions, test_puzzles, test_solutions = get_training_and_test_sets()

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

k = 1000
for i in range(10000):
    sess.run(train_step, feed_dict={data: training_puzzles, y_: training_solutions})
    if i % 100 == 0:
        print("Batch {} complete".format(i))

correct_prediction = tf.equal(
    tf.argmax(
        tf.reshape(y, (-1, 4, 4, 4)),
        2),
    tf.argmax(
        tf.reshape(y_, (-1, 4, 4, 4)),
        2))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
accuracy = sess.run(accuracy, feed_dict={data: test_puzzles, y_: test_solutions})
W = sess.run(W1)
b = sess.run(b1)
incorrect = 0
for i in range(len(test_puzzles)):
    guessed_board = sess.run(y, feed_dict={data: [test_puzzles[i]]})

    if not np.array_equal(
            convert_to_normal(guessed_board.reshape((4, 4, 4))),
            convert_to_normal(test_solutions[i].reshape(4, 4, 4))):
        incorrect += 1
        if incorrect > 4:
            break
        print()
        print("Board:\n", convert_to_normal(test_puzzles[i].reshape((4, 4, 4)), ones=True))
        print("Guess:\n", convert_to_normal(guessed_board.reshape((4, 4, 4))))
        print("Answer:\n", convert_to_normal(test_solutions[i].reshape((4, 4, 4))))

print("Accuracy = ", accuracy)
