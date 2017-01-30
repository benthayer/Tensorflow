from experiments.sudoku_2x2_generator import generate, convert_to_classes, convert_to_normal, remove_nums
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


n = 10000
boards = np.zeros((n, 4 ** 3))
answer_boards = np.zeros((n, 4 ** 3))
for i in range(n):
    # each iteration takes about 2.82 for generation/removal, 3.57 with saving
    board = generate()
    answer_boards[i] = convert_to_classes(board).reshape((4 ** 3))
    remove_nums(board)
    class_board = convert_to_classes(board)
    boards[i] = class_board.reshape((4 ** 3))


init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

for i in range(1000):
    batch_xs, batch_ys = boards[i*10:(i+1)*10], answer_boards[i*10:(i+1)*10]
    sess.run(train_step, feed_dict={data: batch_xs, y_: batch_ys})
    if i % 100 == 0:
        print("Batch {} complete".format(i))

sess.run(train_step, feed_dict={data: boards, y_: answer_boards})

correct_prediction = tf.equal(
    tf.argmax(
        tf.reshape(y, (-1, 4, 4, 4)),
        2),
    tf.argmax(
        tf.reshape(y_, (-1, 4, 4, 4)),
        2))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print("Accuracy: ", sess.run(accuracy, feed_dict={data: boards, y_: answer_boards}))
W = sess.run(W1)
b = sess.run(b1)
for i in range(5):
    guessed_board = np.matmul(boards[i], W) + b
    print(convert_to_normal(answer_boards[i].reshape((4, 4, 4))))
    print(convert_to_normal(guessed_board.reshape((4, 4, 4))))
