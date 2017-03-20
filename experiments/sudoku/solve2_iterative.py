import numpy as np
import tensorflow as tf

from experiments.sudoku.gen2 import convert_to_normal, get_training_and_test_sets


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
for i in range(1000):
    sess.run(train_step, feed_dict={data: training_puzzles, y_: training_solutions})
    if i % 100 == 0:
        print("Batch {} complete".format(i))

print("Finished")

correct_prediction = tf.equal(
    tf.argmax(
        tf.reshape(y, (-1, 4, 4, 4)),
        2),
    tf.argmax(
        tf.reshape(y_, (-1, 4, 4, 4)),
        2))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
# accuracy = sess.run(accuracy, feed_dict={data: test_puzzles, y_: test_solutions})
W = sess.run(W1)
b = sess.run(b1)


def softmax(x):
    scoreMatExp = np.exp(x)
    return scoreMatExp / np.sum(scoreMatExp)


def expand(x):
    x = x.copy()
    i = 0
    while i < len(x):
        if np.sum(x[i:i+4]) == 1:
            x[i:i+4] = 1
        i += 4
    return x


incorrect = 0
for i in range(len(test_puzzles)):
    test_puzzle = test_puzzles[i]
    guess = test_puzzle.copy()
    while np.sum(guess) < 4**2:
        y = np.matmul(guess, W) + b
        y = softmax(y)
        y -= expand(guess)
        best_guess = np.argmax(y)
        guess[best_guess] = 1

    if not np.array_equal(
            convert_to_normal(guess.reshape((4, 4, 4))),
            convert_to_normal(test_solutions[i].reshape(4, 4, 4))):
        incorrect += 1
        if incorrect > 4:
            break
        print("Puzzle #{}".format(i))
        print("Board:\n", convert_to_normal(test_puzzles[i].reshape((4, 4, 4)), ones=True))
        print("Guess:\n", convert_to_normal(guess.reshape((4, 4, 4))))
        print("Answer:\n", convert_to_normal(test_solutions[i].reshape((4, 4, 4))))

print("Accuracy = ", accuracy)
