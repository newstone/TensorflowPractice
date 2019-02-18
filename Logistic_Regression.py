import tensorflow as tf
import numpy as np
import pandas as pd

tf.set_random_seed(777)
df = pd.read_csv('heart.csv')
#xy = np.genfromtxt('heart.csv', delimiter=',', dtype=np.float32,  encoding='UTF8')[:,:]

msk = np.random.rand(len(df)) < 0.8

train_data = df[msk]
test_data = df[~msk]

x_train_data = train_data.iloc[:, 0:-1].values
y_train_data = train_data['target'].values
y_train_data = y_train_data.reshape([-1, 1])

x_train_data[:,0] = (x_train_data[:,0] - x_train_data[:,0].mean())/x_train_data[:,0].std()
x_train_data[:,3] = (x_train_data[:,3] - x_train_data[:,3].mean())/x_train_data[:,3].std()
x_train_data[:,4] = (x_train_data[:,4] - x_train_data[:,4].mean())/x_train_data[:,4].std()
x_train_data[:,7] = (x_train_data[:,7] - x_train_data[:,7].mean())/x_train_data[:,7].std()

x_test_data = test_data.iloc[:, 0:-1].values
y_test_data = test_data['target'].values
y_test_data = y_test_data.reshape([-1, 1])

x_test_data[:,0] = (x_test_data[:,0] - x_test_data[:,0].mean())/x_test_data[:,0].std()
x_test_data[:,3] = (x_test_data[:,3] - x_test_data[:,3].mean())/x_test_data[:,3].std()
x_test_data[:,4] = (x_test_data[:,4] - x_test_data[:,4].mean())/x_test_data[:,4].std()
x_test_data[:,7] = (x_test_data[:,7] - x_test_data[:,7].mean())/x_test_data[:,7].std()

X = tf.placeholder(tf.float32, shape=[None, 13])
Y = tf.placeholder(tf.float32, shape=[None, 1])

W1 = tf.get_variable("W1", shape=[13,12], initializer=tf.contrib.layers.xavier_initializer())
b1 = tf.Variable(tf.random_normal([12]))
L1 = tf.matmul(X, W1) + b1

W2 = tf.get_variable("W2", shape=[12, 12], initializer=tf.contrib.layers.xavier_initializer())
b2 = tf.Variable(tf.random_normal([12]))
L2 = tf.matmul(L1, W2) + b2

W3 = tf.get_variable("W3", shape=[12,8], initializer=tf.contrib.layers.xavier_initializer())
b3 = tf.Variable(tf.random_normal([8]))
L3 = tf.matmul(L2, W3) + b3

W4 = tf.get_variable("W4", shape=[8, 8], initializer=tf.contrib.layers.xavier_initializer())
b4 = tf.Variable(tf.random_normal([8]))
L4 = tf.matmul(L3, W4) + b4

W5 = tf.get_variable("W5", shape=[8, 8], initializer=tf.contrib.layers.xavier_initializer())
b5 = tf.Variable(tf.random_normal([8]))
L5 = tf.matmul(L4, W5) + b5

W6 = tf.get_variable("W6", shape=[8, 4], initializer=tf.contrib.layers.xavier_initializer())
b6 = tf.Variable(tf.random_normal([4]))
L6 = tf.matmul(L5, W6) + b6

W7 = tf.get_variable("W7", shape=[4, 4], initializer=tf.contrib.layers.xavier_initializer())
b7 = tf.Variable(tf.random_normal([4]))
L7 = tf.matmul(L6, W7) + b7

W8 = tf.get_variable("W8", shape=[4, 4], initializer=tf.contrib.layers.xavier_initializer())
b8 = tf.Variable(tf.random_normal([4]))
L8 = tf.matmul(L7, W8) + b8

W9 = tf.get_variable("W9", shape=[4, 4], initializer=tf.contrib.layers.xavier_initializer())
b9 = tf.Variable(tf.random_normal([4]))
L9 = tf.matmul(L8, W9) + b9

W10 = tf.Variable(tf.random_normal([4, 1]))
b10 = tf.Variable(tf.random_normal([1]))
hypothesis =  tf.sigmoid(tf.matmul(L9, W10) + b10)

#cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1 - Y) * tf.log(1 - hypothesis))
cost = tf.nn.sigmoid_cross_entropy_with_logits(logits=hypothesis, labels = Y)

optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)

predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float32))

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(3001):
    _, cost_val = sess.run([optimizer, cost], feed_dict={X: x_train_data, Y: y_train_data})
    if step % 100 == 0:
        print("Step:" ,step, "\tCost:", cost_val[0])

_, c, a = sess.run([hypothesis, predicted, accuracy], feed_dict={X: x_test_data, Y: y_test_data})

for i in range(c.shape[0]):
    print("\nHypothesis: ", c[i], "\tCorrect (Y): ", y_test_data[i], "\t ", y_test_data[i] == c[i])

print("\nAccuracy: ", a)
