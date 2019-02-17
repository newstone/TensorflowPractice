import tensorflow as tf
import numpy as np

tf.set_random_seed(777)

xy = np.genfromtxt('StudentsPerformance.csv', delimiter=',', dtype=np.float32,  encoding='UTF8')[:,:]

x_data = xy[1:,0:-1]
y_data = xy[1:, [-1]]

x_data[:,5] = (x_data[:,5] - x_data[:,5].mean())/x_data[:,5].std()
x_data[:,6] = (x_data[:,6] - x_data[:,6].mean())/x_data[:,6].std()

y_std = y_data[:].std()
y_mean = y_data[:].mean()

y_data[:] = (y_data[:] - y_data[:].mean())/y_data[:].std()

x_train = x_data[0:700, :]
y_train = y_data[0:700]

x_test = x_data[700:, :]
y_test = y_data[700:]


X = tf.placeholder(tf.float32, shape=[None, 7])
Y = tf.placeholder(tf.float32, shape=[None, 1])


W1 = tf.get_variable("W1", shape=[7,7], initializer=tf.contrib.layers.xavier_initializer())
b1 = tf.Variable(tf.random_normal([7]))
L1 = tf.matmul(X, W1) + b1

W2 = tf.get_variable("W2", shape=[7, 7], initializer=tf.contrib.layers.xavier_initializer())
b2 = tf.Variable(tf.random_normal([7]))
L2 = tf.matmul(L1, W2) + b2

W3 = tf.get_variable("W3", shape=[7,4], initializer=tf.contrib.layers.xavier_initializer())
b3 = tf.Variable(tf.random_normal([4]))
L3 = tf.matmul(L2, W3) + b3

W4 = tf.get_variable("W4", shape=[4, 4], initializer=tf.contrib.layers.xavier_initializer())
b4 = tf.Variable(tf.random_normal([4]))
L4 = tf.matmul(L3, W4) + b4

W16 = tf.get_variable("W16", shape=[4, 8], initializer=tf.contrib.layers.xavier_initializer())
b16 = tf.Variable(tf.random_normal([8]))
L16 = tf.matmul(L4, W16) + b16


W17 = tf.get_variable("W17", shape=[8, 8], initializer=tf.contrib.layers.xavier_initializer())
b17 = tf.Variable(tf.random_normal([8]))
L17 = tf.matmul(L16, W17) + b17


W18 = tf.get_variable("W18", shape=[8, 4], initializer=tf.contrib.layers.xavier_initializer())
b18 = tf.Variable(tf.random_normal([4]))
L18 = tf.matmul(L17, W18) + b18


W19 = tf.get_variable("W19", shape=[4, 4], initializer=tf.contrib.layers.xavier_initializer())
b19 = tf.Variable(tf.random_normal([4]))
L19 = tf.matmul(L18, W19) + b19

W20 = tf.Variable(tf.random_normal([4, 1]))
b20 = tf.Variable(tf.random_normal([1]))
hypothesis =  tf.matmul(L19, W20) + b20

cost = tf.reduce_sum(tf.square(hypothesis - Y))

optimizer = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(3001):
    _, cost_val = sess.run([optimizer, cost], feed_dict={X: x_train, Y: y_train})
    if step % 100 == 0:
        print("Step:" ,step, "\tCost:", cost_val)

h, yy = sess.run([hypothesis, Y], feed_dict={X:x_test, Y:y_test})

h[:] *= y_std
yy[:] *= y_std
h[:] += y_mean
yy[:] += y_mean

result = abs(h[:]-yy[:])

max = 0
min = 1e+10

for i in range(h.shape[0]):
    if max < result[i]:
        max = result[i]
    elif min > result[i]:
        min = result[i]

    print("예측한 값: ", h[i], "   실제 값: ",yy[i], "    두 값의 차: ", result[i])

print("최소값: ", min, "최대값: ", max, "오차 평균: ", result.mean())
