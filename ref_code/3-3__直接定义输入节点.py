# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

train_X =np.float32( np.linspace(-1, 1, 100))
train_Y = 2 * train_X + np.random.randn(*train_X.shape) * 0.3
plt.plot(train_X, train_Y, 'ro', label='Original data')
plt.legend()
plt.show()

W = tf.Variable(tf.random_normal([1]), name="weight") #没有占位符预设了，直接输入数据
b = tf.Variable(tf.zeros([1]), name="bias")
z = tf.multiply(W, train_X)+ b

cost =tf.reduce_mean( tf.square(train_Y - z))
learning_rate = 0.01
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

init = tf.global_variables_initializer()
training_epochs = 20
display_step = 2

with tf.Session() as sess:
    sess.run(init)

    for epoch in range(training_epochs):
        for (x, y) in zip(train_X, train_Y):
            sess.run(optimizer)

        if epoch % display_step == 0:
            loss = sess.run(cost)
            print ("Epoch:", epoch+1, "cost=", loss,"W=", sess.run(W), "b=", sess.run(b))

    print (" Finished!")
    print ("cost=", sess.run(cost), "W=", sess.run(W), "b=", sess.run(b))
