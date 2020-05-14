# -*- coding:utf-8 -*-
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt #Python的2D绘图库

plotdata = { "batchsize":[], "loss":[] } #待绘制的内容：批处理次数；损失值

def moving_average(a, w=10): #接收plotdata["loss"]
    if len(a) < w: #如果误差少于10个
        return a[:] #返回全部误差
    return [val if idx < w else sum(a[(idx-w):idx])/w for idx, val in enumerate(a)] 
    #idx是序号，val是值。如果当前误差值的序号<10，直接返回误差值，否则取序号10后面全部的误差值的和，除以10

#生成模拟数据
train_X = np.linspace(-1, 1, 100) #输入值，tuple类型, (100, 1)
train_Y = 2 * train_X + np.random.randn(*train_X.shape) * 0.3 #期望输出值，加入噪声
#显示模拟数据点
plt.plot(train_X, train_Y, 'ro', label='Original data')
plt.legend()
plt.show()

# 创建模型
X = tf.placeholder("float") #占位符，输入值
Y = tf.placeholder("float") #占位符，期望输出值
W = tf.Variable(tf.random_normal([1]), name="weight") #符合正态分布的权值，(1, 1)
b = tf.Variable(tf.zeros([1]), name="bias") #0值偏置，(1, 1)
# 前向结构
z = tf.multiply(X, W)+ b #实际输出值

#反向优化
cost =tf.reduce_mean(tf.square(Y - z)) #均方差函数，误差
learning_rate = 0.01 #学习率
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost) #保持误差最小的情况下，以当前学习率进行梯度下降训练

init = tf.global_variables_initializer() #初始化全局变量
training_epochs = 20 #训练次数
display_step = 2 #每2次输出1次

with tf.Session() as sess:
    sess.run(init)
    
    for epoch in range(training_epochs): #取20次训练中的每一次
        for (x, y) in zip(train_X, train_Y): #打包，100次
            sess.run(optimizer, feed_dict={X: x, Y: y}) #绘制实际输入值和输出值的对应关系，并做反向训练，计算并逼近w和b

        #显示训练中的详细信息
        if epoch % display_step == 0:
            loss = sess.run(cost, feed_dict={X: train_X, Y:train_Y}) #取100个样本的均方差，作为当次训练的误差值，w和b取当次训练的逼近值
            print ("Epoch:", epoch+1, "cost=", loss,"W=", sess.run(W), "b=", sess.run(b))
            if not (loss == "NA" ): #如果误差值不够小，就把当前的次数和误差值添加列表
                plotdata["batchsize"].append(epoch)
                plotdata["loss"].append(loss)

    print (" Finished!")
    print ("cost=", sess.run(cost, feed_dict={X: train_X, Y: train_Y}), "W=", sess.run(W), "b=", sess.run(b))

    #图形显示
    plt.plot(train_X, train_Y, 'ro', label='Original data')
    plt.plot(train_X, sess.run(W) * train_X + sess.run(b), label='Fitted line')
    plt.legend()
    plt.show()
    
    plotdata["avgloss"] = moving_average(plotdata["loss"])
    plt.figure(1)
    plt.subplot(211)
    plt.plot(plotdata["batchsize"], plotdata["avgloss"], 'b--')  #绘制每次训练对应的误差值
    plt.xlabel('Minibatch number')
    plt.ylabel('Loss')
    plt.title('Minibatch run vs. Training loss')     
    plt.show()

    print ("x=0.2，z=", sess.run(z, feed_dict={X: 0.2}))
