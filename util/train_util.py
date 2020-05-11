import tensorflow as tf
import tensorflow.contrib.slim as slim
# import numpy as np

def losses(logits, labels):
    #定义损失函数
    print(logits.shape)
    logits = tf.reshape(logits,[-1,logits.shape[-1]])
    labels = tf.reshape(labels,[-1,labels.shape[-1]])
    loss = slim.losses.softmax_cross_entropy(logits,labels)
    loss = tf.reduce_mean(loss)
    return loss

def evalution(logits, labels):
    # with tf.variable_scope('accuracy') as scope:
    logits = slim.softmax(logits)
    classes = tf.argmax(logits,axis=1)
    logits = tf.reshape(logits,[-1,logits.shape[-1]])
    labels = tf.reshape(labels,[-1,labels.shape[-1]])
    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.cast(classes,tf.int32),labels),tf.float32))

    return accuracy

def training(logits,learning_rate,loss):
    global_step = tf.Variable(0,name = "global_step",trainable=False)
    #定义优化算子及衰减优化算子
    optimizer = tf.train.RMSPropOptimizer(learning_rate=0.0001)
    train_op = optimizer.minimize(loss, global_step=global_step)
    return train_op
