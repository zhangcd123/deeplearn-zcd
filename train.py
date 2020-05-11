

import tensorflow as tf
import tensorflow.contrib.slim as slim
import os

from nets import lenet5
from util import train_util
from datasets import minist
import numpy as np

import argparse

# tf.train.Optimizer
# tf.train.GradientDescentOptimizer
# tf.train.AdadeltaOptimizer
# tf.train.AdagtadOptimizer
# tf.train.AdagradDAOptimizer
# tf.train.MomentumOptimizer
# tf.train.AdamOptimizer
# tf.train.FtrlOptimizer
# tf.train.ProximalGradientDescentOptimizer
# tf.train.ProximalAdagradOptimizer
# tf.train.RMSProOptimizer
#TensorFlow API tf.train文档
parser = argparse.ArgumentParser()
parser.add_argument('--num_epochs', type=int, default=6, help='Number of epochs to train for')
parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate for train')
parser.add_argument('--classes', type=int, default=10, help='classes')
parser.add_argument('--batch_size', type=int, default=4, help='batch size for train')
parser.add_argument('--logs_dir', type=str, default="./logs", help='path to save model')
parser.add_argument('--dataset_dir',type=str,default="./datasets/data_minist",help='path for image and labels')
parser.add_argument('--img_size',type=int,default=28,help='path for image and labels')
args = parser.parse_args()

def train():

    data_len = 1000

    #定义网络输入
    net_input = tf.placeholder(tf.float32, [None,args.img_size*args.img_size], name='inputs')
    net_output = tf.placeholder(tf.int32, [None,args.classes], name='lable_data')
    # net_input = tf.placeholder(tf.float32,shape=[None,args.crop_height,args.crop_width,3])
    # net_output = tf.placeholder(tf.float32,shape=[None,args.crop_height,args.crop_width,num_classes])
    is_training = tf.placeholder(tf.bool, name='is_training')

    #build net
    input_data = tf.reshape(net_input,[-1,args.img_size,args.img_size,1])
    logits = lenet5.build_net(input_data,classes=args.classes)
    print(logits.shape)
    #compute loss
    train_loss = train_util.losses(logits=logits,labels=net_output)

    #定义优化算子及衰减优化算子
    train_op = train_util.training(train_loss,args.learning_rate,train_loss)

    #测评
    # train_acc = train_util.evalution(logits,labels=net_output)

    summary_op = tf.summary.merge_all()

    sess = tf.Session()
    train_writer = tf.summary.FileWriter(args.logs_dir,sess.graph)
    saver = tf.train.Saver()

    # num_iters = int(data_len/batch_size)

    train_img_batches,train_label_batches,test_img_batches,test_label_batches = minist.prepare_data(args.dataset_dir,args.batch_size)
    num_iters = train_img_batches.shape[0]
    sess.run(tf.global_variables_initializer())

    #train
    for step in range(args.num_epochs):
        for n in range(num_iters):

            with tf.device("/cpu:0"):
                
                input_img = tf.reshape(train_img_batches[n],[train_img_batches[n].shape[0],28,28,1])
                labels_data = tf.reshape(train_label_batches[n],[train_label_batches[n].shape[0],args.classes])
                input_img, labels_data = sess.run([input_img,labels_data])
                # input_img = np.array(input_img)
                # labels_data = np.array(labels_data)


                _,tloss=sess.run([train_op,train_loss]
                                            ,feed_dict={input_data:input_img
                                                        ,net_output:labels_data})
                print('iter %d,train loss=%.2f'%(n,tloss))

        if setp%2 == 0:
            print('Step %d,train loss=%.2f'%(step,tloss))
            # train_writer.add_summary(summary_str,step)

        #保存模型
        if step % 2000 == 0 or (step+1) == MAX_STEP:
            checkpoint_path = os.path.join(args.logs_dir,"model.ckpt")
            saver.save(sess,checkpoint_path,global_step=step)
    
if __name__ == "__main__":
    train()
