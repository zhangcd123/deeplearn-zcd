
import os
import numpy as np
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets

def load_data(dataset_dir):
    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)
    mnist = read_data_sets(dataset_dir, one_hot=True,source_url="http://yann.lecun.com/exdb/mnist/")
    train_img = mnist.train.images
    train_label = mnist.train.labels
    test_img = mnist.test.images
    test_label = mnist.test.labels

    return train_img,train_label,test_img,test_label

def generate_data(batch_size,img,label):
    length = img.shape[0]

    batch_length = int(length/batch_size)
    img_batches = []
    label_batches = []
    for i in range(0,length,batch_size):
        start = i
        end  = i+batch_size
        if end <= length:
            img_batches.append(img[start:end,:])
            label_batches.append(label[start:end,:])
    img_batches = np.array(img_batches)
    label_batches = np.array(label_batches)
    return img_batches,label_batches

def prepare_data(dataset_dir,batch_size):
    train_img,train_label,test_img,test_label = load_data(dataset_dir)
    train_img_batches,train_label_batches = generate_data(batch_size,train_img,train_label)
    test_img_batches,test_label_batches = generate_data(batch_size,test_img,test_label)

    print(train_img_batches.shape,train_label_batches.shape)
    return train_img_batches,train_label_batches,test_img_batches,test_label_batches
    

def test():
    
    train_img,train_label,test_img,test_label = load_data("minist")
    prepare_data("data_minist",4)
    # prepare_data(4,test_img,test_label)

if __name__ == "__main__":
    test()
