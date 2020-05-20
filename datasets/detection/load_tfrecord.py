"""
Description:load tfrecord for train and val
Author:cara-zhang
"""
import tensorflow as tf
import os
from datasets import util
from datasets.detection import data_description
slim = tf.contrib.slim

class LoadTfrecord():
    def __init__(self,dataset_dir,dataset,split_name):
        self.dataset_dir = dataset_dir
        self.dataset = dataset
        self.split_name = split_name

    def load_tfrecord(self):
        reader = tf.TFRecordReader

        keys_to_features = {
            'image/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
            'image/format': tf.FixedLenFeature((), tf.string, default_value='jpeg'),
            'image/height': tf.FixedLenFeature([1], tf.int64),
            'image/width': tf.FixedLenFeature([1], tf.int64),
            'image/channel': tf.FixedLenFeature([1], tf.int64),
            'image/shape': tf.VarLenFeature(dtype=tf.int64),
            'image/object/bbox/xmin': tf.VarLenFeature(dtype=tf.float32),
            'image/object/bbox/ymin': tf.VarLenFeature(dtype=tf.float32),
            'image/object/bbox/xmax': tf.VarLenFeature(dtype=tf.float32),
            'image/object/bbox/ymax': tf.VarLenFeature(dtype=tf.float32),
            'image/object/bbox/label': tf.VarLenFeature(dtype=tf.int64)
        }

        items_to_handlers = {
            'image': slim.tfexample_decoder.Image('image/encoded', 'image/format'),
            'shape': slim.tfexample_decoder.Tensor('image/shape'),
            'bboxes': slim.tfexample_decoder.BoundingBox(
                    ['xmin', 'ymin', 'xmax', 'ymax'], 'image/object/bbox/'),
            'labels': slim.tfexample_decoder.Tensor('image/object/bbox/label')
        }

        decoder = slim.tfexample_decoder.TFExampleDecoder(
            keys_to_features, items_to_handlers)

        labels_to_names = None
        label_map_file = os.path.join(self.dataset_dir,"label_map.txt")
        labels_to_names = util.label_to_class_name(label_map_file)

        tfrecordpath = os.path.join(self.dataset_dir,"/tfrecord/"+self.dataset+"_"+self.split_name+".tfrecord")

        items_to_descriptions = {
            'image': 'An image with shape image_shape.',
            'bboxex': 'label roi with:xmin,ymin,xmax,ymax',
            'label': 'A single integer between 0 and 9.'}
        samples = data_description._DATASETS_INFORMATION[self.dataset].splits_to_sizes[self.split_name]
        num_classes = data_description._DATASETS_INFORMATION[self.dataset].num_classes
        return slim.dataset.Dataset(
                data_sources=tfrecordpath,
                reader=reader,
                decoder=decoder,
                num_samples=samples,
                items_to_descriptions=items_to_descriptions,
                num_classes=num_classes,
                labels_to_names=labels_to_names)